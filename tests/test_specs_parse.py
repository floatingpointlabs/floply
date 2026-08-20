"""Parsing ec2:DescribeInstanceTypes into hardware specs, and the discovery filter.

DescribeInstanceTypes has no gpu-info filter, so every accelerator type comes
back and must be sorted out client-side. Getting the exclusions wrong would put
Inferentia or AMD parts into a catalog that assumes NVIDIA throughput.
"""

import boto3
import pytest
from botocore.stub import Stubber

from src.cost_modelling.pricing.aws_client import (
    fetch_instance_hardware,
    fetch_offered_instance_types,
    parse_instance_type,
)
from src.cost_modelling.pricing.settings import build_settings


def gpu_instance(
    instance_type="p5.48xlarge", gpu="H100", count=8, gpu_mib=81920,
    manufacturer="NVIDIA", vcpus=192, mem_mib=2097152, cards=None,
    network_performance="3200 Gigabit",
):
    raw = {
        "InstanceType": instance_type,
        "CurrentGeneration": True,
        "VCpuInfo": {"DefaultVCpus": vcpus},
        "MemoryInfo": {"SizeInMiB": mem_mib},
        "NetworkInfo": {"NetworkPerformance": network_performance},
        "GpuInfo": {
            "Gpus": [{
                "Name": gpu, "Manufacturer": manufacturer, "Count": count,
                "MemoryInfo": {"SizeInMiB": gpu_mib},
            }],
            "TotalGpuMemoryInMiB": gpu_mib * count,
        },
    }
    if cards is not None:
        raw["NetworkInfo"]["NetworkCards"] = cards
    return raw


# -- Field mapping ----------------------------------------------------------

def test_parses_a_gpu_instance():
    hardware = parse_instance_type(gpu_instance())
    assert hardware.instance_type == "p5.48xlarge"
    assert hardware.gpu == "H100"
    assert hardware.gpu_count == 8
    assert hardware.memory_per_gpu == 80          # 81920 MiB -> GB
    assert hardware.total_gpu_memory == 640
    assert hardware.vcpus == 192
    assert hardware.system_memory == 2048
    assert hardware.network_bandwidth == 3200


def test_display_name_matches_the_previous_hand_written_format():
    """The old aws.yaml display_name was 'p4d.24xlarge (8x A100 40GB)'."""
    hardware = parse_instance_type(
        gpu_instance("p4d.24xlarge", gpu="A100", count=8, gpu_mib=40960))
    assert hardware.display_name == "p4d.24xlarge (8x A100 40GB)"


def test_gpu_count_sums_multiple_gpu_groups():
    raw = gpu_instance()
    raw["GpuInfo"]["Gpus"].append(
        {"Name": "H100", "Manufacturer": "NVIDIA", "Count": 4,
         "MemoryInfo": {"SizeInMiB": 81920}})
    assert parse_instance_type(raw).gpu_count == 12


def test_total_gpu_memory_is_derived_when_absent():
    raw = gpu_instance()
    del raw["GpuInfo"]["TotalGpuMemoryInMiB"]
    assert parse_instance_type(raw).total_gpu_memory == 640


# -- Exclusions -------------------------------------------------------------

def test_non_gpu_instance_is_skipped():
    assert parse_instance_type({"InstanceType": "m5.large", "VCpuInfo": {"DefaultVCpus": 2}}) is None


def test_amd_gpu_is_skipped():
    """g4ad carries Radeon Pro V520, not an NVIDIA part."""
    assert parse_instance_type(
        gpu_instance("g4ad.xlarge", gpu="Radeon Pro V520", manufacturer="AMD")) is None


def test_habana_accelerator_is_skipped():
    assert parse_instance_type(
        gpu_instance("dl1.24xlarge", gpu="Gaudi HL-205", manufacturer="Habana")) is None


def test_inferentia_is_skipped():
    """Inf/Trn report InferenceAcceleratorInfo, so they have no GpuInfo at all."""
    raw = {
        "InstanceType": "inf2.xlarge",
        "InferenceAcceleratorInfo": {"Accelerators": [{"Name": "Inferentia2", "Count": 1}]},
    }
    assert parse_instance_type(raw) is None


def test_zero_gpu_count_is_skipped():
    assert parse_instance_type(gpu_instance(count=0)) is None


# -- Network bandwidth ------------------------------------------------------

def test_prefers_structured_peak_bandwidth():
    hardware = parse_instance_type(gpu_instance(
        cards=[{"PeakBandwidthInGbps": 400.0} for _ in range(8)],
        network_performance="wrong",
    ))
    assert hardware.network_bandwidth == 3200


def test_falls_back_to_baseline_bandwidth():
    hardware = parse_instance_type(gpu_instance(
        cards=[{"BaselineBandwidthInGbps": 100.0}], network_performance="wrong"))
    assert hardware.network_bandwidth == 100


@pytest.mark.parametrize("text, expected", [
    ("400 Gigabit", 400),
    ("3200 Gigabit", 3200),
    ("Up to 25 Gigabit", 25),
    ("100 Gigabit", 100),
    ("Very Low", 0),
])
def test_parses_the_free_text_network_performance(text, expected):
    assert parse_instance_type(gpu_instance(network_performance=text)).network_bandwidth == expected


# -- Discovery filtering ----------------------------------------------------

@pytest.fixture
def ec2_client():
    return boto3.client(
        "ec2", region_name="us-east-1",
        aws_access_key_id="test", aws_secret_access_key="test",
    )


def _stub_types(stub, instances):
    stub.add_response("describe_instance_types", {"InstanceTypes": instances}, None)


def test_discovery_skips_small_and_off_family_instances(ec2_client, monkeypatch):
    """g4dn.xlarge in a large-model training estimator is a UX regression."""
    monkeypatch.setattr(
        "src.cost_modelling.pricing.aws_client._client", lambda *a, **k: ec2_client)

    with Stubber(ec2_client) as stub:
        _stub_types(stub, [
            gpu_instance("p5.48xlarge", gpu="H100", count=8),
            gpu_instance("p4d.24xlarge", gpu="A100", count=8, gpu_mib=40960),
            gpu_instance("g5.xlarge", gpu="A10G", count=1, gpu_mib=24576),
            gpu_instance("p3.2xlarge", gpu="V100", count=1, gpu_mib=16384),
            gpu_instance("m5.large", manufacturer="AMD"),
        ])
        snapshot = fetch_instance_hardware(build_settings())

    assert set(snapshot.instances) == {"p5.48xlarge", "p4d.24xlarge"}
    assert any("allowlist" in w for w in snapshot.warnings)      # g5 dropped
    assert any("fewer than" in w for w in snapshot.warnings)     # p3.2xlarge dropped


def test_dropped_instances_are_reported_not_silently_truncated(ec2_client, monkeypatch):
    monkeypatch.setattr(
        "src.cost_modelling.pricing.aws_client._client", lambda *a, **k: ec2_client)

    with Stubber(ec2_client) as stub:
        _stub_types(stub, [gpu_instance("g6.xlarge", gpu="L4", count=1, gpu_mib=24576)])
        snapshot = fetch_instance_hardware(build_settings())

    assert snapshot.instances == {}
    assert snapshot.warnings, "silently dropping every instance would look like an empty catalog"


def test_discovery_disabled_keeps_everything_nvidia(ec2_client, monkeypatch):
    monkeypatch.setenv("FLOPLY_DISCOVER_INSTANCES", "false")
    monkeypatch.setattr(
        "src.cost_modelling.pricing.aws_client._client", lambda *a, **k: ec2_client)

    with Stubber(ec2_client) as stub:
        _stub_types(stub, [
            gpu_instance("g5.xlarge", gpu="A10G", count=1, gpu_mib=24576),
            gpu_instance("p5.48xlarge", gpu="H100", count=8),
        ])
        snapshot = fetch_instance_hardware(build_settings())

    assert set(snapshot.instances) == {"g5.xlarge", "p5.48xlarge"}


# -- Regional availability --------------------------------------------------

def test_offered_instance_types(ec2_client, monkeypatch):
    monkeypatch.setattr(
        "src.cost_modelling.pricing.aws_client._client", lambda *a, **k: ec2_client)

    with Stubber(ec2_client) as stub:
        stub.add_response(
            "describe_instance_type_offerings",
            {"InstanceTypeOfferings": [
                {"InstanceType": "p5.48xlarge", "LocationType": "region", "Location": "us-east-1"},
                {"InstanceType": "p4d.24xlarge", "LocationType": "region", "Location": "us-east-1"},
            ]},
            {"LocationType": "region",
             "Filters": [{"Name": "location", "Values": ["us-east-1"]}]},
        )
        offered = fetch_offered_instance_types("us-east-1", build_settings())

    assert offered == {"p5.48xlarge", "p4d.24xlarge"}
