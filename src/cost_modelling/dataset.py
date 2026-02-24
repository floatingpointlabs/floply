"""Modality-specific token count and storage size estimation.

All conversion constants are documented here so they can be updated in one
place if tokeniser or encoding assumptions change.
"""

# -- Text ------------------------------------------------------------------

WORDS_TO_BPE_TOKENS = 1.3
"""Average BPE tokens per English word (GPT-2 / LLaMA tokeniser empirical average)."""

TEXT_BYTES_PER_TOKEN = 4
"""Bytes per token when stored as raw uint32 token IDs."""

# -- Image -----------------------------------------------------------------

JPEG_COMPRESSION_RATIO = 10
"""Approximate JPEG compression ratio for natural images (10:1 lossiness)."""

# -- Audio -----------------------------------------------------------------

AUDIO_BYTES_PER_SECOND = 32_000
"""Raw audio bytes per second: 16kHz × 16-bit mono = 32,000 bytes/s."""

ENCODEC_TOKENS_PER_SECOND = 75
"""Token rate for EnCodec 24kHz per codebook level."""

SOUNDSTREAM_TOKENS_PER_SECOND = 50
"""Token rate for SoundStream per codebook level."""

WHISPER_TOKENS_PER_SECOND = 50
"""Token rate for Whisper mel-spectrogram tokenisation."""


# -- Per-sample functions --------------------------------------------------

def tokens_per_text_sample(avg_seq_len_words: int) -> int:
    """Estimate BPE tokens for a text sample of given word count."""
    return int(avg_seq_len_words * WORDS_TO_BPE_TOKENS)


def bytes_per_text_sample(tokens: int) -> int:
    """Estimate raw storage bytes for a tokenised text sample."""
    return tokens * TEXT_BYTES_PER_TOKEN


def tokens_per_image_sample(resolution: int, patch_size: int) -> int:
    """Number of ViT patch tokens for a square image."""
    return (resolution // patch_size) ** 2


def bytes_per_image_sample(resolution: int) -> int:
    """Estimated JPEG-compressed bytes for a square RGB image."""
    return (resolution * resolution * 3) // JPEG_COMPRESSION_RATIO


def tokens_per_audio_sample(
    clip_duration: float,
    base_rate: int,
    num_codebooks: int = 1,
) -> int:
    """Estimate token count for an audio clip.

    Args:
        clip_duration: Clip length in seconds
        base_rate: Tokens per second per codebook level
        num_codebooks: Number of RVQ codebook levels (1 for Whisper)
    """
    return int(clip_duration * base_rate * num_codebooks)


def bytes_per_audio_sample(clip_duration: float) -> int:
    """Estimated raw PCM bytes for an audio clip (16kHz mono 16-bit)."""
    return int(clip_duration * AUDIO_BYTES_PER_SECOND)


def tokens_per_video_sample(
    duration: float,
    fps: int,
    resolution: int,
    patch_size: int,
) -> int:
    """Estimate token count for a video clip (spatial ViT patches × frames)."""
    tokens_per_frame = tokens_per_image_sample(resolution, patch_size)
    num_frames = int(duration * fps)
    return tokens_per_frame * num_frames


def bytes_per_video_sample(duration: float, fps: int, resolution: int) -> int:
    """Estimated JPEG-compressed bytes for a video clip (per-frame JPEG × frames)."""
    bytes_per_frame = bytes_per_image_sample(resolution)
    num_frames = int(duration * fps)
    return bytes_per_frame * num_frames
