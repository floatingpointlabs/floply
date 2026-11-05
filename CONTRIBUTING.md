# Contributing to Floply

Thanks for helping. Floply is a cost estimator, so the most valuable contributions are the
ones that make its numbers more right: a wrong estimate reported, a model or GPU added, a
formula corrected.

## Report a problem

Open a [GitHub issue](https://github.com/floatingpointlabs/floply/issues). For a number
that looks wrong, include:

- the page (Data needed, Budget optimizer or Project cost)
- the model, cluster, region and every input you set
- what Floply said, what you expected, and where your figure comes from (a datasheet, a
  paper, a real training run)

**Security issues:** don't open a public issue. Report them privately through
[GitHub's vulnerability reporting](https://github.com/floatingpointlabs/floply/security/advisories/new)
instead. Only the maintainers can see the report.

## Add a model or a GPU

Neither needs code.

- **A model** is one YAML file in `data/models/`. See [Adding a model](README.md#adding-a-model).
- **A GPU** is one entry in `data/gpu_hardware.yaml`. See [Adding a GPU](README.md#adding-a-gpu).

Link your sources in the pull request: the Hugging Face page or paper for a model, the
NVIDIA datasheet or a measurement for a GPU. We check every figure against its source
before merging, because a guessed throughput is exactly what Floply exists to avoid.

## Change the code

### Setup

Follow [Setup](README.md#setup). Use `FLOPLY_PRICING_FIXTURE=1` unless your change touches
live pricing. `pnpm install` also installs a pre-commit hook that runs Prettier on staged
files.

### Before you open a pull request

```bash
pnpm check      # types
pnpm test       # engine against the golden fixtures
pnpm test:e2e   # browser tests (first run: pnpm exec playwright install chromium)
```

All three must pass.

### Changing the maths

`fixtures/*.json` pins the output of every engine function, so `pnpm test` fails when an
estimate changes. That's intended. If the change is deliberate:

1. Run `pnpm gen-fixtures`.
2. Read the fixture diff and check that only the numbers you meant to move have moved.
3. In the pull request, explain why they moved and cite the source for the new formula
   or figure.

Never regenerate fixtures just to make a failing test pass.

### Pull requests

- Keep each pull request to one change.
- Say what changed and why. For a UI change, add a before and after screenshot.
- `lib/engine/` stays free of UI and server imports, so the maths can be tested on its own.

## Licence

Floply is licensed under [Apache 2.0](LICENSE). By opening a pull request, you agree that
your contribution is licensed under the same terms. There is no separate contributor
agreement to sign.
