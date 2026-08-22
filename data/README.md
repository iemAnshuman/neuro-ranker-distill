# Data directory

MS MARCO data is not distributed with this repository. Microsoft makes the
dataset available for non-commercial research under its own terms:

https://microsoft.github.io/msmarco/

The preparation command requires explicit acknowledgement of those terms and
records source URLs, profile limits, record counts, and SHA-256 digests in a
manifest. Generated files live under `data/<profile>/` and are ignored by Git.

For an offline pipeline check that does not use MS MARCO, generate the synthetic
fixture profile:

```bash
neuro-ranker prepare --config configs/fixture.json --output data/fixture
```

For MS MARCO smoke or artifact profiles, pass `--accept-msmarco-terms` only
after reviewing and accepting Microsoft's terms:

```bash
neuro-ranker prepare \
  --config configs/smoke.json \
  --output data/smoke \
  --accept-msmarco-terms
```
