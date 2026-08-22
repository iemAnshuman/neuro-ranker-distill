# NeuroRanker Distill

NeuroRanker is a reproducible teacher-to-student reranking experiment on the
MS MARCO passage-ranking task. A pinned cross-encoder scores positive/negative
training triplets. A mean-pooled bi-encoder learns the teacher's score margin,
then reranks the official candidate set. The same student is exported to FP32
and dynamically quantized INT8 ONNX, with numerical validation against PyTorch.

This repository does not claim a fixed accuracy or speedup. Measured results,
hardware, dataset profile, candidate depth, software versions, and artifact
digests belong in [RESULTS.md](RESULTS.md) after a verified run.

## What is reproducible

- model IDs are pinned to immutable Hugging Face revisions;
- dataset preparation requires explicit acceptance of Microsoft's MS MARCO
  terms and writes SHA-256 manifests;
- `fixture`, `smoke`, and `artifact` profiles have fixed record/query limits;
- training fixes Python, NumPy, and PyTorch seeds and records its inputs;
- evaluation reports candidate-set MRR, nDCG, recall, and a TREC-format run;
- ONNX export fails unless FP32 and INT8 numerical checks pass;
- Rostam jobs capture Git, Slurm, GPU, module, package, and lock provenance.

## Quick local check

Python 3.10–3.13 is supported. The fixture profile is synthetic and does not
download MS MARCO.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --require-hashes -r requirements-runtime.lock
python -m pip install --no-deps -e .

neuro-ranker prepare --config configs/fixture.json --output data/fixture
neuro-ranker verify-data --data-dir data/fixture
pytest
```

The end-to-end fixture still downloads the pinned teacher and student models:

```bash
neuro-ranker run \
  --config configs/fixture.json \
  --work-dir runs/fixture \
  --repo-root .
```

## MS MARCO profiles

| Profile | Train triplets | Eval queries | Candidates/query | Purpose |
|---|---:|---:|---:|---|
| `fixture` | 8 synthetic | 2 synthetic | 4 | Offline code and test checks |
| `smoke` | 512 | 32 | 100 | Rostam end-to-end validation |
| `artifact` | 100,000 | 6,980 | 1,000 | Reported research run |

MS MARCO is provided for non-commercial research under Microsoft's terms.
Review those terms before running:

```bash
neuro-ranker prepare \
  --config configs/smoke.json \
  --output data/smoke \
  --accept-msmarco-terms
```

The preparation command streams the official training-triplet and top-1000 dev
archives and does not commit licensed data.

## Rostam

Rostam setup and Slurm commands are in [REPRODUCIBILITY.md](REPRODUCIBILITY.md).
The short path is:

```bash
bash scripts/bootstrap_rostam.sh
MSMARCO_TERMS_ACCEPTED=1 sbatch slurm/smoke.sbatch
```

Do not report results from the smoke profile as benchmark results. The artifact
profile and its generated `run_manifest.json` are the evidence for RESULTS.md.

## Service

After a successful export:

```bash
neuro-ranker serve \
  --model runs/fixture/export/student.int8.onnx \
  --tokenizer runs/fixture/export \
  --host 127.0.0.1 \
  --port 8000
```

`GET /health` reports model readiness. `POST /rerank` accepts `query`, `texts`,
and `top_k`. The service loads the ONNX model during application startup, so
unit tests can inject a deterministic fake encoder without model downloads.

## License

Code is MIT licensed. MS MARCO and downloaded model weights retain their own
terms and licenses; they are not covered by this repository's MIT license.

