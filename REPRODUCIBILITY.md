# Reproducing NeuroRanker on Rostam

## Scope and acceptance criteria

A run is complete only when all of these files exist and verify:

1. `data/manifest.json` with matching record counts, byte sizes, and SHA-256 digests.
2. `teacher_scores.manifest.json` naming the teacher revision and input/output digests.
3. `student/artifact.json` naming the student revision, seed, training digest, and losses.
4. `evaluation/metrics.json` and `evaluation/student.run.tsv`.
5. `export/export_report.json` with passing FP32 and INT8 numerical checks.
6. `provenance.json` with a clean Git SHA, GPU, Slurm, modules, packages, and lock digests.
7. `run_manifest.json`, which embeds the preceding evidence.

The fixture and smoke profiles prove execution. Only the `artifact` profile is
eligible for reported research metrics.

## Storage

Rostam home storage is small. Keep the checkout, virtual environment, caches,
datasets, models, and runs under `/work`:

```bash
export NEURORANKER_ROOT="$WORK/neuro-ranker-distill"
export NEURORANKER_VENV="$WORK/venvs/neuro-ranker-0.2"
export HF_HOME="$WORK/.cache/neuro-ranker/huggingface"
export TORCH_HOME="$WORK/.cache/neuro-ranker/torch"
```

The scripts use these paths and never install packages with `pip --user`.

## Environment setup

From the repository root:

```bash
bash scripts/bootstrap_rostam.sh
source "$WORK/venvs/neuro-ranker-0.2/bin/activate"
neuro-ranker doctor --repo-root . --output runs/setup-provenance.json
pytest
ruff check .
```

`requirements-runtime.lock` and `requirements-ci.lock` are generated with
hashes. Installation uses `--require-hashes`; changing a package or archive
requires regenerating and reviewing the lock.

## Container

The Apptainer definition uses the immutable manifest-list digest for the tested
`nvidia/cuda:12.8.1-cudnn-runtime-ubuntu24.04` base and the hashed runtime lock:

```bash
apptainer build neuro-ranker.sif containers/neuro-ranker.def
sha256sum neuro-ranker.sif
apptainer test --nv neuro-ranker.sif
```

Record the SIF digest in the run notes. Rostam execution currently uses the
locked virtual environment so source changes can be tested without rebuilding
the image; the container is the portable release environment.

## Data profiles

Microsoft's MS MARCO terms are at https://microsoft.github.io/msmarco/.
The batch scripts require `MSMARCO_TERMS_ACCEPTED=1`; this is an explicit user
acknowledgement, not an automatic acceptance by the software.

The preparation code streams these official resources:

- `triples.train.small.tar.gz` for query/positive/negative training records;
- `top1000.dev.tar.gz` for the official BM25 candidate order and passage text;
- `qrels.dev.tsv` for relevance labels.

The downloader checks each official file's reviewed length, ETag, last-modified
value, and (when supplied by Microsoft) Content-MD5 before reading it. Derived
JSONL is deterministic for a fixed profile and source content, and every
derived file is checked before training.

## Slurm runs

Fixture GPU-path validation:

```bash
sbatch slurm/fixture.sbatch
```

MS MARCO smoke validation:

```bash
MSMARCO_TERMS_ACCEPTED=1 sbatch slurm/smoke.sbatch
```

Artifact run:

```bash
MSMARCO_TERMS_ACCEPTED=1 sbatch slurm/artifact.sbatch
```

Each job targets Rostam's A100 partition, uses Python 3.12, sets deterministic
CUDA configuration, and writes to a unique directory under
`$WORK/neuro-ranker-runs/`. Rostam's current A100 partition does not advertise
a per-GPU GRES, so the scripts do not pass an invalid `--gres` request; the
application uses CUDA device 0 on the assigned node.

## Evaluation contract

The first-stage baseline is the order in MS MARCO's `top1000.dev` file. The
student reranks exactly the candidate depth named by the profile. The report
computes MRR@10, nDCG@10, and recall@10 over the labels present in that candidate
set and records the scope. It does not claim full-corpus retrieval quality.

The artifact profile uses 6,980 dev queries and 1,000 candidates per query,
matching the standard MS MARCO dev-small reranking setup. Preserve the generated
TREC run file so external evaluators can verify the ranking.

## Publishing results

Copy measurements to RESULTS.md only after checking:

```bash
neuro-ranker verify-data --data-dir "$RUN_DIR/data"
git status --short
sha256sum "$RUN_DIR/run_manifest.json"
```

Report the Git SHA, Slurm job ID, GPU, profile, candidate depth, model revisions,
lock digests, metrics, runtime, and run-manifest digest. State failed runs and
methodological limitations; do not reuse the old unverified 97%, 10×, or 8 ms claims.
