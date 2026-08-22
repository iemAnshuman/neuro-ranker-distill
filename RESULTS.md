# Results

No artifact-profile result has been verified yet.

## Synthetic fixture validation (not a benchmark)

The complete pipeline was run twice from clean commit
`5345602726bd6d8e46957036d88847406ee73d96` on Rostam A100 job IDs `186909`
and `186910`. Both jobs completed, and these core artifacts were byte-identical:

| Artifact | SHA-256 |
|---|---|
| prepared training JSONL | `f49a16a4a5aeca90963b349980e14bc52f3978a91ae695160f97a03648ddae05` |
| teacher scores | `b234aad577965cd0750658db084b996e9ea83113ed525a590be6f0e98cd02c39` |
| student weights | `251f9412c2a9b96e57ba7a975d39cbb906c6f379e32948a50d7f28e99dfa4f6e` |
| TREC run | `b772b02d7352d60fd856c34884649f912e47a2d88e3835d2b8ca306735c42c6f` |
| FP32 ONNX | `585cb1d62b6afe7b50af4632b1586bb4a7f5c2a6461c6870c639f80d8fbe5dac` |
| INT8 ONNX | `a1f78aa0cb0c1f09eaa0474cffbccbd98ffc624c4c95235ee4fe442a4b601044` |

FP32 ONNX maximum absolute error was `1.94e-7`. Selective per-channel dynamic
INT8 achieved minimum embedding cosine `0.99143` over the validation corpus and
reduced the ONNX file size by `23.34%`. These values validate the artifact
machinery only; the tiny synthetic ranking metrics are not research results.

### Portable image validation

Rostam job `186946` built the Apptainer image through the clean-tree wrapper
from commit `08a0dddde199ab8da5b90bd5b6d381e548d7fe40`. The image label records that
revision, the embedded test confirmed that Git metadata was excluded, and the
fixture preparation and verification passed. Dependent A100 job `186947`
confirmed PyTorch `2.7.1+cu128`, CUDA runtime `12.8`, an NVIDIA A100-PCIE-40GB,
the embedded source revision, and a CUDA tensor sum of `28`. The finished SIF is
`10,779,668,480` bytes with SHA-256
`1d8082e803250b6f58975c1a5c02587e10b7789bbf626a55899d732c97ebdc69`.

The previous README's accuracy, model-size, speedup, and latency numbers were not
connected to a reproducible run manifest, so they are intentionally not carried
forward. This file will be updated from a completed Rostam artifact run with:

- Git commit and Slurm job ID;
- GPU name and driver;
- data profile, query count, and candidate depth;
- teacher and student model revision hashes;
- MRR@10, nDCG@10, and recall@10 for baseline and student;
- FP32 and INT8 ONNX validation results;
- elapsed time and run-manifest SHA-256;
- known limitations and failed-run notes.
