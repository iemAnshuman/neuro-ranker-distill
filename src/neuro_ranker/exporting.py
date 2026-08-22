from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from onnxruntime import InferenceSession
from onnxruntime.quantization import QuantType, quantize_dynamic

from .data import sha256_file
from .model import BiEncoder, load_tokenizer


class ExportEncoder(torch.nn.Module):
    def __init__(self, model: BiEncoder):
        super().__init__()
        self.model = model

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        return self.model.encode(input_ids, attention_mask)


def _cosine_rows(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    numerator = (left * right).sum(axis=1)
    denominator = np.linalg.norm(left, axis=1) * np.linalg.norm(right, axis=1)
    return numerator / np.clip(denominator, 1e-12, None)


def export_student(
    model_dir: str | Path,
    output_dir: str | Path,
    *,
    max_length: int,
) -> dict[str, Any]:
    model_path = Path(model_dir)
    output = Path(output_dir)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"refusing to overwrite export directory: {output}")
    output.mkdir(parents=True, exist_ok=True)
    tokenizer = load_tokenizer(model_path)
    model = BiEncoder(str(model_path)).cpu().eval()
    wrapper = ExportEncoder(model).eval()
    tokenizer.save_pretrained(output)

    texts = ["neural information retrieval", "a passage about ranking models"]
    tokens = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )
    input_ids = tokens["input_ids"]
    attention_mask = tokens["attention_mask"]
    onnx_path = output / "student.onnx"
    torch.onnx.export(
        wrapper,
        (input_ids, attention_mask),
        onnx_path,
        input_names=["input_ids", "attention_mask"],
        output_names=["embeddings"],
        dynamic_axes={
            "input_ids": {0: "batch", 1: "sequence"},
            "attention_mask": {0: "batch", 1: "sequence"},
            "embeddings": {0: "batch"},
        },
        opset_version=17,
        dynamo=False,
    )

    with torch.inference_mode():
        reference = wrapper(input_ids, attention_mask).numpy()
    ort_inputs = {
        "input_ids": input_ids.numpy(),
        "attention_mask": attention_mask.numpy(),
    }
    fp32_output = InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"]).run(
        None, ort_inputs
    )[0]
    fp32_max_error = float(np.max(np.abs(reference - fp32_output)))
    if fp32_max_error > 1e-4:
        raise RuntimeError(f"ONNX validation failed: max error {fp32_max_error}")

    quantized_path = output / "student.int8.onnx"
    quantize_dynamic(str(onnx_path), str(quantized_path), weight_type=QuantType.QInt8)
    int8_output = InferenceSession(str(quantized_path), providers=["CPUExecutionProvider"]).run(
        None, ort_inputs
    )[0]
    int8_cosines = _cosine_rows(reference, int8_output)
    minimum_cosine = float(int8_cosines.min())
    if minimum_cosine < 0.99:
        raise RuntimeError(f"INT8 validation failed: minimum cosine {minimum_cosine}")

    report = {
        "schema_version": 1,
        "source_model_artifact_sha256": sha256_file(model_path / "artifact.json"),
        "fp32": {
            "path": onnx_path.name,
            "sha256": sha256_file(onnx_path),
            "max_absolute_error": fp32_max_error,
        },
        "int8": {
            "path": quantized_path.name,
            "sha256": sha256_file(quantized_path),
            "minimum_cosine_similarity": minimum_cosine,
        },
        "validation_texts": texts,
        "opset": 17,
    }
    (output / "export_report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return report
