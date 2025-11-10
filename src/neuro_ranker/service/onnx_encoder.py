import onnxruntime as ort
import numpy as np
from transformers import AutoTokenizer


class OnnxEncoder:
    def __init__(self, onnx_path: str, model_name: str):
        self.sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
        self.tok = AutoTokenizer.from_pretrained(model_name)

    def embed(self, texts, max_len=256):
        enc = self.tok(
            texts,
            padding=True,
            truncation=True,
            max_length=max_len,
            return_tensors="np",
        )
        outs = self.sess.run(
            None,
            {"input_ids": enc["input_ids"], "attention_mask": enc["attention_mask"]},
        )
        last = outs[0]  # [B, L, H]
        attn = enc["attention_mask"][:, :, None]
        vec = (last * attn).sum(axis=1) / np.clip(attn.sum(axis=1), 1e-6, None)
        return vec.astype(np.float32)
