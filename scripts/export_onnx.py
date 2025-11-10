import argparse
import torch
import onnx
import sys
import os

# --- Start of Fix ---
# Add the project's root directory to the Python path
# This allows the script to find the 'src' module
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, project_root)
# --- End of Fix ---

from src.neuro_ranker.trainer_student import BiEncoder

p = argparse.ArgumentParser()
p.add_argument("--ckpt", required=True)
p.add_argument("--model_name", default="sentence-transformers/all-MiniLM-L6-v2")
p.add_argument("--onnx", required=True)
args = p.parse_args()

# Create the output directory if it doesn't exist
os.makedirs(os.path.dirname(args.onnx), exist_ok=True)

ckpt = torch.load(args.ckpt, map_location="cpu")
model = BiEncoder(args.model_name)
model.load_state_dict(ckpt["model"])
model.eval()

# Export encoder that maps input_ids, attention_mask → pooled vector
dummy_ids = torch.randint(0, 100, (1, 256), dtype=torch.long)
dummy_attn = torch.ones_like(dummy_ids)

torch.onnx.export(
    model.encoder,
    (dummy_ids, dummy_attn),
    args.onnx,
    input_names=["input_ids", "attention_mask"],
    output_names=["last_hidden_state", "pooler_output"],
    opset_version=17,
    dynamic_axes={
        "input_ids": {0: "batch", 1: "seq"},
        "attention_mask": {0: "batch", 1: "seq"},
    },
)
print("Exported to", args.onnx)
