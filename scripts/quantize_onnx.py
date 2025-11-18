import argparse
import os
from onnxruntime.quantization import quantize_dynamic, QuantType

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="Path to input ONNX model")
    parser.add_argument("--output", required=True, help="Path to output quantized model")
    args = parser.parse_args()

    print(f"Quantizing {args.model} -> {args.output} ...")
    
    quantize_dynamic(
        model_input=args.model,
        model_output=args.output,
        weight_type=QuantType.QUInt8
    )
    print("Quantization complete.")

if __name__ == "__main__":
    main()
