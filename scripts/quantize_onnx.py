import argparse
from onnxruntime.quantization import quantize_dynamic, QuantType

p = argparse.ArgumentParser()
p.add_argument('--in_onnx', required=True)
p.add_argument('--out_onnx', required=True)
args = p.parse_args()

quantize_dynamic(args.in_onnx, args.out_onnx, weight_type=QuantType.QInt8)
print('INT8 written to', args.out_onnx)