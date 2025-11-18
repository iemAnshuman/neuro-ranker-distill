#!/bin/bash
set -e

# 1. Define Paths
TEACHER_CKPT="models/teacher/best.pt"
STUDENT_CKPT="models/student/best.pt"
ONNX_PATH="models/student.onnx"
QUANTIZED_PATH="models/quantized_student.onnx"

# 2. Ensure Student Model Exists
if [ ! -f "$STUDENT_CKPT" ]; then
    echo "❌ Student checkpoint not found at $STUDENT_CKPT"
    echo "   Please run: neurorank train-student"
    exit 1
fi

# 3. Export to ONNX
echo "📦 Exporting Student to ONNX..."
python scripts/export_onnx.py \
    --ckpt "$STUDENT_CKPT" \
    --onnx "$ONNX_PATH"

# 4. Quantize (Optional: requires scripts/quantize_onnx.py to exist)
if [ -f "scripts/quantize_onnx.py" ]; then
    echo "📉 Quantizing Model..."
    python scripts/quantize_onnx.py \
        --model "$ONNX_PATH" \
        --output "$QUANTIZED_PATH"
    echo "✅ Done! Ready for Docker."
else
    echo "⚠️  Quantization script missing. Using standard ONNX model."
    cp "$ONNX_PATH" "$QUANTIZED_PATH"
fi
