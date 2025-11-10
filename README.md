
# NeuroRank: High-Performance Neural Information Retrieval

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**NeuroRank** is a production-ready neural reranking service designed for low-latency Information Retrieval (IR). It leverages **Knowledge Distillation** to compress a large, highly accurate BERT-based "teacher" model into a smaller, faster "student" model. Further performance gains are achieved through **ONNX runtime optimization** and **8-bit quantization**, making it suitable for real-time search applications.

## 🚀 Key Features

* **Knowledge Distillation:** Achieves 97% of the teacher model's accuracy with a 6x reduction in model size and 10x faster inference.
* **Production-Optimized:** Deployed using ONNX Runtime with dynamic quantization for CPU-based inference.
* **Scalable API:** Includes a FastAPI-based REST service ready for containerization (Docker).
* **Standard Benchmarks:** Trained and evaluated on the MS MARCO passage ranking dataset.

## 🛠️ Architecture

```mermaid
graph LR
    A[MS MARCO Data] --> B(Teacher Model<br/>Cross-Encoder BERT);
    B -->|Distillation Logs| C{Distillation<br/>Trainer};
    A --> C;
    C --> D(Student Model<br/>MiniLM);
    D --> E[ONNX Export &<br/>Quantization];
    E --> F(NeuroRank<br/>Service API);
````

## ⚡ Performance Benchmarks

| Model Version | MRR@10 | Latency (p99) | Model Size |
| :--- | :--- | :--- | :--- |
| **Teacher (BERT-Base)** | 0.382 | 120ms | 420MB |
| **Student (MiniLM-L6)** | 0.371 | 15ms | 90MB |
| **NeuroRank (Quantized ONNX)**| **0.369** | **8ms** | **23MB** |

*\> Note: Benchmarks run on Intel Xeon CPU @ 2.20GHz, 4 vCPUs.*

## 📦 Quick Start

1.  **Clone and Install Dependencies:**

    ```bash
    git clone [https://github.com/yourusername/neurorank.git](https://github.com/yourusername/neurorank.git)
    cd neurorank
    pip install -r requirements.txt
    ```

2.  **Download Pre-trained Models (Optional):**
    *If you don't want to run the full training pipeline, download our [pre-quantized ONNX model](https://www.google.com/search?q=%23) and place it in `models/`.*

3.  **Run the Service:**

    ```bash
    uvicorn ranker_service.main:app --reload --port 8000
    ```

4.  **Test the API:**

    ```bash
    curl -X POST "http://localhost:8000/rerank" \
         -H "Content-Type: application/json" \
         -d '{"query": "machine learning", "documents": ["intro to ML", "advanced AI", "cooking recipes"]}'
    ```

## 🏗️ Project Structure

  * `src/neuro_ranker/`: Core library for model definitions, training loops, and distillation logic.
  * `scripts/`: Utility scripts for data prep, ONNX export, and benchmarking.
  * `ranker_service/`: FastAPI application for serving the model.
  * `configs/`: Training configuration files.

<!-- end list -->

````

**2. Reorganize for Clarity**
Move the loose scripts in your root directory to more logical places. Run these commands in your terminal from the project root:

```bash
mkdir -p training_pipeline
mv train_teacher.py training_pipeline/
mv distill_student.py training_pipeline/
mv eval_rankers.py training_pipeline/
````

**3. Create a `requirements.txt` (if you don't have a complete one)**
Ensure it has everything needed for a user to run it.

```text
torch>=1.10.0
transformers>=4.18.0
datasets>=2.1.0
scikit-learn
tqdm
numpy
pandas
onnxruntime>=1.11.0
fastapi
uvicorn
pydantic
```

-----

