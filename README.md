# NeuroRanker: A Production-Grade Neural Ranking System

**NeuroRanker** is a complete, end-to-end system for building and deploying a state-of-the-art, two-stage neural search ranking pipeline. It is designed to deliver significant quality improvements over traditional keyword-based search (like BM25) while maintaining the low latency required for production environments.

This project demonstrates the **Teacher-Student Distillation** paradigm, where a large, highly accurate "teacher" model is used to train a small, lightning-fast "student" model suitable for real-time inference.

---

## Core Concepts

The system follows a modern **Learning to Rank (LTR)** architecture:

1. **Candidate Generation**:
   A fast, traditional search method (BM25) first retrieves a broad set of potentially relevant documents (e.g., the top 100). This *"narrows the haystack."*

2. **Re-ranking**:
   A sophisticated neural network then intelligently re-ranks these top 100 candidates to produce the final, high-quality ordering for the user.

This project focuses on building and optimizing the **re-ranking stage** using a powerful distillation technique.

---

## System Architecture

The pipeline is composed of several key modules:

* **Data Processing**: Scripts to convert the raw MS MARCO dataset into an efficient format for training.
* **BM25 Indexer**: A robust indexer built on Pyserini (Lucene) for fast candidate generation.
* **Teacher Model**: A powerful but slow Cross-Encoder (`microsoft/MiniLM-L12-H384-uncased`) that achieves high accuracy by analyzing the query and a passage simultaneously.
* **Student Model**: A fast and efficient Bi-Encoder (`sentence-transformers/all-MiniLM-L6-v2`) that generates embeddings for the query and passages independently, making it suitable for real-time search.
* **Training Orchestration**: Scripts to manage the full training and distillation workflow.
* **Serving API**: A production-ready FastAPI service that serves the optimized student model via the ONNX Runtime for high-throughput, low-latency inference.

---

## Reproducibility: A Step-by-Step Guide

This project is designed to be **fully reproducible**. The recommended environment for handling the large-scale MS MARCO dataset is **Google Colab Pro with a GPU runtime**.

---

### Phase 0: Prerequisites

* **Git**: To clone the repository.
* **Python 3.10+**: The required Python version.
* **Google Account**: With sufficient Google Drive storage (\~50 GB recommended for all artifacts).

---

### Phase 1: Environment Setup

Clone the repository:

```bash
git clone https://github.com/your-username/neuro-ranker-distill.git
cd neuro-ranker-distill
```

Set up a virtual environment (local):

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

> **Note:** The following steps are for the recommended Google Colab workflow.

---

### Phase 2: Data Acquisition (Google Drive)

**Download the MS MARCO Passage Ranking Dataset:**

* `collection.tar.gz`
* `queries.tar.gz`
* `qrels.train.tsv`

**Upload to Google Drive:**
Create a folder `ms_marco_project` and a subfolder `raw_data` inside it. Upload the three files there.

---

### Phase 3: The Full Cloud-Based Pipeline

The following steps should be run **inside a Google Colab Pro notebook with GPU runtime**.

**Connect to Drive & Clone Repo:**

```python
from google.colab import drive
drive.mount('/content/drive', force_remount=True)

!git clone https://github.com/your-username/neuro-ranker-distill.git
%cd neuro-ranker-distill
!pip install -r requirements.txt
```

**Create project folders in Drive:**

```bash
!mkdir -p "/content/drive/MyDrive/ms_marco_project/processed_data"
!mkdir -p "/content/drive/MyDrive/ms_marco_project/index"
!mkdir -p "/content/drive/MyDrive/ms_marco_project/models"
```

**Process raw data:**

```bash
!python scripts/prepare_msmarco.py \
  --collection_path "/content/drive/MyDrive/ms_marco_project/raw_data/collection.tsv" \
  --queries_path "/content/drive/MyDrive/ms_marco_project/raw_data/queries.train.tsv" \
  --out_dir "/content/drive/MyDrive/ms_marco_project/processed_data"
```

**Build BM25 index:**

```bash
# Set up a compatible Java version for Pyserini
!apt-get install openjdk-21-jdk-headless -qq > /dev/null
!update-alternatives --set java /usr/lib/jvm/java-21-openjdk-amd64/bin/java

!python scripts/build_bm25_index.py \
  --collection "/content/drive/MyDrive/ms_marco_project/processed_data/passages.jsonl" \
  --index_dir "/content/drive/MyDrive/ms_marco_project/index"
```

**Train the Teacher Model:**

```bash
# Prepare relevance file
!cp "/content/drive/MyDrive/ms_marco_project/raw_data/qrels.train.tsv" "/content/drive/MyDrive/ms_marco_project/processed_data/qrels.tsv"

# Start training
!python train_teacher.py \
  --data_dir "/content/drive/MyDrive/ms_marco_project/processed_data" \
  --out_dir "/content/drive/MyDrive/ms_marco_project/models/teacher"
```

**Distill the Student Model:**

```bash
!python distill_student.py \
  --data_dir "/content/drive/MyDrive/ms_marco_project/processed_data" \
  --teacher "/content/drive/MyDrive/ms_marco_project/models/teacher/best.pt" \
  --out_dir "/content/drive/MyDrive/ms_marco_project/models/student"
```

---

### Phase 4: Evaluation and Serving

**Run evaluation:**

```bash
!python eval_rankers.py \
  --data_dir "/content/drive/MyDrive/ms_marco_project/processed_data" \
  --bm25_index "/content/drive/MyDrive/ms_marco_project/index" \
  --teacher "/content/drive/MyDrive/ms_marco_project/models/teacher/best.pt" \
  --student "/content/drive/MyDrive/ms_marco_project/models/student/best.pt"
```

**Export student model for production:**

```bash
!mkdir -p "/content/drive/MyDrive/ms_marco_project/models/onnx"

!python scripts/export_onnx.py \
  --ckpt "/content/drive/MyDrive/ms_marco_project/models/student/best.pt" \
  --onnx "/content/drive/MyDrive/ms_marco_project/models/onnx/student.onnx"

!python scripts/quantize_onnx.py \
  --in_onnx "/content/drive/MyDrive/ms_marco_project/models/onnx/student.onnx" \
  --out_onnx "/content/drive/MyDrive/ms_marco_project/models/onnx/student.int8.onnx"
```

**Run the API Service (Locally):**

```bash
# Set the environment variable to point to your model
export STUDENT_ONNX=out/student.int8.onnx

# Run the server
uvicorn src.neuro_ranker.service.app:app --host 0.0.0.0 --port 8000
```

**Test with curl:**

```bash
curl -X POST http://localhost:8000/rerank \
-H 'Content-Type: application/json' \
-d '{"query":"what is a neural ranking model","texts":["Neural ranking models learn to order documents given a query.","Inverse propensity scoring reweights clicks to debias position."]}'
```

---

## Directory Structure

```
├── configs/              # YAML configuration files
├── data/                 # Placeholder for datasets (use .gitignore)
├── out/                  # Output for exported models (e.g., ONNX)
├── ranker_service/       # FastAPI application for serving
├── runs/                 # Output for training runs and model checkpoints
├── scripts/              # Helper scripts for data prep, indexing, etc.
├── src/                  # Core Python source code for the ranking models
├── .gitignore            # Specifies files to ignore in Git
├── distill_student.py    # Main script for student model distillation
├── eval_rankers.py       # Script to evaluate and compare rankers
├── requirements.txt      # Project dependencies
└── train_teacher.py      # Main script for teacher model training
```


