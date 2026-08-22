FROM python:3.12-slim

WORKDIR /opt/neuro-ranker
COPY requirements-runtime.lock pyproject.toml README.md LICENSE ./
COPY src ./src
COPY ranker_service ./ranker_service
RUN python -m pip install --no-cache-dir --require-hashes -r requirements-runtime.lock \
    && python -m pip install --no-cache-dir --no-deps .

ENV NEURORANKER_ONNX=/models/student.int8.onnx
ENV NEURORANKER_TOKENIZER=/models
EXPOSE 8000
CMD ["uvicorn", "ranker_service.main:app", "--host", "0.0.0.0", "--port", "8000"]

