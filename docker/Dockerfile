# Development image for running the service, migrations, scripts and tests.
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# CPU-only PyTorch for the local sentence-transformers embedding provider (the default PyPI
# wheel bundles CUDA libraries and is several GB larger).
RUN pip install torch --index-url https://download.pytorch.org/whl/cpu

# Dependency layer: only re-built when pyproject.toml changes.
COPY pyproject.toml README.md ./
RUN mkdir app && touch app/__init__.py && pip install -e ".[dev]"

# Hugging Face model cache. Models are downloaded on first use into a named volume
# (docker-compose.yml), never baked into the image or committed to git.
ENV HF_HOME=/opt/hf-cache
RUN mkdir -p /opt/hf-cache

COPY . .

RUN useradd --create-home --uid 1000 appuser && chown -R appuser /app /opt/hf-cache
USER appuser

EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
