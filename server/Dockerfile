FROM python:3.12-slim

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /app

# Copy dependency files first (layer caching)
COPY pyproject.toml ./
COPY uv.lock* ./

# Install all runtime dependencies (no dev extras, no editable install)
RUN uv pip install --system --no-cache \
        fastapi \
        "uvicorn[standard]" \
        websockets \
        pydantic \
        torch \
        transformers \
        huggingface-hub \
        numpy \
        scikit-learn \
        python-dotenv \
        pyyaml \
        openai \
        httpx

# Copy the full project
COPY . .

# Install the package itself (no deps, already installed above)
RUN uv pip install --system --no-cache --no-deps -e .

# Environment variables (overridden at runtime)
ENV PYTHONPATH=/app
ENV PORT=8000
ENV HF_HOME=/app/.cache/huggingface

# Optional: pre-download ESM2 weights at build time.
# Requires --build-arg HF_TOKEN=<your_token> or network access to HF hub.
# If this fails (no token / no network), the weights are downloaded on first call.
ARG HF_TOKEN=""
RUN if [ -n "$HF_TOKEN" ]; then \
      HF_TOKEN=$HF_TOKEN python3 -c "\
from transformers import EsmTokenizer, EsmModel; \
EsmTokenizer.from_pretrained('facebook/esm2_t6_8M_UR50D'); \
EsmModel.from_pretrained('facebook/esm2_t6_8M_UR50D'); \
print('ESM2 weights cached.')"; \
    else \
      echo "HF_TOKEN not provided — ESM2 weights will be downloaded on first request."; \
    fi

EXPOSE 8000

CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "8000", "--log-level", "info"]
