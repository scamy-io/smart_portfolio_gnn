# Stage 1: Build dependencies
FROM python:3.10-slim as builder

WORKDIR /app

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

# Install dependencies into a virtualenv
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir torch==2.1.2+cpu --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir torch_scatter torch_sparse torch_geometric -f https://data.pyg.org/whl/torch-2.1.2+cpu.html && \
    pip install --no-cache-dir "numpy<2.0.0" && \
    pip install --no-cache-dir -r requirements.txt
# Stage 2: Runtime
FROM python:3.10-slim

WORKDIR /app

# Create a non-root user
RUN useradd -m -r appuser

# Copy virtualenv from builder
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy source code
COPY src/ src/
COPY dashboard/ dashboard/
COPY scripts/ scripts/
COPY configs/ configs/

# Create necessary directories for data/reports/alerts
RUN mkdir -p data/processed/graph_snapshots \
    data/streaming/incoming/processed \
    reports \
    alerts && \
    chown -R appuser:appuser /app

USER appuser

EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
  CMD curl --fail http://localhost:8501/_stcore/health || exit 1

# Default command to run dashboard
CMD ["streamlit", "run", "dashboard/app.py", "--server.port", "8501", "--server.headless", "true"]
