# Lightning Sample

## Background

Pytorch Lightning sample used to experiment with MLflow and other similar MLOps tooling.

Prepare environment:

```bash
uv venv -p 3.12
# Install PyTorch first to ensure we get a specific version
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
uv pip install lightning
uv pip install nvidia-ml-py psutil
```

Execute

```
export MLFLOW_TRACKING_URI=http://some.host.name.here:5000
export MLFLOW_ENABLE_SYSTEM_METRICS_LOGGING=true
uv run main.py
```
