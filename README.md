# Lightning Sample

## Background

Pytorch Lightning sample used to experiment with MLflow and other similar MLOps tooling.

Prepare environment:

```bash
uv venv -p 3.12
# Install PyTorch first to ensure we get a specific version
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
# Pinning to 2.5.0 due to this bug
# https://github.com/mlflow/mlflow/issues/15111
uv pip install lightning==2.5.0
uv pip install pytorch-lightning==2.5.0
# Only -skinny is required as we don't need the entire server package
uv pip install mlflow-skinny
uv pip install nvidia-ml-py psutil
```

Execute

```
export MLFLOW_TRACKING_URI=http://some.host.name.here:5000
export MLFLOW_ENABLE_SYSTEM_METRICS_LOGGING=true
uv run main.py
```
