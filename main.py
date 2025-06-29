# Toy example copied from pytorch-lightning GitHub README
import os
import platform

import torch, torch.nn as nn, torch.utils.data as data, torchvision as tv, torch.nn.functional as F
import lightning as L

import pynvml

from lightning.pytorch.callbacks import DeviceStatsMonitor
from mlflow.system_metrics.system_metrics_monitor import SystemMetricsMonitor
from pytorch_lightning.loggers import CSVLogger, MLFlowLogger

MATMUL_PRECISION = "medium" 
torch.set_float32_matmul_precision(MATMUL_PRECISION)

# Adopted from sample provided in https://github.com/Lightning-AI/pytorch-lightning/issues/20563
class MLFlowSystemMonitorCallback(L.Callback):
    def on_fit_start(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        mlf_logger = next((l for l in trainer.loggers if isinstance(l, MLFlowLogger)), None)

        if mlf_logger:
            self.system_monitor = SystemMetricsMonitor(
                run_id=mlf_logger.run_id,
            )
            self.system_monitor.start()
        else:
            print("MLFlowLogger not found. Skipping system metrics logging.")
        
    def on_fit_end(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        if self.system_monitor:
            self.system_monitor.finish()

# --------------------------------
# Step 1: Define a LightningModule
# --------------------------------
# A LightningModule (nn.Module subclass) defines a full *system*
# (ie: an LLM, diffusion model, autoencoder, or simple image classifier).


class LitAutoEncoder(L.LightningModule):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(28 * 28, 128), nn.ReLU(), nn.Linear(128, 3))
        self.decoder = nn.Sequential(nn.Linear(3, 128), nn.ReLU(), nn.Linear(128, 28 * 28))
        self.save_hyperparameters()  # save hyperparameters for reproducibility

    def forward(self, x):
        # in lightning, forward defines the prediction/inference actions
        embedding = self.encoder(x)
        return embedding

    def training_step(self, batch, batch_idx):
        # training_step defines the train loop. It is independent of forward
        x, _ = batch
        x = x.view(x.size(0), -1)
        z = self.encoder(x)
        x_hat = self.decoder(z)
        loss = F.mse_loss(x_hat, x)
        self.log("train_loss", loss)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer


# Helper function to log specific environment details
# It takes the logger instance as an argument.
# (Implementation provided by Gemini AI)
def log_environment_details(logger: MLFlowLogger):
    """Logs key environment and hardware details to the MLFlow run."""
    print("Logging custom environment parameters...")
    
    # The logger object gives us access to the underlying MLFlow client
    # This is the "Lightning way" to add custom tags or params.
    client = logger.experiment
    run_id = logger.run_id

    current_precision = torch.get_float32_matmul_precision()
    client.log_param(run_id, "float32_matmul_precision", current_precision)

    # Log as parameters for easy viewing
    client.log_param(run_id, "python_version", platform.python_version())
    client.log_param(run_id, "pytorch_version", torch.__version__)
    client.log_param(run_id, "lightning_version", L.__version__)
    
    if torch.cuda.is_available():
        client.log_param(run_id, "cuda_version", torch.version.cuda)
        try:
            pynvml.nvmlInit()
            driver_version = pynvml.nvmlSystemGetDriverVersion()
            client.log_param(run_id, "nvidia_driver_version", driver_version)
            gpu_name = pynvml.nvmlDeviceGetName(pynvml.nvmlDeviceGetHandleByIndex(0))
            client.log_param(run_id, "gpu_name", gpu_name)
            pynvml.nvmlShutdown()
        except pynvml.NVMLError as e:
            client.log_param(run_id, "pynvml_error", str(e))

def main() -> None:
    # -------------------
    # Step 2a: Define data
    # -------------------
    dataset = tv.datasets.MNIST(".", download=True, transform=tv.transforms.ToTensor())
    train, val = data.random_split(dataset, [55000, 5000])

    # -------------------
    # Step 2b: Configure loggers (not present in upstream example)
    # -------------------
    loggers = [CSVLogger("logs/", name="autoencoder")]

    # Read environment variables to determine if MLFlow tracking is enabled
    # and if device stats should be logged.
    is_mlflow_tracking_enabled = os.getenv("MLFLOW_TRACKING_URI") is not None
    are_device_stats_enabled = os.environ.get("MLFLOW_ENABLE_SYSTEM_METRICS_LOGGING", "false").lower() == "true"

    if is_mlflow_tracking_enabled:
        # If MLFLOW_TRACKING_URI is set, use MLFlowLogger
        # Otherwise, skip MLFlowLogger
        mlf_logger = MLFlowLogger(experiment_name="autoencoder_experiment", log_model=True)
        mlf_logger.experiment.get_run(mlf_logger.run_id)  # Ensure the run is created
        log_environment_details(mlf_logger)  # Log environment details
        loggers.append(mlf_logger)
    
    callbacks = []
    if are_device_stats_enabled:
        # If MLFLOW_ENABLE_SYSTEM_METRICS_LOGGING is set to true, add DeviceStatsMonitor
        callbacks.append(DeviceStatsMonitor(cpu_stats=True))
        callbacks.append(MLFlowSystemMonitorCallback())

    # -------------------
    # Step 3: Train
    # -------------------
    autoencoder = LitAutoEncoder()
    trainer = L.Trainer(logger=loggers, callbacks=callbacks)
    trainer.fit(
        autoencoder,
        data.DataLoader(train),
        data.DataLoader(dataset=val, num_workers=7)
    )

if __name__ == "__main__":
    main()