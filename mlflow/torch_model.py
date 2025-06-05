import os
import json
import torch

import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

import mlflow
import mlflow.pytorch
import matplotlib.pyplot as plt
import torch.nn.functional as F
from dotenv import load_dotenv, find_dotenv
from mlflow.tracking import MlflowClient
from mlflow.models.signature import infer_signature

# Load environment variables

mlflow.set_tracking_uri(uri="https://ard-mlflow.slac.stanford.edu")
"""
dotenv_path = find_dotenv(".env")
if dotenv_path:
    logger.info(f"Loading environment from: {dotenv_path}")
    load_dotenv(dotenv_path)
else:
    logger.warning("Could not find .env file")

mlflow_uri = os.getenv("MLFLOW_TRACKING_URI")
if not mlflow_uri:
    logger.error("MLFLOW_TRACKING_URI not found in environment. Check your .env file.")
    raise EnvironmentError("MLFLOW_TRACKING_URI not found in environment.")
else:
    logger.info(f"MLFLOW_TRACKING_URI = {mlflow_uri}")
    mlflow.set_tracking_uri(mlflow_uri)
"""
EXPERIMENT_NAME = "lcls-injector-ML"
mlflow.set_experiment(EXPERIMENT_NAME)
client = MlflowClient()
logger.info(f"Using MLflow experiment: {EXPERIMENT_NAME}")

# Get the next "Torch Model Run#" name
experiment = client.get_experiment_by_name("lcls-injector-ML")
all_runs = client.search_runs(experiment_ids=[experiment.experiment_id])

run_numbers = []
for run in all_runs:
    run_name_tag = run.data.tags.get("mlflow.runName", "")
    if run_name_tag.startswith("Torch Model Run"):
        try:
            num = int(run_name_tag.split("Torch Model Run")[-1].strip())
            run_numbers.append(num)
        except ValueError:
            continue

next_run_number = max(run_numbers, default=0) + 1
run_name = f"Torch Model Run{next_run_number}"
logger.info(f"Starting new MLflow run: {run_name}")

# Load base model and transformers
logger.info("Loading model and transformers...")
model = torch.load("../model/model.pt")
input_sim_to_nn = torch.load("../model/input_sim_to_nn.pt")
output_sim_to_nn = torch.load("../model/output_sim_to_nn.pt")

# Define transformed model
class TransformedModel(torch.nn.Module):
    def __init__(self, model, input_transformer, output_transformer):
        super(TransformedModel, self).__init__()
        self.model = model
        self.input_transformer = input_transformer
        self.output_transformer = output_transformer

    def forward(self, x):
        x = self.input_transformer(x)
        x = self.model(x)
        x = self.output_transformer.untransform(x)
        return x

logger.info("Wrapping model with transformers...")
# Create transformed model
transformed_model = TransformedModel(
    model=model, 
    input_transformer=input_sim_to_nn,
    output_transformer=output_sim_to_nn,
).to(torch.double)

# Start MLflow run
with mlflow.start_run(run_name=run_name):
    logger.info("Logging parameters to MLflow...")

    # Log parameters
    mlflow.log_param("model_type", str(type(model)))
    mlflow.log_param("input_transformer_type", str(type(input_sim_to_nn)))
    mlflow.log_param("output_transformer_type", str(type(output_sim_to_nn)))

    # Load data
    logger.info("Loading input/output data...")
    inputs_small = torch.load("../info/inputs_small.pt")
    outputs_small = torch.load("../info/outputs_small.pt")
    logger.info(f"Input shape: {inputs_small.shape}")
    logger.info(f"Output shape: {outputs_small.shape}")

    # Get predictions and calculate error
    logger.info("Running model inference...")
    with torch.no_grad():
        predictions = transformed_model(inputs_small)
    mae = torch.mean(torch.abs(predictions - outputs_small)).item()
    logger.info(f"Mean Absolute Error: {mae:.6f}")
    mlflow.log_metric("mean_absolute_error", mae)

    # Log model
    logger.info("Logging model to MLflow...")
    signature = infer_signature(inputs_small[:1].detach().cpu().numpy(), predictions[:1].detach().cpu().numpy())
    mlflow.pytorch.log_model(
    transformed_model,
    artifact_path="transformed_model",
    input_example=inputs_small[:1].detach().cpu().numpy(),
    signature=signature
)
    # Create and log plot
    logger.info("Generating comparison plot...")
    nrows, ncols = 3, 2
    fig, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=(10, 15))
    for i in range(nrows * ncols):
        ax_i = ax[i // ncols, i % ncols]
        if i < outputs_small.shape[1]:
            sort_idx = torch.argsort(outputs_small[:, i])
            x_axis = torch.arange(outputs_small.shape[0])
            ax_i.plot(x_axis, outputs_small[sort_idx, i], "C0x", label="outputs")
            ax_i.plot(x_axis, predictions[sort_idx, i], "C1x", label="predictions")
            ax_i.legend()
    ax[-1, -1].axis('off')
    fig.tight_layout()
    
    fig_name = f"{run_name.replace(' ', '_').lower()}_comparison_plot.png"
    mlflow.log_figure(fig, fig_name)
    logger.info(f"Logged figure: {fig_name}")
    plt.close()

# End run if still active
if mlflow.active_run() is not None:
    mlflow.end_run()
    logger.info("MLflow run ended.")
