import torch
import matplotlib.pyplot as plt
import os
import logging
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from lume_model.utils import variables_from_yaml
from lume_model.models import TorchModel, TorchModule
from dotenv import load_dotenv, find_dotenv
import mlflow
import mlflow.pytorch
from mlflow.models.signature import infer_signature
from mlflow.tracking import MlflowClient

mlflow.set_tracking_uri(uri="https://ard-mlflow.slac.stanford.edu")
"""
# Load environment variables
dotenv_path = find_dotenv(".env")
if dotenv_path:
    logger.info(f"Loading environment from: {dotenv_path}")
    load_dotenv(dotenv_path)
else:/sdf/home/g/gopikab/lcls-ml/lcls_cu_injector_ml_model/k2eg

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
    if run_name_tag.startswith("LUME Model Run"):
        try:
            num = int(run_name_tag.split("LUME Model Run")[-1].strip())
            run_numbers.append(num)
        except ValueError:
            continue

next_run_number = max(run_numbers, default=0) + 1
run_name = f"LUME Model Run{next_run_number}"
logger.info(f"Starting new MLflow run: {run_name}")

with mlflow.start_run(run_name=run_name):
    # Load transformers
    input_sim_to_nn = torch.load("../model/input_sim_to_nn.pt")
    output_sim_to_nn = torch.load("../model/output_sim_to_nn.pt")
    logging.info("Loaded transformers")

     # Load variable specifications
    input_vars, output_vars = variables_from_yaml("../model/sim_variables.yml")
    logging.info("Loaded input and output variables")

    # Create and wrap model
    lume_model = TorchModel(
        model="../model/model.pt",
        input_variables=input_vars,
        output_variables=output_vars,
        input_transformers=[input_sim_to_nn],
        output_transformers=[output_sim_to_nn],
    )
    lume_module = TorchModule(
        model=lume_model,
        input_order=lume_model.input_names,
        output_order=lume_model.output_names,
    )
    logging.info("Created and wrapped TorchModel")

    # Log parameters
    mlflow.log_param("model_path", "../model/model.pt")
    mlflow.log_param("input_transformer", str(type(input_sim_to_nn)))
    mlflow.log_param("output_transformer", str(type(output_sim_to_nn)))
    #mlflow.log_param("input_vars", input_vars)
    #mlflow.log_param("output_vars", output_vars)
    with open("input_vars.json", "w") as f:
        json.dump([v.__dict__ for v in input_vars], f, indent=2)
        mlflow.log_artifact("input_vars.json")

    with open("output_vars.json", "w") as f:
        json.dump([v.__dict__ for v in output_vars], f, indent=2)
        mlflow.log_artifact("output_vars.json")


    mlflow.log_param("num_input_vars", len(input_vars))
    mlflow.log_param("num_output_vars", len(output_vars))
    logging.info("Logged parameters")

    # Log YAML artifacts
    mlflow.log_artifact("../model/sim_variables.yml")
    mlflow.log_artifact("../model/sim_model.yml")
    mlflow.log_artifact("../model/sim_module.yml")
    logging.info("Logged YAML files as artifacts")

     # Load sample inputs and make predictions
    inputs_small = torch.load("../info/inputs_small.pt")
    outputs_small = torch.load("../info/outputs_small.pt")
    with torch.no_grad():
        predictions = lume_module(inputs_small)
    logging.info("Performed inference on sample inputs")

     # Log performance metric
    mae = torch.mean(torch.abs(predictions - outputs_small)).item()
    mlflow.log_metric("mean_absolute_error", mae)
    logging.info(f"Logged MAE: {mae}")

       # Infer and log model signature
    signature = infer_signature(inputs_small.numpy(), predictions.numpy())
    mlflow.pytorch.log_model(lume_module, artifact_path="lume_module", signature=signature)
    logging.info("Logged PyTorch model with inferred signature")

     # Plot predictions vs actuals
    nrows, ncols = 3, 2
    fig, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=(10, 15))
    for i, output_name in enumerate(lume_module.output_order):
        ax_i = ax[i // ncols, i % ncols]
        if i < outputs_small.shape[1]:
            sort_idx = torch.argsort(outputs_small[:, i])
            x_axis = torch.arange(outputs_small.shape[0])
            ax_i.plot(x_axis, outputs_small[sort_idx, i], "C0x", label="outputs")
            ax_i.plot(x_axis, predictions[sort_idx, i], "C1x", label="predictions")
            ax_i.legend()
            ax_i.set_title(output_name)
    ax[-1, -1].axis('off')
    fig.tight_layout()

    fig_name = f"{run_name.replace(' ', '_').lower()}_comparison_plot.png"
    mlflow.log_figure(fig, fig_name)
    logging.info(f"Logged figure: {fig_name}")
    plt.close()
    logging.info("Saved and logged comparison plot")

os.remove("input_vars.json")
os.remove("output_vars.json")

# End run if still active
if mlflow.active_run() is not None:
    mlflow.end_run()
    logger.info("MLflow run ended.")
