import yaml
import os
import torch
import logging
import json
import math

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from lume_model.utils import variables_from_yaml
from lume_model.models import TorchModel, TorchModule
import k2eg
import matplotlib.pyplot as plt
import mlflow
import mlflow.pytorch
from mlflow.models.signature import infer_signature
from mlflow.tracking import MlflowClient

mlflow.set_tracking_uri("https://ard-mlflow.slac.stanford.edu")
print("Tracking to:", mlflow.get_tracking_uri())

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
    if run_name_tag.startswith("Live K2EG LUME Model Run"):
        try:
            num = int(run_name_tag.split("Live K2EG LUME Model Run")[-1].strip())
            run_numbers.append(num)
        except ValueError:
            continue

next_run_number = max(run_numbers, default=0) + 1
run_name = f"Live K2EG LUME Model Run{next_run_number}"
logger.info(f"Starting new MLflow run: {run_name}")


with mlflow.start_run(run_name=run_name):

    # Load transformers and model
    logger.info("Loading transformers and model components...")
    input_variables, output_variables = variables_from_yaml("../model/pv_variables.yml")
    lume_module = TorchModule("../model/pv_module.yml")

    # Read live input from K2EG
    logger.info("Reading input PVs from live EPICS data using K2EG...")
    k2eg_client = k2eg.dml('lcls', 'app-three')
    pv_map_path = os.path.join("..", "info", "pv_mapping.json")
    pv_names = lume_module.model.input_names

    input_parameter_values = {}
    for pv_name in pv_names:
        input_parameter_values[pv_name] = input_variables[pv_names.index(pv_name)].default_value
        if pv_name not in ['CAMR:IN20:186:R_DIST', 'Pulse_length'] and not input_variables[pv_names.index(pv_name)].is_constant:
            try:
                input_parameter_values[pv_name] = k2eg_client.get('ca://' + pv_name, 5.0)["value"]
            except Exception as e:
                logger.warning(f"Failed to get PV {pv_name}: {e}. Using default.")

    try:
        in_xrms_value = k2eg_client.get('ca://CAMR:IN20:186:XRMS')["value"]
        in_yrms_value = k2eg_client.get('ca://CAMR:IN20:186:YRMS')["value"]
        rdist = math.sqrt(in_xrms_value ** 2 + in_yrms_value ** 2)
        input_parameter_values['CAMR:IN20:186:R_DIST'] = rdist
    except Exception as e:
        logger.error(f"Failed to compute R_DIST: {e}. Using default.")

    for pv_name, val in input_parameter_values.items():
        logger.info(f"Input Dict: {pv_name} → {val}")

    with open("input_parameters.txt", "w") as f:
        for pv_name, val in input_parameter_values.items():
            f.write(f"{pv_name} → {val}\n")

    mlflow.log_artifact("input_parameters.txt")
    os.remove("input_parameters.txt")

    # Predict
    input_tensor = torch.tensor(list(input_parameter_values.values())).to(
        dtype=lume_module.model.dtype,
        device=lume_module.model.device
    ).unsqueeze(0)

    with torch.no_grad():
        predictions = lume_module(input_tensor)

    logger.info(f"Predictions: {predictions}")


    # Log predictions 

    """lume_module.register_to_mlflow(
            input_tensor,
            artifact_path="lcls_injector_torch_module",
            registered_model_name="lcls_injector_torch_module", # not necessary but required for adding tags/aliases
            tags={"type":"surrogate_injector"}, # example tag, if desired
            run_name=run_name)"""
    # Plot predictions
    logger.info("Plotting predictions...")
    nrows, ncols = 3, 2
    fig, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=(10, 15))
    for i, output_name in enumerate(lume_module.output_order):
        ax_i = ax[i // ncols, i % ncols]
        ax_i.plot(predictions[i].detach().numpy(), "C1x", label="predictions")
        ax_i.legend()
        ax_i.set_title(output_name)
    ax[-1, -1].axis('off')
    fig.tight_layout()

    fig_name = f"{run_name.replace(' ', '_').lower()}_prediction_plot.png"
    mlflow.log_figure(fig, fig_name)
    logging.info(f"Logged figure: {fig_name}")
    plt.close()
    logging.info("Saved and logged plot")

    # Log model to registry
    

    k2eg_client.close()
    logger.info("Inference complete and logged to MLflow.")

# End run if still active
if mlflow.active_run() is not None:
    mlflow.end_run()
    logger.info("MLflow run ended.")
