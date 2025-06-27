import os
import time
import math
import json
import torch
import logging
import mlflow
from lume_model.utils import variables_from_yaml
from lume_model.models import TorchModule
import k2eg
from p4p.client.thread import Context

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# MLflow setup
mlflow.set_tracking_uri("https://ard-mlflow.slac.stanford.edu")
EXPERIMENT_NAME = "lcls-injector-ML"
mlflow.set_experiment(EXPERIMENT_NAME)
logger.info(f"Using MLflow experiment: {EXPERIMENT_NAME}")

client = mlflow.tracking.MlflowClient()
experiment = client.get_experiment_by_name(EXPERIMENT_NAME)

# Get next run name
all_runs = client.search_runs(experiment_ids=[experiment.experiment_id])
run_numbers = []

for run in all_runs:
    tag = run.data.tags.get("mlflow.runName", "")
    if tag.startswith("Live K2EG LUME Model Run"):
        try:
            num = int(tag.replace("Live K2EG LUME Model Run", "").strip())
            run_numbers.append(num)
        except ValueError:
            continue

next_run_number = max(run_numbers, default=0) + 1
run_name = f"Live K2EG LUME Model Run{next_run_number}"
logger.info(f"Starting MLflow run: {run_name}")

# Load model
input_variables, output_variables = variables_from_yaml("../model/pv_variables.yml")
lume_module = TorchModule("../model/pv_module.yml")
pv_names = lume_module.model.input_names
k2eg_client = k2eg.dml('lcls', 'app-three')

step = 0
with mlflow.start_run(run_name=run_name) as run:
    logger.info("Started MLflow parent run.")

    try:
        while True:
            input_parameter_values = {}

            for pv_name in pv_names:
                val = input_variables[pv_names.index(pv_name)].default_value
                if pv_name not in ['CAMR:IN20:186:R_DIST', 'Pulse_length'] and not input_variables[pv_names.index(pv_name)].is_constant:
                    try:
                        val = k2eg_client.get('ca://' + pv_name, 5.0)["value"]
                    except Exception as e:
                        logger.warning(f"Failed to get PV {pv_name}: {e}")
                input_parameter_values[pv_name] = val

            try:
                in_xrms = k2eg_client.get('ca://CAMR:IN20:186:XRMS')["value"]
                in_yrms = k2eg_client.get('ca://CAMR:IN20:186:YRMS')["value"]
                rdist = math.sqrt(in_xrms ** 2 + in_yrms ** 2)
                input_parameter_values['CAMR:IN20:186:R_DIST'] = rdist
            except Exception as e:
                logger.error(f"Failed to compute R_DIST: {e}")

            logger.info("Input values:")
            for k, v in input_parameter_values.items():
                logger.info(f"{k} → {v}")

            # Make predictions
            input_tensor = torch.tensor(list(input_parameter_values.values())).to(
                dtype=lume_module.model.dtype,
                device=lume_module.model.device
            ).unsqueeze(0)

            with torch.no_grad():
                predictions = lume_module(input_tensor)

            # Write predictions to EPICS
            logger.info("Writing predictions back to EPICS...")
            try:
                ctxt = Context("pva")
                ctxt.put("SIOC:SYS0:ML06:AO001",predictions[0].item())
                ctxt.put("SIOC:SYS0:ML06:AO002",predictions[1].item())
                ctxt.put("SIOC:SYS0:ML06:AO003",predictions[2].item())
                ctxt.put("SIOC:SYS0:ML06:AO004",predictions[3].item())
                ctxt.put("SIOC:SYS0:ML06:AO005",predictions[4].item())
            except Exception as e:
                logger.error(f"Failed to write predictions to EPICS: {e}")


            # Log predictions as metrics 
            for i, name in enumerate(lume_module.model.output_names):
                metric_name = name.replace(":", "_")
                value = predictions[i].item() if isinstance(predictions[i], torch.Tensor) else predictions[i]
                mlflow.log_metric(metric_name, value, step=step)
                logger.info(f"Logged metric: {metric_name} = {value}")

            # Wait for 1 minute
            step+=1
            time.sleep(60)

    except KeyboardInterrupt:
        logger.info("Interrupted by user.")

    finally:
        k2eg_client.close()
        logger.info("Stopped live monitoring.")

if mlflow.active_run():
    mlflow.end_run()

