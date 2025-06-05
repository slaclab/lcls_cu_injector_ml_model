import yaml
import os
import torch
import logging
import json
import math
import matplotlib.pyplot as plt
from lume_model.utils import variables_from_yaml
from lume_model.models import TorchModel, TorchModule
import k2eg

# Setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load transformers and model
logger.info("Loading transformers and model components...")
input_variables, output_variables = variables_from_yaml("../model/pv_variables.yml")
lume_module = TorchModule("../model/pv_module.yml")

# Read live input from K2EG
logger.info("Reading input PVs from live EPICS data using K2EG...")
k2eg_client = k2eg.dml('lcls', 'app-three')
pv_map_path = os.path.join("..", "info", "pv_mapping.json")
pv_names = lume_module.model.input_names # list of PV names in correct order

input_parameter_values = {}
for pv_name in pv_names:
    # Initialize to default values
    input_parameter_values[pv_name] = input_variables[pv_names.index(pv_name)].default_value

    # Set available and non-constant inputs to PV values
    if pv_name not in ['CAMR:IN20:186:R_DIST', 'Pulse_length'] and not input_variables[pv_names.index(pv_name)].is_constant:
        try:
            input_parameter_values[pv_name] = k2eg_client.get('pva://' + pv_name, 5.0)["value"]
        except Exception as e:
            logger.warning(f"Failed to get PV {pv_name}: {e}. Value kept at default value.")

try:
    in_xrms_value = k2eg_client.get('pva://CAMR:IN20:186:XRMS')["value"]
    in_yrms_value = k2eg_client.get('pva://CAMR:IN20:186:YRMS')["value"]
    rdist = math.sqrt(in_xrms_value ** 2 + in_yrms_value ** 2)
    input_parameter_values['CAMR:IN20:186:R_DIST'] = rdist
except Exception as e:
    logger.error(f"Failed to compute R_DIST: {e}. Value kept at default value.")  

# Should always be in order now
for pv_name, val in input_parameter_values.items(): 
    logger.info(f"Input Dict: {pv_name} → {val}")

# Create tensor for torch module
input_tensor = torch.tensor(list(input_parameter_values.values())).to(dtype=lume_module.model.dtype, device=lume_module.model.device).unsqueeze(0)

# Predict
with torch.no_grad():
    predictions = lume_module(input_tensor)

logger.info(f"Predictions: {predictions}")

# Plot
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

plot_path = "epics_plot_lume.png"
plt.savefig(plot_path)
plt.close()
logger.info(f"Plot saved: {plot_path}")

# Write predictions to EPICS
'''logger.info("Writing predictions back to EPICS...")
try:
    k2eg_client.put('pva://LUME:OTRS:IN20:571:XRMS', predictions[0, 0].item(), 5.0)
    k2eg_client.put('pva://LUME:OTRS:IN20:571:YRMS', predictions[0, 1].item(), 5.0)
except Exception as e:
    logger.error(f"Failed to write predictions to EPICS: {e}")
'''
k2eg_client.close()
logger.info("Done.")
