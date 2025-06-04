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

# Load transformers and model components
logger.info("Loading transformers and model components...")
input_sim_to_nn = torch.load("model/input_sim_to_nn.pt")
output_sim_to_nn = torch.load("model/output_sim_to_nn.pt")
input_pv_to_sim = torch.load("model/input_pv_to_sim.pt")
output_pv_to_sim = torch.load("model/output_pv_to_sim.pt")
input_variables, output_variables = variables_from_yaml("model/pv_variables.yml")
lume_model = TorchModel("model/pv_model.yml")
lume_module = TorchModule("model/pv_module.yml")

# Read live input from K2EG
logger.info("Reading input PVs from live EPICS data using K2EG...")
k2eg_client = k2eg.dml('lcls', 'app-three')
pv_map_path = os.path.join("info", "pv_mapping.json")
pv_names = json.load(open(pv_map_path))['pv_name_to_sim_name']
input_parameter_values = {'CAMR:IN20:186:R_DIST': None, 'Pulse_length': 1.8550514181818183}

for pv_name in pv_names.keys():
    if pv_name not in input_parameter_values and 'OTRS' not in pv_name and ':' in pv_name:
        input_parameter_values[pv_name] = None

k2eg_pvs_to_monitor = ['ca://' + pv for pv in input_parameter_values.keys() if
                       pv not in ['CAMR:IN20:186:R_DIST', 'Pulse_length']]

for pv_name in k2eg_pvs_to_monitor:
    try:
        input_parameter_values[pv_name.replace('ca://', '')] = k2eg_client.get(pv_name, 5.0)["value"]
    except Exception as e:
        logger.warning(f"Failed to get PV {pv_name}: {e}")
        input_parameter_values[pv_name.replace('ca://', '')] = 0.0

try:
    in_xrms_value = k2eg_client.get('ca://CAMR:IN20:186:XRMS')["value"]
    in_yrms_value = k2eg_client.get('ca://CAMR:IN20:186:YRMS')["value"]
    rdist = math.sqrt(in_xrms_value ** 2 + in_yrms_value ** 2)
    input_parameter_values['CAMR:IN20:186:R_DIST'] = rdist
except Exception as e:
    logger.error(f"Failed to compute R_DIST: {e}")
    input_parameter_values['CAMR:IN20:186:R_DIST'] = 0.0

with open("model/pv_variables.yml", "r") as f:
    yaml_data = yaml.safe_load(f)

input_variable_names = list(yaml_data["input_variables"].keys())
ordered_input_values = []

for pv_name in input_variable_names:
    if pv_name not in input_parameter_values:
        raise KeyError(f"Missing PV value for '{pv_name}' in input_parameter_values!")

    val = input_parameter_values[pv_name]
    ordered_input_values.append(val)
    logger.info(f"Ordered Input PV: {pv_name} → {val}")


input_tensor = torch.tensor(ordered_input_values, dtype=torch.float32).unsqueeze(0)
#input_sim_tensor = input_pv_to_sim.transform(input_tensor)
#inputs_small = input_sim_to_nn.transform(input_sim_tensor)


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
    ax_i.plot(predictions[0].detach().numpy(), "C1x", label="predictions")
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

