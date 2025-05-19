import json
import pprint
import torch

# model info
with open('info/model.json', 'r') as f:
    model_info = json.load(f)
print("Model Info")
pprint.pprint(model_info)
print()

# normalization (corresponding to sim_to_nn transformers)
with open('info/normalization.json', 'r') as f:
    normalization_info = json.load(f)
print("Normalization Info")
pprint.pprint(normalization_info)
print()

# pv mapping (corresponding to pv_to_sim transformers)
with open('info/pv_mapping.json', 'r') as f:
    pv_info = json.load(f)
print("PV Info")
pprint.pprint(pv_info)
print()

# example data
inputs_small = torch.load("info/inputs_small.pt")
outputs_small = torch.load("info/outputs_small.pt")

print("Input shape (n_samples, n_dim):", inputs_small.shape)
print("Output shape (n_samples, n_dim):", outputs_small.shape)



