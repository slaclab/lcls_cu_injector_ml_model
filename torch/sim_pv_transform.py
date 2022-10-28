from botorch.models.transforms import Normalize
import torch


def create_transformer(pv_info, sim_key_list):
    """creates a transformer to map simulation values to pv's, call untransform to
    reverse"""
    sim_to_pv_scale = [pv_info["sim_to_pv_factor"][ele] for ele in sim_key_list]
    pv_key_list = [pv_info["sim_name_to_pv_name"][ele] for ele in sim_key_list]

    transformer = Normalize(len(sim_key_list))
    transformer.ranges = 1.0 / torch.tensor(sim_to_pv_scale, dtype=torch.double)
    transformer.key_mapping = pv_info["sim_name_to_pv_name"]
    transformer.sim_key_list = sim_key_list
    transformer.pv_key_list = pv_key_list
    transformer.eval()

    return transformer