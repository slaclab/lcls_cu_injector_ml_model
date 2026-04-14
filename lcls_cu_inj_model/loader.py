"""Model loading utilities"""

import os
from pathlib import Path
from lume_torch.models import TorchModel


def get_resource_path(filename):
    """Get the absolute path to a resource file."""
    package_dir = Path(__file__).parent
    resource_path = package_dir / "resources" / filename
    
    if not resource_path.exists():
        raise FileNotFoundError(f"Resource file not found: {resource_path}")
    
    return str(resource_path)


def load_model(use_cpu=False):
    """
    Load the LCLS FEL TorchModel.
    
    Args:
        use_cpu (bool): If True, loads the CPU version of the model.
                       Default is False (uses GPU version).
    
    Returns:
        TorchModel: Loaded model instance ready for inference.
    
    Example:
        >>> from lcls_fel_model import load_model
        >>> model = load_model()
        >>> # or for CPU
        >>> model = load_model(use_cpu=True)
    """
    config_path = get_resource_path("model_config.yaml")
    
    # Load the model using the config
    model = TorchModel(config_path)
    
    return model