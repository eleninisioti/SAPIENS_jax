import numpy as np
import jax.numpy as jnp

def preprocess_dict(data):
    """
    Recursively preprocess dictionary values for YAML serialization.
    Converts JAX or NumPy arrays to lists, and handles other custom types.
    """
    if isinstance(data, (np.ndarray, jnp.ndarray)):
        return data.tolist()  # Convert arrays to Python lists
    elif isinstance(data, dict):
        return {k: preprocess_dict(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [preprocess_dict(item) for item in data]
    elif isinstance(data, tuple):
        return tuple(preprocess_dict(item) for item in data)
    else:
        return data