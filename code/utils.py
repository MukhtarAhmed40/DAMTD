import os
import numpy as np
import tensorflow as tf
import yaml
import random

def load_config(path):
    with open(path, 'r') as f:
        return yaml.safe_load(f)

def seed_everything(seed=42):
    import os, random
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)
