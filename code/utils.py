import yaml, os
def load_config(path):
    return yaml.safe_load(open(path))
