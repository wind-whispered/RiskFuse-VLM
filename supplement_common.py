"""Resolve the supplementary workspace from the repository's unique config."""
from copy import deepcopy
from pathlib import Path

from hmqr.config import load_config


def load_supplement_config(path='configs/default.yaml'):
    cfg=load_config(path)
    cfg.raw=deepcopy(cfg.raw)
    settings=cfg.raw['release_data']['supplement']

    def merge(destination,source):
        for key,value in source.items():
            if isinstance(value,dict) and isinstance(destination.get(key),dict):
                merge(destination[key],value)
            else:
                destination[key]=deepcopy(value)

    merge(cfg.raw,settings['overrides'])
    workspace=cfg.abspath(settings['workspace'])
    workspace.mkdir(parents=True,exist_ok=True)
    return cfg,workspace
