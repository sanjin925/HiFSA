from copy import deepcopy
import os
import yaml

from utils.hifsa_defaults import (
    build_hifsa_defaults,
    merge_config,
    validate_public_config,
    validate_resolved_config,
)

def normalize(img):
    import numpy as np

    img = img - np.min(img)
    img = img / (np.max(img) + 1e-8)
    return img

def read_text(filename):
    import pandas as pd

    df = pd.read_excel(filename)

    # row is a dictionary: {'Image': ..., 'Ground Truth': ..., 'Description': ...}
    return df.to_dict(orient="records")

class CfgNode(dict):
    """
    CfgNode represents an internal node in the configuration tree. It's a simple
    dict-like container that allows for attribute-based access to keys.
    """

    def __init__(self, init_dict=None, key_list=None, new_allowed=False):
        # Recursively convert nested dictionaries in init_dict into CfgNodes
        init_dict = {} if init_dict is None else deepcopy(init_dict)
        key_list = [] if key_list is None else key_list
        for k, v in init_dict.items():
            if type(v) is dict:
                # Convert dict to CfgNode
                init_dict[k] = CfgNode(v, key_list=key_list + [k])
        super(CfgNode, self).__init__(init_dict)

    def __getattr__(self, name):
        if name in self:
            return self[name]
        else:
            raise AttributeError(name)

    def __setattr__(self, name, value):
        self[name] = value

    def __str__(self):
        def _indent(s_, num_spaces):
            s = s_.split("\n")
            if len(s) == 1:
                return s_
            first = s.pop(0)
            s = [(num_spaces * " ") + line for line in s]
            s = "\n".join(s)
            s = first + "\n" + s
            return s

        r = ""
        s = []
        for k, v in sorted(self.items()):
            seperator = "\n" if isinstance(v, CfgNode) else " "
            attr_str = "{}:{}{}".format(str(k), seperator, str(v))
            attr_str = _indent(attr_str, 2)
            s.append(attr_str)
        r += "\n".join(s)
        return r

    def __repr__(self):
        return "{}({})".format(self.__class__.__name__, super(CfgNode, self).__repr__())
    
    def merge_from_list(self, opts):
        """
        Merge config options from a list like:
        ["MODEL.CLIP_MODEL", "unimedclip", "TRAIN.BATCH_SIZE", "16"]
        """
        opts = [] if opts is None else list(opts)
        if len(opts) % 2:
            raise ValueError("Command-line overrides must be KEY VALUE pairs")

        for dotted_key, raw_value in zip(opts[0::2], opts[1::2]):
            parts = dotted_key.split(".")
            node = self
            for part in parts[:-1]:
                if part not in node or not isinstance(node[part], CfgNode):
                    raise KeyError(f"Unknown configuration key: {dotted_key}")
                node = node[part]
            leaf = parts[-1]
            if leaf not in node:
                raise KeyError(f"Unknown configuration key: {dotted_key}")
            node[leaf] = yaml.safe_load(raw_value)

        validate_resolved_config(self)


def load_cfg_from_cfg_file(file: str):
    assert os.path.isfile(file) and file.endswith('.yaml'), \
        '{} is not a yaml file'.format(file)

    with open(file, 'r', encoding='utf-8') as f:
        cfg_from_file = yaml.safe_load(f)

    validate_public_config(cfg_from_file)
    dataset_name = cfg_from_file["DATASET"]["NAME"]
    resolved = merge_config(build_hifsa_defaults(dataset_name), cfg_from_file)
    validate_resolved_config(resolved)
    return CfgNode(resolved)


def to_plain_dict(value):
    if isinstance(value, dict):
        return {key: to_plain_dict(item) for key, item in value.items()}
    if isinstance(value, list):
        return [to_plain_dict(item) for item in value]
    return value


def dump_resolved_config(cfg, path):
    with open(path, "w", encoding="utf-8") as handle:
        yaml.safe_dump(
            to_plain_dict(cfg),
            handle,
            allow_unicode=True,
            sort_keys=False,
        )
