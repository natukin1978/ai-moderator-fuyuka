import json
import os

import global_value as g


def read_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def write_json(data: any, path: str):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

def read_config(name: str = "config.json"):
    if not os.path.isabs(name):
        name = os.path.join(g.base_dir, name)
    if not os.path.isfile(name):
        # 無いならひな形を参照
        name += ".template"
        if not os.path.isfile(name):
            return {}
    return read_json(name)

def write_config(data: any, name: str = "config.json"):
    if not os.path.isabs(name):
        name = os.path.join(g.base_dir, name)
    write_json(data, name)
