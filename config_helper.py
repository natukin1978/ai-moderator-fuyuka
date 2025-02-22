import json
import os

import global_value as g


def readConfig(name: str = "config.json"):
    json_file_path = os.path.join(g.base_dir, name)
    with open(json_file_path, "r", encoding="utf-8") as f:
        return json.load(f)
