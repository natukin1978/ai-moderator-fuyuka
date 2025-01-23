import json
import os


def readConfig(name: str = "config.json"):
    current_dir = os.path.dirname(os.path.abspath(__file__))
    json_file_path = os.path.join(current_dir, name)
    with open(json_file_path, "r", encoding="utf-8") as f:
        return json.load(f)
