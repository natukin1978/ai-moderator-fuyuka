import os


def readText(name: str):
    current_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(current_dir, name)
    if not os.path.isfile(file_path):
        # 無いならひな形を参照
        file_path += ".template"
        if not os.path.isfile(file_path):
            return ""
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read()
