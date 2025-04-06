import os

import global_value as g


def read_text(name: str) -> str:
    if not os.path.isabs(name):
        name = os.path.join(g.base_dir, name)
    if not os.path.isfile(name):
        # 無いならひな形を参照
        name += ".template"
        if not os.path.isfile(name):
            return ""
    with open(name, "r", encoding="utf-8") as f:
        return f.read()


def read_texts(name: str) -> list[str]:
    return read_text(name).splitlines()


def read_text_set(name: str) -> set[str]:
    return set(read_texts(name))
