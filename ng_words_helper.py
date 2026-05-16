from config_helper import read_config


def read_ng_words(name: str = "ng_words.json"):
    result = read_config(name)
    if not result:
        return []
    return result
