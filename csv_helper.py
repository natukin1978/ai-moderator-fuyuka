import csv
import os


def read_csv_to_list(path: str) -> list[list[any]]:
    if not os.path.isfile(path):
        return {}
    with open(path, encoding="utf-8") as f:
        reader = csv.reader(f)
        return [row for row in reader]
