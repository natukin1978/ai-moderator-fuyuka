import os
import sys
from unittest import TestLoader, TextTestRunner

import global_value as g

g.base_dir = os.path.join(os.path.dirname(os.path.abspath(sys.argv[0])), "tests")


def main(path):
    loader = TestLoader()
    test = loader.discover(path)
    runner = TextTestRunner()
    runner.run(test)


if __name__ == "__main__":
    main("tests")
