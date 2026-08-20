# One-time setup: creates the dataSet/trainingData and dataSet/testingData
# folder trees, with one subfolder per class (blank "0" + A-Z).

import os
import string

BASE = "dataSet"
SPLITS = ["trainingData", "testingData"]
LABELS = ["0"] + list(string.ascii_uppercase)


def ensure(path):
    if not os.path.exists(path):
        os.makedirs(path)


def main():
    ensure(BASE)
    for split in SPLITS:
        ensure(os.path.join(BASE, split))
        for label in LABELS:
            ensure(os.path.join(BASE, split, label))
    print("Dataset folder structure ready under:", BASE)


if __name__ == "__main__":
    main()
