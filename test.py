import numpy as np
import gzip

import json
with gzip.open("apple_train.jgz", "r") as fin:
    annotation = json.loads(fin.read())

print(annotation['110_13052_23173'][0])

with gzip.open("/mimer/NOBACKUP/groups/snic2022-6-266/davnords/co3d_anno/cleaned/apple_train.jgz", "r") as fin:
    annotation = json.loads(fin.read())

print(annotation['110_13052_23173'][0])