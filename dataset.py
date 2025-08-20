# Load dataset and print
import os
import sys
from datasets import load_dataset, load_from_disk

ds = load_from_disk('/export/fs06/kramesh3/datasets/wikipedia-large')
print(ds)