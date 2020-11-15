import os
from imutils import paths

RESULT_FILE = "test_result.txt"
# RESULT_FILE = "/mnt/DATA/GRADUATION_RESEARCH/Jetson_Sources/carsmartcam/build/bin/test_result_TensorRT_fp32.txt"
LABEL_FILE = "labels.txt"

def read_file(file):
    with open(file, 'r') as infile:
        results = infile.readlines()
        results = [line for line in results if len(line) > 0]
        results = [line.split() for line in results]
        results = [(os.path.basename(x[0]), int(x[1])) for x in results]
        results.sort()
    return results

results = read_file(RESULT_FILE)
labels = read_file(LABEL_FILE)

# Extract object ids
results = [x[1] for x in results]
labels = [x[1] for x in labels]

# Eval
from sklearn.metrics import f1_score
f1 = f1_score(labels, results, average='micro')
print("F1: {}".format(f1))