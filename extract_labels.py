import os
from imutils import paths

class_list = ['eosl', 'other', 'sl10', 'sl100', 'sl110', 'sl120', 'sl20', 'sl30', 'sl40', 'sl5', 'sl50', 'sl60', 'sl70', 'sl80', 'sl90']
class_to_id = {class_list[i]:i  for i in range(len(class_list))}

TEST_FOLDER = "dataset/test"
OUTPUT_FILE = "labels.txt"

test_classes = next(os.walk(TEST_FOLDER))[1]
with open(OUTPUT_FILE, "w") as outfile:
    for c in test_classes:
        class_dir = os.path.join(TEST_FOLDER, c)
        images = list(paths.list_images(class_dir))
        for i in images:
            outfile.write("{} {}\n".format(
                os.path.basename(i),
                class_to_id[c]
            ))
