import os
from pathlib import Path

DATASET_DIR = '/mnt/e/Data/mjsynth'
data_path = Path(DATASET_DIR) / 'mnt' / 'ramdisk' / 'max' / '90kDICT32px'
output_path = Path('output')

os.makedirs(str(output_path), exist_ok=True)

with open(str(data_path / 'annotation_train.txt')) as fp:
    lines = fp.readlines()

train_fp = open(str(output_path / 'train_list.txt'), 'w')
for line in lines:
    imgpath = line.strip().split(' ')[0]
    label = imgpath.split('/')[-1].split('_')[1].lower()
    label = label + '$'
    label = ':'.join(label)
    imgpath = str(data_path / imgpath)
    output = ' '.join([imgpath, label])
    train_fp.write(output)

train_fp.close()


with open(str(data_path / 'annotation_test.txt')) as fp:
    lines = fp.readlines()

test_fp = open(str(output_path / 'test_list.txt'), 'w')
for line in lines:
    imgpath = line.strip().split(' ')[0]
    label = imgpath.split('/')[-1].split('_')[1].lower()
    label = label + '$'
    label = ':'.join(label)
    imgpath = str(data_path / imgpath)
    output = ' '.join([imgpath, label])
    test_fp.write(output)

test_fp.close()

with open(str(data_path / 'annotation_test.txt')) as fp:
    lines = fp.readlines()

val_fp = open(str(output_path / 'val_list.txt'), 'w')
for line in lines:
    imgpath = line.strip().split(' ')[0]
    label = imgpath.split('/')[-1].split('_')[1].lower()
    label = label + '$'
    label = ':'.join(label)
    imgpath = str(data_path / imgpath)
    output = ' '.join([imgpath, label])
    val_fp.write(output)

val_fp.close()
