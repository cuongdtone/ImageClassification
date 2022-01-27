import os
import shutil
from glob import glob
import random
import argparse

dataset = 'dataset_example/'
splitted_dataset = 'dataset/'
train_ratio = 0.8
val_ratio = 0.1
test_ratio = 0.1

def train_test_split(dataset, splitted_dataset, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1):
    while os.path.exists(splitted_dataset):
        print("Directory already exists !")
        if input("Delete (D) or Skip(S) ?") == 'D':
            shutil.rmtree(splitted_dataset)
        else:
            print("Exit !")
            return 0

    for i in ['', '/train', 'val', '/test']:
        os.mkdir(splitted_dataset + i)

    list_dir_class = glob(dataset + '/*')
    list_class = []
    for i in list_dir_class:
        temp = i.split('/')[-1]
        list_class.append(temp)
        for j in ['/train', 'val', '/test']:
            os.mkdir(splitted_dataset + j + '/' + temp)

    for i in list_class:
        list_img = glob(dataset +'/' + i + '/*')
        random.shuffle(list_img)
        length = len(list_img)
        for idx in range(length):
            if idx<round(train_ratio*length):
                shutil.copy(list_img[idx], splitted_dataset + '/train/' + i)
            elif idx<round((train_ratio+val_ratio)*length):
                shutil.copy(list_img[idx], splitted_dataset + '/val/' + i)
            else:
                shutil.copy(list_img[idx], splitted_dataset + '/test/' + i)
    print("Complete !")
    return 0
if __name__ == '__main__':
    train_test_split(dataset, splitted_dataset)




