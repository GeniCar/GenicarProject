import os
import torch
import torchvision
import torchvision.datasets as datasets


ROOT_DATASET = 'video_datasets'


# 2 class data set (getting closer, getting farther away)
def return_something(modality):

    if modality == 'RGB':
        filename_categories = 'something/category.txt'
        root_data = 'video_datasets/something/data'
        filename_imglist_train = 'something/train_videofolder.txt'
        filename_imglist_val = 'something/val_videofolder.txt'
        prefix = '{:05d}.jpg'
    
    else:
        print('no such modality:'+modality)

    return filename_categories, filename_imglist_train, filename_imglist_val, root_data, prefix



# 3 class data set (getting closer, getting farther away ,accident)
def return_drive(modality):
    if modality == 'RGB':
        filename_categories = 'drive/category.txt'
        root_data = 'video_datasets/drive/data'
        filename_imglist_train = 'drive/train_videofolder.txt'
        filename_imglist_val = 'drive/val_videofolder.txt'
        prefix = '{:05d}.jpg'

    else:
        print('no such modality:'+modality)

    return filename_categories, filename_imglist_train, filename_imglist_val, root_data, prefix





def return_dataset(dataset, modality):
    dict_single = {'something':return_something,'drive':return_drive}
    if dataset in dict_single:
        file_categories, file_imglist_train, file_imglist_val, root_data, prefix = dict_single[dataset](modality)
    else:
        raise ValueError('Unknown dataset '+dataset)

    file_imglist_train = os.path.join(ROOT_DATASET, file_imglist_train)
    file_imglist_val = os.path.join(ROOT_DATASET, file_imglist_val)
    file_categories = os.path.join(ROOT_DATASET, file_categories)
    with open(file_categories) as f:
        lines = f.readlines()
    categories = [item.rstrip() for item in lines]
    return categories, file_imglist_train, file_imglist_val, root_data, prefix