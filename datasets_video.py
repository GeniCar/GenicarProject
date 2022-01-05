import os

ROOT_DATASET = 'video_datasets'


def return_dataset():
    root_data = os.path.join(ROOT_DATASET, 'drive/data')
    file_imglist_train = os.path.join(ROOT_DATASET, 'drive/train_videofolder.txt')
    file_imglist_val = os.path.join(ROOT_DATASET, 'drive/val_videofolder.txt')
    file_categories = os.path.join(ROOT_DATASET, 'drive/category.txt')

    prefix = '{:05d}.jpg'

    with open(file_categories) as f:
        lines = f.readlines()
    categories = [item.rstrip() for item in lines]

    return categories, file_imglist_train, file_imglist_val, root_data, prefix
