from glob import glob
import os
import cv2
import random
import string

total_file = f'video_datasets/drive/total_dataset.txt'
train_file = f'video_datasets/drive/train_videofolder.txt'
val_file = f'video_datasets/drive/val_videofolder.txt'


def random_char(n):
    return ''.join(random.choice(string.ascii_letters) for x in range(n)).upper()


def make_dir(mydir):
    check_folder = os.path.isdir(mydir)

    # If folder doesn't exist, then create it.
    if not check_folder:
        os.makedirs(mydir)


def extract_frame(input_file, output_file):
    output_dir = f'video_datasets/drive/{output_file}'
    make_dir(output_dir)

    vid = cv2.VideoCapture(input_file)
    success, image = vid.read()

    # resize image
    image = cv2.resize(image, (256, 256))

    frame_num = 0

    while success:
        file_name = f'{frame_num:05d}.jpg'
        cv2.imwrite(os.path.join(output_dir, file_name), image)
        success, image = vid.read()
        frame_num += 1

    return 0, frame_num


def write_total_dataset(videos, label):
    for v in videos:
        output_file = random_char(5)
        print(f'class{label} {v} -> {output_file}')
        start_frame, end_frame = extract_frame(v, output_file)

        with open(total_file, "a") as file:
            file.write(f'{output_file} {end_frame} {label}\n')

def split_dataset():
    file = open(total_file, "r")
    data = file.read().splitlines()
    file.close()

    random.shuffle(data)

    num_data = len(data)

    train_dataset = data[:int(num_data * 0.7)]
    val_dataset = data[int(num_data * 0.7):]

    return train_dataset, val_dataset

def write_dataset(train_dataset, val_dataset)
    with open(train_file, "w+") as file:
        file.write('\n'.join(train_dataset))

    with open(val_file, "w+") as file:
        file.write('\n'.join(val_dataset))


if __name__ == "__main__":
    class1_videos = glob('./가까워짐/*.mp4')
    class2_videos = glob('./멀어짐/*.mp4')
    class3_videos = glob('./사고/*.mp4')

    write_total_dataset(class1_videos, 1)
    write_total_dataset(class2_videos, 2)
    write_total_dataset(class3_videos, 3)

    split_dataset()
    write_dataset()

