import shutil
import os
from sklearn.model_selection import train_test_split


# Utility function to move images
def move_files_to_folder(list_of_files, destination_folder):
    for f in list_of_files:
        try:
            shutil.move(f, destination_folder)
        except:
            print(f)
            assert False


# Read images and annotations
images = [os.path.join('data/obj/', x) for x in os.listdir('data/obj/') if x[-3:] == "png"]
annotations = [os.path.join('data/obj/', x) for x in os.listdir('data/obj/') if x[-3:] == "txt"]

images.sort()
annotations.sort()

# Split the dataset into train-valid-test splits
train_images, val_images, train_annotations, val_annotations = train_test_split(images, annotations, test_size=0.2,
                                                                                random_state=1)
val_images, test_images, val_annotations, test_annotations = train_test_split(val_images, val_annotations,
                                                                              test_size=0.5, random_state=1)

# Move the splits into their folders
move_files_to_folder(train_images, 'data/images/train/')
move_files_to_folder(val_images, 'data/images/val/')
move_files_to_folder(test_images, 'data/images/test/')
move_files_to_folder(train_annotations, 'data/labels/train/')
move_files_to_folder(val_annotations, 'data/labels/val/')
move_files_to_folder(test_annotations, 'data/labels/test/')
