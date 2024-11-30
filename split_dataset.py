import os
import shutil
import random


def create_dataset_split(original_data_dir, output_data_dir):
    #
    os.makedirs(os.path.join(output_data_dir, 'train'), exist_ok=True)
    os.makedirs(os.path.join(output_data_dir, 'val'), exist_ok=True)
    os.makedirs(os.path.join(output_data_dir, 'test'), exist_ok=True)

    #
    for class_name in os.listdir(original_data_dir):
        class_folder = os.path.join(original_data_dir, class_name)

        if os.path.isdir(class_folder):  #
            images = os.listdir(class_folder)
            random.shuffle(images)  #

            #
            total_images = len(images)
            train_count = int(total_images * 0.7)
            val_count = int(total_images * 0.2)
            test_count = total_images - train_count - val_count  #

            print(f": {class_name}, : {total_images}, : {train_count}, : {val_count}, : {test_count}")


            os.makedirs(os.path.join(output_data_dir, 'train', class_name), exist_ok=True)
            os.makedirs(os.path.join(output_data_dir, 'val', class_name), exist_ok=True)
            os.makedirs(os.path.join(output_data_dir, 'test', class_name), exist_ok=True)
            
            for i, image in enumerate(images):
                # print(image)
                image_path = os.path.join(class_folder, image)
                # print(image_path)
                if i < train_count:
                    shutil.copy(image_path, os.path.join(output_data_dir, 'train', class_name))
                elif i < train_count + val_count:
                    shutil.copy(image_path, os.path.join(output_data_dir, 'val', class_name))
                else:
                    shutil.copy(image_path, os.path.join(output_data_dir, 'test', class_name))


original_data_dir = './AbNormal_Classification_dataset/original_image'
output_data_dir = './AbNormal_Classification_dataset/split_new_new'

create_dataset_split(original_data_dir, output_data_dir)