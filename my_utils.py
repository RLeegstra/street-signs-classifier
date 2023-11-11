from sklearn.model_selection import train_test_split
from keras.preprocessing.image import ImageDataGenerator
import shutil
import os
import glob
import csv

def split_data(path_to_data, path_to_train, path_to_val, split_size=0.1):

    folders = os.listdir(path_to_data)

    for folder in folders:
        full_path = os.path.join(path_to_data, folder) # 
        images_paths = glob.glob(os.path.join(full_path, '*.png')) # load files with .png extension from certain folder

        x_train, x_val = train_test_split(images_paths, test_size=split_size) # split list of image paths into train and val groups

        for x in x_train:
            path_to_folder = os.path.join(path_to_train, folder)

            if not os.path.isdir(path_to_folder): # check if folderpath exists
                os.makedirs(path_to_folder)
        
            shutil.copy(x, path_to_folder)

        for x in x_val:
            path_to_folder = os.path.join(path_to_val, folder)

            if not os.path.isdir(path_to_folder): # check if folderpath exists
                os.makedirs(path_to_folder)
        
            shutil.copy(x, path_to_folder)




def order_test_set(path_to_images, path_to_csv): # create dictionary with name of the image + corresponding label. NOT REQUIRED

    dict_testset = {}


    try:
        with open(path_to_csv, 'r') as csvfile: # try to open csv-file

            reader = csv.reader(csvfile, delimiter=',')

            for i, row in enumerate(reader):

                if i == 0: # Dont do anything with the header row (we do not want this in our dict)
                    continue

                img_name = row[-1].replace('Test/', '') # grab last item in row (= path). remove Test/ from path to keep only the png-name
                label = row[-2] # grab second last item in row (= ClassID)

                path_to_folder = os.path.join(path_to_images, label)

                if not os.path.isdir(path_to_folder):
                    os.makedirs(path_to_folder)
                
                img_full_path = os.path.join(path_to_images, img_name)
                shutil.move(img_full_path, path_to_folder) # move images paths into path_to_folder

    except:
        print('[INFO] : Error reading csv file')


    finally:
        return dict_testset








def create_generators(batch_size, train_data_path, val_data_path, test_data_path):

    preprocessor = ImageDataGenerator( # pre-process images first before entering DL model
        rescale = 1 / 255.
    )

    train_generator = preprocessor.flow_from_directory(
        train_data_path,
        class_mode = 'categorical', # means we need categorical crossentropy for optimization 
        target_size = (60,60), # remember in the model, we want 60x60 images as input
        color_mode = 'rgb',
        shuffle = True, # randomization usually good
        batch_size = batch_size
    )

    test_generator = preprocessor.flow_from_directory(
        test_data_path,
        class_mode = 'categorical',
        target_size = (60,60), 
        color_mode = 'rgb',
        shuffle = False, # Sometimes not shuffled in test & val sets
        batch_size = batch_size
    )

    val_generator = preprocessor.flow_from_directory(
        val_data_path,
        class_mode = 'categorical',
        target_size = (60,60), 
        color_mode = 'rgb',
        shuffle = False,
        batch_size = batch_size
    )

    return train_generator, val_generator, test_generator

