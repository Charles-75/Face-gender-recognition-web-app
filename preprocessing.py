from utils import get_size, gender
from tqdm import tqdm
from glob import glob
import scipy.io
import numpy as np
import pandas as pd
import pickle
import cv2
import os

if not os.path.exists('./data/parameters'):
    os.makedirs('./data/parameters')


def preprocess_wiki():
    mat = scipy.io.loadmat('./data/wiki.mat')
    wiki = mat['wiki']
    wiki_genders = wiki[0][0][3][0]
    wiki_full_path = wiki[0][0][2][0]

    genders = []
    for i in range(len(wiki_genders)):
        if wiki_genders[i] == 1:
            genders.append('male')
        else:
            genders.append('female')

    paths = []
    for path in wiki_full_path:
        paths.append('wiki/' + path[0])

    data = np.vstack((genders, paths)).T
    wiki_df = pd.DataFrame(data=data, columns=['gender', 'path'])
    wiki_df.to_csv('./data/meta.csv', index=False)

    output_dir_male = "./data/male"
    output_dir_female = "./data/female"

    if not os.path.exists(output_dir_male):
        os.makedirs(output_dir_male)
    if not os.path.exists(output_dir_female):
        os.makedirs(output_dir_female)

    with tqdm(total=wiki_df.shape[0], desc="Preprocessing images") as pbar:
        for i, image in enumerate(wiki_df.values):
            try:
                img = cv2.imread(image[1], 1)
                if image[0] == 'male':
                    cv2.imwrite('./data/male/' + str(i) + '.jpg', img)
                else:
                    cv2.imwrite('./data/female/' + str(i) + '.jpg', img)
            except Exception as e:
                print(f"Failed to preprocess {image[0]} image {i}, error message : {str(e)}")
                pass
            pbar.update(1)


def preprocess_images():
    female_paths = glob('./data/crop/female_crop/*.png')
    male_paths = glob('./data/crop/male_crop/*.png')
    paths = female_paths + male_paths

    df = pd.DataFrame(data=paths, columns=['path'])
    # Add size (first dimension) for each image in the DataFrame
    df['size'] = df['path'].apply(get_size)
    # Remove outliers
    df = df[df['size'] > 60]
    # Add gender for each image in the DataFrame
    df['gender'] = df['path'].apply(gender)
    # Making the dataset balanced (50% male - 50% female)
    n = df['gender'].value_counts()[0] - df['gender'].value_counts()[1]
    df.drop(df[df['gender'] == 'male'].head(n).index, inplace=True)
    # Resize then convert image to structure data and add it to the DataFrame
    df['structure_data'] = df['path'].apply(resize_img)
    # Put structure data in columns in a new DataFrame
    df_images = df['structure_data'].apply(pd.Series)
    # Concat with gender column
    df_images = pd.concat((df['gender'], df_images), axis=1)
    pickle.dump(df_images, open('./data/parameters/df_images_100_100', 'wb'))


def preprocess_data():
    df_images = pickle.load(open('./data/parameters/df_images_100_100', 'rb'))
    # Removing missing values
    df_images.dropna(axis=0, inplace=True)
    # Split data into two parts : independent features and dependent feature
    X = df_images.iloc[:,1:].values
    y = df_images.iloc[:,0].values
    # Scale data with MinMaxScaler technique (X.min() = 0 and X.max() = 255)
    X = X / X.max()
    y = np.where(y == 'female', 1, 0)
    np.savez('./data/parameters/X_y_10000_norm.npz', X, y)


def resize_img(path_to_resize):
    try:
        # Read image
        img = cv2.imread(path_to_resize)
        # Convert image into grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # Resize into 100 x 100 array
        size = gray.shape[0]
        if size >= 100:
            gray_re = cv2.resize(gray, (100,100), cv2.INTER_AREA)  # Shrink image
        else:
            gray_re = cv2.resize(gray, (100,100), cv2.INTER_CUBIC)  # Enlarge image
        # Flatten image (1x10,000)
        return gray_re.flatten()
    except Exception as e:
        print(f"Failed to resize image, error message : {str(e)}")
        return None

