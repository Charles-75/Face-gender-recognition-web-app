from glob import glob
from tqdm import tqdm
import cv2
import os

# Load CascadeClassifier model to detect face
haar = cv2.CascadeClassifier('./models/haarcascade_frontalface_default.xml')


def crop_images():
    output_dir_female_crop = "./data/crop/female_crop"
    output_dir_male_crop = "./data/crop/male_crop"

    if not os.path.exists(output_dir_female_crop):
        os.makedirs(output_dir_female_crop)
    if not os.path.exists(output_dir_male_crop):
        os.makedirs(output_dir_male_crop)

    female_paths = glob('./data/female/*.jpg')
    male_paths = glob('./data/male/*.jpg')

    with tqdm(total=len(female_paths), desc="Crop female images progress") as pbar:
        for i, path in enumerate(female_paths):
            try:
                crop_image(path, 'female', i)
            except Exception as e:
                print(f"ERROR: Failed to preprocess female image {i}, error message : {str(e)}")
                pass
            pbar.update(1)

    with tqdm(total=len(male_paths), desc="Crop male images progress") as pbar:
        for i, path in enumerate(male_paths):
            try:
                crop_image(path, 'male', i)
            except Exception as e:
                print(f"ERROR: Failed to preprocess male image {i}, error message : {str(e)}")
                pass
            pbar.update(1)


def crop_image(path, gender, i):
    img = cv2.imread(path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = haar.detectMultiScale(gray, 1.5, 5)
    for x, y, w, h in faces:
        crop_img = img[y:y+h, x:x+w]
        cv2.imwrite(f'./data/crop/{gender}_crop/{gender}_{i}.png', crop_img)

