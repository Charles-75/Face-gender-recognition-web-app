from glob import glob
import cv2
import pickle
import os


# Load CascadeClassifier model to detect face
haar = cv2.CascadeClassifier('./models/haarcascade_frontalface_default.xml')

# Load all models and preprocess mean
svm_path = glob('./models/svm*')[0]
pca_path = glob('./models/pca*')[0]
mean = pickle.load(open('./models/mean_preprocess.pickle', 'rb'))
model_svm = pickle.load(open(svm_path, 'rb'))
model_pca = pickle.load(open(pca_path, 'rb'))
print('Model loaded sucessfully')

# Settings
gender_pre = ['Male', 'Female']
font = cv2.FONT_HERSHEY_SIMPLEX

PREDICT_FOLDER = './static/predict'
if not os.path.exists(PREDICT_FOLDER):
    os.makedirs(PREDICT_FOLDER)


def pipeline_model(path, filename, color='bgr'):
    # Read image in cv2
    img = cv2.imread(path)
    # Convert image into gray scale
    if color == 'bgr':
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # Crop the face (using haar cascade classifier)
    faces = haar.detectMultiScale(gray, 1.5, 3)
    for x, y, w, h in faces:
        # Drawing a rectangle around the face
        cv2.rectangle(img, (x,y), (x+w, y+h), (255,255,0), 2)
        # Crop the image
        crop_image = gray[y:y+h, x:x+w]
        # Normalization
        crop_image = crop_image / 255.0
        # Resize images (100,100)
        if crop_image.shape[1] > 100:
            crop_image_resize = cv2.resize(crop_image, (100,100), cv2.INTER_AREA)   # Shrink image
        else:
            crop_image_resize = cv2.resize(crop_image, (100, 100), cv2.INTER_CUBIC)  # Enlarge image
        # Flattening image (1x10,000)
        crop_image_reshape = crop_image_resize.reshape(1,10000)
        # Substract with mean
        crop_image_mean = crop_image_reshape - mean
        # Get eigen image
        eigen_image = model_pca.transform(crop_image_mean)
        # Pass eigen image to svm model
        results = model_svm.predict_proba(eigen_image)[0]
        # Get the class with highest prediction (0 => male or 1 => female)
        prediction = results.argmax()
        score = results[prediction]
        text = "%s : %0.2f"%(gender_pre[prediction], score)
        cv2.putText(img, text, (x,y), font, 1, (255,255,0), 2)

    cv2.imwrite(f'./static/predict/{filename}', img)

