import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2

import tqdm as tq

import face_recognition
from sklearn.metrics import f1_score

csv_path = str(sys.argv[1])
img_path = str(sys.argv[2]) + '/'
try: num_samples = int(sys.argv[3])
except: num_samples = None

try: training = sys.argv[4]
except: training = 0

data = pd.read_csv(csv_path)
data = data[:num_samples]

# Function to find and crop out the faces in the images
def face_crop(df, path_to_images, scaleFactor):
    col = df.columns
    m = len(df) 
    cropped_images_rgb1 = []
    cropped_images_rgb2 = []
    image_names_rgb1 = []
    image_names_rgb2 = []

    cascPath = './haarcascade_frontalface_alt.xml'

    print("Cropping the images...")
    for i in tq.tqdm(range(m)):
        image1 = plt.imread(path_to_images + df[col[0]][i])
        image2 = plt.imread(path_to_images + df[col[1]][i])

        faceCascade = cv2.CascadeClassifier(cascPath)
        gray1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)

        faces1 = faceCascade.detectMultiScale(gray1, scaleFactor=scaleFactor, minNeighbors=5, minSize=(100, 100))
        faces2 = faceCascade.detectMultiScale(gray2, scaleFactor=scaleFactor, minNeighbors=5, minSize=(100, 100))

        for (x, y, w, h) in faces1:
            image1 = cv2.resize(image1[y:y+h,x:x+w], (256,256))

        for (x, y, w, h) in faces2:
            image2 = cv2.resize(image2[y:y+h,x:x+w], (256,256))
            
        cropped_images_rgb1.append(image1)
        image_names_rgb1.append(df[col[0]][i])

        cropped_images_rgb2.append(image2)
        image_names_rgb2.append(df[col[1]][i])

    print("Cropping completed")
    return cropped_images_rgb1, image_names_rgb1, cropped_images_rgb2, image_names_rgb2

cropped_images1, image_names1, cropped_images2, image_names2 = face_crop(data, img_path, 1.05)

# This function returns the encoding of the argument image, using the face_recognition module
def get_encoding(image):
    height, width, _ = image.shape
    face_location = (0, width, height, 0)
    image_encoding = face_recognition.face_encodings(image, known_face_locations = [face_location])
    return image_encoding

# This function scans through the DataFrame and returns the list of unique "image ids" in the column number specified
def get_uniques(df, col_num):
    col = df.columns
    unique_image_ids = pd.unique(df[col[col_num]])
    unique_images = []
    unique_image_encodings = []

    return unique_image_ids, unique_images, unique_image_encodings   

# This function computes and stores the encoding of an image so that if the image is repeated, it could use the same encoding to save computation time
def get_unique_encodings(df, col_num, name_list, img_list):
    unique_image_ids, unique_images, unique_image_encodings = get_uniques(df, col_num)

    print("Working....")
    for i in tq.tqdm(unique_image_ids):
        img = img_list[name_list.index(i)]
        img_encoding = get_encoding(img)

        unique_images.append(img)
        unique_image_encodings.append(img_encoding)
    print(f"Computed face encodings for column \"{df.columns[col_num]}\"")

    return unique_image_ids, unique_images, unique_image_encodings

# The master function which compares the images referenced by the id in the input csv file and returns a dataframe containing the predicted labels. This also prints the accuracy and f1 score of the predictions (for training)
def compare_faces(df, show_img = False, training = False):
    col = df.columns
    results = []
    diff_abs = []
    img_col0 = []
    img_col1 = []

    unique_image_ids_col0, unique_images_col0, unique_image_encodings_col0 = get_unique_encodings(df, 0, image_names1, cropped_images1)
    unique_image_ids_col1, unique_images_col1, unique_image_encodings_col1 = get_unique_encodings(df, 1, image_names2, cropped_images2)

    for i in tq.tqdm(range(len(df))):
        index_col0 = list(unique_image_ids_col0).index(df[col[0]][i])
        index_col1 = list(unique_image_ids_col1).index(df[col[1]][i])
        #print("Comparing", df[col[0]][i], df[col[1]][i])
        
        img0 = unique_images_col0[index_col0]
        img0_encoding = unique_image_encodings_col0[index_col0][0]

        img1 = unique_images_col1[index_col1]
        img1_encoding = unique_image_encodings_col1[index_col1][0]

        img_col0.append(img0)
        img_col1.append(img1)

        result = int(face_recognition.compare_faces([img0_encoding], img1_encoding)[0])
        results.append(result)
        if training == 1: diff_abs.append(np.abs(result - df[col[2]][i]))

    if training == 1: 
        df_results = pd.DataFrame([results, diff_abs], index = ['predicted labels', 'absolute difference']).transpose()
        print(f"Accuracy : {1 - np.sum(diff_abs)/len(diff_abs)}")
        print(f"f1 score : {f1_score(results, df[col[2]])}")

    else: df_results = pd.DataFrame(results, columns = ['predicted labels']) 

    if show_img: 
        fig, axes = plt.subplots(3, 2)
        
        fig.suptitle("A few image pairs")        
        for i in range(3):
            axes[i,0].imshow(img_col0[i])
            axes[i,0].axes.get_xaxis().set_visible(False)
            axes[i,0].axes.get_yaxis().set_visible(False)
            
            axes[i,1].imshow(img_col1[i])
            axes[i,1].axes.get_xaxis().set_visible(False)
            axes[i,1].axes.get_yaxis().set_visible(False)
        plt.show()

    df_out = pd.concat([df, df_results], axis = 1)
    return df_out

output = compare_faces(data, show_img = False, training = training)
if training == 0: output.to_csv('./results.csv', index = None)