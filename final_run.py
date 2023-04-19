# Machine Learning and Data Science Imports (basics)
import tensorflow as tf;

import pandas as pd; pd.options.mode.chained_assignment = None; pd.set_option('display.max_columns', None);
import numpy as np
import sklearn
import cv2
import mediapipe as mp

# Built-In Imports (mostly don't worry about these)
from collections import Counter
from datetime import datetime
from zipfile import ZipFile
from glob import glob
import warnings
import requests
import hashlib
import imageio
import IPython
import sklearn
import urllib
import zipfile
import pickle
import random
import shutil
import string
import json
import math
import time
import gzip
import ast
import sys
import io
import os
import gc
import re


ROWS_PER_FRAME = 543  # number of landmarks per frame
def load_relevant_data_subset(pq_path):
    data_columns = ['x', 'y', 'z']
    data = pd.read_parquet(pq_path, columns=data_columns)
    n_frames = int(len(data) / ROWS_PER_FRAME)
    data = data.values.reshape(n_frames, ROWS_PER_FRAME, len(data_columns))
    return data.astype(np.float32)

# Manually add decoder
decoder = {v:k for k, v in {'tv': 0, 'after': 1, 'airplane': 2, 'all': 3, 'alligator': 4, 'animal': 5, 'another': 6, 'any': 7, 'apple': 8, 'arm': 9, 'aunt': 10, 'awake': 11, 'backyard': 12, 'bad': 13, 'balloon': 14, 
           'bath': 15, 'because': 16, 'bed': 17, 'bedroom': 18, 'bee': 19, 'before': 20, 'beside': 21, 'better': 22, 'bird': 23, 'black': 24, 'blow': 25, 'blue': 26, 'boat': 27, 'book': 28, 'boy': 29, 
           'brother': 30, 'brown': 31, 'bug': 32, 'bye': 33, 'callonphone': 34, 'can': 35, 'car': 36, 'carrot': 37, 'cat': 38, 'cereal': 39, 'chair': 40, 'cheek': 41, 'child': 42, 'chin': 43, 'chocolate': 44, 
           'clean': 45, 'close': 46, 'closet': 47, 'cloud': 48, 'clown': 49, 'cow': 50, 'cowboy': 51, 'cry': 52, 'cut': 53, 'cute': 54, 'dad': 55, 'dance': 56, 'dirty': 57, 'dog': 58, 'doll': 59, 'donkey': 60, 
           'down': 61, 'drawer': 62, 'drink': 63, 'drop': 64, 'dry': 65, 'dryer': 66, 'duck': 67, 'ear': 68, 'elephant': 69, 'empty': 70, 'every': 71, 'eye': 72, 'face': 73, 'fall': 74, 'farm': 75, 'fast': 76, 
           'feet': 77, 'find': 78, 'fine': 79, 'finger': 80, 'finish': 81, 'fireman': 82, 'first': 83, 'fish': 84, 'flag': 85, 'flower': 86, 'food': 87, 'for': 88, 'frenchfries': 89, 'frog': 90, 'garbage': 91, 
           'gift': 92, 'giraffe': 93, 'girl': 94, 'give': 95, 'glasswindow': 96, 'go': 97, 'goose': 98, 'grandma': 99, 'grandpa': 100, 'grass': 101, 'green': 102, 'gum': 103, 'hair': 104, 'happy': 105, 'hat': 106, 
           'hate': 107, 'have': 108, 'haveto': 109, 'head': 110, 'hear': 111, 'helicopter': 112, 'hello': 113, 'hen': 114, 'hesheit': 115, 'hide': 116, 'high': 117, 'home': 118, 'horse': 119, 'hot': 120, 'hungry': 121, 
           'icecream': 122, 'if': 123, 'into': 124, 'jacket': 125, 'jeans': 126, 'jump': 127, 'kiss': 128, 'kitty': 129, 'lamp': 130, 'later': 131, 'like': 132, 'lion': 133, 'lips': 134, 'listen': 135, 'look': 136, 
           'loud': 137, 'mad': 138, 'make': 139, 'man': 140, 'many': 141, 'milk': 142, 'minemy': 143, 'mitten': 144, 'mom': 145, 'moon': 146, 'morning': 147, 'mouse': 148, 'mouth': 149, 'nap': 150, 'napkin': 151, 
           'night': 152, 'no': 153, 'noisy': 154, 'nose': 155, 'not': 156, 'now': 157, 'nuts': 158, 'old': 159, 'on': 160, 'open': 161, 'orange': 162, 'outside': 163, 'owie': 164, 'owl': 165, 'pajamas': 166, 'pen': 167,
           'pencil': 168, 'penny': 169, 'person': 170, 'pig': 171, 'pizza': 172, 'please': 173, 'police': 174, 'pool': 175, 'potty': 176, 'pretend': 177, 'pretty': 178, 'puppy': 179, 'puzzle': 180, 'quiet': 181, 'radio': 182, 
           'rain': 183, 'read': 184, 'red': 185, 'refrigerator': 186, 'ride': 187, 'room': 188, 'sad': 189, 'same': 190, 'say': 191, 'scissors': 192, 'see': 193, 'shhh': 194, 'shirt': 195, 'shoe': 196, 'shower': 197, 
           'sick': 198, 'sleep': 199, 'sleepy': 200, 'smile': 201, 'snack': 202, 'snow': 203, 'stairs': 204, 'stay': 205, 'sticky': 206, 'store': 207, 'story': 208, 'stuck': 209, 'sun': 210, 'table': 211, 'talk': 212, 
           'taste': 213, 'thankyou': 214, 'that': 215, 'there': 216, 'think': 217, 'thirsty': 218, 'tiger': 219, 'time': 220, 'tomorrow': 221, 'tongue': 222, 'tooth': 223, 'toothbrush': 224, 'touch': 225, 'toy': 226, 
           'tree': 227, 'uncle': 228, 'underwear': 229, 'up': 230, 'vacuum': 231, 'wait': 232, 'wake': 233, 'water': 234, 'wet': 235, 'weus': 236, 'where': 237, 'white': 238, 'who': 239, 'why': 240, 'will': 241, 
           'wolf': 242, 'yellow': 243, 'yes': 244, 'yesterday': 245, 'yourself': 246, 'yucky': 247, 'zebra': 248, 'zipper': 249}.items()}


interpreter = tf.lite.Interpreter(model_path='model.tflite')
found_signatures = list(interpreter.get_signature_list().keys())
prediction_fn = interpreter.get_signature_runner("serving_default")

data = pd.read_csv('train_isl_table.csv')

output = prediction_fn(inputs=load_relevant_data_subset("100015657.parquet"))
sign = np.argmax(output["outputs"])

print("PRED : ", decoder[sign])

"""# tested using data generated from video"""

# !pip install mediapipe
# !pip install pyarrow
# !pip install fastparquet


# Initialize Mediapipe holistic model
mp_holistic = mp.solutions.holistic
holistic = mp_holistic.Holistic()

# OpenCV video capture
cap = cv2.VideoCapture('blow.mp4')

# Initialize empty DataFrame to store landmark data
landmark_data = pd.DataFrame(columns=['frame', 'label', 'x', 'y', 'z'])

while True:
    ret, frame = cap.read()

    if not ret:
        break

    # Convert frame to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Detect landmarks using Mediapipe holistic model
    results = holistic.process(rgb_frame)

    # Extract landmark data from results
    frame_num = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
    for label, landmark_list in zip(['face', 'left_hand','pose' ,'right_hand'], [results.face_landmarks.landmark, results.left_hand_landmarks.landmark,results.pose_landmarks.landmark,results.right_hand_landmarks.landmark]):
        for landmark in landmark_list:
            x, y, z = landmark.x, landmark.y, landmark.z
            landmark_data = landmark_data.append({'frame': frame_num, 'label': label, 'x': x, 'y': y, 'z': z}, ignore_index=True)

    # Draw landmarks on frame for visualization (optional)
#     holistic.draw_landmarks(frame, results.face_landmarks, mp_holistic.FACE_CONNECTIONS)
#     holistic.draw_landmarks(frame, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
#     holistic.draw_landmarks(frame, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)
#     holistic.draw_landmarks(frame, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
    

    # Display frame with landmarks (optional)
    # cv2.imshow('Landmarks', frame)
    # if cv2.waitKey(1) & 0xFF == ord('q'):
    #     break

# # Release video capture and close all OpenCV windows
# cap.release()
# cv2.destroyAllWindows()

# Save landmark data to Parquet file
landmark_data.to_parquet('landmarks_data.parquet', index=False)



output = prediction_fn(inputs=load_relevant_data_subset("landmarks_data.parquet"))
sign = np.argmax(output["outputs"])

print("PRED : ", decoder[sign])
