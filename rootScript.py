import tkinter as tk
from tkinter import *
from tkinter import filedialog
from PIL import ImageTk, Image
import os
import cv2
import sys
import logging as log
import datetime as dt
from time import sleep
from tkinter import ttk
from keras.models import Sequential
from keras.layers import Conv2D, ZeroPadding2D, Activation, Input, concatenate
from keras.models import Model
from keras.layers.normalization import BatchNormalization
from keras.layers.pooling import MaxPooling2D, AveragePooling2D
from keras.layers.merge import Concatenate
from keras.layers.core import Lambda, Flatten, Dense
from keras.initializers import glorot_uniform
from keras.engine.topology import Layer
from keras import backend as K
K.set_image_data_format('channels_first')
import matplotlib.pyplot as plt
import cv2
import os
import numpy as np
from numpy import genfromtxt
import pandas as pd
import tensorflow as tf
from fr_utils import *
from inceptionblock import *
import keras.backend.tensorflow_backend as tfback
import cv2
import sys
import logging as log
import datetime as dt
from time import sleep
print("tf.__version__ is", tf.__version__)
print("tf.keras.__version__ is:", tf.keras.__version__)
#import face_recognition as fr
import os
import cv2
#import face_recognition
import numpy as np
from time import sleep
import subprocess


root = tk.Tk()
global id
up = False
####################################Hamzas Code Starts Here###########################################



from gtts import gTTS 
from playsound import playsound   
# This module is imported so that we can  
# play the converted audio 
import os 

def sound(message):
# The text that you want to convert to audio 
    mytext = message
    language = 'en'
    myobj = gTTS(text=mytext, lang=language, slow=True) 
    myobj.save("welcomehere.mp3") 
    playsound("welcomehere.mp3")
    os.remove("welcomehere.mp3")
# Playing the converted file 


def _get_available_gpus():
    """Get a list of available gpu devices (formatted as strings).

    # Returns
        A list of available GPU devices.
    """
    #global _LOCAL_DEVICES
    if tfback._LOCAL_DEVICES is None:
        devices = tf.config.list_logical_devices()
        tfback._LOCAL_DEVICES = [x.name for x in devices]
    return [x for x in tfback._LOCAL_DEVICES if 'device:gpu' in x.lower()]

tfback._get_available_gpus = _get_available_gpus
def triplet_loss(y_true, y_pred, alpha = 0.2):
    """
    Implementation of the triplet loss as defined by formula (3)
    
    Arguments:
    y_true -- true labels, required when you define a loss in Keras, you don't need it in this function.
    y_pred -- python list containing three objects:
            anchor -- the encodings for the anchor images, of shape (None, 128)
            positive -- the encodings for the positive images, of shape (None, 128)
            negative -- the encodings for the negative images, of shape (None, 128)
    
    Returns:
    loss -- real number, value of the loss
    """
    
    anchor, positive, negative = y_pred[0], y_pred[1], y_pred[2]
    
    ### START CODE HERE ### (≈ 4 lines)
    # Step 1: Compute the (encoding) distance between the anchor and the positive
    pos_dist = tf.reduce_sum(tf.square(tf.subtract(anchor,positive)),-1)
    # Step 2: Compute the (encoding) distance between the anchor and the negative
    neg_dist = tf.reduce_sum((tf.square(tf.subtract(anchor,negative))),-1)
    # Step 3: subtract the two previous distances and add alpha.
    basic_loss = tf.add(tf.subtract(pos_dist,neg_dist),alpha)
    # Step 4: Take the maximum of basic_loss and 0.0. Sum over the training examples.
    loss = tf.reduce_sum(tf.maximum(basic_loss,0))
    ### END CODE HERE ###
    
    return loss
def Intializer():
    global FRmodel
    FRmodel = faceRecoModel(input_shape=(3, 96, 96))
    print("Total Params:", FRmodel.count_params())
    import time
    seconds = time.time()
    FRmodel.compile(optimizer = 'adam', loss = triplet_loss, metrics = ['accuracy'])
    print("Model Has Compiled Now weights will be loaded")
    load_weights_from_FaceNet(FRmodel)
    seconds2 = time.time()
    print("Minutes used to load data =",  (seconds2 - seconds)/60)
    import glob
    global database
    database = {}
    for file in glob.glob("train/*"):
        person_name = os.path.splitext(os.path.basename(file))[0]
        print(file)
    #image_file = cv2.imread(file, 1)
        database[person_name] = img_to_encoding(file,FRmodel) 
        
        
        
        
        
        
#This code is loading Data#############################################################
Intializer()  
######End of Loading Code################################################






  
from PIL import Image  
import PIL  
# GRADED FUNCTION: who_is_it
def who_is_it(image_path, database, model):
    """
    Implements face recognition for the office by finding who is the person on the image_path image.
    
    Arguments:
    image_path -- path to an image
    database -- database containing image encodings along with the name of the person on the image
    model -- your Inception model instance in Keras
    
    Returns:
    min_dist -- the minimum distance between image_path encoding and the encodings from the database
    identity -- string, the name prediction for the person on image_path
    """
    
    ### START CODE HERE ### 
    
    ## Step 1: Compute the target "encoding" for the image. Use img_to_encoding() see example above. ## (≈ 1 line)
    encoding = img_to_encoding(image_path,model)
    
    ## Step 2: Find the closest encoding ##
    
    # Initialize "min_dist" to a large value, say 100 (≈1 line)
    min_dist = 100
    
    # Loop over the database dictionary's names and encodings.
    for (name, db_enc) in database.items():
        
        # Compute L2 distance between the target "encoding" and the current db_enc from the database. (≈ 1 line)
        dist = np.linalg.norm(db_enc - encoding)

        # If this distance is less than the min_dist, then set min_dist to dist, and identity to name. (≈ 3 lines)
        if dist < min_dist:
            min_dist = dist
            identity = name

    ### END CODE HERE ###
    
    if min_dist > 0.7:
        print("Not in the database.")
        print ( "the distance is " + str(min_dist))
        playsound("dknow.mp3")
        identity2 = identity
        identity = "Not in the database. The closest picture is "+identity2
    else:
        print ("it's " + str(identity) + ", the distance is " + str(min_dist))
        playsound(str(identity) + ".mp3")
    img = Image.open(image_path)  
    #plt.imshow(img)
    #plt.show()
    return identity
################################### Hamza's code End Here###########################################
id = StringVar()
root.title("AI PROJECT")
global canvas
canvas = tk.Canvas(root, height = 500, width = 400, bg = "white")
canvas.pack()
def Upload():
   global fname
   fname = filedialog.askopenfilename(initialdir = "/Users/HP/AI Project/test/", title = "Select the Picture",filetype = (("jpg", "*.jpg"),("All Files","*.*")))
   global up
   up = True
   img = Image.open(fname)
   img = img.resize((300, 300), Image.ANTIALIAS)
   img = ImageTk.PhotoImage(img)
   global panel
   panel = Label(canvas, image=img)
   panel.image = img
   panel.pack()
def jhonCallBack():
	cascPath = "haarcascade_frontalface_default.xml"
	faceCascade = cv2.CascadeClassifier(cascPath)
	log.basicConfig(filename='webcam.log',level=log.INFO)

	video_capture = cv2.VideoCapture(0)
	anterior = 0

	while True:
	    if not video_capture.isOpened():
	        print('Unable to load camera.')
	        sleep(5)
	        pass

	    # Capture frame-by-frame
	    ret, frame = video_capture.read()

	    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

	    faces = faceCascade.detectMultiScale(
	        gray,
	        scaleFactor=1.1,
	        minNeighbors=5,
	        minSize=(30, 30)
	    )

	    # Draw a rectangle around the faces
	    for (x, y, w, h) in faces:
	        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

	    if anterior != len(faces):
	        anterior = len(faces)
	        log.info("faces: "+str(len(faces))+" at "+str(dt.datetime.now()))


	    # Display the resulting frame
	    cv2.imshow('Video', frame)


	    if cv2.waitKey(1) & 0xFF == ord('q'):
	        break

	    # Display the resulting frame
	    cv2.imshow('Video', frame)

	# When everything is done, release the capture
	video_capture.release()
	cv2.destroyAllWindows()

def HamzaCallBack():
    if(up == True):
        id2 = who_is_it(fname,database,FRmodel)
        id.set(id2)
    else:
        id.set("Please Upload a Image First")
def AliCallBack():
    subprocess.check_call('M.Alilibs/mtcnn_final1.py')
    subprocess.check_call('M.Alilibs/mtcnn_final2.py')

def Clear():
    id.set("")
    panel.config(image = "")
def jhonPictureCallback():
	print(classify_face("test.jpg"))
###################################Jhons Code################################################
def get_encoded_faces():
	encoded = {}

	for dirpath, dnames, fnames in os.walk("./faces"):
	    for f in fnames:
	        if f.endswith(".jpg") or f.endswith(".png"):
	            face = fr.load_image_file("faces/" + f)
	            encoding = fr.face_encodings(face)[0]
	            encoded[f.split(".")[0]] = encoding

	return encoded


def unknown_image_encoded(img):
    """
    encode a face given the file name
    """
    face = fr.load_image_file("faces/" + img)
    encoding = fr.face_encodings(face)[0]

    return encoding


def classify_face(im):
    """
    will find all of the faces in a given image and label
    them if it knows what they are

    :param im: str of file path
    :return: list of face names
    """
    faces = get_encoded_faces()
    faces_encoded = list(faces.values())
    known_face_names = list(faces.keys())

    img = cv2.imread(im, 1)
    #img = cv2.resize(img, (0, 0), fx=0.5, fy=0.5)
    #img = img[:,:,::-1]
 
    face_locations = face_recognition.face_locations(img)
    unknown_face_encodings = face_recognition.face_encodings(img, face_locations)

    face_names = []
    for face_encoding in unknown_face_encodings:
        # See if the face is a match for the known face(s)
        matches = face_recognition.compare_faces(faces_encoded, face_encoding)
        name = "Unknown"

        # use the known face with the smallest distance to the new face
        face_distances = face_recognition.face_distance(faces_encoded, face_encoding)
        best_match_index = np.argmin(face_distances)
        if matches[best_match_index]:
            name = known_face_names[best_match_index]

        face_names.append(name)

        for (top, right, bottom, left), name in zip(face_locations, face_names):
            # Draw a box around the face
            cv2.rectangle(img, (left-20, top-20), (right+20, bottom+20), (255, 0, 0), 2)

            # Draw a label with a name below the face
            cv2.rectangle(img, (left-20, bottom -15), (right+20, bottom+20), (255, 0, 0), cv2.FILLED)
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(img, name, (left -20, bottom + 15), font, 1.0, (255, 255, 255), 2)


    # Display the resulting image
    while True:

        cv2.imshow('Photo', img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            return face_names
 ###################################END##############################################
 
Identitylabel = Label(root,textvariable = id)
Identitylabel.pack()
buttonUpload = tk.Button(root,text = "Upload File",padx = 0.3, pady = 0.15, fg = "white", bg = "#0000CD", command = Upload)
buttonUpload.pack()
buttonJhon = tk.Button(root, text = "Jhon's code", padx = 0.3, pady = 0.15, fg = "white", bg = "#0000CD", command = jhonCallBack)
buttonJhon.pack()
buttonJhon1 = tk.Button(root, text = "Jhon's picture recognition code", padx = 0.3, pady = 0.15, fg = "white", bg = "#0000CD", command = jhonPictureCallback)
buttonJhon1.pack()
buttonHamza = tk.Button(root, text = "Hamza's code", padx = 0.3, pady = 0.15, fg = "white", bg = "#0000CD", command = HamzaCallBack)
buttonHamza.pack()
buttonAli = tk.Button(root, text = "Ali's code", padx = 0.3, pady = 0.15, fg = "white", bg = "#0000CD", command = AliCallBack)
buttonAli.pack()
buttonclear = tk.Button(root, text = "Clear Image", padx = 0.3, pady = 0.15, fg = "white", bg = "#0000CD", command = Clear)
buttonclear.pack()

root.mainloop()

