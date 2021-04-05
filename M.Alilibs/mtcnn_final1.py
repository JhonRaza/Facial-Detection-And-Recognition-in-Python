#!/usr/bin/python3.7
from keras.models import load_model
from PIL import Image
from numpy import asarray
from mtcnn.mtcnn import MTCNN
from os import listdir
from matplotlib import pyplot
from os.path import isdir


# load the model
filename = r'/facenet_keras.h5'
required_size=(160, 160)
model = load_model(filename)
# summarize input and output shape
print(model.inputs)
print(model.outputs)


# extract a single face from a given photograph
def extract_face(filename, required_size):
	# load image from file
	image = Image.open(filename)
	# convert to RGB, if needed
	image = image.convert('RGB')
	# convert to array
	pixels = asarray(image)
	# create the detector, using default weights
	detector = MTCNN()
	# detect faces in the image
	results = detector.detect_faces(pixels)
	# extract the bounding box from the first face
	x1, y1, width, height = results[0]['box']
	# bug fix
	x1, y1 = abs(x1), abs(y1)
	x2, y2 = x1 + width, y1 + height
	# extract the face
	face = pixels[y1:y2, x1:x2]
	# resize pixels to the model size
	image = Image.fromarray(face)
	image = image.resize(required_size)
	face_array = asarray(image)
	return face_array
 
#2nd program

# extract a single face from a given photograph
def extract_face(filename, required_size=(160, 160)):
	# load image from file
	image = Image.open(filename)
	# convert to RGB, if needed
	image = image.convert('RGB')
	# convert to array
	pixels = asarray(image)
	# create the detector, using default weights
	detector = MTCNN()
	# detect faces in the image
	results = detector.detect_faces(pixels)
	# extract the bounding box from the first face
	x1, y1, width, height = results[0]['box']
	# bug fix
	x1, y1 = abs(x1), abs(y1)
	x2, y2 = x1 + width, y1 + height
	# extract the face
	face = pixels[y1:y2, x1:x2]
	# resize pixels to the model size
	image = Image.fromarray(face)
	image = image.resize(required_size)
	face_array = asarray(image)
	return face_array
 
# specify folder to plot
folder = r'/New-Dataset/train/leo_messi'
i = 1
# enumerate files
for filename in listdir(folder):
	# path
	path = folder + "/" + filename
	# get face
	face = extract_face(path)
	print(i, face.shape)
	# plot
	pyplot.subplot(2, 7, i)
	pyplot.axis('off')
	pyplot.imshow(face)
	i += 1
pyplot.show()

#3rd program
 
from os import listdir
from os.path import isdir
from PIL import Image
from matplotlib import pyplot
from numpy import savez_compressed
from numpy import asarray
from mtcnn.mtcnn import MTCNN
 
# extract a single face from a given photograph
def extract_face(filename, required_size=(160, 160)):
	# load image from file
	image = Image.open(filename)
	# convert to RGB, if needed
	image = image.convert('RGB')
	# convert to array
	pixels = asarray(image)
	# create the detector, using default weights
	detector = MTCNN()
	# detect faces in the image
	results = detector.detect_faces(pixels)
	# extract the bounding box from the first face
	x1, y1, width, height = results[0]['box']
	# bug fix
	x1, y1 = abs(x1), abs(y1)
	x2, y2 = x1 + width, y1 + height
	# extract the face
	face = pixels[y1:y2, x1:x2]
	# resize pixels to the model size
	image = Image.fromarray(face)
	image = image.resize(required_size)
	face_array = asarray(image)
	return face_array
 
# load images and extract faces for all images in a directory
def load_faces(directory):
	faces = list()
	# enumerate files
	for filename in listdir(directory):
		# path
		path = directory + "/" + filename
		# get face
		face = extract_face(path)
		# store
		faces.append(face)
	return faces
 
# load a dataset that contains one subdir for each class that in turn contains images
def load_dataset(directory):
	X, y = list(), list()
	# enumerate folders, on per class
	for subdir in listdir(directory):
		# path
		path = directory + "/" + subdir
		# skip any files that might be in the dir
		if not isdir(path):
			continue
		# load all faces in the subdirectory
		faces = load_faces(path)
		# create labels
		labels = [subdir for _ in range(len(faces))]
		# summarize progress
		print('>loaded %d examples for class: %s' % (len(faces), subdir))
		# store
		X.extend(faces)
		y.extend(labels)
	return asarray(X), asarray(y)

def load_faces1(directory):
	faces = list()
	# enumerate files
#	for filename in listdir(directory):
		# path
#		path = directory + "/" + filename
		# get face 
	face = extract_face(directory)
		# store
	faces.append(face)
	return faces
 
# load a dataset that contains one subdir for each class that in turn contains images
def load_dataset1(directory):
	X, y = list(), list()
	# enumerate folders, on per class
#	for subdir in listdir(directory):
		# path
#		path = directory + "/" + subdir
		# skip any files that might be in the dir
#		if not isdir(path):
#			continue
		# load all faces in the subdirectory
	faces = load_faces1(directory)
		# create labels
	labels = len(faces)
		# summarize progress
	print('>loaded one examples')
		# store
	X.extend(faces)
	y.extend(labels)
	return asarray(X), asarray(y)
 
# load train dataset
directory1 = r'/New-Dataset/train'
directory2 = r'/New-Dataset/val'
trainX, trainy = load_dataset(directory1)
print(trainX.shape, trainy.shape)
# load test dataset
testX, testy = load_dataset(directory2)
# save arrays to one file in compressed format
savez_compressed(r'/New-Dataset.npz', trainX, trainy, testX, testy)