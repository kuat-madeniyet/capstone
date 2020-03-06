# set the matplotlib backend so figures can be saved in the background
import matplotlib
matplotlib.use("Agg")

# import the necessary packages
import keras
import matplotlib.pyplot as plt
import numpy as np
import argparse
import os
import cv2
import random
import shutil
import Augmentor


from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing.image import save_img
from keras.callbacks import LearningRateScheduler
from keras.optimizers import RMSprop
from keras.utils import np_utils
from keras.models import load_model
from keras import models
from keras import layers
from keras import optimizers
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from imutils import paths
from PIL import Image
from keras.optimizers import Adagrad

'''import tensorflow as tf  
from keras.backend.tensorflow_backend import set_session  
config = tf.ConfigProto()  
config.gpu_options.allow_growth = True  # dynamically grow the memory used on the GPU  
config.log_device_placement = True  # to log device placement (on which device the operation ran)  
                                    # (nothing gets printed in Jupyter, only if you run it standalone)
sess = tf.Session(config=config) 
set_session(sess)  # set this TensorFlow session as the default session for Keras  
'''
#FUNCTIONS TO IMPORT DIFFERENT MODELS


def func1(shape):
	from keras.applications import ResNet50
	BS = 16
	conv_base = ResNet50(weights = 'imagenet',
                 	include_top = False,
                 	input_shape = (shape[0],shape[1],3))
	return BS,conv_base

def func2(shape):
	from keras.applications import ResNet101
	BS = 16
	conv_base = ResNet101(weights = 'imagenet',
                 	include_top = False,
                 	input_shape = (shape[0],shape[1],3))
	return BS,conv_base

def func3(shape):
	from keras.applications import ResNet152
	BS = 8
	conv_base = ResNet152(weights = 'imagenet',
                 	include_top = False,
                 	input_shape = (shape[0],shape[1],3))
	return BS,conv_base

def func4(shape):
	from keras.applications import DenseNet121
	BS = 32
	conv_base = DenseNet121(weights = 'imagenet',
                 	include_top = False,
                 	input_shape = (shape[0],shape[1],3))
	return BS,conv_base

def func5(shape):
	from keras.applications import DenseNet201
	BS = 16
	conv_base = DenseNet201(weights = 'imagenet',
                 	include_top = False,
                 	input_shape = (shape[0],shape[1],3))
	return BS,conv_base

def func6(shape):
	from keras.applications import Xception
	BS = 16
	conv_base = Xception(weights = 'imagenet',
                 	include_top = False,
                 	input_shape = (shape[0],shape[1],3))
	return BS,conv_base

def func7(shape):
	from keras.applications import InceptionV3
	BS = 32
	conv_base = InceptionV3(weights = 'imagenet',
                 	include_top = False,
                 	input_shape = (shape[0],shape[1],3))
	return BS,conv_base

def func8(shape):
	from keras.applications import VGG16
	BS = 16
	conv_base = VGG16(weights = 'imagenet',
                 	include_top = False,
                 	input_shape = (shape[0],shape[1],3))
	return BS,conv_base

def func9(shape):
	from keras.applications import VGG19
	BS = 16
	conv_base = VGG19(weights = 'imagenet',
                 	include_top = False,
                 	input_shape = (shape[0],shape[1],3))
	return BS,conv_base






BASE_PATH = ['TMAD/','breakhis_m_subtypes_aug/','breakhis_b_subtypes_aug/','breakhis_binary']
dest = ['bladder','breast','lung','lymphoma',
		'ductal_carcinoma','lobular_carcinoma','mucinous_carcinoma','papillary_carcinoma',
		'adenosis','fibroadenoma','phyllodes_tumor','tubular_adenoma',
		]
dest1 = ['0','1']
i = 3
ii = -4
k = 1    #FOR SEPARATE FIGURES

for bp in BASE_PATH:
	
	BASE_SUBPATH = BASE_PATH[i]

	subdest = dest[(ii+4):(ii+8)]

	TRAIN_PATH = os.path.sep.join([BASE_SUBPATH, "train/"])
	VAL_PATH = os.path.sep.join([BASE_SUBPATH, "valid/"])
	TEST_PATH = os.path.sep.join([BASE_SUBPATH, "test/"])

	NUM_EPOCHS = 100
	#INIT_LR = 1e-2

	model_names = ['Xception']

	model_names1 = ['DenseNet201']

	for name in model_names:

		print(i)

		#if(i == 2 and name == 'InceptionV3'):
		#	continue
	

		if ((name == 'InceptionV4') or (name =='InceptionV3') or (name =='InceptionV2') or (name =='Xception')):
			shape = [299,299]
		else:
			shape = [224,224]

		if (name == 'ResNet50'): BS,conv_base = func1(shape)
		elif (name == 'ResNet101'): BS,conv_base = func2(shape)
		elif (name == 'ResNet152'): BS,conv_base = func3(shape)
		elif (name == 'DenseNet121'): BS,conv_base = func4(shape)
		elif (name == 'DenseNet201'): BS,conv_base = func5(shape)
		elif (name == 'Xception'): BS,conv_base = func6(shape)
		elif (name == 'InceptionV3'): BS,conv_base = func7(shape)
		elif (name == 'VGG16'): BS,conv_base = func8(shape)
		elif (name == 'VGG19'): BS,conv_base = func9(shape)

		# determine the total number of image paths in training, validation,
		# and testing directories
		trainPaths = list(paths.list_images(TRAIN_PATH))
		totalTrain = len(trainPaths)
		totalVal = len(list(paths.list_images(VAL_PATH)))
		totalTest = len(list(paths.list_images(TEST_PATH)))

		# account for skew in the labeled data
		
		trainLabels1 = [0 for p in range(len(list(paths.list_images(TRAIN_PATH + dest1[0]))))]
		trainLabels2 = [1 for p in range(len(list(paths.list_images(TRAIN_PATH + dest1[1]))))]
		#trainLabels3 = [2 for p in range(len(list(paths.list_images(TRAIN_PATH + subdest[2]))))]
		#trainLabels4 = [3 for p in range(len(list(paths.list_images(TRAIN_PATH + subdest[3]))))]
		
		trainLabels = []

		for j in range(len(trainLabels1)):
			trainLabels.append(trainLabels1[j])
	    
		for j in range(len(trainLabels2)):
			trainLabels.append(trainLabels2[j])

		#for j in range(len(trainLabels3)):
		#	trainLabels.append(trainLabels3[j])

		#for j in range(len(trainLabels4)):
		#	trainLabels.append(trainLabels4[j])

		#print(trainLabels)

		trainLabels = np_utils.to_categorical(trainLabels)
		classTotals = trainLabels.sum(axis=0)
		classWeight = classTotals.max() / classTotals

		train_datagen = ImageDataGenerator(rescale = 1 / 255.0)
		val_datagen = ImageDataGenerator(rescale = 1 / 255.0)

		trainGen = train_datagen.flow_from_directory(
			TRAIN_PATH,
			class_mode="categorical",
			target_size=(shape[0],shape[1]),
			color_mode="rgb",
			shuffle=True,
			batch_size=BS)
		valGen = val_datagen.flow_from_directory(
			VAL_PATH,
			class_mode="categorical",
			target_size=(shape[0],shape[1]),
			color_mode="rgb",
			shuffle=False,
			batch_size=BS)
		testGen = val_datagen.flow_from_directory(
			TEST_PATH,
			class_mode="categorical",
			target_size=(shape[0],shape[1]),
			color_mode="rgb",
			shuffle=False,
			batch_size=BS)

		#Feature extraction with data augmentation
		model = models.Sequential()
		model.add(conv_base)
		model.add(layers.Flatten())
		model.add(layers.Dropout(0.5))
		model.add(layers.Dense(256, activation = 'relu'))
		model.add(layers.Dense(2, activation = 'softmax'))

		conv_base.trainable = True
		'''set_trainable = False
		print("\n\n\n"+str(conv_base.layers))'''
		count = 0
		for layer in conv_base.layers:
				#if(count > 39):
					layer.trainable = True
				#count = count + 1
		model.summary()
			
		model.compile(loss = 'categorical_crossentropy',
	        		optimizer = optimizers.adam(lr=0.0001),
	        		metrics = ['acc'])


		#es = EarlyStopping(monitor='val_accuracy', mode='max', min_delta=5)
		mcp_save = keras.callbacks.callbacks.ModelCheckpoint(str(i)+"_img_Best_model#"+str(NUM_EPOCHS)+'_'+ str(name) + ".h5",monitor='val_acc', verbose=0, 
			save_best_only=True, save_weights_only=False, mode='max', period=1)

		H = model.fit_generator(
			trainGen,
			steps_per_epoch=totalTrain // BS,
			validation_data=valGen,
			validation_steps=totalVal // BS,
			class_weight=classWeight,
			epochs=NUM_EPOCHS,
			callbacks = [mcp_save])


		
		# save model and architecture to single file
		#model.save("noaug_model#"+str(NUM_EPOCHS)+'_'+ str(name) + ".h5")
		model.save(str(i)+"_img_AllLayer_model#"+str(NUM_EPOCHS)+'_'+ str(name) + ".h5")
		print("Saved model to disk")

		

		testGen.reset()
		model = load_model(str(i)+"_img_Best_model#"+str(NUM_EPOCHS)+'_'+ str(name) + ".h5")
		predIdxs = model.predict_generator(testGen,verbose=2,
						steps=(totalTest // BS) + 1)
		# for each image in the testing set we need to find the index of the
		# label with corresponding largest predicted probability
		predIdxs = np.argmax(predIdxs, axis=1)


		# show a nicely formatted classification report
		print(classification_report(testGen.classes, predIdxs,
			target_names=testGen.class_indices.keys()))

		cm = confusion_matrix(testGen.classes, predIdxs)                               #CHANGED EVERY testGen to valGen
		total = sum(sum(cm))
		acc = (cm[0, 0] + cm[1, 1]) / total

		sensitivity0 = cm[0, 0] / (cm[0, 0] + cm[0, 1])
		sensitivity1 = cm[1, 1] / (cm[1, 1] + cm[1, 0])
		#sensitivity2 = cm[2, 2] / (cm[2, 2] + cm[2, 0] + cm[2,1] + cm[2,3])
		#sensitivity3 = cm[3, 3] / (cm[3, 3] + cm[3, 0] + cm[3,1] + cm[3,2])
		sensitivity = (sensitivity1+sensitivity0)/2

		#fname1 = "noaug_modelHistory#"+str(NUM_EPOCHS)+'_'+ str(name) + ".txt"
		fname1 = str(i)+"_img_AllLayer_modelHistory#"+str(NUM_EPOCHS)+'_'+ str(name) + ".txt"
		f = open(fname1,"w+")
		f.write("\nValidation acc: " + str(round(H.history['val_acc'][NUM_EPOCHS-1],4)) + "; Training acc: " + str(round(H.history['acc'][NUM_EPOCHS-1],4)) + 
			"\nSensitivity (avg): " + str(round(sensitivity,4)) + "; Test acc: " + str(round(acc,4)))
		f.write("\nSensitivity0: {:.4f}".format(sensitivity0))
		f.write("\nSensitivity1: {:.4f}".format(sensitivity1))
		#f.write("\nSensitivity2: {:.4f}".format(sensitivity2))
		#f.write("\nSensitivity3: {:.4f}".format(sensitivity3))
		f.write("\n\n" + classification_report(testGen.classes, predIdxs,
								target_names=testGen.class_indices.keys()))
		f.write("\n" + str(cm))
		f.close()

		print(cm)
		print("test acc: {:.4f}".format(acc))
		print("sensitivity0: {:.4f}".format(sensitivity0))
		print("sensitivity1: {:.4f}".format(sensitivity1))

		#print("sensitivity2: {:.4f}".format(sensitivity2))
		#print("sensitivity3: {:.4f}".format(sensitivity3))
		print("sensitivity (avg): {:.4f}".format(sensitivity))

		plt.figure(k)
		plt.plot(H.history['acc'])
		plt.plot(H.history['val_acc'])
		plt.title('model accuracy')
		plt.ylabel('accuracy')
		plt.xlabel('epoch')
		plt.legend(['train', 'val'], loc='upper left')
		#plt.savefig('noaug_modelHistory#'+str(NUM_EPOCHS)+'_'+ str(name)+'_accuracy.png')
		plt.savefig(str(i)+'_img_AllLayer_modelHistory#'+str(NUM_EPOCHS)+'_'+ str(name)+'_accuracy.png')

		plt.figure(k+1)
		plt.plot(H.history['loss'])
		plt.plot(H.history['val_loss'])
		plt.title('model loss')
		plt.ylabel('loss')
		plt.xlabel('epoch')
		plt.legend(['loss', 'val_loss'], loc='upper left')
		#plt.savefig('noaug_modelHistory#'+str(NUM_EPOCHS)+'_'+ str(name)+'_loss.png')
		plt.savefig(str(i)+'_img_AllLayer_modelHistory#'+str(NUM_EPOCHS)+'_'+ str(name)+'_loss.png')

		k = k + 2
		
	i = i + 1

	ii = ii + 4