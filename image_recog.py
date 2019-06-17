from __future__ import absolute_import, division, print_function

#import matplotlib.pylab as plt
"""
More can be found here
https://www.tensorflow.org/tutorials/images/hub_with_keras#run_it_on_a_single_image
"""

import tensorflow as tf
import tensorflow_hub as hub
import matplotlib.pylab as plt
from tensorflow.keras import layers
import tensorflow.keras.backend as K
import numpy as np
import PIL.Image as Image


data_root = tf.keras.utils.get_file(
  'flower_photos','https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz',
   untar=True)


#Rescale the image as it expect float inputs in the [0,1] range
image_generator = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1/255)
#image_data = image_generator.flow_from_directory(str(data_root))

#Download the classifier
classifier_url = "https://tfhub.dev/google/imagenet/mobilenet_v2_100_224/classification/2" #@param {type:"string"}
def classifier(x):
  classifier_module = hub.Module(classifier_url)
  return classifier_module(x)
  
IMAGE_SIZE = hub.get_expected_image_size(hub.Module(classifier_url))
classifier_layer = layers.Lambda(classifier, input_shape = IMAGE_SIZE+[3])
classifier_model = tf.keras.Sequential([classifier_layer])
classifier_model.summary()
image_data = image_generator.flow_from_directory(str(data_root), target_size=IMAGE_SIZE)




sess = K.get_session()
init = tf.global_variables_initializer()

sess.run(init)

#Inference example

#Get the file by using tf.keras.utils.get_file('image_name','image_url')
#Then Image.open(variable_above).resize(IMAGE_SIZE)

bush_house = tf.keras.utils.get_file('bush_house.jpg','https://en.wikipedia.org/wiki/Bush_House#/media/File:Bush_House,_Aldwych_(geograph_4238525).jpg')
bush_house = Image.open(bush_house).resize(IMAGE_SIZE)

bush_house = np.array(bush_house)/255.0
#bush_house.shape
 
 #Predict the class
 result = classifier_model.predict(bush_house[np.newaxis])
 predicted_class = np.argmax(result[0],axis=-1)

 #Decode the prediction class
 labels_path = tf.keras.utils.get_file('ImageNetLabels.txt','https://storage.googleapis.com/download.tensorflow.org/data/ImageNetLabels.txt')
 imagenet_labels = np.array(open(labels_path).read().splitlines())

 predicted_class_name = imagenet_labels[predicted_class]
 print("predicted_class_name")


