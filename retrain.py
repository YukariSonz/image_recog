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


feature_extractor_url = "https://tfhub.dev/google/imagenet/mobilenet_v2_100_224/feature_vector/2" #@param {type:"string"}


def feature_extractor(x):
    feature_extractor_module = hub.Module(feature_extractor_url)
    return feature_extractor_module(x)


IMAGE_SIZE = hub.get_expected_image_size(hub.Module(feature_extractor_url))

image_data = image_generator.flow_from_directory(str(data_root), target_size=IMAGE_SIZE)
features_extractor_layer = layers.Lambda(feature_extractor, input_shape=IMAGE_SIZE+[3])

#Freeze the variables in the feature extractor layer so that the trainning only modifies the new classifier layer
features_extractor_layer.trainable = False

model = tf.keras.Sequential([
  features_extractor_layer,
  layers.Dense(image_data.num_classes, activation='softmax')
])
model.summary()

#Initialize the TFHub module
init = tf.global_variables_initializer()
sess.run(init)

###Train the model

#Training process configuration
model.compile(
  optimizer=tf.train.AdamOptimizer(), 
  loss='categorical_crossentropy',
  metrics=['accuracy'])


#Use .fit to train the model
class CollectBatchStats(tf.keras.callbacks.Callback):
  def __init__(self):
    self.batch_losses = []
    self.batch_acc = []
    
  def on_batch_end(self, batch, logs=None):
    self.batch_losses.append(logs['loss'])
    self.batch_acc.append(logs['acc'])


steps_per_epoch = image_data.samples//image_data.batch_size
batch_stats = CollectBatchStats()
model.fit((item for item in image_data), epochs=1, 
                    steps_per_epoch=steps_per_epoch,
                    callbacks = [batch_stats])


#Check the predicition by urself


#Export the model
export_path = tf.contrib.saved_model.save_keras_model(model, "./saved_models")
export_path

