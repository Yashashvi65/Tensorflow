#WITHOUT CALLBACK
import tensorflow as tf
import numpy as np
from tensorflow import keras
fashion_mnist=keras.datasets.fashion_mnist
(train_images,train_labels),(test_images,test_labels)=fashion_mnist.load_data()

import matplotlib.pyplot as plt
plt.imshow(train_images[0])
print (train_images[0])
print(train_labels[0])

train_images=train_images/255.0
test_images=test_images/255.0

model=tf.keras.models.Sequential([tf.keras.layers.Flatten(),tf.keras.layers.Dense(128,activation=tf.nn.relu),tf.keras.layers.Dense(10,activation=tf.nn.softmax)])

model.compile(optimizer=tf.keras.optimizers.Adam(),loss='sparse_categorical_crossentropy')
model.fit(train_images,train_labels,epochs=10)

model.evaluate(test_images,test_labels)



#WITH CALLBACK
import tensorflow as tf
import numpy as np
from tensorflow import keras
class myCallBack(tf.keras.callbacks.EarlyStopping):
  def on_epoch_end(self,epoch,logs={}):
    if(logs.get('loss')<0.4):
      print("\nReached 60% accuracy so cancelling training")
      self.model.stop_training=True
callbacks=myCallBack()
fashion_mnist=keras.datasets.fashion_mnist
(train_images,train_labels),(test_images,test_labels)=fashion_mnist.load_data()

train_images=train_images/255.0
test_images=test_images/255.0
model=tf.keras.models.Sequential([tf.keras.layers.Flatten(),tf.keras.layers.Dense(128,activation=tf.nn.relu),tf.keras.layers.Dense(10,activation=tf.nn.softmax)])
model.compile(optimizer=tf.keras.optimizers.Adam(),loss='sparse_categorical_crossentropy')
model.fit(train_images,train_labels,epochs=10,callbacks=[callbacks])
model.evaluate(test_images,test_labels)

