import tensorflow as tf
print(tf.__version__)
#load dataset
mnist=tf.keras.datasets.fashion_mnist
(training_images,training_labels),(test_images,test_labels)=mnist.load_data()
training_images=training_images.reshape(60000,28,28,1)
training_images=training_images/255.0
test_images=test_images.reshape(10000,28,28,1)
test_images=test_images/255.0


model=tf.keras.models.Sequential([tf.keras.layers.Conv2D(64,(3,3),activation='relu',input_shape=(28,28,1)),tf.keras.layers.MaxPooling2D(2,2),
                                  tf.keras.layers.Conv2D(64,(3,3),activation='relu',input_shape=(28,28,1)),tf.keras.layers.MaxPooling2D(2,2),tf.keras.layers.Flatten()
                                  ,tf.keras.layers.Dense(128,activation='relu'),tf.keras.layers.Dense(10,activation='softmax')])
                                  
model.compile(optimizer='adam',loss='sparse_categorical_crossentropy')
model.summary()

#Summary
Model: "sequential"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
conv2d_4 (Conv2D)            (None, 26, 26, 64)        640       
_________________________________________________________________
max_pooling2d_4 (MaxPooling2 (None, 13, 13, 64)        0         
_________________________________________________________________
conv2d_5 (Conv2D)            (None, 11, 11, 64)        36928     
_________________________________________________________________
max_pooling2d_5 (MaxPooling2 (None, 5, 5, 64)          0         
_________________________________________________________________
flatten_2 (Flatten)          (None, 1600)              0         
_________________________________________________________________
dense (Dense)                (None, 128)               204928    
_________________________________________________________________
dense_1 (Dense)              (None, 10)                1290      
=================================================================
Total params: 243,786
Trainable params: 243,786
Non-trainable params: 0

model.fit(training_images,training_labels,epochs=5)
test_loss=model.evaluate(test_images,test_labels)

Epoch 1/5
1875/1875 [==============================] - 79s 42ms/step - loss: 0.4340
Epoch 2/5
1875/1875 [==============================] - 78s 42ms/step - loss: 0.2916
Epoch 3/5
1875/1875 [==============================] - 78s 42ms/step - loss: 0.2450
Epoch 4/5
1875/1875 [==============================] - 78s 41ms/step - loss: 0.2144
Epoch 5/5
1875/1875 [==============================] - 78s 42ms/step - loss: 0.1879
313/313 [==============================] - 4s 12ms/step - loss: 0.2479

#VISUALIZATION
import matplotlib.pyplot as plt
f,axarr=plt.subplots(3,4)
FIRST_IMAGE=4
SECOND_IMAGE=10
THIRD_IMAGE=26
CONVOLUTION_NUMBER=2
from tensorflow.keras import models
layer_outputs=[layer.output for layer in model.layers]
activation_model=tf.keras.models.Model(inputs=model.input,outputs=layer_outputs)
for x in range(0,4):
 f1=activation_model.predict(test_images[FIRST_IMAGE].reshape(1,28,28,1))[x]
 axarr[0,x].imshow(f1[0,:,:,CONVOLUTION_NUMBER],cmap='inferno')
 axarr[0,x].grid(False)
 f2=activation_model.predict(test_images[SECOND_IMAGE].reshape(1,28,28,1))[x]
 axarr[1,x].imshow(f1[0,:,:,CONVOLUTION_NUMBER],cmap='inferno')
 axarr[1,x].grid(False)
 f3=activation_model.predict(test_images[THIRD_IMAGE].reshape(1,28,28,1))[x]
 axarr[2,x].imshow(f1[0,:,:,CONVOLUTION_NUMBER],cmap='inferno')
 axarr[2,x].grid(False)
