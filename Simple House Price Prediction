import tensorflow as tf
import numpy as np
from tensorflow import keras
model = tf.keras.Sequential([keras.layers.Dense(units = 1, input_shape = [1])])
model.compile(optimizer = 'sgd', loss = 'mean_squared_error')
xs = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], dtype = float)
ys = np.array([100, 150, 200, 250, 300, 350], dtype = float)
model.fit(xs, ys/100, epochs = 1000)
print(model.predict([7.0]))
#My Output=4.003(in hundreds of thousand)
