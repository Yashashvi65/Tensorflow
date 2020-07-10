import tensorflow as tf
import numpy as np
from tensorflow import keras


model=keras.Sequential([keras.layers.Dense(units=1,input_shape=[1])])
model.compile(optimizer='sgd',loss='mean_squared_error')



x=np.array([-1.0,0.0,1.0,2.0,3.0,4.0],dtype=float)
y=np.array([-3.0,-1.0,1.0,3.0,5.0,7.0],dtype=float)
#have a look at the relationship bw x and y(Yes you got it right it is y=2x-1)
model.fit(x,y,epochs=500)#500 iterations

print(model.predict([10.0]))
#you might think the output to be 19 but its a value very close to 19.My output was [[18.986021]]
