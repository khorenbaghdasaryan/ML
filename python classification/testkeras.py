#loading the cifar10 dataset
from keras.datasets import cifar10
import matplotlib.pyplot as plt
     
#load train and test dataset 
(train_X,train_Y),(test_X,test_Y) = cifar10.load_data()

#summarize loaded dataset
print('Train: X=%s, y=%s' % (train_X.shape, train_Y.shape))
print('Test: X=%s, y=%s' % (test_X.shape, test_Y.shape))

#plot first few images
plt.figure(figsize=(20,10))
for i in range(9):
    #define subplot
    plt.subplot(330 + 1 + i)
    #plot raw pixel data
    plt.imshow(train_X[i])
#show the figure
plt.show()

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.constraints import maxnorm
from keras.optimizers import SGD
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.utils import np_utils

#convert from integers to floats    
train_x = train_X.astype('float32')
test_X = test_X.astype('float32')
#normalize to range 0-1
train_X = train_X / 255.0
test_X = test_X / 255.0

train_Y=np_utils.to_categorical(train_Y)
test_Y=np_utils.to_categorical(test_Y)
     
num_classes=test_Y.shape[1]

#define cnn model
model=Sequential()
model.add(Conv2D(32,(3,3),input_shape = (32,32,3),
                          padding='same', activation='relu',
                          kernel_constraint = maxnorm(3)))
model.add(Dropout(0.2))
model.add(Conv2D(64,(3,3),activation='relu',
                          padding='same',
                          kernel_constraint=maxnorm(3)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(512,activation='relu',kernel_constraint=maxnorm(3)))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))
    
sgd=SGD(lr=0.01,momentum=0.9,decay=(0.01/25),nesterov=False)
 
model.compile(loss='categorical_crossentropy', optimizer=sgd,
              metrics=['accuracy'])

model.fit(train_X,train_Y, validation_data=(test_X,test_Y),
          epochs=10, batch_size=32)
    
_,acc=model.evaluate(test_X,test_Y)
print(acc*100)
    
model.save("model1_cifar_10epoch.h5")
    
results={
   0:'aeroplane',
   1:'automobile',
   2:'bird',
   3:'cat',
   4:'deer',
   5:'dog',
   6:'frog',
   7:'horse',
   8:'ship',
   9:'truck'
    }
from PIL import Image
import numpy as np
im=Image.open("horse.jpg")
# the input image is required to be in the shape of dataset, i.e (32,32,3)
     
im=im.resize((32,32))
im=np.expand_dims(im,axis=0)
im=np.array(im)
pred=model.predict_classes([im])[0]
print(pred,results[pred])