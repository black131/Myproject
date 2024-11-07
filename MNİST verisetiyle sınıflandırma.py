#MNİST verisetiyle sınıflandırma
import tensorflow 
from tensorflow.python.client import device_lib
device_lib.list_local_devices()
#Uygulama adımları
from __future__ import print_function
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.models import load_model
from keras.layers import Dense,Dropout,Flatten
from keras.layers import Conv2D,MaxPooling2D
from keras import backend as K
import matplotlib.pyplot as plt

(x_train,y_train),(x_test,y_test)=mnist.load_data()
plt.figure(figsize=(14,14))
x,y=10,4
for i in range(40):
    plt.subplot(x,y,i+1)
    plt.imshow(x_train[i])
plt.show() 
batch_size=128
num_classes=10
epochs=5 #Bazende 12 olabiliyor.genelde 12 kullanılır.

img_rows,img_cls=28,28
if K.image_data_format()== 'channels_first':
    x_train=x_train.reshape(x_train.shape[0],1,img_rows,img_cls,1)
    x_test=x_test.reshape(x_test.shape[0],1,img_rows,img_cls,1)
    input_shape=(1,img_rows,img_cls,1)

#Sınıf vektörlerini ikiliye dönüştürüp encoding etme işlemi
y_train=keras.utils.to_categorical(y_train,num_classes)
y_test=keras.utils.to_categorical(y_test,num_classes)

#Model oluşturma
model=Sequential()
#1.katman
model.add(Conv2D(32,kernel_size=(3,3),activation='relu',input_shape=input_shape))
model.add(Conv2D(64,kernel_size=(3,3),activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128,activation='relu'))
model.add(Dropout(0.50))
model.add(Dense(num_classes,activation="softmax"))
model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])

#Eğitme işlemleri
model.fit(x_train,y_train,
          batch_size=128,
          epochs=5,verbose=1,
          validation_data=(x_test,y_test))
#Test işlemi ve ekrana yazdırma
score=model.evaluate(x_test,y_test,verbose=0 )
print("Test Loss: ",score[0])
print("Test Accuracy: ",score[1])

#Rastgele deger için test işlemi
test_image=x_test[32]
y_test[32]
plt.imshow(test_image.reshape(28,28))
test_data=x_test[32].reshape(1,28,28,1)
pre=model_test.predict(test_data,batch_size=1)
preds=model_test.predict_classes(test_data)
prob=model_test.predict_proba(test_data)
print(preds,prob)