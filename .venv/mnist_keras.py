import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import SGD, Adam
from keras.utils import to_categorical
from keras.datasets import mnist
from keras.callbacks import ModelCheckpoint
import matplotlib.pyplot as plt


# load the data set
(X_train,Y_train),(X_test,Y_test)=mnist.load_data()
# print(X_train.shape, X_test.shape)
# print(X_train[0])

X_train=X_train.reshape(X_train.shape[0],784)
X_test= X_test.reshape(X_test.shape[0],784)

Y_train = to_categorical(Y_train)
Y_test = to_categorical(Y_test)
print(Y_train.shape)

#normalization 
# print(X_train.max())
X_train = X_train/255
# print(X_train.max())

X_test= X_test/255
# print(X_test.max())

model= Sequential()
model.add(Dense(10,input_dim=784,activation='softmax'))
model.compile(optimizer=SGD(learning_rate=0.001),loss='categorical_crossentropy',metrics=['accuracy'])
cp  =  ModelCheckpoint('my_model.keras',monitor='val_loss',save_best_only=True,mode='min')
res = model.fit(X_train,Y_train, epochs=5, batch_size=32,validation_split=0.2,callbacks=[cp])
model.evaluate(X_test,Y_test)

plt.plot(res.history['loss'],label='training loss')
plt.plot(res.history['val_loss'],label='Validation loss')
plt.legend()
plt.show()

# print(out)




