import keras as keras
from keras.layers import Dense
from keras.models import Sequential
import scipy.io
import numpy as np
import cv2 as cv
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

hoda_data = scipy.io.loadmat('Data_hoda_full.mat')

data = hoda_data['Data'].reshape(-1)
labels = hoda_data['labels'].reshape(-1)

data_resized = np.array([cv.resize(img, dsize=(5, 5)) for img in data])
data_norm = data_resized / 255
data_norm = data_norm.reshape(60000, 25)

x_train, x_test, y_train, y_test = train_test_split(data_norm, labels)

print("size of training dataset is: " + str(x_train.shape))
print("size of test dataset is: " + str(x_test.shape))

classes_num = 10
y_train_cat = keras.utils.to_categorical(y_train, classes_num)
y_test_cat = keras.utils.to_categorical(y_test, classes_num)

model = Sequential(
    [Dense(50, activation='relu', input_shape=(25,)), Dense(50, activation='relu'), Dense(50, activation='relu'),
     Dense(10, activation='softmax')])
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
hist = model.fit(x_train, y_train_cat, batch_size=512, epochs=100, verbose=1)

loss, accuracy = model.evaluate(x_test, y_test_cat)
print(accuracy)
print(loss)

plt.plot(hist.history['accuracy'])
plt.plot(hist.history['loss'])
plt.legend(['training', 'validation'], loc='upper left')
plt.show()

prediction = model.predict_classes(x_test)
for i in range(0, 100):
    print(f'predicted digit is : {prediction[i]}  and label of digit is : {y_test[i]}')