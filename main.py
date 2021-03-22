from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from mlxtend.data import loadlocal_mnist
import matplotlib.pyplot as plt
import numpy as np

train_img, train_label = loadlocal_mnist(images_path='files/train-images-idx3-ubyte',
                                         labels_path='files/train-labels-idx1-ubyte')

test_img, test_label = loadlocal_mnist(images_path='files/t10k-images.idx3-ubyte',
                                       labels_path='files/t10k-labels-idx1-ubyte')

print("size of training dataset is: " + str(train_img.shape))
print("size of test dataset is: " + str(test_img.shape))

model = Sequential([Dense(64, activation='relu'), Dense(64, activation='relu'), Dense(10, activation='softmax')])
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
hist = model.fit(train_img, train_label, epochs=100)

model.save_weights('model.h5')
# model.load_weights('model.h5')
fig = plt.figure()
plt.hist(train_label, histtype='bar', rwidth=0.8)
fig.suptitle('Class distribution in Data', fontsize=15)
plt.xlabel('classes')
plt.ylabel('count')
plt.show()

loss, accuracy = model.evaluate(test_img, test_label)
print(accuracy)
print(loss)
plt.plot(hist.history['accuracy'])
plt.plot(hist.history['loss'])
plt.legend(['training', 'validation'], loc='upper left')
plt.show()

predictions = model.predict(test_img)

for i in range(0, 100):
    print(f'predicted digit is : {np.argmax(predictions[i])}  and label of digit is : {test_label[i]}')


