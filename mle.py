import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from sklearn.metrics import classification_report, confusion_matrix

train_dir = os.path.join("plant_dataset/train")
test_dir = os.path.join("plant_dataset/test")

IMG_WIDTH = 224
IMG_HEIGHT = 224
BATCH_SIZE = 32

train_data = ImageDataGenerator(rescale=1./255, rotation_range=0.2, vertical_flip=True, fill_mode='nearest')
test_data = ImageDataGenerator(rescale=1./255)

train_set = train_data.flow_from_directory(train_dir, target_size=(IMG_WIDTH, IMG_HEIGHT), batch_size=BATCH_SIZE, class_mode='categorical')
test_set = test_data.flow_from_directory(test_dir, target_size=(IMG_WIDTH, IMG_HEIGHT), batch_size=BATCH_SIZE, shuffle=False, class_mode='categorical')

model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(IMG_WIDTH, IMG_HEIGHT, 3)),
    tf.keras.layers.Conv2D(16, (3, 3), padding='same', activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(32, (3, 3), padding='same', activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.Dense(38, activation='softmax')
])

print("Total classes in training set:", len(train_set.class_indices))
print("Total classes in training set:", len(test_set.class_indices))



model.compile(loss='categorical_crossentropy', optimizer=tf.keras.optimizers.RMSprop(learning_rate=0.001), metrics=['acc'])

history_model = model.fit(train_set, steps_per_epoch=train_set.samples // BATCH_SIZE, epochs=16, validation_data=test_set, validation_steps=test_set.samples // BATCH_SIZE, verbose=1, callbacks=[tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)])

model.save("first_one.h5")

plt.plot(history_model.history['acc'])
plt.plot(history_model.history['val_acc'])
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()

plt.plot(history_model.history['loss'])
plt.plot(history_model.history['val_loss'])
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()

y_test = test_set.classes
pred = model.predict(test_set, steps=test_set.samples // BATCH_SIZE + 1, verbose=1).argmax(axis=1)

print(classification_report(y_test, pred, target_names=list(test_set.class_indices.keys())))

conf = confusion_matrix(y_test, pred)
plt.figure(figsize=(10,10))
plt.imshow(conf, cmap=plt.cm.Blues)
plt.colorbar()
ticks = np.arange(len(test_set.class_indices))
plt.xticks(ticks, list(test_set.class_indices.keys()), rotation=45)
plt.yticks(ticks, list(test_set.class_indices.keys()))
plt.show()
