
# Importing necessary libraries
import keras
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.preprocessing.image import ImageDataGenerator

# Initialising the LeNet Architecture
lenet = Sequential()

# Convolution Layer--1
lenet.add(Conv2D(6, kernel_size=(5, 5), input_shape=(32, 32, 3), activation="relu"))

# Pooling Layer--1
lenet.add(MaxPooling2D(pool_size=(2, 2)))

# Convolution Layer--2
lenet.add(Conv2D(16, kernel_size=(5, 5), activation="relu"))

# Pooling Layer--2
lenet.add(MaxPooling2D(pool_size=(2, 2)))

# Flattening
lenet.add(Flatten())

# Forward Connection---Fully connected Neural Network
lenet.add(Dense(units=120, activation="relu"))
# Hidden Layer
lenet.add(Dense(units=84, activation="relu"))
# Output Layer
lenet.add(Dense(units=4, activation="softmax"))


# Compiling the CNN
lenet.compile(optimizer=keras.optimizers.Adam(), loss=keras.metrics.categorical_crossentropy, metrics=["accuracy"])

# Pre-processing the data
train_datagen = ImageDataGenerator(rescale=1./255, shear_range=0.2, zoom_range=0.2, horizontal_flip=True)
test_datagen = ImageDataGenerator(rescale=1./255)

# Assigning train and test data
training_set = train_datagen.flow_from_directory(r"E:\Deep_Learning\CNN_Overview\Player_Dataset",
                                                 target_size=(32, 32), batch_size=128, class_mode="categorical")
test_set = test_datagen.flow_from_directory(r"E:\Deep_Learning\CNN_Overview\Player_Dataset",
                                            target_size=(32, 32), batch_size=128, class_mode="categorical")

# Fitting the model
model = lenet.fit_generator(training_set, steps_per_epoch=2000, epochs=3, validation_data=test_set, validation_steps=2000)

# Saving the model
lenet.save("lenet-model.h5")
print("Model Saved to disk")

# Printing the values of the classes
print(training_set.class_indices)
