# Import necessary libraries
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.preprocessing.image import ImageDataGenerator

# Initialising the Convolutional Neural Network(CNN)
classifier = Sequential()

# Convolution Layer
classifier.add(Conv2D(32, (3, 3), input_shape=(64, 64, 3), activation="relu"))

# Pooling Layer
classifier.add(MaxPooling2D(pool_size=(2, 2)))

# Flattening
classifier.add(Flatten())

# Forward Connection--->>Fully Connected Neural Network
classifier.add(Dense(units=128, activation="relu"))
# Output Layer
classifier.add(Dense(units=4, activation="softmax"))

# Compiling the CNN
classifier.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

# Pre-processing the data
train_datagen = ImageDataGenerator(rescale=1./255, shear_range=0.2, zoom_range=0.2, horizontal_flip=True)
test_datagen = ImageDataGenerator(rescale=1./255)

# Assigning train and test data
training_set = train_datagen.flow_from_directory(r"E:\Deep_Learning\CNN_Overview\Player_Dataset",
                                                 target_size=(64, 64), batch_size=32, class_mode="categorical")
test_set = test_datagen.flow_from_directory(r"E:\Deep_Learning\CNN_Overview\Player_Dataset",
                                            target_size=(64, 64), batch_size=32, class_mode="categorical")

# Fitting the model
model = classifier.fit_generator(training_set, steps_per_epoch=2000, epochs=3, validation_data=test_set, validation_steps=2000)

# Saving the model
classifier.save("cnn-model.h5")
print("Model saved to disk")

# Printing the values of the classes
print(training_set.class_indices)
