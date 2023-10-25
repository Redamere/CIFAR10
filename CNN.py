import tensorflow as tf
from tensorflow import keras
from keras import datasets, layers, models
import matplotlib.pyplot as plt
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator

#Load and split datset
(train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data() #dataset object

#normalize pixel values between 0 and 1 -- preprocessing

train_images, test_images = train_images/255.0, test_images/255.0

class_names =['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

#look at one image
IMG_INDEX = 2

# plt.imshow(train_images[IMG_INDEX], cmap=plt.cm.binary)
# plt.xlabel(class_names[train_labels[IMG_INDEX][0]])
# plt.show()

#CNN Architecture
model = models.Sequential()
model.add(layers.Conv2D(32, (3,3), activation='relu', input_shape=(32, 32, 3 )))
""" First layer -- 32 is amount of filters, (3,3) is sample size (how big are the filters), input_shape is what to expect in the first layer -- 32x32x3 
    following layers do not need input shape, they will predict the shape based on the first layer
"""
model.add(layers.MaxPooling2D((2,2))) # 2x2 sample size with a stride of 2
model.add(layers.Conv2D(64, (3,3), activation='relu'))
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Conv2D(64, (3,3), activation='relu'))

# print(model.summary())

#Add dense layers

model.add(layers.Flatten()) #takes the photo pixels and make it into a flat line of pixels 
model.add(layers.Dense(64, activation='relu')) #creates a 64 neuron dense layer that connects the pixels to an activation functino of rectified linear unit (relu)
model.add(layers.Dense(10)) #output layer, 10 neurons because it is the amount of classes we have 

#print(model.summary())

#-----Time to train the model-----#

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy']
)

history = model.fit(train_images, train_labels, epochs=7, validation_data=(test_images,test_labels))

#-----Evaluating the model-----#

test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)

print(test_acc)

