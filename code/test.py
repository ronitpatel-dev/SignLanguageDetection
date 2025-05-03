# Import necessary libraries
from keras.preprocessing import image
from keras.models import load_model
import numpy as np
import os

# Load the pre-trained Keras model
model = load_model('model.h5')
print("Model Loaded Successfully")

# Define a function to classify an image using the loaded model
def classify(img_file):
    img_name = img_file

    # Load the image and resize it to match the model's expected input size (256x256)
    test_image = image.load_img(img_name, target_size=(256, 256), grayscale=True)

    # Convert image to numpy array
    test_image = image.img_to_array(test_image)

    # Expand dimensions to create a batch of size 1 (model expects batches)
    test_image = np.expand_dims(test_image, axis=0)

    # Make prediction using the model
    result = model.predict(test_image)

    # Extract prediction probabilities and find the class with the highest probability
    arr = np.array(result[0])
    print(arr)
    maxx = np.amax(arr)
    max_prob = arr.argmax(axis=0)
    max_prob = max_prob + 1  # Shift index to match class label

    # List of class names corresponding to model output
    classes = ["NONE", "ONE", "TWO", "THREE", "FOUR", "FIVE"]

    # Get the result as a class name
    result = classes[max_prob - 1]

    # Print the image filename and the predicted result
    print(img_name, result)

# Set the path to the folder containing images to classify
path = 'D:/MasterClass/Artificial_Intelligence/Day13/Dataset/val/TWO'

# Initialize a list to store image file paths
files = []

# Walk through the directory and collect all PNG image paths
for r, d, f in os.walk(path):
    for file in f:
        if '.png' in file:
            files.append(os.path.join(r, file))

# Loop through each file and classify it using the model
for f in files:
    classify(f)
    print('\n')  # Print newline for separation between outputs
