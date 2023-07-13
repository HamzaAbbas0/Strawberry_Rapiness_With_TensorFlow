import cv2
import numpy as np
from keras.models import load_model

# Load the Keras model and labels
model = load_model('converted_keras/keras_model.h5')
with open('converted_keras/labels.txt', 'r') as f:
    labels = f.read().splitlines()

# Load the image
image_path = 'images/369.jpg'
image = cv2.imread(image_path)

# Preprocess the image
# Perform any necessary preprocessing on the image before passing it to the model

# Resize the image to match the input size of the model
image = cv2.resize(image, (224, 224))

# Convert the image to the input format expected by the model (e.g., RGB, normalized)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
image = image.astype(np.float32) / 255.0

# Perform prediction with the model
# Reshape the image to match the input shape expected by the model
image = np.expand_dims(image, axis=0)

# Make the prediction
predictions = model.predict(image)

# Get the predicted class label
class_index = np.argmax(predictions)
class_label = labels[class_index]

# Display the result
image = cv2.cvtColor(image[0], cv2.COLOR_RGB2BGR)
cv2.putText(image, class_label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
cv2.imshow('Image Detection', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
