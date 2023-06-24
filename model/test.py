import tensorflow as tf
import tf_dataset
import cv2 as cv
import numpy as np
import os
from matplotlib import pyplot as plt

#Load the test dataset
def test_data_load(files, path, image_ext=".jpg"):
    X = []
    Y = []

    for file in files:
        img = cv.imread(os.path.join(path, file + image_ext), cv.IMREAD_GRAYSCALE)
        box = np.array([0, 0, 0, 0], dtype=float)

        img, box = tf_dataset.format_image(img, box)
        img = img.astype(float) / 255.
        box = np.asarray(box, dtype=float) / tf_dataset.input_size
        label = np.append(box, 0)

        X.append(img)
        Y.append(label)

    X = np.array(X)

    X = np.expand_dims(X, axis=3)

    X = tf.convert_to_tensor(X, dtype=tf.float32)

    Y = tf.convert_to_tensor(Y, dtype=tf.float32)

    result = tf.data.Dataset.from_tensor_slices((X, Y))

    result = result.map(tf_dataset.format_instance, num_parallel_calls=tf.data.AUTOTUNE)
    result = result.batch(1)
    result = result.repeat()

    return result

# Trained model
saved_model_path = 'saved_model'
model = tf.saved_model.load(saved_model_path)

# Name of the image test file without extension
img_test_name = 'bee'

# Directory containing test image
test_dir = 'test'

# Load of the test dataset
test_ds = test_data_load([img_test_name], test_dir)

# Get the first image as a list
image = list(test_ds.take(1).as_numpy_iterator())[0][0]

# Apply the model on the image
predictions = model(image)

# Retrieve the predicted bounding box from the predictions and scale it
predicted_box = predictions[1][0] * tf_dataset.input_size

# Cast to tf record
predicted_box = tf.cast(predicted_box, tf.int32)

# Retrieve the index of the label predicted
predicted_label_index = np.argmax(predictions[0][0])

# Retrieve the accuracy of the prediction
accuracy = np.max(predictions[0][0])

# Get the image
image = image[0]

# Scale, cast and color the image
image = image.astype("float") * 255.0
image = image.astype(np.uint8)
image_color = cv.cvtColor(image, cv.COLOR_GRAY2RGB)

color = (255, 255, 0)
img_label = "bee"
# print box is yellow if a bee is predicted and green if it is a cat
if predicted_label_index == 1:
    color = (0, 255, 0)
    img_label = "cat"

# Convert the TensorFlow tensor predicted_box to a numpy array
predicted_box_n = predicted_box.numpy()


# Draw a rectangle around the predicted bounding box
cv.rectangle(image_color, predicted_box_n, color, 2)

# Draw a filled rectangle at the bottom of the predicted bounding box
cv.rectangle(image_color, (predicted_box_n[0], predicted_box_n[1] + predicted_box_n[3] - 20),
             (predicted_box_n[0] + predicted_box_n[2], predicted_box_n[1] + predicted_box_n[3]), color, -1)

# Add a text label on the image near the bottom of the bounding box
cv.putText(image_color, img_label, (predicted_box_n[0] + 5, predicted_box_n[1] + predicted_box_n[3] - 5),
           cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0))

plt.imshow(image_color)
plt.axis("off")
plt.show()

# Print the accurency
print("Accuracy : " + str(accuracy))
