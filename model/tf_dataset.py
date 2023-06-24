import tensorflow as tf
import cv2 as cv
import numpy as np
import os

# Define the root directory of the dataset
dataset_dir = "project_final"

# Define the directory for labels within the dataset
labels_dir = os.path.join(dataset_dir, "labels")

# Define the directory for images within the dataset
images_dir = os.path.join(dataset_dir, "images")

# Define the input size for the images
input_size = 244

# Define the number of classes in the dataset
nb_classes = 2

# Define the batch size for training and evaluation
batch_size = 32


# Split the dataset into 2 parts : training (80%) and validation (20%)
def list_files(image_ext='.jpg', split_percentage=80):
    files = []

    discarded = 0

    for filename in os.listdir(labels_dir):

        if filename.endswith(".txt"):

            with open(labels_dir + "/" + filename, 'r') as fp:
                lines = fp.readlines()
                if len(lines) > 1:
                    discarded += 1
                    continue

            strip = filename[0:len(filename) - len(".txt")]

            image_path = images_dir + "/" + strip + image_ext
            if os.path.isfile(image_path):
                files.append(strip)

    size = len(files)

    split_training = int(split_percentage * size / 100)

    return files[0:split_training], files[split_training:]


training_files, validation_files = list_files()

# Format the image to the wanted size
def format_image(img, box):
    height, width = img.shape
    max_size = max(height, width)
    r = max_size / input_size
    new_width = int(width / r)
    new_height = int(height / r)
    new_size = (new_width, new_height)
    resized = cv.resize(img, new_size, interpolation=cv.INTER_LINEAR)
    new_image = np.zeros((input_size, input_size), dtype=np.uint8)
    new_image[0:new_height, 0:new_width] = resized

    x, y, w, h = box[0], box[1], box[2], box[3]
    new_box = [int((x - 0.5 * w) * width / r), int((y - 0.5 * h) * height / r), int(w * width / r),
               int(h * height / r)]
    return new_image, new_box


#Load the data from the project
def data_load(files, image_ext=".jpg"):
    X = []
    Y = []

    for file in files:
        img = cv.imread(os.path.join(images_dir, file + image_ext), cv.IMREAD_GRAYSCALE)

        with open(labels_dir + "/" + file + ".txt", 'r') as fp:
            line = fp.readlines()[0]
            k = int(line[0])

            box = np.array(line[1:].split(), dtype=float)

        img, box = format_image(img, box)
        img = img.astype(float) / 255.
        box = np.asarray(box, dtype=float) / input_size
        label = np.append(box, k)

        X.append(img)
        Y.append(label)

    X = np.array(X)

    X = np.expand_dims(X, axis=3)

    X = tf.convert_to_tensor(X, dtype=tf.float32)

    Y = tf.convert_to_tensor(Y, dtype=tf.float32)

    result = tf.data.Dataset.from_tensor_slices((X, Y))

    return result


# Loading the data for the training dataset and the validation dataset
raw_train_ds = data_load(training_files)
raw_validation_ds = data_load(validation_files)


# The function returns the original image and a tuple containing the one-hot encoding of the label and the coordinates of the bounding box
def format_instance(image, label):
    return image, (tf.one_hot(int(label[4]), nb_classes), [label[0], label[1], label[2], label[3]])

#Format and shuffle the training dataset
def tune_training_ds(dataset):
    dataset = dataset.map(format_instance, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.shuffle(1024, reshuffle_each_iteration=True)
    dataset = dataset.repeat()  # The dataset be repeated indefinitely.
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    return dataset

#Training dataset
train_ds = tune_training_ds(raw_train_ds)

#Format the validation dataset
def tune_validation_ds(dataset):
    dataset = dataset.map(format_instance, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.batch(len(validation_files) // 4)
    dataset = dataset.repeat()
    return dataset

#Validation dataset
validation_ds = tune_validation_ds(raw_validation_ds)
