import tensorflow as tf
import tf_dataset as ds


def build_feature_extractor(inputs):
    # Convolutional layer 1
    # Apply 16 filters with a kernel size of 3x3 and ReLU activation
    # The input shape is (ds.input_size, ds.input_size, 1)
    x = tf.keras.layers.Conv2D(32, kernel_size=3, activation='relu', input_shape=(ds.input_size, ds.input_size, 1))(inputs)
    x = tf.keras.layers.AveragePooling2D(2, 2)(x)

    # Convolutional layer 2
    # Apply 32 filters with a kernel size of 3x3 and ReLU activation
    x = tf.keras.layers.Conv2D(64, kernel_size=3, activation='relu')(x)
    x = tf.keras.layers.AveragePooling2D(2, 2)(x)


    # Convolutional layer 3
    # Apply 64 filters with a kernel size of 3x3 and ReLU activation
    x = tf.keras.layers.Conv2D(64, kernel_size=3, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    x = tf.keras.layers.AveragePooling2D(2, 2)(x)

    # Return the output tensor
    return x

def build_model_adaptor(inputs):
    # Flatten the input tensor
    # Convert the multi-dimensional input tensor into a 1D tensor
    x = tf.keras.layers.Flatten()(inputs)

    # Fully connected layer
    # Apply a dense layer with 64 units and ReLU activation
    x = tf.keras.layers.Dense(64, activation='relu')(x)

    # Return the output tensor
    return x

def build_classifier_head(inputs):
    # Fully connected layer
    # Apply a dense layer with ds.nb_classes units and softmax activation
    # The output represents the predicted probabilities for each class
    return tf.keras.layers.Dense(ds.nb_classes, activation='softmax', name='classifier_head')(inputs)


def build_regressor_head(inputs):
    # Fully connected layer
    # Apply a dense layer with 4 units
    # The output represents the regression predictions
    return tf.keras.layers.Dense(units=4, name='regressor_head')(inputs)


def build_model(inputs):
    # Build the feature extractor
    feature_extractor = build_feature_extractor(inputs)

    # Build the model adaptor
    model_adaptor = build_model_adaptor(feature_extractor)

    # Build the classification head
    classification_head = build_classifier_head(model_adaptor)

    # Build the regression head
    regressor_head = build_regressor_head(model_adaptor)

    # Create the overall model
    # Combine the inputs and the outputs of the classification and regression heads
    model = tf.keras.Model(inputs=inputs, outputs=[classification_head, regressor_head])

    # Compile the model
    # Use Adam optimizer, categorical cross-entropy loss for classification, and mean squared error loss for regression
    # Track accuracy for classification and mean squared error for regression
    model.compile(optimizer=tf.keras.optimizers.Adam(),
                  loss={'classifier_head': 'categorical_crossentropy', 'regressor_head': 'mse'},
                  metrics={'classifier_head': 'accuracy', 'regressor_head': 'mse'})

    # Return the built model
    return model


# Construct the model
# Create the model using the build_model function, specifying the input shape
model = build_model(tf.keras.layers.Input(shape=(ds.input_size, ds.input_size, 1,)))

# Print model summary
# Display a summary of the model's architecture
model.summary()

# Set the number of epochs
EPOCHS = 15

# Train the model
# Fit the model using the training dataset, specifying the number of steps per epoch,
# validation dataset, validation steps, and the number of epochs
model.fit(ds.train_ds,
          steps_per_epoch=(len(ds.training_files) // ds.batch_size),
          validation_data=ds.validation_ds, validation_steps=1,
          epochs=EPOCHS)

# Define the path to save the trained model
saved_model_path = 'saved_model'

# Save the model in SavedModel format
tf.saved_model.save(model, saved_model_path)
