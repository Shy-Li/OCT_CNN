import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import os
from numpy.random import seed
import numpy as np
import glob
import pandas as pd

seed(1337)
tf.random.set_seed(1337)

image_size = (256,128)
batch_size = 32
regularizer = tf.keras.regularizers.L2(1e-5)

train_ds1 = tf.keras.preprocessing.image_dataset_from_directory(
    "Bench_Train_Cropped_1.5e6_512",
    seed=1337,
    image_size=image_size,
    batch_size=batch_size,
    color_mode="grayscale",
)

train_ds2 = tf.keras.preprocessing.image_dataset_from_directory(
    "Catheter_cropped_512/training",
    seed=1337,
    image_size=image_size,
    batch_size=batch_size,
    color_mode="grayscale",
)

train_ds = train_ds1.concatenate(train_ds2)#.concatenate(train_ds3).concatenate(train_ds4)

val_ds1 = tf.keras.preprocessing.image_dataset_from_directory(
    "Bench_val_set",
    seed=1337,
    image_size=image_size,
    batch_size=batch_size,
    color_mode="grayscale",
)
val_ds2 = tf.keras.preprocessing.image_dataset_from_directory(
    "Catheter_cropped_512/val",
    seed=1337,
    image_size=image_size,
    batch_size=batch_size,
    color_mode="grayscale",
)

val_ds = val_ds1.concatenate(val_ds2)
sum1 = 0


def make_model(input_shape, num_classes):
    inputs = keras.Input(shape=input_shape)
    # Entry block
    x = layers.experimental.preprocessing.Normalization()(inputs)
    x = layers.Conv2D(4, 3, strides=2, padding="same",kernel_regularizer=regularizer)(x)
    x = layers.Activation("relu")(x)

    previous_block_activation = x  # Set aside residual

    for size in [8,16,32,64]:
        x = layers.Activation("relu")(x)
        x = layers.SeparableConv2D(size, 3, padding="same",kernel_regularizer=regularizer)(x)

        x = layers.Activation("relu")(x)
        x = layers.SeparableConv2D(size, 3, padding="same",kernel_regularizer=regularizer)(x)

        x = layers.MaxPooling2D(3, strides=2, padding="same")(x)

        # Project residual
        residual = layers.Conv2D(size, 1, strides=2, padding="same",kernel_regularizer=regularizer)(
            previous_block_activation
        )
        x = layers.add([x, residual])  # Add back residual
        previous_block_activation = x  # Set aside next residual

    x = layers.SeparableConv2D(128, 3, padding="same",kernel_regularizer=regularizer)(x)
    x = layers.Activation("relu")(x)

    x = layers.GlobalAveragePooling2D()(x)
    if num_classes == 2:
        activation = "sigmoid"
        units = 1
    else:
        activation = "softmax"
        units = num_classes

    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(units, activation=activation,kernel_regularizer=regularizer)(x)
    return keras.Model(inputs, outputs)


model = make_model(input_shape=image_size + (1,), num_classes=2)
# keras.utils.plot_model(model, show_shapes=True)

epochs = 200
isFile = os.path.isdir('models/models_train_on_both3') 
if isFile == False:
    os.mkdir('models/models_train_on_both3')
reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2,
                              patience=5, min_lr=1e-8)
earlyStopping = keras.callbacks.EarlyStopping(
    monitor='val_loss',
    min_delta=0,
    patience=20,
    verbose=0,
    mode="auto",
    baseline=None,
    restore_best_weights=False,)
mcp_save = keras.callbacks.ModelCheckpoint(
    "models/models_train_on_both3/save_at_{epoch}.h5", 
                                           save_best_only=False, 
                                           monitor='val_loss', 
                                           mode='min'
                                           )

model.compile(
    optimizer=keras.optimizers.Adam(5e-5),
    loss = "binary_crossentropy",
    metrics=keras.metrics.AUC(),
)

# class weight
neg = len(glob.glob("Bench_Train_Cropped_1.5e6_512/Cancer/*.png"))\
    + len(glob.glob("Catheter_cropped_512/training/Cancer/*.png"))
pos =  len(glob.glob("Bench_Train_Cropped_1.5e6_512/Normal/*.png"))\
    + len(glob.glob("Catheter_cropped_512/training/Normal/*.png"))
total = neg + pos
weight_for_0 = (1 / neg) * (total / 2.0)
weight_for_1 = (1 / pos) * (total / 2.0)
class_weight = {0: weight_for_0, 1: weight_for_1}

# train 
history = model.fit(
    train_ds, epochs=epochs, 
    callbacks=[earlyStopping,mcp_save,reduce_lr], 
        validation_data=val_ds, 
        class_weight=class_weight
    )

# convert the history.history dict to a pandas DataFrame:     
hist_df = pd.DataFrame(history.history) 

with open('both3.csv', mode='w') as f:
    hist_df.to_csv(f)
