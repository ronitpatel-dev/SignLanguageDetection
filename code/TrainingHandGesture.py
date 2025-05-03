import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout, Dense, Flatten
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# ------------------ Configuration ------------------
IMG_SIZE = 256
BATCH_SIZE = 8
EPOCHS = 25
NUM_CLASSES = 6
INPUT_SHAPE = (IMG_SIZE, IMG_SIZE, 1)
CLASS_NAMES = ['NONE', 'ONE', 'TWO', 'THREE', 'FOUR', 'FIVE']

TRAIN_DIR = os.path.join("Internship on AI", "Day  - Hand Gesture Recognition using DL", "HandGestureDataset", "train")
TEST_DIR = os.path.join("Internship on AI", "Day  - Hand Gesture Recognition using DL", "HandGestureDataset", "test")
MODEL_SAVE_PATH = "model.h5"

# ------------------ Model Definition ------------------
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=INPUT_SHAPE),
    MaxPooling2D(pool_size=(2, 2)),

    Conv2D(64, (3, 3), activation='relu'),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),

    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),

    Conv2D(256, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),

    Flatten(),
    Dense(150, activation='relu'),
    Dropout(0.25),
    Dense(NUM_CLASSES, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# ------------------ Data Preparation ------------------
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=12.,
    width_shift_range=0.2,
    height_shift_range=0.2,
    zoom_range=0.15,
    horizontal_flip=True
)

val_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    color_mode='grayscale',
    batch_size=BATCH_SIZE,
    classes=CLASS_NAMES,
    class_mode='categorical',
    shuffle=True
)

val_generator = val_datagen.flow_from_directory(
    TEST_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    color_mode='grayscale',
    batch_size=BATCH_SIZE,
    classes=CLASS_NAMES,
    class_mode='categorical',
    shuffle=True
)

# ------------------ Callbacks ------------------
callbacks = [
    EarlyStopping(monitor='val_loss', patience=10, verbose=1),
    ModelCheckpoint(filepath=MODEL_SAVE_PATH, monitor='val_loss', save_best_only=True, verbose=1)
]

# ------------------ Model Training ------------------
model.fit(
    train_generator,
    steps_per_epoch=len(train_generator),
    epochs=EPOCHS,
    validation_data=val_generator,
    validation_steps=len(val_generator),
    callbacks=callbacks
)
