import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def create_generators(train_dir, val_dir, test_dir, img_size=(224,224), batch_size=32):

    """
    This function sets up data pipelines for training, validation, and test datasets using Keras 'ImageDataGenerator'. 
    It handles:
    - Preprocessing (rescaling pixel values)
    - Data augmenentation (random rotations, shifts, flips for training data)
    - Directory-based data loading
    """
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True)
    
    val_test_datagen=ImageDataGenerator(rescale=1./255)
    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=img_size,
        batch_size=batch_size,
        class_mode='categorical'
    )

    val_generator = val_test_datagen.flow_from_directory(
        val_dir,
        target_size=img_size,
        batch_size=batch_size,
        class_mode='categorical'
    )

    test_generator = val_test_datagen.flow_from_directory(
        test_dir,
        target_size=img_size,
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=False #predictions line up with filenames for evaluation later
    )


    return train_generator, val_generator, test_generator