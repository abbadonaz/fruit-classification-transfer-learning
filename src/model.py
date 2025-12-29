from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Flatten, Dense, Dropout

def build_model(num_classes, img_size=(224,224,3)):
    base_model = VGG16(weights="imagenet", include_top=False, input_shape=img_size)

    # this will freeze initial layers as they have already been trained on ImageNet
    for layer in base_model.layers:
        layer.trainable = False

    # Add custom layers on top of the base model
    x = Flatten()(base_model.output)
    x = Dense(256, activation="relu")(x)
    x = Dropout(0.5)(x)
    output = Dense(num_classes, activation="softmax")(x) # Softmax for multi-class classification

    model = Model(inputs=base_model.input, outputs=output)
    return model