from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, Conv2DTranspose, Concatenate


def deconvnet(input_shape=(256, 256, 3), num_classes=1):
    inputs = Input(shape=input_shape)

    # Encoder
    conv1 = Conv2D(64, (3, 3), activation='relu', padding='same')(inputs)
    conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = Conv2D(128, (3, 3), activation='relu', padding='same')(pool1)
    conv4 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv3)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv4)

    conv5 = Conv2D(256, (3, 3), activation='relu', padding='same')(pool2)
    conv6 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv5)
    conv7 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv6)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv7)

    conv8 = Conv2D(512, (3, 3), activation='relu', padding='same')(pool3)

    # Decoder
    up9 = Concatenate()([Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same')(conv8), conv7])
    conv9 = Conv2D(256, (3, 3), activation='relu', padding='same')(up9)
    conv10 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv9)
    conv11 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv10)

    up12 = Concatenate()([Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(conv11), conv4])
    conv12 = Conv2D(128, (3, 3), activation='relu', padding='same')(up12)
    conv13 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv12)

    up14 = Concatenate()([Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(conv13), conv2])
    conv14 = Conv2D(64, (3, 3), activation='relu', padding='same')(up14)
    conv15 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv14)

    # Output
    outputs = Conv2D(num_classes, (1, 1), activation='sigmoid')(conv15)

    model = Model(inputs=inputs, outputs=outputs)
    return model
