from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, BatchNormalization, Activation


def segnet(input_size=(256, 256, 3), num_classes=1):
    inputs = Input(input_size)

    # Encoder
    conv1 = Conv2D(64, 3, padding='same')(inputs)
    conv1 = BatchNormalization()(conv1)
    conv1 = Activation('relu')(conv1)
    conv2 = Conv2D(64, 3, padding='same')(conv1)
    conv2 = BatchNormalization()(conv2)
    conv2 = Activation('relu')(conv2)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = Conv2D(128, 3, padding='same')(pool1)
    conv3 = BatchNormalization()(conv3)
    conv3 = Activation('relu')(conv3)
    conv4 = Conv2D(128, 3, padding='same')(conv3)
    conv4 = BatchNormalization()(conv4)
    conv4 = Activation('relu')(conv4)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv4)

    conv5 = Conv2D(256, 3, padding='same')(pool2)
    conv5 = BatchNormalization()(conv5)
    conv5 = Activation('relu')(conv5)
    conv6 = Conv2D(256, 3, padding='same')(conv5)
    conv6 = BatchNormalization()(conv6)
    conv6 = Activation('relu')(conv6)
    conv7 = Conv2D(256, 3, padding='same')(conv6)
    conv7 = BatchNormalization()(conv7)
    conv7 = Activation('relu')(conv7)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv7)

    # Decoder
    up1 = UpSampling2D(size=(2, 2))(pool3)
    conv8 = Conv2D(256, 3, padding='same')(up1)
    conv8 = BatchNormalization()(conv8)
    conv8 = Activation('relu')(conv8)
    conv9 = Conv2D(256, 3, padding='same')(conv8)
    conv9 = BatchNormalization()(conv9)
    conv9 = Activation('relu')(conv9)
    conv10 = Conv2D(256, 3, padding='same')(conv9)
    conv10 = BatchNormalization()(conv10)
    conv10 = Activation('relu')(conv10)

    up2 = UpSampling2D(size=(2, 2))(conv10)
    conv11 = Conv2D(128, 3, padding='same')(up2)
    conv11 = BatchNormalization()(conv11)
    conv11 = Activation('relu')(conv11)
    conv12 = Conv2D(128, 3, padding='same')(conv11)
    conv12 = BatchNormalization()(conv12)
    conv12 = Activation('relu')(conv12)

    up3 = UpSampling2D(size=(2, 2))(conv12)
    conv13 = Conv2D(64, 3, padding='same')(up3)
    conv13 = BatchNormalization()(conv13)
    conv13 = Activation('relu')(conv13)
    conv14 = Conv2D(64, 3, padding='same')(conv13)
    conv14 = BatchNormalization()(conv14)
    conv14 = Activation('relu')(conv14)

    # Classifier
    conv15 = Conv2D(num_classes, 1, activation='sigmoid')(conv14)

    model = Model(inputs=inputs, outputs=conv15)

    return model
