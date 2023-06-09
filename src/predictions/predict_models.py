import numpy as np
from tensorflow.keras.metrics import BinaryIoU

from models.deconvnet_model import deconvnet
from models.dice_coefficient import dice_coef
from models.segnet_model import segnet
from models.unet_model import unet


def init_model_from_name(model_name_):
    if model_name_ == "U-Net":
        model_ = unet()
    elif model_name_ == "SegNet":
        model_ = segnet()
    elif model_name_ == "DeconvNet":
        model_ = deconvnet()
    return model_


def predict_model(model_name, model_path_, ground_data):
    model_ = init_model_from_name(model_name)
    model_.load_weights(model_path_)

    prediction_ = model_.predict(np.array([ground_data]))

    return prediction_


def evaluate_model(model_name_, model_path_, ground_data, true_mask):
    model_ = init_model_from_name(model_name_)
    model_.compile(optimizer='adam', loss='binary_crossentropy',
                   metrics=['accuracy', BinaryIoU(), dice_coef],
                   run_eagerly=True)

    model_.load_weights(model_path_)

    evaluation_results = model_.evaluate(np.array([ground_data]), np.array([true_mask]))
    metrics_names = model_.metrics_names
    return dict(zip(metrics_names, evaluation_results))
