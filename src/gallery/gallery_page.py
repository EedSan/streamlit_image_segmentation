import json
import os

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import rasterio
import streamlit as st
from PIL import Image
from pandas import json_normalize
from streamlit_image_select import image_select

from image_processing import get_img_762bands_from_path
from predictions.predict_models import evaluate_model
from predictions.predict_page import prediction_to_pyplot, predict_with_model_name, MODELS


def plot_mask_from_path(mask_path_):
    img = mpimg.imread(mask_path_)
    fig = plt.figure(frameon=False, figsize=(10, 10), dpi=50)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)
    ax.imshow(img, aspect='auto')
    return fig


def eval_model(model_name, ground_data_, true_mask):
    model_path_ = MODELS[model_name]
    evaluation_results_ = evaluate_model(model_name, model_path_, ground_data_, true_mask)
    return evaluation_results_


def get_mask_arr(path):
    img = rasterio.open(path).read().transpose((1, 2, 0))
    seg = np.float32(img)
    return seg


def display_images():
    data_ = json.load(open(os.path.dirname(__file__) + "/test_images_masks.json"))
    df = json_normalize(data_["data"])
    images_ = df["image"].values.tolist()
    masks_ = df["mask"].values.tolist()
    images_ = list(map(lambda img_element_: np.uint8(get_img_762bands_from_path(img_element_) * 255), images_))

    st.markdown("""<h2><font color="#00FFFF"> Gallery page </font> </h2>""", unsafe_allow_html=True)

    image_idx_ = image_select(label="Select image...", images=images_, use_container_width=False, return_value='index')
    # if image_idx_ is not None or st.session_state.load_state:
    #     st.session_state.load_state = True

    cols = st.columns(3, gap='small')
    st.markdown(
        """
        <style>
        img {
            cursor: pointer;
            transition: all .2s ease-in-out;
        }
        img:hover {
            transform: scale(1.1); 
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
    with cols[0]:
        st.markdown(f"### Test image")
        st.image(Image.fromarray(images_[image_idx_]).resize(size=(500, 500)))
    with cols[1]:
        st.markdown(f"### True mask of test image")
        fig = plot_mask_from_path(masks_[image_idx_])
        st.pyplot(fig)
    with cols[2]:
        ground_data_ = np.float32(images_[image_idx_]) / 255
        model_name = st.selectbox(f"Choose a model to build prediction for test image:", list(MODELS.keys()))

        prediction_ = predict_with_model_name(model_name, ground_data_)
        prediction_plot_ = prediction_to_pyplot(prediction_)
        st.pyplot(prediction_plot_)
        true_mask_arr_ = get_mask_arr(masks_[image_idx_])
        evaluation_results = eval_model(model_name, ground_data_, true_mask_arr_)
        expander_1 = st.expander(f"Expand to show metrics evaluated for this image")
        for key, val in evaluation_results.items():
            expander_1.write(f'{key} : {val:.3f}')
