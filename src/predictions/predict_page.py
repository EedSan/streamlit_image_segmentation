import os
import streamlit as st
from matplotlib import pyplot as plt

from image_processing import image_processing
from predictions.predict_models import predict_model

MODELS = {
    # "U-Net": "./resources/models/unet/unet-19-epochs-weights.h5",
    "U-Net": "./resources/models/unet/unet-11-epochs-weights.h5",
    "SegNet": './resources/models/segnet/segnet-29-epochs-weights.h5',
    "DeconvNet": "./resources/models/deconvnet/deconvnet-16-epochs-weights.h5"
}


# todo add logger

def predict_with_model_name(model_name_, data_):
    model_path_ = MODELS[model_name_]
    print(model_name_)
    prediction_ = predict_model(model_name_, model_path_, data_)
    return prediction_


def predictions_page():
    st.markdown("""<h2><font color="#00FFFF"> Predict page </font> </h2>""", unsafe_allow_html=True)
    st.write(os.listdir("./"))
    st.markdown(f"### Choose an image...")
    uploaded_file_ = st.file_uploader("1", type='tif', label_visibility='collapsed')
    if uploaded_file_ is not None:
        ground_data, bands_msg_ = image_processing(uploaded_file_)
        st.markdown(f'#### {bands_msg_}')

        model_name = st.selectbox("Choose a model:", list(MODELS.keys()))
        prediction_ = predict_with_model_name(model_name, ground_data)

        prediction_plot_ = prediction_to_pyplot(prediction_)

        images = {"User Image": ground_data, "Segmentation Result": prediction_plot_}
        cols = st.columns(len(images))

        for i, (title, image) in enumerate(images.items()):
            cols[i].markdown(f'### {title}')
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
            if title == list(images.keys())[0]:
                cols[i].image(image, use_column_width=True)
            else:
                cols[i].pyplot(image)


def prediction_to_pyplot(prediction_):
    fig_, ax = plt.subplots()
    fig_.patch.set_visible(False)
    plt.set_cmap('viridis')
    ax.axis('off')
    ax.imshow(prediction_[0], interpolation='nearest')
    return fig_
