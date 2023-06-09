import streamlit as st

# Define all your pages as separate functions
from information_page.information_page import default_page
from display_plots.display_history_plots import display_history_metrics_plots
from predictions.predict_page import predictions_page
from gallery.gallery_page import display_images
import streamlit_nested_layout

st.set_page_config(layout="wide")

pages = {
    "Information Page": default_page,
    "Predict Page": predictions_page,
    "Gallery Page": display_images,
    "Metrics Plots Page": display_history_metrics_plots
}

if 'current_page' not in st.session_state:
    st.session_state['current_page'] = 'Information Page'

for page in pages.keys():
    if st.sidebar.button(page):
        st.session_state['current_page'] = page

pages[st.session_state['current_page']]()
