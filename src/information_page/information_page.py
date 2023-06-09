import os

import streamlit as st


def default_page():
    with open(os.path.dirname(__file__) + '/about_predict.html', 'r') as f:
        about_predict_info_ = f.read()
    with open(os.path.dirname(__file__) + '/about_gallery.html', 'r') as f:
        about_gallery_info_ = f.read()
    with open(os.path.dirname(__file__) + '/about_plots.html', 'r') as f:
        about_plots_info_ = f.read()

    st.markdown("<h1 style='text-align: center; color: green;'>Wildfire Detection</h1>", unsafe_allow_html=True)
    st.markdown("""
    <h3>
    This is the
    <font color="#00FFFF">information page </font>
    for
    <font color="#228B22">Wildfire Detection from Satellite Images </font>
    project
    </h3>""", unsafe_allow_html=True)

    expander_1 = st.expander(f"""#### _About Project_ """)
    expander_1.markdown(f""" 
    \nThis is bachelor diploma thesis project made by Eugene Sazonenko. Faculty of Applied Mathematics, Igor Sikorsky Kyiv Polytechnic Institute, Ukraine.
    \nThe goal of this project is to develop a system for detecting fires from satellite images using semantic segmentation algorithms, and to compare the effectiveness of basic neural network architectures for solving this problem.
    \nIn this work, we investigated basic algorithms for semantic segmentation using neural networks. We learn the U-Net, SegNet, and DeconvNet networks on LandSat-8 images from this project https://doi.org/10.1016/j.isprsjprs.2021.06.002.
    ...
    """)

    expander_results = st.expander(f"""#### _Results of experiments_ """)
    expander_results.markdown(
        f"""The implemented networks demonstrate high performance on data from this dataset ...""")

    expander_faq = st.expander(f"""#### _Frequently Asked Questions_ """)

    predict_page_usage_expander = expander_faq.expander(f"""#### About `Predict` page""")
    predict_page_usage_expander.markdown(about_predict_info_, unsafe_allow_html=True)

    gallery_page_usage_expander = expander_faq.expander(f"""#### About `Gallery` page""")
    gallery_page_usage_expander.markdown(about_gallery_info_, unsafe_allow_html=True)

    plots_page_usage_expander = expander_faq.expander(f"""#### About `History Plots` page""")
    plots_page_usage_expander.markdown(about_plots_info_, unsafe_allow_html=True)

    create_models_faq_expander(expander_faq)
    create_metrics_faq_expander(expander_faq)


def create_metrics_faq_expander(mother_expander):
    metrics_expander = mother_expander.expander(f"""#### About Metrics""")
    metrics_expander.markdown("""<h2><font color="purple">Understanding the Metrics</font></h2>""", unsafe_allow_html=1)

    metrics_expander.markdown("""<li><h3><strong><font color="green">Loss </font></strong></h3>""", unsafe_allow_html=1)
    metrics_expander.markdown(r'''$$Loss = \min_{f}\frac{1}{N}\sum_{i=1}^{N}L_{\theta}(f(x_i))+\lambda R(f)$$''')
    metrics_expander.markdown(
        """This is a measure of how well the neural network's predictions match the actual values. A lower loss value indicates better performance. Formula of loss is from doi.org/10.1007/s40745-020-00253-5</li>""",
        unsafe_allow_html=1)

    metrics_expander.markdown("""<li><h3><strong><font color="green">Pixel Accuracy</font></strong></h3>""",
                              unsafe_allow_html=1)
    metrics_expander.markdown(
        r'''$$Accuracy = {True Positive+True Negative \over True Positive+True Negative+False Positive+False Negative}.$$''')
    metrics_expander.markdown(
        """This is a measure of how many individual pixels the network correctly classified, as a percentage of the total number of pixels. A higher pixel accuracy indicates better performance.</li>""",
        unsafe_allow_html=True)

    metrics_expander.markdown(
        """<li><h3><strong><font color="green">IoU (IntersectionOver Union)</font></strong></h3>""",
        unsafe_allow_html=1)
    metrics_expander.markdown(r'''$$IoU = {True Positive \over True Positive+True Negative+False Positive}.$$''')
    metrics_expander.markdown(
        """This is another measure of overlap between the network's predicted segmentation and the actual segmentation, calculated slightly differently from the IoU. A higher Dice Coefficient indicates better performance.</li>""",
        unsafe_allow_html=True)

    metrics_expander.markdown("""<li><h3><strong><font color="green">Dice Coefficient</font></strong></h3>""",
                              unsafe_allow_html=1)
    metrics_expander.markdown(r'''$$Dice = {2 True Positive \over 2 True Positive+True Negative+False Positive}.$$''')
    metrics_expander.markdown(
        """This is another measure of overlap between the network's predicted segmentation and the actual segmentation, calculated slightly differently from the IoU. A higher Dice Coefficient indicates better performance.</li>""",
        unsafe_allow_html=True)


def create_models_faq_expander(mother_expander):
    models_expander = mother_expander.expander(f"""#### About Models""")
    models_expander.markdown(f"""This project uses the Tensorflow library to implement neural networks of the <font color="green">U-Net</font>, <font color="green">SegNet</font> and <font color="green">DeconvNet</font> architectures. 
    You can view the code of the implemented neural networks by selecting from the menu below""", unsafe_allow_html=1)
    print(os.listdir("./models"))
    with open('./models/unet_model.py', 'r') as file:
        unet_code = file.read()
    with open('../models/segnet_model.py', 'r') as file:
        segnet_code = file.read()
    with open('../src/models/deconvnet_model.py', 'r') as file:
        deconvnet_code = file.read()

    code_dict = {
        "Hide Code": "#Code is hidden",
        'U-Net': unet_code,
        'SegNet': segnet_code,
        'DeconvNet': deconvnet_code,
    }

    model_name_ = models_expander.selectbox('Select a model:', list(code_dict.keys()))
    models_expander.code(code_dict[model_name_], language='python')
