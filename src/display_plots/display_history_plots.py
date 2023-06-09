import os

import pandas as pd
from PIL import Image
import json
import streamlit as st

paths_file_ = open(os.path.dirname(__file__) + "/history_plots_path.json")
paths_ = json.load(paths_file_)
explanations_file_ = open(os.path.dirname(__file__) + "/metrics_explanations.json")
explanations_ = json.load(explanations_file_)


def display_history_metrics_plots():
    df = pd.DataFrame(paths_)
    # df = df.rename(columns=df.iloc[0]).drop(df.index[0])
    df.index.name = 'metric'
    st.markdown("""<h2><font color="#00FFFF"> Metrics history comparison </font> </h2>""", unsafe_allow_html=True)
    metrics_list = list(df.index)
    st.markdown(
        """
        <style>
        img {
            cursor: pointer;
            transition: all .2s ease-in-out;
        }
        img:hover {
            transform: scale(1.15); 
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    for metric_idx, metric in enumerate(metrics_list):
        row = df.loc[[metric]]
        expander_1 = st.expander(metric)
        cols = expander_1.columns(len(row.columns))
        for idx, (column_name, column_val) in enumerate(row.items()):
            img_ = Image.open(column_val.iloc[0])
            cols[idx].markdown(f'### {column_name}')
            cols[idx].image(img_, use_column_width=True)
        expander_2 = expander_1.expander(f"Expand for {metric} metric explanation")
        expander_2.markdown(f"Explanation: {explanations_[metric]}")
