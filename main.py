import os
import streamlit as st
import pandas as pd
from streamlit_option_menu import option_menu

#Main Page
EXAMPLE_NO = 3


def streamlit_menu(example=3):
    if example == 3:
        selected = option_menu(
            menu_title=None,  # required
            options=["Home", "Regression", "Classification", "About"],
            icons=["house", "hourglsass-split", "exsclamation", "people-fill"],
            menu_icon="cast",  # optional
            default_index=0,  # optional
            orientation="horizontal",
            styles={
                "container": {"padding": "0!important", "background-color": "#fafafa"},
                "icon": {"color": "orange", "font-size": "25px"},
                "nav-link": {
                    "font-size": "25px",
                    "text-align": "left",
                    "margin": "0px",
                    "--hover-color": "#eee",
                },
                "nav-link-selected": {"background-color": "green"},
            },
        )
        return selected


selected = streamlit_menu(example=EXAMPLE_NO)

if selected == "Home":
    st.title(f"{selected}")
if selected == "Supervised Learning":
    st.title(f"{selected}")
if selected == "Unsupervised Learning":
    st.title(f"{selected}")
if selected == "About":
    st.title(f"{selected}")
if selected == "Contact":
    st.title(f"{selected}")

# Hiding the hamburger Logo by default
# st.markdown(""" <style>
# #MainMenu {visibility: hidden;}
# footer {visibility: hidden;}
# </style> """, unsafe_allow_html=True)

# Limiting Scrolling
padding = 0
st.markdown(f""" <style>
    .reportview-container .main .block-container{{
        padding-top: {padding}rem;
        padding-right: {padding}rem;
        padding-left: {padding}rem;
        padding-bottom: {padding}rem;
    }} </style> """, unsafe_allow_html=True)



st.title("""
DANA - A Fun place to play around with Machine Learning algorithms! 
""")
# Can be used wherever a "file-like" object is accepted:
uploaded_file = st.file_uploader("")
if uploaded_file is not None:
    dataframe = pd.read_csv(uploaded_file)
    st.write(dataframe)

supervised = st.selectbox('Select one option:', ['', 'Summary of my Data', 'Clean the data', 'Data Analysis', 'Data Distribution'], format_func=lambda x: 'Select an option' if x == '' else x)

if supervised:
    st.success('Yay! 🎉')
else:
    st.warning('No option is selected')
# print("you selected: ",option)