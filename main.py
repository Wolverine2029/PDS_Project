import os
import streamlit as st
import pandas as pd
import pandas_profiling
import seaborn as sns
import tkinter
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')
from streamlit_option_menu import option_menu
from streamlit_pandas_profiling import st_profile_report


# Main Page
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
    numeric_cols = dataframe.select_dtypes(include=['number']).columns
    non_numeric_cols = dataframe.select_dtypes(exclude=['number']).columns

supervised = st.selectbox('Select one option:',
                          ['', 'Data Report', 'Summary of my Data', 'Check for Null Values in the data',
                            'Data Analysis','Data Distribution'], format_func=lambda x: 'Select an option' if x == '' else x)

if supervised:
    st.success('Yay! 🎉')
else:
    st.warning('No option is selected')
# print("you selected: ",option)
if supervised == 'Data Report':
    pr = dataframe.profile_report()
    st_profile_report(pr)

if supervised == 'Summary of my Data':
    summary = dataframe.describe()
    st.write(summary)

if supervised == 'Check for Null Values in the data':
    nullData = dataframe.isnull().sum()
    st.write(nullData)
    nullDataPercentage = dataframe.isnull().mean()
    Clean_DF = dataframe.copy()

    cleanDF_option = st.selectbox('Select one option:',
                                  ['', 'Remove columns with Nulls values', 'Impute the null with constant values',
                                   'Impute the null with statistics'],
                                  format_func=lambda x: 'Select an option' if x == '' else x)
    if cleanDF_option == 'Remove columns with Nulls values':
        # Removing the columns with null values but only those columns that have null value percentage greater than 30% in the column.
        nullDataPercentage[nullDataPercentage > .3]
        CleanDF_less_missing_columns = dataframe.loc[:,
                                       nullDataPercentage <= .3].copy()  # equivalent to df.drop(columns=pct_missing[pct_missing > .3].index)
        # CleanDF_less_missing_columns.shape
        st.write("The updated DataSet is")
        st.write(CleanDF_less_missing_columns.head())
    if cleanDF_option == 'Impute the null with constant values':
        CleanDF_replace_constant = dataframe.copy()
        # numeric_cols = dataframe.select_dtypes(include=['number']).columns
        # non_numeric_cols = dataframe.select_dtypes(exclude=['number']).columns
        CleanDF_replace_constant[numeric_cols] = CleanDF_replace_constant[numeric_cols].fillna(0)
        CleanDF_replace_constant[non_numeric_cols] = CleanDF_replace_constant[non_numeric_cols].fillna('NA')
        CleanDF_constant = CleanDF_replace_constant.head()

        # st.write(CleanDF_constant)
    if cleanDF_option == 'Impute the null with statistics':
        CleanDF_replace_statistics = dataframe.copy()

        cleanDF_option_replace = st.selectbox('Select one option:',
                                              ['', 'Mean', 'Median', 'Mode'],
                                              format_func=lambda x: 'Select an option' if x == '' else x)

        st.write("This option only works on Numerical Columns")
        if cleanDF_option_replace == 'Mean':
            mean = CleanDF_replace_statistics[numeric_cols].mean()
            CleanDF_replace_statistics[numeric_cols] = CleanDF_replace_statistics[numeric_cols].fillna(mean)
            # st.write(CleanDF_replace_statistics.head())
        if cleanDF_option_replace == 'Median':
            med = CleanDF_replace_statistics[numeric_cols].median()
            CleanDF_replace_statistics[numeric_cols] = CleanDF_replace_statistics[numeric_cols].fillna(med)
            # st.write(CleanDF_replace_statistics.head())
        if cleanDF_option_replace == 'Mode':
            mode = CleanDF_replace_statistics[numeric_cols].mode()
            CleanDF_replace_statistics[numeric_cols] = CleanDF_replace_statistics[numeric_cols].fillna(mode)
            # st.write(CleanDF_replace_statistics.head())


    if (cleanDF_option == 'Remove columns with Nulls values'):
        Clean_DF = CleanDF_less_missing_columns.copy()
    if (cleanDF_option == 'Impute the null with constant values'):
        Clean_DF = CleanDF_replace_constant.copy()
    if (cleanDF_option == 'Impute the null with statistics'):
        Clean_DF = CleanDF_replace_statistics.copy()
    if ((cleanDF_option == 'Impute the null with statistics') and (cleanDF_option_replace == 'Mean')):
        Clean_DF = CleanDF_replace_statistics.copy()
    if ((cleanDF_option == 'Impute the null with statistics') and (cleanDF_option_replace == 'Median')):
        Clean_DF = CleanDF_replace_statistics.copy()
    if ((cleanDF_option == 'Impute the null with statistics') and (cleanDF_option_replace == 'Mode')):
        Clean_DF = CleanDF_replace_statistics.copy()

    st.write("Please find the cleaned dataset below:")
    st.write(Clean_DF.head())

    # Clean_DF.corr()
    # # plotting correlation heatmap
    # dataplot = sns.heatmap(Clean_DF.corr(), cmap="YlGnBu", annot=True)
    #
    # # displaying heatmap
    # s = plt.show()
    # st.write(s)

