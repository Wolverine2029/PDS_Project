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
import plotly.express as px
import functions


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
    st.write("Data Analysis")


    # if supervised == 'Data Analysis':
    #     # # edaDF_option = st.selectbox('Select one option:',
    #     #                               ['', 'Descriptive Analysis', 'Target Analysis'],
    #     #                               format_func=lambda x: 'Select an option' if x == '' else x)
    #
    edaDF_option = ['Descriptive Analysis', 'Target Analysis',
                       'Distribution of Numerical Columns', 'Count Plots of Categorical Columns',
                       'Box Plots', 'Outlier Analysis', 'Variance of Target with Categorical Columns']
    functions.bar_space(3)
    vizuals = st.multiselect("Choose which visualizations you want to see 👇", edaDF_option)

    if 'Descriptive Analysis' in vizuals:
        st.subheader('Descriptive Analysis:')
        st.dataframe(Clean_DF.describe())

    if 'Target Analysis' in vizuals:
        st.subheader("Select target column:")
        target_column = st.selectbox("", Clean_DF.columns, index=len(Clean_DF.columns) - 1)

        st.subheader("Histogram of target column")
        fig = px.histogram(Clean_DF, x=target_column)
        c1, c2, c3 = st.columns([0.5, 2, 0.5])
        c2.plotly_chart(fig)

    num_columns = Clean_DF.select_dtypes(exclude='object').columns
    cat_columns = Clean_DF.select_dtypes(include='object').columns

    if 'Distribution of Numerical Columns' in vizuals:

        if len(num_columns) == 0:
            st.write('There is no numerical columns in the data.')
        else:
            selected_num_cols = functions.multiselect_container('Choose columns for Distribution plots:',
                                                                        num_columns, 'Distribution')
            st.subheader('Distribution of numerical columns')
            i = 0
            while (i < len(selected_num_cols)):
                c1, c2 = st.columns(2)
                for j in [c1, c2]:

                    if (i >= len(selected_num_cols)):
                        break

                    fig = px.histogram(Clean_DF, x=selected_num_cols[i])
                    j.plotly_chart(fig, use_container_width=True)
                    i += 1
    if 'Count Plots of Categorical Columns' in vizuals:

        if len(cat_columns) == 0:
            st.write('There is no categorical columns in the data.')
        else:
            selected_cat_cols = functions.multiselect_container('Choose columns for Count plots:', cat_columns,
                                                                        'Count')
            st.subheader('Count plots of categorical columns')
            i = 0
            while (i < len(selected_cat_cols)):
                c1, c2 = st.columns(2)
                for j in [c1, c2]:

                    if (i >= len(selected_cat_cols)):
                        break

                    fig = px.histogram(Clean_DF, x=selected_cat_cols[i], color_discrete_sequence=['indianred'])
                    j.plotly_chart(fig)
                    i += 1
    if 'Box Plots' in vizuals:
        if len(num_columns) == 0:
            st.write('There is no numerical columns in the data.')
        else:
            selected_num_cols = functions.multiselect_container('Choose columns for Box plots:', num_columns,
                                                                        'Box')
            st.subheader('Box plots')
            i = 0
            while (i < len(selected_num_cols)):
                c1, c2 = st.columns(2)
                for j in [c1, c2]:

                    if (i >= len(selected_num_cols)):
                        break

                    fig = px.box(Clean_DF, y=selected_num_cols[i])
                    j.plotly_chart(fig, use_container_width=True)
                    i += 1

    if 'Outlier Analysis' in vizuals:
        st.subheader('Outlier Analysis')
        c1, c2, c3 = st.columns([1, 2, 1])
        c2.dataframe(functions.number_of_outliers(Clean_DF))

    if 'Variance of Target with Categorical Columns' in vizuals:

        df_1 = Clean_DF.dropna()

        high_cardi_columns = []
        normal_cardi_columns = []

        for i in cat_columns:
            if (Clean_DF[i].nunique() > Clean_DF.shape[0] / 10):
                high_cardi_columns.append(i)
            else:
                normal_cardi_columns.append(i)

        if len(normal_cardi_columns) == 0:
            st.write('There is no categorical columns with normal cardinality in the data.')
        else:

            st.subheader('Variance of target variable with categorical columns')
            model_type = st.radio('Select Problem Type:', ('Regression', 'Classification'), key='model_type')
            selected_cat_cols = functions.multiselect_container('Choose columns for Category Colored plots:',
                                                                        normal_cardi_columns, 'Category')

            if 'Target Analysis' not in vizuals:
                target_column = st.selectbox("Select target column:", Clean_DF.columns, index=len(Clean_DF.columns) - 1)

            i = 0
            while (i < len(selected_cat_cols)):

                if model_type == 'Regression':
                    fig = px.box(df_1, y=target_column, color=selected_cat_cols[i])
                else:
                    fig = px.histogram(df_1, color=selected_cat_cols[i], x=target_column)

                st.plotly_chart(fig, use_container_width=True)
                i += 1

            if high_cardi_columns:
                if len(high_cardi_columns) == 1:
                    st.subheader('The following column has high cardinality, that is why its boxplot was not plotted:')
                else:
                    st.subheader(
                        'The following columns have high cardinality, that is why its boxplot was not plotted:')
                for i in high_cardi_columns:
                    st.write(i)

                st.write('<p style="font-size:140%">Do you want to plot anyway?</p>', unsafe_allow_html=True)
                answer = st.selectbox("", ('No', 'Yes'))

                if answer == 'Yes':
                    for i in high_cardi_columns:
                        fig = px.box(df_1, y=target_column, color=i)
                        st.plotly_chart(fig, use_container_width=True)

# if supervised == 'Data Analysis':
#     # summary = dataframe.describe()
#     # st.write(summary)



