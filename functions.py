import io
import pandas as pd
import streamlit as st


def bar_space(num_lines=1):
    for _ in range(num_lines):
        st.write("")


def multiselect_container(massage, arr, key):
    container = st.container()
    select_all_button = st.checkbox("Select all for " + key + " plots")
    if select_all_button:
        selected_num_cols = container.multiselect(massage, arr, default=list(arr))
    else:
        selected_num_cols = container.multiselect(massage, arr, default=arr[0])

    return selected_num_cols


def number_of_outliers(Clean_DF):
    Clean_DF = Clean_DF.select_dtypes(exclude='object')

    Q1 = Clean_DF.quantile(0.25)
    Q3 = Clean_DF.quantile(0.75)
    IQR = Q3 - Q1

    ans = ((Clean_DF < (Q1 - 1.5 * IQR)) | (Clean_DF > (Q3 + 1.5 * IQR))).sum()
    Clean_DF = pd.DataFrame(ans).reset_index().rename(columns={'index': 'column', 0: 'count_of_outliers'})
    return Clean_DF




