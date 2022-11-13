import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn.metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, confusion_matrix
import streamlit as st
from streamlit_option_menu import option_menu
import numpy as np
import matplotlib

matplotlib.use('TkAgg')
from streamlit_option_menu import option_menu
from streamlit_pandas_profiling import st_profile_report
import plotly.express as px
import functions

EXAMPLE_NO = 3


def streamlit_menu(example=3):
    if example == 3:
        with st.sidebar:
            selected = option_menu(
                menu_title=None,  # required
                options=["Home", "Logistic Regression", "KNN", "Random Forest", "Naive Bayes", "Visualizations", "About"],
                # icons=["house", "hourglass-split", "hourglass-split", "hourglass-split", "hourglass-split", "people-fill"],
               # icons=["house", " ", " ", " ", " ", " ", "people-fill"],
               # menu_icon="cast",  # optional
               # default_index=0,  # optional
                orientation="vertical",
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
    st.title("Welcome to Classification Central")
if selected == "Logistic Regression":
    st.title(f"{selected}")
if selected == "KNN":
    st.title(f"{selected}")
if selected == "Random Forest":
    st.title(f"{selected}")
if selected == "Naive Bayes":
    st.title(f"{selected}")
if selected == "Visualizations":
    st.title(f"{selected}")
if selected == "DIY":
    st.title(f"{selected}")
if selected == "About":
    st.title(f"{selected}")
if selected == "Contact":
    st.title(f"{selected}")

url = "https://raw.githubusercontent.com/Wolverine2029/PDS_Project/main/data.csv"
df = pd.read_csv(url)

if selected == "Logistic Regression":
    Logistic_Regression = st.selectbox('Logistic Regression: Select one option:',
                                       ['', 'Step 1: Check for Null Values', 'Step 2: Summarize Data',
                                        'Step 3: Convert Categorical to Numerical values',
                                        'Step 4: Create your Test and Training Sets and Create your Model',
                                        'Step 5: Run your model and check your Model Accuracy',
                                        'Step 6: Prediction Results'],
                                       format_func=lambda x: 'Select an option' if x == '' else x)
    if Logistic_Regression == 'Step 1: Check for Null Values':
        st.markdown("# Null Values")
        st.markdown("""
                    Let's determine if there are any null values in the data.
                    In this step, we print out all of the column names.  A true value represents 
                    a null value and will need to be located and deleted. 
                    """)
        if st.button('Code'):
            st.write("df.isnull().any()")
        st.text(df.isnull().any())
        if st.button('Explanation'):
            st.write("""
                    At the bottom of the column name list, we see that there
                    are null values signified by the 'true' statement.  All of the null
                    values are located in the 'Unnamed: 32' column and thus we will drop the entire column.
                    Do some research on your dataset to determine if you have an entire null column, or just a 
                    few null values within several of the columns.  If this is the case, don't drop the whole column
                    for your dataset.  Instead, just drop the nulls in each column.  This can be done with a df.dropna()
                    command. 
                    """)
        st.markdown("# Check for null values: Next step")
        st.markdown("""
                    Now, we will delete the null values in our dataset. 
                    There are many ways to handle Null values, go to our DIY section to learn more ways to deal with them!
                    """)
        if st.button('Code '):
            st.write("df.drop('Unnamed: 32', axis=1)")
        df = df.drop('Unnamed: 32', axis=1)
        if st.button('Explanation '):
            st.write("Once we perform the df.drop function on our identified null values, "
                     "we have removed all nulls from our dataset.  If you have a different dataset, change "
                     "the column name to reflect your dataset.  If you are not dropping an entire column but just several rows, "
                     "use the df.dropna() command.")
        st.markdown(""
                    "")
        st.markdown("# Yay! We successfully removed the Null values")
        st.markdown(""" 
                Let's move on to Step 2!
                """)
    if Logistic_Regression == 'Step 2: Summarize Data':
        df = df.drop('Unnamed: 32', axis=1)
        st.markdown("# Summarize Data")
        st.markdown("""
                Summarizing data in this case means to generate descriptive statistics, which includes insight pertaining to
                central tendency, dispersion, and the shape of a dataset's distribution.
                """)
        if st.button('Code'):
            st.write("df.describe()")
        st.text(df.describe())
        st.markdown(""
                    "")
        st.markdown("# Yay! You have successfully summarized your data. Do you see anything interesting in your dataset?")
        st.markdown(""" 
                    Let's move on to Step 3
                    """)
    if Logistic_Regression == 'Step 3: Convert Categorical to Numerical values':
        df = df.drop('Unnamed: 32', axis=1)
        st.markdown("# Convert Categorical to Numerical for our Model")
        st.markdown("""
                    Binary classification uses 0's and 1's to perform the classification analysis.  
                    0 represents the positive value and 1 represents the negative value.  A majority of datasets
                    are going to contain the qualitative classifier instead of the 0 and 1 values.  As such, when
                    performing classification, it is necessary to write a code that will replace the classifiers
                    in the dataset.    
                    """)
        if st.button('   Code'):
            st.write("df.diagnosis.replace(['M', 'B'], [1, 0], inplace = True)")
        df.diagnosis.replace(["M", "B"], [1, 0], inplace=True)
        if st.button('   Explanation'):
            st.write("""
                    For the breast cancer dataset used in this tutorial, the goal is to classify 'Benign' (i.e., 1, 
                    the negative value) and 'Malignant' (i.e., 0, the positive value) tumors.  These are represented 
                    by 'M' and 'B' in the dataset under the 'diagnosis' column.  If performing classification analysis 
                    on a different dataset, determine your dataset's classifiers and its respective column name.  For 
                    best practice, place the positive value first within the brackets.
                    """)
        st.markdown("# Convert Categorical to Numerical: Next step")
        st.markdown("""
                    Let's check if our Categorical values are co Numerical have been replaced with the 0 and 1 values.      
                    """)
        if st.button('Code  '):
            st.write("df.head()")
        st.text(df.head())
        if st.button('Explanation  '):
            st.write("""
                    We can see that the diagnosis column has been replaced with 0's and 1's, meaning that our 
                    replace() function was successful.
                    """)
        st.markdown("# Convert Categorical to Numerical: Next step")
        st.markdown("""
                Now, lets see the counts of your 0 and 1 values.  It is a best practice to have
                a good representation of both.  If there is large discrepancy between classifier counts percentage wise, 
                we have different ways to deal with class imbalance and we can work around it or we can look for a new dataset. 
                """)
        if st.button(' Code  '):
            st.write("df.groupby(['diagnosis']).diagnosis.count()")
        st.text(df.groupby(["diagnosis"]).diagnosis.count())
        if st.button(' Explanation  '):
            st.write("""
                    We can see that there are 212 Benign (1) tumors and 357 Malignant (0) 
                    tumors in the breast cancer dataset.  This is an acceptable representation of 
                    both classifiers, thus we can proceed with this dataset. 
                    Remember, if you are performing classification analysis on a different dataset, 
                    insert your classifier column name into the Part 3 code above. 
                    """)
        st.markdown("# Convert Categorical to Numerical: Next step")
        st.markdown("""
                    Let's now visualize the counts of our categorical values.  When working with data, it is always
                    best to visualize if you can!  Charts, graphs, maps, etc help the viewer to understand the data in
                    a more clear and efficient way.      
                    """)
        if st.button(' Code   '):
            st.write("sns.countplot('diagnosis', data=df)")
            st.write("plt.show()")
        sns.countplot('diagnosis', data=df)
        st.set_option('deprecation.showPyplotGlobalUse', False)
        st.pyplot()
        if st.button(' Explanation   '):
            st.write("""
                    We can see that there is a good representation of both classifiers, where 0=malignant
                    and 1=benign. 
                    Remember, if you are performing classification analysis on a different dataset, 
                    insert your classifier column name into the above code. 
                    """)
        st.markdown(""
                    "")
        st.markdown("# Yay 🎉")
        st.markdown(""" 
                       You have completed Logistic Regression, Convert Categorical to Numerical values!
                       You are ready to move on to Step 4
                       """)
    if Logistic_Regression == 'Step 4: Create your Test and Training Sets and Create your Model':
        df.isnull().any()
        df = df.drop('Unnamed: 32', axis=1)
        df.diagnosis.replace(["M", "B"], [1, 0], inplace=True)
        st.markdown("# Create your Test and Training Set and Create your Model")
        st.markdown("""
                    Let's ensure that X only takes on the independent variables in the dataset.      
                    """)
        if st.button('Code'):
            st.write("X = df.drop(['id','diagnosis'],axis=1)")
        if st.button('Explanation'):
            st.write("""
                    Since the diagnosis column contains the classifiers, we want to remove this column from the X variable.  We 
                    won't neglect the diagnosis column for long, however, as we will utilize these values for the y variable!
                    We remove the id column as it is just an identifier, and does not play a role in tumor classification.  
                    If working with your own dataset, make sure you remove the irrelevant columns pertaining to your dataset!
                    """)
        X = df.drop(['id', 'diagnosis'], axis=1)
        st.markdown("# Create your Test and Training Set and Create your Model: Next step")
        st.markdown("""
                    Let's ensure that Y only takes on the classifier column in the dataset.      
                    """)
        if st.button('Code '):
            st.write("Y = df.iloc[:, 1]")
        if st.button('Explanation '):
            st.write("""
                    Since the diagnosis column contains the classifiers, we consider this the dependent variable column.  
                    If working with your own dataset, make sure you do some research on indexes so that you can accurately 
                    store your classifier column!
                    """)
        Y = df.iloc[:, 1]
        st.markdown("# Create your Test and Training Set and Create your Model: Next step")
        st.markdown("""
                    It is time to split the data into test and training sets in a standard 80% Training data and 20% Test data ratio.
                    """)
        if st.button(' Code'):
            st.write(
                "xTrain, xTest, yTrain, yTest = sklearn.model_selection.train_test_split(X, Y, train_size=0.8, random_state =20221023)")
        xTrain, xTest, yTrain, yTest = sklearn.model_selection.train_test_split(X, Y, train_size=0.8,
                                                                                random_state=20221023)
        if st.button(' Explanation'):
            st.write("""
                    This will not return any output, but will instead assign data to variables that will return output later on 
                    in the classification process. 
                    You train the model using the training set, and test the model using the testing set. Training the model means to 
                    create the model, whereas testing the model means to test the accuracy of the model.  Training and testing sets
                    are created for both the X and Y variables.  
                    It is common to split the test and training sets by 20/80 percent, however, you should do some research on what is 
                    best for your model!  
                    The random_state number sets the seed so that you calculate the same accuracy each time.  Any number can be used, 
                    but many people prefer to use today's date.
                    """)
        st.markdown("# Create your Test and Training Set and Create your Model: Next step")
        st.markdown("""
                    Create a model that performs logistic regression.      
                    """)
        if st.button(' Code '):
            st.write("regressionModel = LogisticRegression(solver='newton-cg')")
        regressionModel = LogisticRegression(solver='newton-cg')
        st.markdown("# Create your Test and Training Set and Create your Model: next step")
        st.markdown("""
                    It is time for our model to train! This happens using the fit() method.  This allows the regressor to "study"
                    the data and "learn" from it.  This step will not return any output, but will be used later on 
                    in the classification process.       
                    """)
        if st.button('  Code'):
            st.write("regressionModel.fit(xTrain, yTrain)")
        regressionModel.fit(xTrain, yTrain)
        st.markdown("# Create your Test and Training Set and Create your Model: next step")
        st.markdown("""
                    Now that we have created our model and trained it, Part 6 will use our testing dataset to test the model.  
                    This step will not return any output, but will be used later on in the classification process.
                    """)
        if st.button('Code  '):
            st.write("predRegression =regressionModel.predict(xTest)")
        predRegression = regressionModel.predict(xTest)
        st.markdown(""
                    "")
        st.markdown("# Yay 🎉")
        st.markdown(""" 
                    You have successfully learned to split your dataset into test and training and create a logistic regression model to perform classification!
                    Let's proceed to Step 5
                    """)
    if Logistic_Regression == 'Step 5: Run your model and check your Model Accuracy':
        df = df.drop('Unnamed: 32', axis=1)
        df.diagnosis.replace(["M", "B"], [1, 0], inplace=True)
        X = df.drop(['id', 'diagnosis'], axis=1)
        Y = df.iloc[:, 1]
        xTrain, xTest, yTrain, yTest = sklearn.model_selection.train_test_split(X, Y, train_size=0.8,
                                                                                random_state=20221023)
        regressionModel = LogisticRegression(solver='newton-cg')
        regressionModel.fit(xTrain, yTrain)
        predRegression = regressionModel.predict(xTest)
        st.markdown("# Check your Model Accuracy")
        st.markdown("""
                    Let's see the actual and predicted classifiers in array form. Can you spot the 
                    differences between the actual and predicted set?      
                    """)
        if st.button('Code'):
            st.write("print(yTest.values)")
            st.write("print(predRegression)")
        st.text("Actual breast cancer : ")
        st.text(yTest.values)
        st.text("\nPredicted breast cancer : ")
        st.text(predRegression)
        st.markdown("# Check your Model Accuracy: Next step")
        st.markdown("""
                    Now, calculate the accuracy score.  As it may sound, the accuracy score declares how 
                    accurately the model classifies the classifiers.     
                    """)
        if st.button('Code '):
            st.write("accuracy_score(yTest, predRegression) * 100")
        st.text("Accuracy Score: ")
        st.text(accuracy_score(yTest, predRegression) * 100)
        if st.button("Explanation "):
            st.write("""
                    Accuracy score is calculated by dividing the number of correct predictions by the total prediction number.
                    """)
        st.markdown("# Yay 🎉")
        st.markdown(""" 
                    You have learned to check your model's accuracy.
                    You are ready to move on to Step 6
                    """)
    if Logistic_Regression == 'Step 6: Prediction Results':
        df = df.drop('Unnamed: 32', axis=1)
        df.diagnosis.replace(["M", "B"], [1, 0], inplace=True)
        X = df.drop(['id', 'diagnosis'], axis=1)
        Y = df.iloc[:, 1]
        xTrain, xTest, yTrain, yTest = sklearn.model_selection.train_test_split(X, Y, train_size=0.8,
                                                                                random_state=20221023)
        regressionModel = LogisticRegression(solver='newton-cg')
        regressionModel.fit(xTrain, yTrain)
        predRegression = regressionModel.predict(xTest)
        if st.button('Code'):
            st.write("""plt.figure(figsize=(16, 14))""")
            st.write("""plt.subplot(3, 3, 1)""")
            st.write(
                """sns.heatmap(sklearn.metrics.confusion_matrix(yTest, predRegression), annot=True).set(title='Logistic Regression')""")
            st.write("""plt.show()""")
        fig = plt.figure(figsize=(16, 14)),
        plt.subplot(3, 3, 1)
        sns.heatmap(sklearn.metrics.confusion_matrix(yTest, predRegression), annot=True).set(
            title='Logistic Regression')
        st.set_option('deprecation.showPyplotGlobalUse', False)
        st.pyplot(fig)
        predVals = pd.DataFrame(data={'truth': yTest, 'regression': predRegression})
        if st.button("Explanation"):
            st.write("""
                    This is called a Confusion Matrix! Confused? Let us break it down for you, the top left value represents the true positive classifications.  This means you predicted positive and it's 
                    true.  In the case of our dataset, you predicted that a tumor is malignant and it actually is. 
                    The top right value represents the false positive classifications.  This is also known as a Type I error.
                    This means you predicted positive and it's false. In the case of our dataset, you predicted that a tumor is malignant
                    but it is actually benign.
                    The bottom left value represents the false negative classifications.  This is also known as a Type II error. 
                    This means you predicted negative and it's false.  In the case of our dataset, you predicted
                    that a tumor is benign but it is actually malignant.  
                    The bottom right value represents the true negative classifications.This means you predicted negative
                    and it's true.  In the case of our dataset, you predicted that a tumor is benign and it is indeed benign.   
                    """)
        st.markdown("# Yay 🎉")
        st.markdown(""" 
                    You learned to check your model prediction results!
                    You are now a Logistic Regression classification Pro! Check out our DIY section to play around with your own dataset.
                    """)

if selected == "KNN":
    KNN = st.selectbox('KNN: Select one option:',
                                   ['', 'Step 1: Check for Null Values', 'Step 2: Summarize Data',
                                    'Step 3: Convert Categorical to Numerical values',
                                    'Step 4: Create your Test and Training Sets',
                                    'Step 5: Run your model and check your Model Accuracy',
                                    'Step 6: Prediction Results'],
                       format_func=lambda x: 'Select an option' if x == '' else x)
    if KNN == 'Step 1: Check for Null Values':
        st.markdown("# Null Values")
        st.markdown("""
                            Let us try to determine if there are any null values in the data.
                            In this step, we print out all of the column names.  A true value represents 
                            a null value and will need to be located and deleted. 
                            """)
        if st.button('Code'):
            st.write("df.isnull().any()")
        st.text(df.isnull().any())
        if st.button('Explanation'):
            st.write("""
                            At the bottom of the column name list, we see that there
                            are 32 null values, signified by the 'true' statement.  All of the null
                            values are located in the 'Unnamed: 32 column and thus we will drop the entire column.
                            Do some research on your dataset to determine if you have an entire null column, or just a 
                            few null values within several of the columns.  If this is the case, don't drop the whole column
                            for your dataset.  Instead, just drop the nulls in each column.  This can be done with a df.dropna()
                            command. 
                            """)
        st.markdown("# Null Values: Next step")
        st.markdown("""
                            Now, lets delete the null values in our dataset.
                            """)
        if st.button('Code '):
            st.write("df.drop('Unnamed: 32', axis=1)")
        df = df.drop('Unnamed: 32', axis=1)
        if st.button('Explanation '):
            st.write("Once we perform the df.drop function on our identified null values, "
                     "we have removed all nulls from our dataset.  If you have a different dataset, change "
                     "the column name to reflect your dataset.  If you are not dropping an entire column but just several rows, "
                     "use the df.dropna() command.")
        st.markdown(""
                    "")
        st.markdown("# Yay 🎉")
        st.markdown(""" 
                    You have completed KNN, Null Values!
                    You are ready to move on to Step 2
                    """)
    if KNN == 'Step 2: Summarize Data':
        df = df.drop('Unnamed: 32', axis=1)
        st.markdown("# Summarize Data")
        st.markdown("""
                    Summarizing data in this case means to generate descriptive statistics, which includes insight pertaining to
                    central tendency, dispersion, and the shape of a dataset's distribution.
                    """)
        if st.button('Code'):
            st.write("df.describe()")
        st.text(df.describe())
        st.markdown(""
                    "")
        st.markdown("# Yay 🎉")
        st.markdown(""" 
                    You have completed KNN, Summarize Data!
                    You are ready to move on to the 'Replace Classifier With 0 and 1' step!
                    """)
    if KNN == 'Step 3: Convert Categorical to Numerical values':
        df = df.drop('Unnamed: 32', axis=1)
        st.markdown("# Convert Categorical to Numerical")
        st.markdown("""
                    Binary classification uses 0's and 1's to perform the classification analysis.  
                    0 represents the positive value and 1 represents the negative value.  A majority of datasets
                    are going to contain the qualitative classifier instead of the 0 and 1 values.  As such, when
                    performing classification, it is necessary to write a code that will replace the classifiers
                    in the dataset.    
                    """)
        if st.button('Code'):
            st.write("df.diagnosis.replace(['M', 'B'], [1, 0], inplace = True)")
        df.diagnosis.replace(["M", "B"], [1, 0], inplace=True)
        if st.button('Explanation'):
            st.write("""
                    For the breast cancer dataset used in this tutorial, the goal is to classify 'Benign' (i.e., 1, 
                    the negative value) and 'Malignant' (i.e., 0, the positive value) tumors.  These are represented 
                    by 'M' and 'B' in the dataset under the 'diagnosis' column.  If performing classification analysis 
                    on a different dataset, determine your dataset's classifiers and its respective column name.  For 
                    best practice, place the positive value first within the brackets.
                    """)
        st.markdown("# Convert Categorical to Numerical")
        st.markdown("""
                    Lets check that your classifiers have been replaced with the 0 and 1 values.      
                    """)
        if st.button('Code '):
            st.write("df.head()")
        st.text(df.head())
        if st.button('Explanation '):
            st.write("""
                    We can see that the diagnosis column has been replaced with 0's and 1's, meaning that our 
                    replace() function was successful.
                    """)
        st.markdown("# Convert Categorical to Numerical")
        st.markdown("""
                    Now, lets see the counts of your 0 and 1 classifiers.  It is a best practice to have
                    a good representation of both classifiers.  If there is large discrepancy between classifier counts percentage wise, 
                    perhaps look for a different dataset.      
                    """)
        if st.button(' Code'):
            st.write("df.groupby(['diagnosis']).diagnosis.count()")
        st.text(df.groupby(["diagnosis"]).diagnosis.count())
        if st.button(' Explanation'):
            st.write("""
                    We can see that there are 212 Benign (1) tumors and 357 Malignant (0) 
                    tumors in the breast cancer dataset.  This is an acceptable representation of 
                    both classifiers, thus we can proceed with this dataset. 
                    Remember, if you are performing classification analysis on a different dataset, 
                    insert your classifier column name into the Part 3 code above. 
                    """)
        st.markdown("# Convert Categorical to Numerical")
        st.markdown("""
                    Here, let us visualize the counts of your classifiers.  When working with data, it is always
                    best to visualize if you can!  Charts, graphs, maps, etc help the viewer to understand the data in
                    a more clear and efficient way.      
                    """)
        if st.button('Code  '):
            st.write("sns.countplot('diagnosis', data=df)")
            st.write("plt.show()")
        sns.countplot('diagnosis', data=df)
        st.set_option('deprecation.showPyplotGlobalUse', False)
        st.pyplot()
        if st.button('Explanation  '):
            st.write("""
                    We can see that there is a good representation of both classifiers, where 0=malignant
                    and 1=benign. 
                    Remember, if you are performing classification analysis on a different dataset, 
                    insert your classifier column name into the Part 4 code above. 
                    """)
        st.markdown(""
                    "")
        st.markdown("# Yay🎉")
        st.markdown(""" 
                    You have completed KNN, Replace Classifier!
                    You are ready to move on to Step 4
                    """)
    if KNN == 'Step 4: Create your Test and Training Sets':
        df.isnull().any()
        df = df.drop('Unnamed: 32', axis=1)
        df.diagnosis.replace(["M", "B"], [1, 0], inplace=True)
        st.markdown("# Create your Test and Training Set")
        st.markdown("""
                   Let's ensure that X only takes on the independent variables in the dataset.      
                    """)
        if st.button('Code'):
            st.write("X = df.drop(['id','diagnosis'],axis=1)")
        if st.button('Explanation'):
            st.write("""
                   Since the diagnosis column contains the classifiers, we want to remove this column from the X variable.  We 
                   won't neglect the diagnosis column for long, however, as we will utilize these values for the y variable!
                   We remove the id column as it is just an identifier, and does not play a role in tumor classification.  
                   If working with your own dataset, make sure you remove the irrelevant columns pertaining to your dataset!
                   """)
        X = df.drop(['id', 'diagnosis'], axis=1)
        st.markdown("# Create your Test and Training Set")
        st.markdown("""
                    Let's ensure that Y only takes on the classifier column in the dataset.      
                    """)
        if st.button('Code '):
            st.write("Y = df.iloc[:, 1]")
        if st.button('Explanation '):
            st.write("""
                    Since the diagnosis column contains the classifiers, we consider this the dependent variable column.  
                    If working with your own dataset, make sure you do some research on indexes so that you can accurately 
                    store your classifier column!
                    """)
        Y = df.iloc[:, 1]
        st.markdown("# Create your Test and Training Set")
        st.markdown("""
                    Let's splits the data into test and training sets using the X and Y variables we created previously.       
                    """)
        if st.button(' Code'):
            st.write(
                "xTrain, xTest, yTrain, yTest = sklearn.model_selection.train_test_split(X, Y, train_size=0.8, random_state =20221023)")
        xTrain, xTest, yTrain, yTest = sklearn.model_selection.train_test_split(X, Y, train_size=0.8,
                                                                                random_state=20221023)
        if st.button(' Explanation'):
            st.write("""
                   Part 3 will not return any output, but will instead assign data to variables that will return output later on 
                   in the classification process. 
                   You train the model using the training set, and test the model using the testing set. Training the model means to 
                   create the model, whereas testing the model means to test the accuracy of the model.  Training and testing sets
                   are created for both the X and Y variables.  
                   It is common to split the test and training sets by 20/80 percent, however, you should do some research on what is 
                   best for your model!  
                   The random_state number sets the seed so that you calculate the same accuracy each time.  Any number can be used, 
                   but many people prefer to use today's date.
                   """)
        st.markdown("# Create your Test and Training Set: next step")
        st.markdown("""
                    Creates a model that performs KNN Classification.      
                    """)
        if st.button('  Code'):
            st.write("KNNModel = KNeighborsClassifier(n_neighbors=5,leaf_size=1, weights='uniform')")
        KNNModel = KNeighborsClassifier(n_neighbors=5, leaf_size=1, weights='uniform')
        if st.button('  Explanation'):
            st.write("""
                   The n_neighbors parameter declares the number of neighbors.  5 is the default.  Do some research to find
                   the best option for your dataset.  
                   Uniform weights means that all points in each neighborhood are weighed equally.  Do some research to find the best
                   option for your dataset. 
                   """)
        st.markdown("# Create your Test and Training Set: Next step")
        st.markdown("""
                    Now, lets fits the KNN model to the training set.  This allows the model to "study"
                    the data and "learn" from it.  This step will not return any output, but will be used later on 
                    in the classification process.       
                    """)
        if st.button('Code   '):
            st.write("KNNModel.fit(xTrain,yTrain)")
        KNNModel.fit(xTrain, yTrain)
        st.markdown("# Create your Test and Training Set: next step")
        st.markdown("""
                    Now that we have created our model and trained it, Part 6 will use our testing dataset to test the model.  
                    This step will not return any output, but will be used later on in the classification process.
                    """)
        if st.button('Code    '):
            st.write("predKNN =KNNModel.predict(xTest)")
        predKNN = KNNModel.predict(xTest)
        st.markdown(""
                    "")
        st.markdown("# Yay 🎉")
        st.markdown(""" 
                    You have completed KNN, Test and Training Data!
                    You are ready to move on to Step 5
                    """)
    if KNN == 'Step 5: Run your model and check your Model Accuracy':
        df = df.drop('Unnamed: 32', axis=1)
        df.diagnosis.replace(["M", "B"], [1, 0], inplace=True)
        X = df.drop(['id', 'diagnosis'], axis=1)
        Y = df.iloc[:, 1]
        xTrain, xTest, yTrain, yTest = sklearn.model_selection.train_test_split(X, Y, train_size=0.8,
                                                                                random_state=20221023)
        KNNModel = KNeighborsClassifier(n_neighbors=5, leaf_size=1, weights='uniform')
        KNNModel.fit(xTrain, yTrain)
        predKNN = KNNModel.predict(xTest)
        st.markdown("# Run your model and check your Model Accuracy: next step")
        st.markdown("""
                     See the actual and predicted classifiers in array form.  Can you spot the 
                    differences between the actual and predicted set?      
                    """)
        if st.button('Code'):
            st.write("print(yTest.values)")
            st.write("print(predKNN)")
        st.text("Actual breast cancer : ")
        st.text(yTest.values)
        st.text("\nPredicted breast cancer : ")
        st.text(predKNN)
        st.markdown("#Run your model and check your Model Accuracy: next step")
        st.markdown("""
                    This code allows you to calculate the accuracy score.  As it may sound, the accuracy score declares how 
                    accurately the model classifies the classifiers.     
                    """)
        if st.button(' Code'):
            st.write("accuracy_score(yTest, predKNN) * 100")
        st.text("Accuracy Score: ")
        st.text(accuracy_score(yTest, predKNN) * 100)
        if st.button("Explanation"):
            st.write("""
                    Accuracy score is calculated by dividing the number of correct predictions by the total prediction number.
                    """)
        st.markdown("# Yay 🎉")
        st.markdown(""" 
                    You have completed KNN, Accuracy Score!
                    You are ready to move on to Step 6
                    """)
    if KNN == 'Step 6: Prediction Results':
        df = df.drop('Unnamed: 32', axis=1)
        df.diagnosis.replace(["M", "B"], [1, 0], inplace=True)
        X = df.drop(['id', 'diagnosis'], axis=1)
        Y = df.iloc[:, 1]
        xTrain, xTest, yTrain, yTest = sklearn.model_selection.train_test_split(X, Y, train_size=0.8,
                                                                                random_state=20221023)
        KNNModel = KNeighborsClassifier(n_neighbors=5, leaf_size=1, weights='uniform')
        KNNModel.fit(xTrain, yTrain)
        predKNN = KNNModel.predict(xTest)
        if st.button('Code'):
            st.write("""plt.figure(figsize=(16, 14))""")
            st.write("""plt.subplot(3, 3, 2)""")
            st.write("""sns.heatmap(sklearn.metrics.confusion_matrix(yTest, predKNN), annot=True).set(title= 'KNN')""")
            st.write("""plt.show()""")
        plt.figure(figsize=(16, 14))
        plt.subplot(3, 3, 2)
        sns.heatmap(sklearn.metrics.confusion_matrix(yTest, predKNN), annot=True).set(title='KNN')
        st.set_option('deprecation.showPyplotGlobalUse', False)
        st.pyplot()
        if st.button("Explanation"):
            st.write("""
                    The top left value represents the true positive classifications.  This means you predicted positive and it's 
                    true.  In the case of our dataset, you predicted that a tumor is malignant and it actually is. 
                    The top right value represents the false positive classifications.  This is also known as a Type I error.
                    This means you predicted positive and it's false. In the case of our dataset, you predicted that a tumor is malignant
                    but it is actually benign.
                    The bottom left value represents the false negative classifications.  This is also known as a Type II error. 
                    This means you predicted negative and it's false.  In the case of our dataset, you predicted
                    that a tumor is benign but it is actually malignant.  
                    The bottom right value represents the true negative classifications.This means you predicted negative
                    and it's true.  In the case of our dataset, you predicted that a tumor is benign and it is indeed benign.   
                    """)
        st.markdown("# Yay 🎉")
        st.markdown(""" 
                    You have completed KNN and learnt to see the Prediction Results
                    You are now a KNN classification pro and ready to classify your own dataset! Checkout the DIY option to do it on your own!
                    """)

if selected == "Random Forest":
    Random_Forest = st.selectbox('Random Forest: Select one option:',
                                   ['', 'Step 1: Check for Null Values', 'Step 2: Summarize Data',
                                    'Step 3: Convert Categorical to Numerical values',
                                    'Step 4: Create your Test and Training Sets',
                                    'Step 5: Run your model and check your Model Accuracy',
                                    'Step 6: Prediction Results'],
                                 format_func=lambda x: 'Select an option' if x == '' else x)
    if Random_Forest == 'Step 1: Check for Null Values':
        st.markdown("# Null Values Part 1")
        st.markdown("""
                            Part 1 is to determine if there are any null values in the data.
                            In this step, we print out all of the column names.  A true value represents 
                            a null value and will need to be located and deleted. 
                            """)
        if st.button('Code'):
            st.write("df.isnull().any()")
        st.text(df.isnull().any())
        if st.button('Explanation'):
            st.write("""
                            At the bottom of the column name list, we see that there
                            are 32 null values, signified by the 'true' statement.  All of the null
                            values are located in the 'Unnamed: 32 column and thus we will drop the entire column.
                            Do some research on your dataset to determine if you have an entire null column, or just a 
                            few null values within several of the columns.  If this is the case, don't drop the whole column
                            for your dataset.  Instead, just drop the nulls in each column.  This can be done with a df.dropna()
                            command. 
                            """)
        st.markdown("# Null Values Part 2")
        st.markdown("""
                            Part 2 is to delete the null values in our dataset.
                            """)
        if st.button('Code '):
            st.write("df.drop('Unnamed: 32', axis=1)")
        df = df.drop('Unnamed: 32', axis=1)
        if st.button('Explanation '):
            st.write("Once we perform the df.drop function on our identified null values, "
                     "we have removed all nulls from our dataset.  If you have a different dataset, change "
                     "the column name to reflect your dataset.  If you are not dropping an entire column but just several rows, "
                     "use the df.dropna() command.")
        st.markdown(""
                    "")
        st.markdown("# Yay 🎉!")
        st.markdown(""" 
                    You have completed Random Forest, Null Values!
                    You are ready to move on to the 'Summary of my Data' step!
                    """)
    if Random_Forest == 'Step 2: Summarize Data':
        df = df.drop('Unnamed: 32', axis=1)
        st.markdown("# Summarize Data")
        st.markdown("""
                    Summarizing data in this case means to generate descriptive statistics, which includes insight pertaining to
                    central tendency, dispersion, and the shape of a dataset's distribution.
                    """)
        if st.button('Code'):
            st.write("df.describe()")
        st.text(df.describe())
        st.markdown(""
                    "")
        st.markdown("# Yay 🎉!")
        st.markdown(""" 
                    You have completed Random Forest, Summarize Data!
                    You are ready to move on to the 'Replace Classifier With 0 and 1' step!
                    """)
    if Random_Forest == 'Step 3: Convert Categorical to Numerical values':
        df = df.drop('Unnamed: 32', axis=1)
        st.markdown("# Replace Classifier Part 1")
        st.markdown("""
                    Binary classification uses 0's and 1's to perform the classification analysis.  
                    0 represents the positive value and 1 represents the negative value.  A majority of datasets
                    are going to contain the qualitative classifier instead of the 0 and 1 values.  As such, when
                    performing classification, it is necessary to write a code that will replace the classifiers
                    in the dataset.    
                    """)
        if st.button('Code'):
            st.write("df.diagnosis.replace(['M', 'B'], [1, 0], inplace = True)")
        df.diagnosis.replace(["M", "B"], [1, 0], inplace=True)
        if st.button('Explanation'):
            st.write("""
                    For the breast cancer dataset used in this tutorial, the goal is to classify 'Benign' (i.e., 1, 
                    the negative value) and 'Malignant' (i.e., 0, the positive value) tumors.  These are represented 
                    by 'M' and 'B' in the dataset under the 'diagnosis' column.  If performing classification analysis 
                    on a different dataset, determine your dataset's classifiers and its respective column name.  For 
                    best practice, place the positive value first within the brackets.
                    """)
        st.markdown("# Convert Categorical to Numerical values")
        st.markdown("""
                    Part 2 is for you to check that your classifiers have been replaced with the 0 and 1 values.      
                    """)
        if st.button('Code '):
            st.write("df.head()")
        st.text(df.head())
        if st.button('Explanation '):
            st.write("""
                    We can see that the diagnosis column has been replaced with 0's and 1's, meaning that our 
                    replace() function was successful.
                    """)
        st.markdown("# Convert Categorical to Numerical values")
        st.markdown("""
                    Part 3 is for you to see the counts of your 0 and 1 classifiers.  It is a best practice to have
                    a good representation of both classifiers.  If there is large discrepancy between classifier counts percentage wise, 
                    perhaps look for a different dataset.      
                    """)
        if st.button(' Code'):
            st.write("df.groupby(['diagnosis']).diagnosis.count()")
        st.text(df.groupby(["diagnosis"]).diagnosis.count())
        if st.button(' Explanation'):
            st.write("""
                    We can see that there are 212 Benign (1) tumors and 357 Malignant (0) 
                    tumors in the breast cancer dataset.  This is an acceptable representation of 
                    both classifiers, thus we can proceed with this dataset. 
                    Remember, if you are performing classification analysis on a different dataset, 
                    insert your classifier column name into the Part 3 code above. 
                    """)
        st.markdown("# Convert Categorical to Numerical values")
        st.markdown("""
                    Part 4 is for you visualize the counts of your classifiers.  When working with data, it is always
                    best to visualize if you can!  Charts, graphs, maps, etc help the viewer to understand the data in
                    a more clear and efficient way.      
                    """)
        if st.button('Code  '):
            st.write("sns.countplot('diagnosis', data=df)")
            st.write("plt.show()")
        sns.countplot('diagnosis', data=df)
        st.set_option('deprecation.showPyplotGlobalUse', False)
        st.pyplot()
        if st.button('Explanation  '):
            st.write("""
                    We can see that there is a good representation of both classifiers, where 0=malignant
                    and 1=benign. 
                    Remember, if you are performing classification analysis on a different dataset, 
                    insert your classifier column name into the code above. 
                    """)
        st.markdown(""
                    "")
        st.markdown("# Yay 🎉!")
        st.markdown(""" 
                    You have completed Random Forest, Replace Classifier!
                    You are ready to move on to the 'Test and Training Data' step!
                    """)
    if Random_Forest == 'Step 4: Create your Test and Training Sets':
        df.isnull().any()
        df = df.drop('Unnamed: 32', axis=1)
        df.diagnosis.replace(["M", "B"], [1, 0], inplace=True)
        st.markdown("# Create your Test and Training Sets")
        st.markdown("""
                    Part 1 ensures that X only takes on the independent variables in the dataset.      
                    """)
        if st.button('Code'):
            st.write("X = df.drop(['id','diagnosis'],axis=1)")
        if st.button('Explanation'):
            st.write("""
                    Since the diagnosis column contains the classifiers, we want to remove this column from the X variable.  We 
                    won't neglect the diagnosis column for long, however, as we will utilize these values for the y variable!
                    We remove the id column as it is just an identifier, and does not play a role in tumor classification.  
                    If working with your own dataset, make sure you remove the irrelevant columns pertaining to your dataset!
                    """)
        X = df.drop(['id', 'diagnosis'], axis=1)
        st.markdown("# Create your Test and Training Sets")
        st.markdown("""
                    Part 2 ensures that Y only takes on the classifier column in the dataset.      
                    """)
        if st.button('Code '):
            st.write("Y = df.iloc[:, 1]")
        if st.button('Explanation '):
            st.write("""
                    Since the diagnosis column contains the classifiers, we consider this the dependent variable column.  
                    If working with your own dataset, make sure you do some research on indexes so that you can accurately 
                    store your classifier column!
                    """)
        Y = df.iloc[:, 1]
        st.markdown("# Create your Test and Training Sets")
        st.markdown("""
                    Code splits the data into test and training sets using the X and Y variables we created previously.       
                    """)
        if st.button(' Code'):
            st.write(
                "xTrain, xTest, yTrain, yTest = sklearn.model_selection.train_test_split(X, Y, train_size=0.8, random_state =20221023)")
        xTrain, xTest, yTrain, yTest = sklearn.model_selection.train_test_split(X, Y, train_size=0.8,
                                                                                random_state=20221023)
        if st.button(' Explanation'):
            st.write("""
                   This will not return any output, but will instead assign data to variables that will return output later on 
                    in the classification process. 
                    You train the model using the training set, and test the model using the testing set. Training the model means to 
                    create the model, whereas testing the model means to test the accuracy of the model.  Training and testing sets
                    are created for both the X and Y variables.  
                    It is common to split the test and training sets by 20/80 percent, however, you should do some research on what is 
                    best for your model!  
                    The random_state number sets the seed so that you calculate the same accuracy each time.  Any number can be used, 
                    but many people prefer to use today's date.
                    """)
        st.markdown("# Create your Test and Training Sets")
        st.markdown("""
                    This creates a model that performs Random Forest Classification.      
                    """)
        if st.button('Code   '):
            st.write("randomFModel = RandomForestClassifier()")
        randomFModel = RandomForestClassifier()
        st.markdown("# Create your Test and Training Sets")
        st.markdown("""
                    This fits the Random Forest model to the training set.  This allows the model to "study"
                    the data and "learn" from it.  This step will not return any output, but will be used later on 
                    in the classification process.       
                    """)
        if st.button('  Code '):
            st.write("randomFModel.fit(xTrain,yTrain)")
        randomFModel.fit(xTrain, yTrain)
        st.markdown("# Create your Test and Training Sets")
        st.markdown("""
                    Now that we have created our model and trained it, Part 6 will use our testing dataset to test the model.  
                    This step will not return any output, but will be used later on in the classification process.
                    """)
        if st.button('Code     '):
            st.write("predRandomF =randomFModel.predict(xTest)")
        predRandomF = randomFModel.predict(xTest)
        st.markdown(""
                    "")
        st.markdown("# Yay 🎉!")
        st.markdown(""" 
                    You have completed Random Forest, Test and Training Data!
                    You are ready to move on to step 5
                    """)
    if Random_Forest == 'Step 5: Run your model and check your Model Accuracy':
        df = df.drop('Unnamed: 32', axis=1)
        df.diagnosis.replace(["M", "B"], [1, 0], inplace=True)
        X = df.drop(['id', 'diagnosis'], axis=1)
        Y = df.iloc[:, 1]
        xTrain, xTest, yTrain, yTest = sklearn.model_selection.train_test_split(X, Y, train_size=0.8,
                                                                                random_state=20221023)
        randomFModel = RandomForestClassifier()
        randomFModel.fit(xTrain, yTrain)
        randomFModel.score(xTrain, yTrain)
        predRandomF = randomFModel.predict(xTest)
        st.markdown("# Run your model and check your Model Accuracy")
        st.markdown("""
                    This allows you to see the actual and predicted classifiers in array form.  Can you spot the 
                    differences between the actual and predicted set?      
                    """)
        if st.button('Code'):
            st.write("print(yTest.values)")
            st.write("print(predRandomF)")
        st.text("Actual breast cancer : ")
        st.text(yTest.values)
        st.text("\nPredicted breast cancer : ")
        st.text(predRandomF)
        st.markdown("# Accuracy Score Part 2")
        st.markdown("""
                    This allows you to calculate the accuracy score.  As it may sound, the accuracy score declares how 
                    accurately the model classifies the classifiers.     
                    """)
        if st.button('Code '):
            st.write("accuracy_score(yTest, predRandomF) * 100")
        st.text("Accuracy Score: ")
        st.text(accuracy_score(yTest, predRandomF) * 100)
        if st.button("Explanation  "):
            st.write("""
                    Accuracy score is calculated by dividing the number of correct predictions by the total prediction number.
                    """)
        st.markdown("# Yay 🎉!")
        st.markdown(""" 
                    You have completed Random Forest, Accuracy Score!
                    You are ready to move on to the 'Confusion Matrix' step!
                    """)
    if Random_Forest == 'Step 6: Prediction Results':
        df = df.drop('Unnamed: 32', axis=1)
        df.diagnosis.replace(["M", "B"], [1, 0], inplace=True)
        X = df.drop(['id', 'diagnosis'], axis=1)
        Y = df.iloc[:, 1]
        xTrain, xTest, yTrain, yTest = sklearn.model_selection.train_test_split(X, Y, train_size=0.8,
                                                                                random_state=20221023)
        randomFModel = RandomForestClassifier()
        randomFModel.fit(xTrain, yTrain)
        predRandomF = randomFModel.predict(xTest)
        if st.button('Code'):
            st.write("""plt.figure(figsize=(16, 14))""")
            st.write("""plt.subplot(3, 3, 3)""")
            st.write(
                """sns.heatmap(sklearn.metrics.confusion_matrix(yTest, predRandomF), annot=True).set(title='Random Forest')""")
            st.write("""plt.show()""")
        plt.figure(figsize=(16, 14))
        plt.subplot(3, 3, 3)
        sns.heatmap(sklearn.metrics.confusion_matrix(yTest, predRandomF), annot=True).set(title='Random Forest')
        st.set_option('deprecation.showPyplotGlobalUse', False)
        st.pyplot()
        if st.button("Explanation  "):
            st.write("""
                    The top left value represents the true positive classifications.  This means you predicted positive and it's 
                    true.  In the case of our dataset, you predicted that a tumor is malignant and it actually is. 
                    The top right value represents the false positive classifications.  This is also known as a Type I error.
                    This means you predicted positive and it's false. In the case of our dataset, you predicted that a tumor is malignant
                    but it is actually benign.
                    The bottom left value represents the false negative classifications.  This is also known as a Type II error. 
                    This means you predicted negative and it's false.  In the case of our dataset, you predicted
                    that a tumor is benign but it is actually malignant.  
                    The bottom right value represents the true negative classifications.This means you predicted negative
                    and it's true.  In the case of our dataset, you predicted that a tumor is benign and it is indeed benign.   
                    """)
        st.markdown("# Yay 🎉!")
        st.markdown(""" 
                    You have completed Random Forest, Confusion Matrix!
                    You are now a Random Forest classification pro and ready to classify your own dataset!
                    """)

if selected == "Naive Bayes":
    Naive_Bayes = st.selectbox('Naive Bayes: Select one option:',
                                   ['', 'Step 1: Check for Null Values', 'Step 2: Summarize Data',
                                    'Step 3: Convert Categorical to Numerical values',
                                    'Step 4: Create your Test and Training Sets',
                                    'Step 5: Run your model and check your Model Accuracy',
                                    'Step 6: Prediction Results'],
                               format_func=lambda x: 'Select an option' if x == '' else x)
    if Naive_Bayes == 'Step 1: Check for Null Values':
        st.markdown("# Null Values Part 1")
        st.markdown("""
                            Part 1 is to determine if there are any null values in the data.
                            In this step, we print out all of the column names.  A true value represents 
                            a null value and will need to be located and deleted. 
                            """)
        if st.button('Code'):
            st.write("df.isnull().any()")
        st.text(df.isnull().any())
        if st.button('Explanation'):
            st.write("""
                            At the bottom of the column name list, we see that there
                            are 32 null values, signified by the 'true' statement.  All of the null
                            values are located in the 'Unnamed: 32 column and thus we will drop the entire column.
                            Do some research on your dataset to determine if you have an entire null column, or just a 
                            few null values within several of the columns.  If this is the case, don't drop the whole column
                            for your dataset.  Instead, just drop the nulls in each column.  This can be done with a df.dropna()
                            command. 
                            """)
        st.markdown("# Null Values Part 2")
        st.markdown("""
                            Part 2 is to delete the null values in our dataset.
                            """)
        if st.button('Code '):
            st.write("df.drop('Unnamed: 32', axis=1)")
        df = df.drop('Unnamed: 32', axis=1)
        if st.button('Explanation '):
            st.write("Once we perform the df.drop function on our identified null values, "
                     "we have removed all nulls from our dataset.  If you have a different dataset, change "
                     "the column name to reflect your dataset.  If you are not dropping an entire column but just several rows, "
                     "use the df.dropna() command.")
        st.markdown(""
                    "")
        st.markdown("# Yay 🎉!")
        st.markdown(""" 
                    You have completed Naive Bayes, Null Values!
                    You are ready to move on to the 'Summary of my Data' step!
                    """)
    if Naive_Bayes == 'Step 2: Summarize Data':
        df = df.drop('Unnamed: 32', axis=1)
        st.markdown("# Summarize Data")
        st.markdown("""
                    Summarizing data in this case means to generate descriptive statistics, which includes insight pertaining to
                    central tendency, dispersion, and the shape of a dataset's distribution.
                    """)
        if st.button('Code'):
            st.write("df.describe()")
        st.text(df.describe())
        st.markdown(""
                    "")
        st.markdown("# Yay 🎉!")
        st.markdown(""" 
                    You have completed Naive Bayes, Summarize Data!
                    You are ready to move on to the 'Replace Classifier With 0 and 1' step!
                    """)
    if Naive_Bayes == 'Step 3: Convert Categorical to Numerical values':
        df = df.drop('Unnamed: 32', axis=1)
        st.markdown("# Convert Categorical to Numerical values")
        st.markdown("""
                    Binary classification uses 0's and 1's to perform the classification analysis.  
                    0 represents the positive value and 1 represents the negative value.  A majority of datasets
                    are going to contain the qualitative classifier instead of the 0 and 1 values.  As such, when
                    performing classification, it is necessary to write a code that will replace the classifiers
                    in the dataset.    
                    """)
        if st.button('Code'):
            st.write("df.diagnosis.replace(['M', 'B'], [1, 0], inplace = True)")
        df.diagnosis.replace(["M", "B"], [1, 0], inplace=True)
        if st.button('Explanation'):
            st.write("""
                    For the breast cancer dataset used in this tutorial, the goal is to classify 'Benign' (i.e., 1, 
                    the negative value) and 'Malignant' (i.e., 0, the positive value) tumors.  These are represented 
                    by 'M' and 'B' in the dataset under the 'diagnosis' column.  If performing classification analysis 
                    on a different dataset, determine your dataset's classifiers and its respective column name.  For 
                    best practice, place the positive value first within the brackets.
                    """)
        st.markdown("# Convert Categorical to Numerical values")
        st.markdown("""
                    This is for you to check that your classifiers have been replaced with the 0 and 1 values.      
                    """)
        if st.button('Code '):
            st.write("df.head()")
        st.text(df.head())
        if st.button('Explanation '):
            st.write("""
                    We can see that the diagnosis column has been replaced with 0's and 1's, meaning that our 
                    replace() function was successful.
                    """)
        st.markdown("# Convert Categorical to Numerical values")
        st.markdown("""
                    This is for you to see the counts of your 0 and 1 classifiers.  It is a best practice to have
                    a good representation of both classifiers.  If there is large discrepancy between classifier counts percentage wise, 
                    perhaps look for a different dataset.      
                    """)
        if st.button(' Code'):
            st.write("df.groupby(['diagnosis']).diagnosis.count()")
        st.text(df.groupby(["diagnosis"]).diagnosis.count())
        if st.button(' Explanation'):
            st.write("""
                    We can see that there are 212 Benign (1) tumors and 357 Malignant (0) 
                    tumors in the breast cancer dataset.  This is an acceptable representation of 
                    both classifiers, thus we can proceed with this dataset. 
                    Remember, if you are performing classification analysis on a different dataset, 
                    insert your classifier column name into the Part 3 code above. 
                    """)
        st.markdown("# Convert Categorical to Numerical values")
        st.markdown("""
                    This is for you visualize the counts of your classifiers.  When working with data, it is always
                    best to visualize if you can!  Charts, graphs, maps, etc help the viewer to understand the data in
                    a more clear and efficient way.      
                    """)
        if st.button('Code  '):
            st.write("sns.countplot('diagnosis', data=df)")
            st.write("plt.show()")
        sns.countplot('diagnosis', data=df)
        st.set_option('deprecation.showPyplotGlobalUse', False)
        st.pyplot()
        if st.button('Explanation  '):
            st.write("""
                    We can see that there is a good representation of both classifiers, where 0=malignant
                    and 1=benign. 
                    Remember, if you are performing classification analysis on a different dataset, 
                    insert your classifier column name into the Part 4 code above. 
                    """)
        st.markdown(""
                    "")
        st.markdown("# Yay 🎉!")
        st.markdown(""" 
                    You have completed Naive Bayes, Replace Classifier!
                    You are ready to move on to the 'Test and Training Data' step!
                    """)
    if Naive_Bayes == 'Step 4: Create your Test and Training Sets':
        df.isnull().any()
        df = df.drop('Unnamed: 32', axis=1)
        df.diagnosis.replace(["M", "B"], [1, 0], inplace=True)
        st.markdown("# Create your Test and Training Sets")
        st.markdown("""
                    This ensures that X only takes on the independent variables in the dataset.      
                    """)
        if st.button('Code'):
            st.write("X = df.drop(['id','diagnosis'],axis=1)")
        if st.button('Explanation'):
            st.write("""
                    Since the diagnosis column contains the classifiers, we want to remove this column from the X variable.  We 
                    won't neglect the diagnosis column for long, however, as we will utilize these values for the y variable!
                    We remove the id column as it is just an identifier, and does not play a role in tumor classification.  
                    If working with your own dataset, make sure you remove the irrelevant columns pertaining to your dataset!
                    """)
        X = df.drop(['id', 'diagnosis'], axis=1)
        st.markdown("# Create your Test and Training Sets")
        st.markdown("""
                    This ensures that Y only takes on the classifier column in the dataset.      
                    """)
        if st.button('Code '):
            st.write("Y = df.iloc[:, 1]")
        if st.button('Explanation '):
            st.write("""
                    Since the diagnosis column contains the classifiers, we consider this the dependent variable column.  
                    If working with your own dataset, make sure you do some research on indexes so that you can accurately 
                    store your classifier column!
                    """)
        Y = df.iloc[:, 1]
        st.markdown("# Create your Test and Training Sets")
        st.markdown("""
                    This splits the data into test and training sets using the X and Y variables we created previously.       
                    """)
        if st.button(' Code'):
            st.write(
                "xTrain, xTest, yTrain, yTest = sklearn.model_selection.train_test_split(X, Y, train_size=0.8, random_state =20221023)")
        xTrain, xTest, yTrain, yTest = sklearn.model_selection.train_test_split(X, Y, train_size=0.8,
                                                                                random_state=20221023)
        if st.button(' Explanation'):
            st.write("""
                    This will not return any output, but will instead assign data to variables that will return output later on 
                    in the classification process. 
                    You train the model using the training set, and test the model using the testing set. Training the model means to 
                    create the model, whereas testing the model means to test the accuracy of the model.  Training and testing sets
                    are created for both the X and Y variables.  
                    It is common to split the test and training sets by 20/80 percent, however, you should do some research on what is 
                    best for your model!  
                    The random_state number sets the seed so that you calculate the same accuracy each time.  Any number can be used, 
                    but many people prefer to use today's date.
                    """)
        st.markdown("# Create your Test and Training Sets")
        st.markdown("""
                    This creates a model that performs Naive Bayes Classification.      
                    """)
        if st.button(' Code '):
            st.write("NB = GaussianNB()")
        NB = GaussianNB()
        st.markdown("# Test and Training Data Part 5")
        st.markdown("""
                    Part 5 fits the Naive Bayes model to the training set.  This allows the model to "study"
                    the data and "learn" from it.  This step will not return any output, but will be used later on 
                    in the classification process.       
                    """)
        if st.button('  Code'):
            st.write("NB.fit(xTrain,yTrain)")
        NB.fit(xTrain, yTrain)
        st.markdown("# Create your Test and Training Sets")
        st.markdown("""
                    Now that we have created our model and trained it, Part 6 will use our testing dataset to test the model.  
                    This step will not return any output, but will be used later on in the classification process.
                    """)
        if st.button('Code'):
            st.write("predNB =NB.predict(xTest)")
        predNB = NB.predict(xTest)
        st.markdown(""
                    "")
        st.markdown("# Yay 🎉!")
        st.markdown(""" 
                    You have completed Naive Bayes, Test and Training Data!
                    You are ready to move on to the next step
                    """)
    if Naive_Bayes == 'Step 5: Run your model and check your Model Accuracy':
        df = df.drop('Unnamed: 32', axis=1)
        df.diagnosis.replace(["M", "B"], [1, 0], inplace=True)
        X = df.drop(['id', 'diagnosis'], axis=1)
        Y = df.iloc[:, 1]
        xTrain, xTest, yTrain, yTest = sklearn.model_selection.train_test_split(X, Y, train_size=0.8,
                                                                                random_state=20221023)
        NB = GaussianNB()
        NB.fit(xTrain, yTrain)
        predNB = NB.predict(xTest)
        st.markdown("# Run your model and check your Model Accuracy")
        st.markdown("""
                    this allows you to see the actual and predicted classifiers in array form.  Can you spot the 
                    differences between the actual and predicted set?      
                    """)
        if st.button('Code'):
            st.write("print(yTest.values)")
            st.write("print(predNB)")
        st.text("Actual breast cancer : ")
        st.text(yTest.values)
        st.text("\nPredicted breast cancer : ")
        st.text(predNB)
        st.markdown("#Run your model and check your Model Accuracy")
        st.markdown("""
                    this allows you to calculate the accuracy score.  As it may sound, the accuracy score declares how 
                    accurately the model classifies the classifiers.     
                    """)
        if st.button('Code    '):
            st.write("accuracy_score(yTest, predNB) * 100")
        st.text("Accuracy Score: ")
        st.text(accuracy_score(yTest, predNB) * 100)
        if st.button("Explanation "):
            st.write("""
                    Accuracy score is calculated by dividing the number of correct predictions by the total prediction number.
                    """)
        st.markdown("# Yay 🎉 ")
        st.markdown(""" 
                    You have completed Naive Bayes, Accuracy Score 
                    You are ready to move on to the 'Confusion Matrix' step!
                    """)
    if Naive_Bayes == 'Step 6: Prediction Results':
        df = df.drop('Unnamed: 32', axis=1)
        df.diagnosis.replace(["M", "B"], [1, 0], inplace=True)
        X = df.drop(['id', 'diagnosis'], axis=1)
        Y = df.iloc[:, 1]
        xTrain, xTest, yTrain, yTest = sklearn.model_selection.train_test_split(X, Y, train_size=0.8,
                                                                                random_state=20221023)
        NB = GaussianNB()
        NB.fit(xTrain, yTrain)
        predNB = NB.predict(xTest)
        if st.button('Code'):
            st.write("""
                    plt.figure(figsize=(16, 14))
                    """)
            st.write("""
                    plt.subplot(3, 3, 3)
                    """)
            st.write("""
                    sns.heatmap(sklearn.metrics.confusion_matrix(yTest, predNB), annot=True).set(title='Naive Bayes')
                    """)
            st.write("""
                    plt.show()
                    """)
        plt.figure(figsize=(16, 14))
        plt.subplot(3, 3, 3)
        sns.heatmap(sklearn.metrics.confusion_matrix(yTest, predNB), annot=True).set(title='Naive Bayes')
        st.set_option('deprecation.showPyplotGlobalUse', False)
        st.pyplot()
        if st.button("Explanation"):
            st.write("""
                    The top left value represents the true positive classifications.  This means you predicted positive and it's 
                    true.  In the case of our dataset, you predicted that a tumor is malignant and it actually is. 
                    The top right value represents the false positive classifications.  This is also known as a Type I error.
                    This means you predicted positive and it's false. In the case of our dataset, you predicted that a tumor is malignant
                    but it is actually benign.
                    The bottom left value represents the false negative classifications.  This is also known as a Type II error. 
                    This means you predicted negative and it's false.  In the case of our dataset, you predicted
                    that a tumor is benign but it is actually malignant.  
                    The bottom right value represents the true negative classifications.This means you predicted negative
                    and it's true.  In the case of our dataset, you predicted that a tumor is benign and it is indeed benign.   
                    """)
        st.markdown("# Yay 🎉!")
        st.markdown(""" 
                    You have completed Naive Bayes, Confusion Matrix!
                    You are now a Naive Bayes classification pro and ready to classify your own dataset!
                    """)

if selected == "About":
    st.markdown("# Meet the Classification Central Developers!")
    st.markdown("""
                """)
    st.markdown("""
                Classification Central was created as part of a Comp-Sci 5530 project for UMKC and Hack-a-Roo.  The goal of the project was to create
                a user interface for beginner data scientists learning how to classify data.  By providing the code, the output of the code, and explanations of each 
                of the steps and its corresponding output within the 4 different classification techniques, we hope we have improved the ability of beginner data scientists 
                to understand classification.  As a result of this project, the developers also had the opportunity to learn a new tool: Streamlit, as well as refresh their memory 
                in popular classification techniques.  It is always helpful in industry to be able to have clear knowledge of data science concepts, and as a result of this project, the
                Classification Central developers can simply and accurately explain the basics behind logistic regression, KNN, Random Forest, and Naive Bayes.  
                """)
    st.markdown("# Ally Ryan")
    st.markdown("## UMKC Data Science Graduate Student")
    st.markdown("""
        I am a first year UMKC Data Science Graduate Student.  I have a passion for data science in the 
        sports industry.  In general, I love to learn new things and figure things out.  As such, I often 
        find the field of data science to be like a sometimes frustrating, but always rewarding puzzle. 
        I hope to pursue a career as a biomechanical data scientist that designs running shoes, and/or assist 
        in predictive analytics for high-profile races or training analysis.  Currently, I work as a data privacy and 
        cybersecurity consultant.  Outside of school and work, I am an avid runner and am a middle 
        distance track runner for UMKC. 
        """)
    st.markdown("# Anil Kochera")
    st.markdown("## Computer Science Grad Student ")
    st.markdown("""
            Hello! My name is Anil Kochera and I'm pursuing Masters in CS. I love playing with data and always 
            learning about new things happening in the Data Science field. 
            """)
    st.markdown("# Nabila Hashim")
    st.markdown("## Computer Science Grad Student ")
    st.markdown("""
                I am a graduate student majoring in computer science with emphasis on data science. I'm enthusiastic 
                about data science, especially considering how quickly technology is changing the profession. I enjoy being 
                a part of new technologies and trying out innovative solutions. I'm especially interested in artificial
                intelligence after learning about machine learning in my studies, and I'm eager to use AI in a more practical capacity.
                """)
    st.markdown("# Danny Rider")
    st.markdown("## Computer Science Grad Student")
    st.markdown("""
                Hey! My name is Daniel Rider and I'm pursuing a Masters In CS.  I did my undergraduate in 
                Physics and I'm excited to learn about everything computer science, especially data 
                science and machine learning.  I believe data science is a very innovative field, 
                and I've been amazed at the rapid pace the field has been developing at.  Whether its facial 
                recognition or forecasting sales, Data science is quickly becoming a necessity, and I'm 
                excited to learn more! 
                """)
if selected == "Home":
    st.markdown("""
        Too often, data scientists provide their code but don't provide explanations of what they are doing or what their results say. 
        Classification Central seeks to provide clear explanations of 4 common classification techniques: Logistic Regression, K Nearest Neighbors (KNN),
        Random Forest, and Naive Bayes.  We provide a tutorial using a breast cancer dataset, and provide the code in hopes that you can perform
        classification on your own data set by making slight tweaks.  We also provide detailed explanations of what the code is doing, in hopes that
        you can gain a better understanding of each classification step.
        In each of the classification steps, you will:
        1) Check for and Remove Null Values
        2) Summarize Your Data
        3) Replace the Classifier with 0 and 1
        4) Implement Test and Training Sets
        5) Determine the Accuracy Score
        6) Output a Confusion Matrix
        By going through each of these steps, you will see that it is quite simple to classify your data in many different ways.  Our tutorial is 
        sufficient for any dataset with a binary classifier (i.e., a dataset with two class labels.  This may include diseased vs not diseased, spam vs 
        not spam, etc.).  After this tutorial, you will leave Classification Central a classification pro, and be able to impress people in industry with your ability to classify
        data via 4 different techniques.  
        We hope you enjoy Classification Central, and that we have helped make your data science learning fun and easy!
        """)

    st.markdown("# DANA- Interactive Visualization")
    st.markdown("""
                DANA is Classification Central's Bonus Section that allows for you to upload your own CSV file.  Once
                your file is uploaded, you can remove nulls from your dataset, summarize your data, and perform many 
                different visualizations of your data for optimal understanding. 
                DANA is fully interactive, meaning that you can visualize based on any column name in your dataset.
                """)

    st.markdown("# The following are summaries of the classification techniques covered by Classification Central:")
    st.markdown("""
                """)
    st.markdown("# Logistic Regression")
    st.markdown("""
                Whereas linear regression is often used to predict continuous-valued outcomes (such as the weight of a person or the amount of rainfall),
                logistic regression is used to predict binomial (Y=0 or 1) categorical outcomes.  In logistic regression, there are one or more independent variables
                that determine an outcome.  The statistical methods used to perform the logistic regression are meant to identify the model that best fits the relationship 
                between the dependent and independent variable.  
                """)
    st.markdown("""
                """)
    st.markdown("# KNN")
    st.markdown("""
                The k-nearest neighbors algorithm is a supervised machine learning algorithm that can be used to solve classification problems.  Within the
                algorithm, KNN assumes that similar things are in close proximity of each other.  To determine the similarity between data points, KNN 
                uses math to calculate the distance between the data points.  Python can perform KNN with a few simple lines of code.  However, behind the scenes,
                the are calculations that determine the appropriate number of neighbors, perform calculations pertaining to distances, and for classification, return
                the mode of K labels that result from sorting the distances within each neighbor in ascending order. Whew! Don't worry, our tutorial will make KNN seem
                like a breeze!
                """)
    st.markdown("""
                """)
    st.markdown("# Random Forest")
    st.markdown("""
                A random forest is a classification technique that uses many decision trees.  Therefore, the many decision trees create a forest!
                A decision tree is a collection of binary nodes that split the observations in a way that best separates and distinguishes the two 
                classifiers. In the random forest algorithm, each tree outputs a class prediction and the class with the most votes becomes the model's 
                prediction.    
                """)
    st.markdown("""
                """)
    st.markdown("# Naive Bayes")
    st.markdown("""
                The Naive Bayes classifier is based on the Bayes theorem, which allows us to find the probability of A happening given that B has occured.
                When the presence of on independent variable does not affect another, the classifier is called naive.  For example, if we are looking at weather, 
                we cannot necessarily conclude that high temperatures also mean high humidity.  Statistically when using Naive Bayes, the goal is to find the probability
                of x belonging to some class C.    
                """)

if selected == "Visualizations":
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
                              ['', 'Data Report', 'Summary of my Data', 'Check for Null Values in the data'],
                              format_func=lambda x: 'Select an option' if x == '' else x)

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
                                      ['', 'Remove columns with Nulls values', 'Replace the null with constant values',
                                       'Replace the null with statistics'],
                                      format_func=lambda x: 'Select an option' if x == '' else x)
        if cleanDF_option == 'Remove columns with Nulls values':
            # Removing the columns with null values but only those columns that have null value percentage greater than 30% in the column.
            nullDataPercentage[nullDataPercentage > .3]
            CleanDF_less_missing_columns = dataframe.loc[:,
                                           nullDataPercentage <= .3].copy()  # equivalent to df.drop(columns=pct_missing[pct_missing > .3].index)
            # CleanDF_less_missing_columns.shape
            st.write("The updated DataSet is")
            st.write(CleanDF_less_missing_columns.head())
        if cleanDF_option == 'Replace the null with constant values':
            CleanDF_replace_constant = dataframe.copy()
            # numeric_cols = dataframe.select_dtypes(include=['number']).columns
            # non_numeric_cols = dataframe.select_dtypes(exclude=['number']).columns
            CleanDF_replace_constant[numeric_cols] = CleanDF_replace_constant[numeric_cols].fillna(0)
            CleanDF_replace_constant[non_numeric_cols] = CleanDF_replace_constant[non_numeric_cols].fillna('NA')
            CleanDF_constant = CleanDF_replace_constant.head()

            # st.write(CleanDF_constant)
        if cleanDF_option == 'Replace the null with statistics':
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
        if (cleanDF_option == 'Replace the null with constant values'):
            Clean_DF = CleanDF_replace_constant.copy()
        if (cleanDF_option == 'Replace the null with statistics'):
            Clean_DF = CleanDF_replace_statistics.copy()
        if ((cleanDF_option == 'Replace the null with statistics') and (cleanDF_option_replace == 'Mean')):
            Clean_DF = CleanDF_replace_statistics.copy()
        if ((cleanDF_option == 'Replace the null with statistics') and (cleanDF_option_replace == 'Median')):
            Clean_DF = CleanDF_replace_statistics.copy()
        if ((cleanDF_option == 'Replace the null with statistics') and (cleanDF_option_replace == 'Mode')):
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
                st.write('There are no numerical columns in the data.')
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
                st.write('There are no categorical columns in the data.')
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
                st.write('There are no numerical columns in the data.')
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
                st.write('There are no categorical columns with normal cardinality in the data.')
            else:

                st.subheader('Variance of target variable with categorical columns')
                model_type = st.radio('Select Problem Type:', ('Regression', 'Classification'), key='model_type')
                selected_cat_cols = functions.multiselect_container('Choose columns for Category Colored plots:',
                                                                    normal_cardi_columns, 'Category')

                if 'Target Analysis' not in vizuals:
                    target_column = st.selectbox("Select target column:", Clean_DF.columns,
                                                 index=len(Clean_DF.columns) - 1)

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
                        st.subheader(
                            'The following column has high cardinality, which is why its boxplot was not plotted:')
                    else:
                        st.subheader(
                            'The following columns have high cardinality, which is why its boxplot was not plotted:')
                    for i in high_cardi_columns:
                        st.write(i)

                    st.write('<p style="font-size:140%">Do you want to plot anyway?</p>', unsafe_allow_html=True)
                    answer = st.selectbox("", ('No', 'Yes'))

                    if answer == 'Yes':
                        for i in high_cardi_columns:
                            fig = px.box(df_1, y=target_column, color=i)
                            st.plotly_chart(fig, use_container_width=True)