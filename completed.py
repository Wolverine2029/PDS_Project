import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import preprocessing
import sklearn.model_selection, sklearn.linear_model, sklearn.svm, sklearn.metrics
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from  sklearn.model_selection import GridSearchCV
import streamlit as st
import pandas as pd
from streamlit_option_menu import option_menu

#ain Page
EXAMPLE_NO = 3


def streamlit_menu(example=3):
    if example == 3:
        selected = option_menu(
            menu_title=None,  # required
            options=["Home", "Logistic Regression", "KNN", "Random Forest", "Naive Bayes", "About"],
            icons=["house", "hourglass-split", "hourglass-split", "hourglass-split", "hourglass-split", "people-fill"],
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
    st.title("Welcome to Classification Central")
if selected == "Logistic Regression":
    st.title(f"{selected}")
if selected == "KNN":
    st.title(f"{selected}")
if selected == "Random Forest":
    st.title(f"{selected}")
if selected == "Naive Bayes":
    st.title(f"{selected}")
if selected == "About":
    st.title(f"{selected}")
if selected == "Contact":
    st.title(f"{selected}")
from streamlit_option_menu import option_menu
df = pd.read_csv("/Users/allyryan/Downloads/data.csv")



if selected == "Logistic Regression":
    Logistic_Regression = st.selectbox('Logistic Regression: Select one option:', ['', 'Summary of my Data', 'Replace Classifier With 0 and 1', 'Check for Null Values in the data', 'Create and fit models to test and training data', 'Accuracy Score', 'Confusion Matrix'], format_func=lambda x: 'Select an option' if x == '' else x)
#if Logistic_Regression:
    #st.success('Yay! 🎉')
#else:
    #st.warning('No option is selected')
    if Logistic_Regression == 'Check for Null Values in the data':
        st.markdown("# Null Values Part 1")
        st.markdown("""
                    Part 1 is to determine if there are any null values in the data.
                    In this step, we print out all of the column names.  Any column name 
                    that returns true signifies a null value and will need to be deleted.
                    """)
        if st.button('Click for Part 1 Code'):
            st.write("df.isnull().any()")
        st.text(df.isnull().any())
        if st.button('Part 1 Explanation'):
            st.write("At the bottom of the column name list, we see that there "
                     "are 32 null values, signified by the 'true' statement.  The column is labeled 'Unnamed: 32 "
                     "and we seek to delete that column.")
        st.markdown("# Null Values Part 2")
        st.markdown("""
                    Part 2 is to delete the null values in our dataset.
                    """)
        if st.button('Click for Part 2 Code'):
            st.write("df.drop('Unnamed: 32', axis=1)")
        df = df.drop('Unnamed: 32', axis=1)
        if st.button('Part 2 Explanation'):
            st.write("Once we perform the df.drop function on our identified null values, "
                     "we have removed all nulls from our dataset.")
        st.markdown(""
                    "")
        st.markdown("# Congrats 🎉!!!")
        st.markdown(""" 
                You have completed Logistic Regression, Null Values!!!
                You are ready to move on to the 'Summary of my Data' step!!!
                """)
    if Logistic_Regression == 'Summary of my Data':
        df = df.drop('Unnamed: 32', axis=1)
        st.markdown("# Summarize Data")
        st.markdown("""
                Summarizing data in this case means to generate descriptive statistics, which includes insight pertaining to
                central tendency, dispersion, and the shape of a dataset's distribution.
                """)
        if st.button('Click for Code'):
            st.write("df.describe()")
        st.text(df.describe())
        st.markdown(""
                    "")
        st.markdown("# Congrats 🎉!!!")
        st.markdown(""" 
                    You have completed Logistic Regression, Summarize Data!!!
                    You are ready to move on to the 'Replace Classifier With 0 and 1' step!!!
                    """)
    if Logistic_Regression == 'Replace Classifier With 0 and 1':
        df = df.drop('Unnamed: 32', axis=1)
        st.markdown("# Replace Classifier Part 1")
        st.markdown("""
                    Binary classification uses 0's and 1's to perform the classification analysis.  
                    0 represents the negative value and 1 represents the positive value.  A majority of datasets
                    are going to contain the qualitative classifier instead of the 0 and 1 values.  As such, when
                    performing classification, it is necessary to write a code that will replace the classifiers
                    in the dataset.    
                    """)
        if st.button('Click for Part 1 Code'):
            st.write("df.diagnosis.replace(['M', 'B'], [1, 0], inplace = True)")
        df.diagnosis.replace(["M", "B"], [1, 0], inplace=True)
        if st.button('Part 1 Explanation'):
            st.write("""
                    For the breast cancer dataset used in this tutorial, the goal is to classify 'Benign' (i.e., 1, 
                    the negative value) and 'Malignant' (i.e., 0, the positive value) tumors.  These are represented 
                    by 'M' and 'B' in the dataset under the 'diagnosis' column.  If performing classification analysis 
                    on a different dataset, determine your dataset's classifiers and its respective column name.  For 
                    best practice, place the positive value first within the brackets.
                    """)
        st.markdown("# Replace Classifier Part 2")
        st.markdown("""
                    Part 2 is for you to check that your classifiers have been replaced with the 0 and 1 values.      
                    """)
        if st.button('Click for Part 2 Code'):
            st.write("df.head()")
        st.text(df.head())
        if st.button('Part 2 Explanation'):
            st.write("""
                    We can see that the diagnosis column has been replaced with 0's and 1's, meaning that our 
                    replace() function was successful.
                    """)
        st.markdown("# Replace Classifier Part 3")
        st.markdown("""
                Part 3 is for you to see the counts of your 0 and 1 classifiers.  It is a best practice to have
                a good representation of both classifiers.  If there is large discrepancy between classifier counts percentage wise, 
                perhaps look for a different dataset.      
                """)
        if st.button('Click for Part 3 Code'):
            st.write("df.groupby(['diagnosis']).diagnosis.count()")
        st.text(df.groupby(["diagnosis"]).diagnosis.count())
        if st.button('Part 3 Explanation'):
            st.write("""
                    We can see that there are 212 Benign (1) tumors and 357 Malignant (0) 
                    tumors in the breast cancer dataset.  This is an acceptable representation of 
                    both classifiers, thus we can proceed with this dataset. 
            
                    Remember, if you are performing classification analysis on a different dataset, 
                    insert your classifier column name into the Part 3 code above. 
                    """)
        st.markdown("# Replace Classifier Part 4")
        st.markdown("""
                    Part 4 is for you visualize the counts of your classifiers.  When working with data, it is always
                    best to visualize if you can!  Charts, graphs, maps, etc help the viewer to understand the data in
                    a more clear and efficient way.      
                    """)
        if st.button('Click for Part 4 Code'):
            st.write("sns.countplot('diagnosis', data=df)")
            st.write("plt.show()")
        sns.countplot('diagnosis', data=df)
        st.pyplot(plt.show())
        if st.button('Part 4 Explanation'):
            st.write("""
                    We can see that there is a good representation of both classifiers, where 0=malignant
                    and 1=benign. 

                    Remember, if you are performing classification analysis on a different dataset, 
                    insert your classifier column name into the Part 4 code above. 
                    """)
        st.markdown(""
                    "")
        st.markdown("# Congrats 🎉!!!")
        st.markdown(""" 
                       You have completed Logistic Regression, Replace Classifier!!!
                       You are ready to move on to the 'Test and Training Data' step!!!
                       """)
    if Logistic_Regression ==  'Create and fit models to test and training data':
        df.isnull().any()
        df = df.drop('Unnamed: 32', axis=1)
        df.diagnosis.replace(["M", "B"], [1, 0], inplace=True)
        st.markdown("# Test and Training Data Part 1")
        st.markdown("""
                    Part 1 ensures that X only takes on the independent variables in the dataset.      
                    """)
        if st.button('Click for Part 1 Code'):
            st.write("X = df.drop(['id','diagnosis'],axis=1)")
        if st.button('Part 1 Explanation'):
            st.write("""
                    Since the diagnosis column contains the classifiers, we want to remove this column from the X variable.  We 
                    won't neglect the diagnosis column for long, however, as we will utilize these values for the y variable!
            
                    We remove the id column as it is just an identifier, and does not play a role in tumor classification.  
            
                    If working with your own dataset, make sure you remove the irrelevant columns pertaining to your dataset!
                    """)
        X = df.drop(['id', 'diagnosis'], axis=1)
        st.markdown("# Test and Training Data Part 2")
        st.markdown("""
                    Part 2 ensures that Y only takes on the classifier column in the dataset.      
                    """)
        if st.button('Click for Part 2 Code'):
            st.write("Y = df.iloc[:, 1]")
        if st.button('Part 2 Explanation'):
            st.write("""
                    Since the diagnosis column contains the classifiers, we consider this the dependent variable column.  

                    If working with your own dataset, make sure you do some research on indexes so that you can accurately 
                    store your classifier column!
                    """)
        Y = df.iloc[:, 1]
        st.markdown("# Test and Training Data Part 3")
        st.markdown("""
                    Part 3 splits the data into test and training sets using the X and Y variables we created previously.       
                    """)
        if st.button('Click for Part 3 Code'):
            st.write("xTrain, xTest, yTrain, yTest = sklearn.model_selection.train_test_split(X, Y, train_size=0.8, random_state =20221023)")
        xTrain, xTest, yTrain, yTest = sklearn.model_selection.train_test_split(X, Y, train_size=0.8, random_state =20221023)
        if st.button('Part 3 Explanation'):
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
        st.markdown("# Test and Training Data Part 4")
        st.markdown("""
                    Part 4 creates a model that performs logistic regression.      
                    """)
        if st.button('Click for Part 4 Code'):
            st.write("regressionModel = LogisticRegression(solver='newton-cg')")
        regressionModel = LogisticRegression(solver='newton-cg')
        st.markdown("# Test and Training Data Part 5")
        st.markdown("""
                    Part 5 fits the regression model to the training set.  This allows the regressor to "study"
                    the data and "learn" from it.  This step will not return any output, but will be used later on 
                    in the classification process.       
                    """)
        if st.button('Click for Part 5 Code'):
            st.write("regressionModel.fit(xTrain, yTrain)")
        regressionModel.fit(xTrain, yTrain)
        st.markdown("# Test and Training Data Part 6")
        st.markdown("""
                    Now that we have created our model and trained it, Part 6 will use our testing dataset to test the model.  
                    This step will not return any output, but will be used later on in the classification process.
                    """)
        if st.button('Click for Part 6 Code'):
            st.write("predRegression =regressionModel.predict(xTest)")
        predRegression = regressionModel.predict(xTest)
        st.markdown(""
                    "")
        st.markdown("# Congrats 🎉!!!")
        st.markdown(""" 
                    You have completed Logistic Regression, Test and Training Data!!!
                    You are ready to move on to the 'Accuracy Score' step!!!
                    """)
    if Logistic_Regression == 'Accuracy Score':
        df = df.drop('Unnamed: 32', axis=1)
        df.diagnosis.replace(["M", "B"], [1, 0], inplace=True)
        X = df.drop(['id', 'diagnosis'], axis=1)
        Y = df.iloc[:, 1]
        xTrain, xTest, yTrain, yTest = sklearn.model_selection.train_test_split(X, Y, train_size=0.8, random_state =20221023)
        regressionModel = LogisticRegression(solver='newton-cg')
        regressionModel.fit(xTrain, yTrain)
        predRegression = regressionModel.predict(xTest)
        st.markdown("# Accuracy Score Part 1")
        st.markdown("""
                    Part 1 allows you to see the actual and predicted classifiers in array form.  Can you spot the 
                    differences between the actual and predicted set?      
                    """)
        if st.button('Click for Part 1 Code'):
            st.write("print(yTest.values)")
            st.write("print(predRegression)")
        st.text("Actual breast cancer : ")
        st.text(yTest.values)
        st.text("\nPredicted breast cancer : ")
        st.text(predRegression)
        st.markdown("# Accuracy Score Part 2")
        st.markdown("""
                    Part 2 allows you to calculate the accuracy score.  As it may sound, the accuracy score declares how 
                    accurately the model classifies the classifiers.     
                    """)
        if st.button('Click for Part 2 Code'):
            st.write("accuracy_score(yTest, predRegression) * 100")
            st.text("Accuracy Score: ")
            st.text(accuracy_score(yTest, predRegression) * 100)
        if st.button("Part 2 Explanation"):
            st.write("""
                    Accuracy score is calculated by dividing the number of correct predictions by the total prediction number.
                    """)
        st.markdown("# Congrats 🎉!!!")
        st.markdown(""" 
                    You have completed Logistic Regression, Accuracy Score!!!
                    You are ready to move on to the 'Confusion Matrix' step!!!
                    """)
    if Logistic_Regression == 'Confusion Matrix':
        df = df.drop('Unnamed: 32', axis=1)
        df.diagnosis.replace(["M", "B"], [1, 0], inplace=True)
        X = df.drop(['id', 'diagnosis'], axis=1)
        Y = df.iloc[:, 1]
        xTrain, xTest, yTrain, yTest = sklearn.model_selection.train_test_split(X, Y, train_size=0.8, random_state =20221023)
        regressionModel = LogisticRegression(solver='newton-cg')
        regressionModel.fit(xTrain, yTrain)
        regressionModel.score(xTrain, yTrain)
        predRegression = regressionModel.predict(xTest)
        plt.figure(figsize=(16, 14))
        plt.subplot(3, 3, 1)
        sns.heatmap(sklearn.metrics.confusion_matrix(yTest, predRegression), annot=True).set(title='Logistic Regression')
        st.set_option('deprecation.showPyplotGlobalUse', False)
        st.pyplot(plt.show())
        predVals = pd.DataFrame(data={'truth': yTest, 'regression': predRegression})


if selected == "KNN":
    KNN = st.selectbox('KNN: Select one option:', ['', 'Summary of my Data', 'Replace Classifier With 0 and 1', 'Check for Null Values in the data', 'Create and fit models to test and training data', 'Accuracy Score', 'Confusion Matrix'], format_func=lambda x: 'Select an option' if x == '' else x)
    if KNN == 'Check for Null Values in the data':
        st.markdown("# Null Values Part 1")
        st.markdown("""
                    Part 1 is to determine if there are any null values in the data.
                    In this step, we print out all of the column names.  Any column name 
                    that returns true signifies a null value and will need to be deleted.
                    """)
        if st.button('Click for Part 1 Code'):
            st.write("df.isnull().any()")
        st.text(df.isnull().any())
        if st.button('Part 1 Explanation'):
            st.write("At the bottom of the column name list, we see that there "
                     "are 32 null values, signified by the 'true' statement.  The column is labeled 'Unnamed: 32 "
                     "and we seek to delete that column.")
        st.markdown("# Null Values Part 2")
        st.markdown("""
                    Part 2 is to delete the null values in our dataset.
                    """)
        if st.button('Click for Part 2 Code'):
            st.write("df.drop('Unnamed: 32', axis=1)")
        df = df.drop('Unnamed: 32', axis=1)
        if st.button('Part 2 Explanation'):
            st.write("Once we perform the df.drop function on our identified null values, "
                     "we have removed all nulls from our dataset.")
        st.markdown(""
                    "")
        st.markdown("# Congrats 🎉!!!")
        st.markdown(""" 
                    You have completed KNN, Null Values!!!
                    You are ready to move on to the 'Summary of my Data' step!!!
                    """)
    if KNN == 'Summary of my Data':
        df = df.drop('Unnamed: 32', axis=1)
        st.markdown("# Summarize Data")
        st.markdown("""
                    Summarizing data in this case means to generate descriptive statistics, which includes insight pertaining to
                    central tendency, dispersion, and the shape of a dataset's distribution.
                    """)
        if st.button('Click for Code'):
            st.write("df.describe()")
        st.text(df.describe())
        st.markdown(""
                    "")
        st.markdown("# Congrats 🎉!!!")
        st.markdown(""" 
                    You have completed KNN, Summarize Data!!!
                    You are ready to move on to the 'Replace Classifier With 0 and 1' step!!!
                    """)
    if KNN == 'Replace Classifier With 0 and 1':
        df = df.drop('Unnamed: 32', axis=1)
        st.markdown("# Replace Classifier Part 1")
        st.markdown("""
                    Binary classification uses 0's and 1's to perform the classification analysis.  
                    0 represents the negative value and 1 represents the positive value.  A majority of datasets
                    are going to contain the qualitative classifier instead of the 0 and 1 values.  As such, when
                    performing classification, it is necessary to write a code that will replace the classifiers
                    in the dataset.    
                    """)
        if st.button('Click for Part 1 Code'):
            st.write("df.diagnosis.replace(['M', 'B'], [1, 0], inplace = True)")
        df.diagnosis.replace(["M", "B"], [1, 0], inplace=True)
        if st.button('Part 1 Explanation'):
            st.write("""
                    For the breast cancer dataset used in this tutorial, the goal is to classify 'Benign' (i.e., 1, 
                    the negative value) and 'Malignant' (i.e., 0, the positive value) tumors.  These are represented 
                    by 'M' and 'B' in the dataset under the 'diagnosis' column.  If performing classification analysis 
                    on a different dataset, determine your dataset's classifiers and its respective column name.  For 
                    best practice, place the positive value first within the brackets.
                    """)
        st.markdown("# Replace Classifier Part 2")
        st.markdown("""
                    Part 2 is for you to check that your classifiers have been replaced with the 0 and 1 values.      
                    """)
        if st.button('Click for Part 2 Code'):
            st.write("df.head()")
        st.text(df.head())
        if st.button('Part 2 Explanation'):
            st.write("""
                    We can see that the diagnosis column has been replaced with 0's and 1's, meaning that our 
                    replace() function was successful.
                    """)
        st.markdown("# Replace Classifier Part 3")
        st.markdown("""
                    Part 3 is for you to see the counts of your 0 and 1 classifiers.  It is a best practice to have
                    a good representation of both classifiers.  If there is large discrepancy between classifier counts percentage wise, 
                    perhaps look for a different dataset.      
                    """)
        if st.button('Click for Part 3 Code'):
            st.write("df.groupby(['diagnosis']).diagnosis.count()")
        st.text(df.groupby(["diagnosis"]).diagnosis.count())
        if st.button('Part 3 Explanation'):
            st.write("""
                    We can see that there are 212 Benign (1) tumors and 357 Malignant (0) 
                    tumors in the breast cancer dataset.  This is an acceptable representation of 
                    both classifiers, thus we can proceed with this dataset. 

                    Remember, if you are performing classification analysis on a different dataset, 
                    insert your classifier column name into the Part 3 code above. 
                    """)
        st.markdown("# Replace Classifier Part 4")
        st.markdown("""
                    Part 4 is for you visualize the counts of your classifiers.  When working with data, it is always
                    best to visualize if you can!  Charts, graphs, maps, etc help the viewer to understand the data in
                    a more clear and efficient way.      
                    """)
        if st.button('Click for Part 4 Code'):
            st.write("sns.countplot('diagnosis', data=df)")
            st.write("plt.show()")
        sns.countplot('diagnosis', data=df)
        st.pyplot(plt.show())
        if st.button('Part 4 Explanation'):
            st.write("""
                    We can see that there is a good representation of both classifiers, where 0=malignant
                    and 1=benign. 

                    Remember, if you are performing classification analysis on a different dataset, 
                    insert your classifier column name into the Part 4 code above. 
                    """)
        st.markdown(""
                    "")
        st.markdown("# Congrats 🎉!!!")
        st.markdown(""" 
                    You have completed KNN, Replace Classifier!!!
                    You are ready to move on to the 'Test and Training Data' step!!!
                    """)
    if KNN ==  'Create and fit models to test and training data':
        df.isnull().any()
        df = df.drop('Unnamed: 32', axis=1)
        df.diagnosis.replace(["M", "B"], [1, 0], inplace=True)
        st.markdown("# Test and Training Data Part 1")
        st.markdown("""
                   Part 1 ensures that X only takes on the independent variables in the dataset.      
                    """)
        if st.button('Click for Part 1 Code'):
            st.write("X = df.drop(['id','diagnosis'],axis=1)")
        if st.button('Part 1 Explanation'):
            st.write("""
                   Since the diagnosis column contains the classifiers, we want to remove this column from the X variable.  We 
                   won't neglect the diagnosis column for long, however, as we will utilize these values for the y variable!

                   We remove the id column as it is just an identifier, and does not play a role in tumor classification.  

                   If working with your own dataset, make sure you remove the irrelevant columns pertaining to your dataset!
                   """)
        X = df.drop(['id', 'diagnosis'], axis=1)
        st.markdown("# Test and Training Data Part 2")
        st.markdown("""
                    Part 2 ensures that Y only takes on the classifier column in the dataset.      
                    """)
        if st.button('Click for Part 2 Code'):
            st.write("Y = df.iloc[:, 1]")
        if st.button('Part 2 Explanation'):
            st.write("""
                    Since the diagnosis column contains the classifiers, we consider this the dependent variable column.  

                    If working with your own dataset, make sure you do some research on indexes so that you can accurately 
                    store your classifier column!
                    """)
        Y = df.iloc[:, 1]
        st.markdown("# Test and Training Data Part 3")
        st.markdown("""
                    Part 3 splits the data into test and training sets using the X and Y variables we created previously.       
                    """)
        if st.button('Click for Part 3 Code'):
            st.write("xTrain, xTest, yTrain, yTest = sklearn.model_selection.train_test_split(X, Y, train_size=0.8, random_state =20221023)")
        xTrain, xTest, yTrain, yTest = sklearn.model_selection.train_test_split(X, Y, train_size=0.8,random_state=20221023)
        if st.button('Part 3 Explanation'):
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
        st.markdown("# Test and Training Data Part 4")
        st.markdown("""
                    Part 4 creates a model that performs KNN Classification.      
                    """)
        if st.button('Click for Part 4 Code'):
            st.write("KNNModel = KNeighborsClassifier(n_neighbors=5,leaf_size=1, weights='uniform')")
        KNNModel = KNeighborsClassifier(n_neighbors=5, leaf_size=1, weights='uniform')
        if st.button('Part 4 Explanation'):
            st.write("""
                   The n_neighbors parameter declares the number of neighbors.  5 is the default.  Do some research to find
                   the best option for your dataset.  
                   
                   Uniform weights means that all points in each neighborhood are weighed equally.  Do some research to find the best
                   option for your dataset. 
                   """)
        st.markdown("# Test and Training Data Part 5")
        st.markdown("""
                    Part 5 fits the KNN model to the training set.  This allows the model to "study"
                    the data and "learn" from it.  This step will not return any output, but will be used later on 
                    in the classification process.       
                    """)
        if st.button('Click for Part 5 Code'):
            st.write("KNNModel.fit(xTrain,yTrain)")
        KNNModel.fit(xTrain, yTrain)
        st.markdown("# Test and Training Data Part 6")
        st.markdown("""
                    Now that we have created our model and trained it, Part 6 will use our testing dataset to test the model.  
                    This step will not return any output, but will be used later on in the classification process.
                    """)
        if st.button('Click for Part 6 Code'):
            st.write("predKNN =KNNModel.predict(xTest)")
        predKNN = KNNModel.predict(xTest)
        st.markdown(""
                    "")
        st.markdown("# Congrats 🎉!!!")
        st.markdown(""" 
                    You have completed KNN, Test and Training Data!!!
                    You are ready to move on to the 'Accuracy Score' step!!!
                    """)
    if KNN == 'Accuracy Score':
        df = df.drop('Unnamed: 32', axis=1)
        df.diagnosis.replace(["M", "B"], [1, 0], inplace=True)
        X = df.drop(['id', 'diagnosis'], axis=1)
        Y = df.iloc[:, 1]
        xTrain, xTest, yTrain, yTest = sklearn.model_selection.train_test_split(X, Y, train_size=0.8, random_state=20221023)
        KNNModel = KNeighborsClassifier(n_neighbors=5, leaf_size=1, weights='uniform')
        KNNModel.fit(xTrain, yTrain)
        predKNN = KNNModel.predict(xTest)
        st.markdown("# Accuracy Score Part 1")
        st.markdown("""
                    Part 1 allows you to see the actual and predicted classifiers in array form.  Can you spot the 
                    differences between the actual and predicted set?      
                    """)
        if st.button('Click for Part 1 Code'):
            st.write("print(yTest.values)")
            st.write("print(predKNN)")
        st.text("Actual breast cancer : ")
        st.text(yTest.values)
        st.text("\nPredicted breast cancer : ")
        st.text(predKNN)
        st.markdown("# Accuracy Score Part 2")
        st.markdown("""
                    Part 2 allows you to calculate the accuracy score.  As it may sound, the accuracy score declares how 
                    accurately the model classifies the classifiers.     
                    """)
        if st.button('Click for Part 2 Code'):
            st.write("accuracy_score(yTest, predKNN) * 100")
            st.text("Accuracy Score: ")
            st.text(accuracy_score(yTest, predKNN) * 100)
        if st.button("Part 2 Explanation"):
            st.write("""
                    Accuracy score is calculated by dividing the number of correct predictions by the total prediction number.
                    """)
        st.markdown("# Congrats 🎉!!!")
        st.markdown(""" 
                    You have completed KNN, Accuracy Score!!!
                    You are ready to move on to the 'Confusion Matrix' step!!!
                    """)
    if KNN == 'Confusion Matrix':
        df = df.drop('Unnamed: 32', axis=1)
        df.diagnosis.replace(["M", "B"], [1, 0], inplace=True)
        X = df.drop(['id', 'diagnosis'], axis=1)
        Y = df.iloc[:, 1]
        xTrain, xTest, yTrain, yTest = sklearn.model_selection.train_test_split(X, Y, train_size=0.8, random_state =20221023)
        KNNModel = KNeighborsClassifier(n_neighbors=5, leaf_size=1, weights='uniform')
        KNNModel.fit(xTrain, yTrain)
        predKNN = KNNModel.predict(xTest)
        plt.figure(figsize=(16, 14))
        plt.subplot(3, 3, 2)
        sns.heatmap(sklearn.metrics.confusion_matrix(yTest, predKNN), annot=True).set(title= 'KNN')
        st.set_option('deprecation.showPyplotGlobalUse', False)
        st.pyplot(plt.show())
        predVals = pd.DataFrame(data={'truth': yTest, 'KNN': predKNN})


if selected == "Random Forest":
    Random_Forest = st.selectbox('Random Forest: Select one option:', ['', 'Summary of my Data', 'Replace Classifier With 0 and 1', 'Check for Null Values in the data', 'Create and fit models to test and training data', 'Accuracy Score', 'Confusion Matrix'], format_func=lambda x: 'Select an option' if x == '' else x)
    if Random_Forest == 'Check for Null Values in the data':
        st.markdown("# Null Values Part 1")
        st.markdown("""
                    Part 1 is to determine if there are any null values in the data.
                    In this step, we print out all of the column names.  Any column name 
                    that returns true signifies a null value and will need to be deleted.
                    """)
        if st.button('Click for Part 1 Code'):
            st.write("df.isnull().any()")
        st.text(df.isnull().any())
        if st.button('Part 1 Explanation'):
            st.write("At the bottom of the column name list, we see that there "
                     "are 32 null values, signified by the 'true' statement.  The column is labeled 'Unnamed: 32 "
                     "and we seek to delete that column.")
        st.markdown("# Null Values Part 2")
        st.markdown("""
                    Part 2 is to delete the null values in our dataset.
                    """)
        if st.button('Click for Part 2 Code'):
            st.write("df.drop('Unnamed: 32', axis=1)")
        df = df.drop('Unnamed: 32', axis=1)
        if st.button('Part 2 Explanation'):
            st.write("Once we perform the df.drop function on our identified null values, "
                     "we have removed all nulls from our dataset.")
        st.markdown(""
                    "")
        st.markdown("# Congrats 🎉!!!")
        st.markdown(""" 
                    You have completed Random Forest, Null Values!!!
                    You are ready to move on to the 'Summary of my Data' step!!!
                    """)
    if Random_Forest == 'Summary of my Data':
        df = df.drop('Unnamed: 32', axis=1)
        st.markdown("# Summarize Data")
        st.markdown("""
                    Summarizing data in this case means to generate descriptive statistics, which includes insight pertaining to
                    central tendency, dispersion, and the shape of a dataset's distribution.
                    """)
        if st.button('Click for Code'):
            st.write("df.describe()")
        st.text(df.describe())
        st.markdown(""
                    "")
        st.markdown("# Congrats 🎉!!!")
        st.markdown(""" 
                    You have completed Random Forest, Summarize Data!!!
                    You are ready to move on to the 'Replace Classifier With 0 and 1' step!!!
                    """)
    if Random_Forest == 'Replace Classifier With 0 and 1':
        df = df.drop('Unnamed: 32', axis=1)
        st.markdown("# Replace Classifier Part 1")
        st.markdown("""
                    Binary classification uses 0's and 1's to perform the classification analysis.  
                    0 represents the negative value and 1 represents the positive value.  A majority of datasets
                    are going to contain the qualitative classifier instead of the 0 and 1 values.  As such, when
                    performing classification, it is necessary to write a code that will replace the classifiers
                    in the dataset.    
                    """)
        if st.button('Click for Part 1 Code'):
            st.write("df.diagnosis.replace(['M', 'B'], [1, 0], inplace = True)")
        df.diagnosis.replace(["M", "B"], [1, 0], inplace=True)
        if st.button('Part 1 Explanation'):
            st.write("""
                    For the breast cancer dataset used in this tutorial, the goal is to classify 'Benign' (i.e., 1, 
                    the negative value) and 'Malignant' (i.e., 0, the positive value) tumors.  These are represented 
                    by 'M' and 'B' in the dataset under the 'diagnosis' column.  If performing classification analysis 
                    on a different dataset, determine your dataset's classifiers and its respective column name.  For 
                    best practice, place the positive value first within the brackets.
                    """)
        st.markdown("# Replace Classifier Part 2")
        st.markdown("""
                    Part 2 is for you to check that your classifiers have been replaced with the 0 and 1 values.      
                    """)
        if st.button('Click for Part 2 Code'):
            st.write("df.head()")
        st.text(df.head())
        if st.button('Part 2 Explanation'):
            st.write("""
                    We can see that the diagnosis column has been replaced with 0's and 1's, meaning that our 
                    replace() function was successful.
                    """)
        st.markdown("# Replace Classifier Part 3")
        st.markdown("""
                    Part 3 is for you to see the counts of your 0 and 1 classifiers.  It is a best practice to have
                    a good representation of both classifiers.  If there is large discrepancy between classifier counts percentage wise, 
                    perhaps look for a different dataset.      
                    """)
        if st.button('Click for Part 3 Code'):
            st.write("df.groupby(['diagnosis']).diagnosis.count()")
        st.text(df.groupby(["diagnosis"]).diagnosis.count())
        if st.button('Part 3 Explanation'):
            st.write("""
                    We can see that there are 212 Benign (1) tumors and 357 Malignant (0) 
                    tumors in the breast cancer dataset.  This is an acceptable representation of 
                    both classifiers, thus we can proceed with this dataset. 

                    Remember, if you are performing classification analysis on a different dataset, 
                    insert your classifier column name into the Part 3 code above. 
                    """)
        st.markdown("# Replace Classifier Part 4")
        st.markdown("""
                    Part 4 is for you visualize the counts of your classifiers.  When working with data, it is always
                    best to visualize if you can!  Charts, graphs, maps, etc help the viewer to understand the data in
                    a more clear and efficient way.      
                    """)
        if st.button('Click for Part 4 Code'):
            st.write("sns.countplot('diagnosis', data=df)")
            st.write("plt.show()")
        sns.countplot('diagnosis', data=df)
        st.pyplot(plt.show())
        if st.button('Part 4 Explanation'):
            st.write("""
                    We can see that there is a good representation of both classifiers, where 0=malignant
                    and 1=benign. 

                    Remember, if you are performing classification analysis on a different dataset, 
                    insert your classifier column name into the Part 4 code above. 
                    """)
        st.markdown(""
                    "")
        st.markdown("# Congrats 🎉!!!")
        st.markdown(""" 
                    You have completed Random Forest, Replace Classifier!!!
                    You are ready to move on to the 'Test and Training Data' step!!!
                    """)
    if Random_Forest ==  'Create and fit models to test and training data':
        df.isnull().any()
        df = df.drop('Unnamed: 32', axis=1)
        df.diagnosis.replace(["M", "B"], [1, 0], inplace=True)
        st.markdown("# Test and Training Data Part 1")
        st.markdown("""
                    Part 1 ensures that X only takes on the independent variables in the dataset.      
                    """)
        if st.button('Click for Part 1 Code'):
            st.write("X = df.drop(['id','diagnosis'],axis=1)")
        if st.button('Part 1 Explanation'):
            st.write("""
                    Since the diagnosis column contains the classifiers, we want to remove this column from the X variable.  We 
                    won't neglect the diagnosis column for long, however, as we will utilize these values for the y variable!

                    We remove the id column as it is just an identifier, and does not play a role in tumor classification.  

                    If working with your own dataset, make sure you remove the irrelevant columns pertaining to your dataset!
                    """)
        X = df.drop(['id', 'diagnosis'], axis=1)
        st.markdown("# Test and Training Data Part 2")
        st.markdown("""
                    Part 2 ensures that Y only takes on the classifier column in the dataset.      
                    """)
        if st.button('Click for Part 2 Code'):
            st.write("Y = df.iloc[:, 1]")
        if st.button('Part 2 Explanation'):
            st.write("""
                    Since the diagnosis column contains the classifiers, we consider this the dependent variable column.  

                    If working with your own dataset, make sure you do some research on indexes so that you can accurately 
                    store your classifier column!
                    """)
        Y = df.iloc[:, 1]
        st.markdown("# Test and Training Data Part 3")
        st.markdown("""
                    Part 3 splits the data into test and training sets using the X and Y variables we created previously.       
                    """)
        if st.button('Click for Part 3 Code'):
            st.write("xTrain, xTest, yTrain, yTest = sklearn.model_selection.train_test_split(X, Y, train_size=0.8, random_state =20221023)")
        xTrain, xTest, yTrain, yTest = sklearn.model_selection.train_test_split(X, Y, train_size=0.8,random_state=20221023)
        if st.button('Part 3 Explanation'):
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
        st.markdown("# Test and Training Data Part 4")
        st.markdown("""
                    Part 4 creates a model that performs Random Forest Classification.      
                    """)
        if st.button('Click for Part 4 Code'):
            st.write("randomFModel = RandomForestClassifier()")
        randomFModel = RandomForestClassifier()
        st.markdown("# Test and Training Data Part 5")
        st.markdown("""
                    Part 5 fits the Random Forest model to the training set.  This allows the model to "study"
                    the data and "learn" from it.  This step will not return any output, but will be used later on 
                    in the classification process.       
                    """)
        if st.button('Click for Part 5 Code'):
            st.write("randomFModel.fit(xTrain,yTrain)")
        randomFModel.fit(xTrain, yTrain)
        st.markdown("# Test and Training Data Part 6")
        st.markdown("""
                    Now that we have created our model and trained it, Part 6 will use our testing dataset to test the model.  
                    This step will not return any output, but will be used later on in the classification process.
                    """)
        if st.button('Click for Part 6 Code'):
            st.write("predRandomF =randomFModel.predict(xTest)")
        predRandomF = randomFModel.predict(xTest)
        st.markdown(""
                    "")
        st.markdown("# Congrats 🎉!!!")
        st.markdown(""" 
                    You have completed Random Forest, Test and Training Data!!!
                    You are ready to move on to the 'Accuracy Score' step!!!
                    """)
    if Random_Forest == 'Accuracy Score':
        df = df.drop('Unnamed: 32', axis=1)
        df.diagnosis.replace(["M", "B"], [1, 0], inplace=True)
        X = df.drop(['id', 'diagnosis'], axis=1)
        Y = df.iloc[:, 1]
        xTrain, xTest, yTrain, yTest = sklearn.model_selection.train_test_split(X, Y, train_size=0.8, random_state=20221023)
        randomFModel = RandomForestClassifier()
        randomFModel.fit(xTrain, yTrain)
        randomFModel.score(xTrain, yTrain)
        predRandomF = randomFModel.predict(xTest)
        st.markdown("# Accuracy Score Part 1")
        st.markdown("""
                    Part 1 allows you to see the actual and predicted classifiers in array form.  Can you spot the 
                    differences between the actual and predicted set?      
                    """)
        if st.button('Click for Part 1 Code'):
            st.write("print(yTest.values)")
            st.write("print(predRandomF)")
        st.text("Actual breast cancer : ")
        st.text(yTest.values)
        st.text("\nPredicted breast cancer : ")
        st.text(predRandomF)
        st.markdown("# Accuracy Score Part 2")
        st.markdown("""
                    Part 2 allows you to calculate the accuracy score.  As it may sound, the accuracy score declares how 
                    accurately the model classifies the classifiers.     
                    """)
        if st.button('Click for Part 2 Code'):
            st.write("accuracy_score(yTest, predRandomF) * 100")
            st.text("Accuracy Score: ")
            st.text(accuracy_score(yTest, predRandomF) * 100)
        if st.button("Part 2 Explanation"):
            st.write("""
                    Accuracy score is calculated by dividing the number of correct predictions by the total prediction number.
                    """)
        st.markdown("# Congrats 🎉!!!")
        st.markdown(""" 
                    You have completed Random Forest, Accuracy Score!!!
                    You are ready to move on to the 'Confusion Matrix' step!!!
                    """)
    if Random_Forest == 'Confusion Matrix':
        df = df.drop('Unnamed: 32', axis=1)
        df.diagnosis.replace(["M", "B"], [1, 0], inplace=True)
        X = df.drop(['id', 'diagnosis'], axis=1)
        Y = df.iloc[:, 1]
        xTrain, xTest, yTrain, yTest = sklearn.model_selection.train_test_split(X, Y, train_size=0.8, random_state=20221023)
        randomFModel = RandomForestClassifier()
        randomFModel.fit(xTrain, yTrain)
        randomFModel.score(xTrain, yTrain)
        predRandomF = randomFModel.predict(xTest)
        plt.figure(figsize=(16, 14))
        plt.subplot(3, 3, 3)
        sns.heatmap(sklearn.metrics.confusion_matrix(yTest, predRandomF), annot=True).set(title='Random Forest')
        st.set_option('deprecation.showPyplotGlobalUse', False)
        st.pyplot(plt.show())
        predVals = pd.DataFrame(data={'truth': yTest, 'KNN': predRandomF})

if selected == "Naive Bayes":
    Naive_Bayes = st.selectbox('Naive Bayes: Select one option:', ['', 'Summary of my Data', 'Replace Classifier With 0 and 1', 'Check for Null Values in the data', 'Create and fit models to test and training data', 'Accuracy Score', 'Confusion Matrix'], format_func=lambda x: 'Select an option' if x == '' else x)
    if Naive_Bayes == 'Check for Null Values in the data':
        st.markdown("# Null Values Part 1")
        st.markdown("""
                    Part 1 is to determine if there are any null values in the data.
                    In this step, we print out all of the column names.  Any column name 
                    that returns true signifies a null value and will need to be deleted.
                    """)
        if st.button('Click for Part 1 Code'):
            st.write("df.isnull().any()")
        st.text(df.isnull().any())
        if st.button('Part 1 Explanation'):
            st.write("At the bottom of the column name list, we see that there "
                     "are 32 null values, signified by the 'true' statement.  The column is labeled 'Unnamed: 32 "
                     "and we seek to delete that column.")
        st.markdown("# Null Values Part 2")
        st.markdown("""
                    Part 2 is to delete the null values in our dataset.
                    """)
        if st.button('Click for Part 2 Code'):
            st.write("df.drop('Unnamed: 32', axis=1)")
        df = df.drop('Unnamed: 32', axis=1)
        if st.button('Part 2 Explanation'):
            st.write("Once we perform the df.drop function on our identified null values, "
                     "we have removed all nulls from our dataset.")
        st.markdown(""
                    "")
        st.markdown("# Congrats 🎉!!!")
        st.markdown(""" 
                    You have completed Naive Bayes, Null Values!!!
                    You are ready to move on to the 'Summary of my Data' step!!!
                    """)
    if Naive_Bayes == 'Summary of my Data':
        df = df.drop('Unnamed: 32', axis=1)
        st.markdown("# Summarize Data")
        st.markdown("""
                    Summarizing data in this case means to generate descriptive statistics, which includes insight pertaining to
                    central tendency, dispersion, and the shape of a dataset's distribution.
                    """)
        if st.button('Click for Code'):
            st.write("df.describe()")
        st.text(df.describe())
        st.markdown(""
                    "")
        st.markdown("# Congrats 🎉!!!")
        st.markdown(""" 
                    You have completed Naive Bayes, Summarize Data!!!
                    You are ready to move on to the 'Replace Classifier With 0 and 1' step!!!
                    """)
    if Naive_Bayes == 'Replace Classifier With 0 and 1':
        df = df.drop('Unnamed: 32', axis=1)
        st.markdown("# Replace Classifier Part 1")
        st.markdown("""
                    Binary classification uses 0's and 1's to perform the classification analysis.  
                    0 represents the negative value and 1 represents the positive value.  A majority of datasets
                    are going to contain the qualitative classifier instead of the 0 and 1 values.  As such, when
                    performing classification, it is necessary to write a code that will replace the classifiers
                    in the dataset.    
                    """)
        if st.button('Click for Part 1 Code'):
            st.write("df.diagnosis.replace(['M', 'B'], [1, 0], inplace = True)")
        df.diagnosis.replace(["M", "B"], [1, 0], inplace=True)
        if st.button('Part 1 Explanation'):
            st.write("""
                    For the breast cancer dataset used in this tutorial, the goal is to classify 'Benign' (i.e., 1, 
                    the negative value) and 'Malignant' (i.e., 0, the positive value) tumors.  These are represented 
                    by 'M' and 'B' in the dataset under the 'diagnosis' column.  If performing classification analysis 
                    on a different dataset, determine your dataset's classifiers and its respective column name.  For 
                    best practice, place the positive value first within the brackets.
                    """)
        st.markdown("# Replace Classifier Part 2")
        st.markdown("""
                    Part 2 is for you to check that your classifiers have been replaced with the 0 and 1 values.      
                    """)
        if st.button('Click for Part 2 Code'):
            st.write("df.head()")
        st.text(df.head())
        if st.button('Part 2 Explanation'):
            st.write("""
                    We can see that the diagnosis column has been replaced with 0's and 1's, meaning that our 
                    replace() function was successful.
                    """)
        st.markdown("# Replace Classifier Part 3")
        st.markdown("""
                    Part 3 is for you to see the counts of your 0 and 1 classifiers.  It is a best practice to have
                    a good representation of both classifiers.  If there is large discrepancy between classifier counts percentage wise, 
                    perhaps look for a different dataset.      
                    """)
        if st.button('Click for Part 3 Code'):
            st.write("df.groupby(['diagnosis']).diagnosis.count()")
        st.text(df.groupby(["diagnosis"]).diagnosis.count())
        if st.button('Part 3 Explanation'):
            st.write("""
                    We can see that there are 212 Benign (1) tumors and 357 Malignant (0) 
                    tumors in the breast cancer dataset.  This is an acceptable representation of 
                    both classifiers, thus we can proceed with this dataset. 

                    Remember, if you are performing classification analysis on a different dataset, 
                    insert your classifier column name into the Part 3 code above. 
                    """)
        st.markdown("# Replace Classifier Part 4")
        st.markdown("""
                    Part 4 is for you visualize the counts of your classifiers.  When working with data, it is always
                    best to visualize if you can!  Charts, graphs, maps, etc help the viewer to understand the data in
                    a more clear and efficient way.      
                    """)
        if st.button('Click for Part 4 Code'):
            st.write("sns.countplot('diagnosis', data=df)")
            st.write("plt.show()")
        sns.countplot('diagnosis', data=df)
        st.pyplot(plt.show())
        if st.button('Part 4 Explanation'):
            st.write("""
                    We can see that there is a good representation of both classifiers, where 0=malignant
                    and 1=benign. 

                    Remember, if you are performing classification analysis on a different dataset, 
                    insert your classifier column name into the Part 4 code above. 
                    """)
        st.markdown(""
                    "")
        st.markdown("# Congrats 🎉!!!")
        st.markdown(""" 
                    You have completed Naive Bayes, Replace Classifier!!!
                    You are ready to move on to the 'Test and Training Data' step!!!
                    """)
    if Naive_Bayes ==  'Create and fit models to test and training data':
        ##
        df.isnull().any()
        df = df.drop('Unnamed: 32', axis=1)
        df.diagnosis.replace(["M", "B"], [1, 0], inplace=True)
        st.markdown("# Test and Training Data Part 1")
        st.markdown("""
                    Part 1 ensures that X only takes on the independent variables in the dataset.      
                    """)
        if st.button('Click for Part 1 Code'):
            st.write("X = df.drop(['id','diagnosis'],axis=1)")
        if st.button('Part 1 Explanation'):
            st.write("""
                    Since the diagnosis column contains the classifiers, we want to remove this column from the X variable.  We 
                    won't neglect the diagnosis column for long, however, as we will utilize these values for the y variable!

                    We remove the id column as it is just an identifier, and does not play a role in tumor classification.  

                    If working with your own dataset, make sure you remove the irrelevant columns pertaining to your dataset!
                    """)
        X = df.drop(['id', 'diagnosis'], axis=1)
        st.markdown("# Test and Training Data Part 2")
        st.markdown("""
                    Part 2 ensures that Y only takes on the classifier column in the dataset.      
                    """)
        if st.button('Click for Part 2 Code'):
            st.write("Y = df.iloc[:, 1]")
        if st.button('Part 2 Explanation'):
            st.write("""
                    Since the diagnosis column contains the classifiers, we consider this the dependent variable column.  

                    If working with your own dataset, make sure you do some research on indexes so that you can accurately 
                    store your classifier column!
                    """)
        Y = df.iloc[:, 1]
        st.markdown("# Test and Training Data Part 3")
        st.markdown("""
                    Part 3 splits the data into test and training sets using the X and Y variables we created previously.       
                    """)
        if st.button('Click for Part 3 Code'):
            st.write("xTrain, xTest, yTrain, yTest = sklearn.model_selection.train_test_split(X, Y, train_size=0.8, random_state =20221023)")
        xTrain, xTest, yTrain, yTest = sklearn.model_selection.train_test_split(X, Y, train_size=0.8, random_state=20221023)
        if st.button('Part 3 Explanation'):
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
        st.markdown("# Test and Training Data Part 4")
        st.markdown("""
                    Part 4 creates a model that performs Naive Bayes Classification.      
                    """)
        if st.button('Click for Part 4 Code'):
            st.write("NB = GaussianNB()")
        NB = GaussianNB()
        st.markdown("# Test and Training Data Part 5")
        st.markdown("""
                    Part 5 fits the Naive Bayes model to the training set.  This allows the model to "study"
                    the data and "learn" from it.  This step will not return any output, but will be used later on 
                    in the classification process.       
                    """)
        if st.button('Click for Part 5 Code'):
            st.write("NB.fit(xTrain,yTrain)")
        NB.fit(xTrain, yTrain)
        st.markdown("# Test and Training Data Part 6")
        st.markdown("""
                    Now that we have created our model and trained it, Part 6 will use our testing dataset to test the model.  
                    This step will not return any output, but will be used later on in the classification process.
                    """)
        if st.button('Click for Part 6 Code'):
            st.write("predNB =NB.predict(xTest)")
        predNB = NB.predict(xTest)
        st.markdown(""
                    "")
        st.markdown("# Congrats 🎉!!!")
        st.markdown(""" 
                    You have completed Naive Bayes, Test and Training Data!!!
                    You are ready to move on to the 'Accuracy Score' step!!!
                    """)
    if Naive_Bayes == 'Accuracy Score':
        df = df.drop('Unnamed: 32', axis=1)
        df.diagnosis.replace(["M", "B"], [1, 0], inplace=True)
        X = df.drop(['id', 'diagnosis'], axis=1)
        Y = df.iloc[:, 1]
        xTrain, xTest, yTrain, yTest = sklearn.model_selection.train_test_split(X, Y, train_size=0.8, random_state=20221023)
        NB = GaussianNB()
        NB.fit(xTrain, yTrain)
        predNB = NB.predict(xTest)
        st.markdown("# Accuracy Score Part 1")
        st.markdown("""
                    Part 1 allows you to see the actual and predicted classifiers in array form.  Can you spot the 
                    differences between the actual and predicted set?      
                    """)
        if st.button('Click for Part 1 Code'):
            st.write("print(yTest.values)")
            st.write("print(predNB)")
        st.text("Actual breast cancer : ")
        st.text(yTest.values)
        st.text("\nPredicted breast cancer : ")
        st.text(predNB)
        st.markdown("# Accuracy Score Part 2")
        st.markdown("""
                    Part 2 allows you to calculate the accuracy score.  As it may sound, the accuracy score declares how 
                    accurately the model classifies the classifiers.     
                    """)
        if st.button('Click for Part 2 Code'):
            st.write("accuracy_score(yTest, predNB) * 100")
        st.text("Accuracy Score: ")
        st.text(accuracy_score(yTest, predNB) * 100)
        if st.button("Part 2 Explanation"):
            st.write("""
                    Accuracy score is calculated by dividing the number of correct predictions by the total prediction number.
                    """)
        st.markdown("# Congrats 🎉!!!")
        st.markdown(""" 
                    You have completed Naive Bayes, Accuracy Score!!!
                    You are ready to move on to the 'Confusion Matrix' step!!!
                    """)
    if Naive_Bayes == 'Confusion Matrix':
        df = df.drop('Unnamed: 32', axis=1)
        df.diagnosis.replace(["M", "B"], [1, 0], inplace=True)
        X = df.drop(['id', 'diagnosis'], axis=1)
        Y = df.iloc[:, 1]
        xTrain, xTest, yTrain, yTest = sklearn.model_selection.train_test_split(X, Y, train_size=0.8, random_state=20221023)
        NB = GaussianNB()
        NB.fit(xTrain, yTrain)
        predNB = NB.predict(xTest)
        plt.figure(figsize=(16, 14))
        plt.subplot(3, 3, 3)
        sns.heatmap(sklearn.metrics.confusion_matrix(yTest, predNB), annot=True).set(title='Naive Bayes')
        st.set_option('deprecation.showPyplotGlobalUse', False)
        st.pyplot(plt.show())
        predVals = pd.DataFrame(data={'truth': yTest, 'KNN': predNB})

if selected == "About":
    st.markdown("# Meet the Classification Central Developers!")
    st.markdown("""
                """)
    st.markdown("""
                Classification Central was created as part of a Comp-Sci 5530 project for UMKC...
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
    st.markdown("# Insert Name")
    st.markdown("## Insert Short Description")
    st.markdown("""
            Insert Bio Paragraph 
            """)
    st.markdown("# Insert Name")
    st.markdown("## Insert Short Description")
    st.markdown("""
                Insert Bio Paragraph 
                """)
    st.markdown("# Insert Name")
    st.markdown("## Insert Short Description")
    st.markdown("""
                Insert Bio Paragraph 
                """)
if selected == "Home":
    st.markdown("""
        Too often, data scientists provide their code but don't provide explanations of what they are doing or what their results say. 
        Classification Central seeks to provide clear explanations of 4 common classification techniques: Logistic Regression, K Nearest Neighbors (KNN),
        Random Forest, and Naive Bayes.  We provide a tutorial using a breast cancer dataset, and provide the code in hopes that you can perform
        classification on your own data set by making slight tweaks.  We also provide detailed explanations of what the code is doing, in hopes that
        you can gain a better understanding of each classification step.
        """)
    st.markdown("# The following are summaries of the classification techniques covered by Classification Central:")
    st.markdown("""
        
                    """)
    st.markdown("# Logistic Regression")
    st.markdown("""
                    Enter Log Regression Summary Here
                    """)
    st.markdown("""

                    """)
    st.markdown("# KNN")
    st.markdown("""
                    Enter KNN Summary Here
                    """)
    st.markdown("""

                    """)
    st.markdown("# Random Forest")
    st.markdown("""
                    Enter Random Forest Summary Here
                    """)
    st.markdown("""

                    """)
    st.markdown("# Naive Bayes")
    st.markdown("""
                    Enter Naive Bayes Summary Here
                    """)






