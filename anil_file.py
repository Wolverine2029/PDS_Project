import os
import streamlit as st
import pandas as pd
from streamlit_option_menu import option_menu
from classificationCode import logistic_Regression_model

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

classification = st.selectbox('Select one option:', ['', 'Logistic Regression', 'KNN', 'Naive Bayes', 'Random Forest'], format_func=lambda x: 'Select an option' if x == '' else x)

if classification == 'Logistic Regression':
    st.write("LOL")
    # lr_fit, lr_score, lr_predict, lr_accuracy, heatmap, cf_matrix = logistic_Regression_model()
    # st.write("Classifier LR fit: ")
    # st.write(lr_fit)
    # st.write("Classifier LR Score: ")
    # st.write(lr_score)
    # st.write("Classifier LR Prediction: ")
    # st.write(lr_predict)
    # st.write("Classifier LR Accuracy: ")
    # st.write(lr_accuracy)
    # st.write("Classifier LR HeatMap: ")
    # st.write(heatmap)
    # st.write("Classifier Confusion Matrix: ")
    # st.write(cf_matrix)

# ALLY CODE

# This is a sample Python script.

# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.


def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press ⌘F8 to toggle the breakpoint.


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print_hi('PyCharm')

# See PyCharm help at https://www.jetbrains.com/help/pycharm/

##this file is the streamlit syntax file that uploads into the UI

#streamlit run /Users/allyryan/PycharmProjects/pythonProject10/mainGG.py
# Here I am attempting to make 2 pages on the UI.  I haven't gotten it to work yet.
#import streamlit as st
#import tutorial5530 as app1
#import tryityourself5540 as app2

#PAGES = {
 #   "Tutorial": app1,
  #  "Try it yourself": app2
#}
#st.sidebar.title('Navigation')
#selection = st.sidebar.radio("Go to", list(PAGES.keys()))
#page = PAGES[selection]
#page.app()


import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn.metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from  sklearn.model_selection import GridSearchCV

#put in your own file path name here
df = pd.read_csv("/Users/allyryan/Downloads/data.csv")

# st.title('Classification Tutorial')
# st.text('Print Dataframe')
# st.text("code-> print(df)")
# st.text(df)


st.text("code-> df.head()")
st.text(df.head())
st.text("code-> df.isnull().any()")
st.text(df.isnull().any())
st.text("code-> df.drop('Unnamed: 32', axis=1)")
df = df.drop('Unnamed: 32', axis=1)
st.text("code-> df.head()")
st.text(df.head())
st.text("code-> df.describe()")
st.text(df.describe())
st.text("code-> df.info()")
st.text(df.info())
st.text("code-> df.diagnosis.replace(['M', 'B'], [1, 0], inplace = True)")
st.text(df.diagnosis.replace(["M", "B"], [1, 0], inplace = True))
st.text("code-> df.head()")
st.text(df.head())
st.text("code-> df.groupby(['diagnosis']).diagnosis.count()")
st.text(df.groupby(["diagnosis"]).diagnosis.count())


plt.figure(figsize=(12,8))
data = df.corr()["diagnosis"].sort_values(ascending=False)
indices = data.index
labels = []
corr = []
for i in range(1, len(indices)):
    labels.append(indices[i])
    corr.append(data[i])
sns.barplot(x=corr, y=labels, palette='viridis')
plt.title('Correlation coefficient between different features and Label')
st.set_option('deprecation.showPyplotGlobalUse', False)
st.pyplot(plt.show())

sns.countplot('diagnosis', data=df) #1=M,0=B
st.pyplot(plt.show())

###
X = df.drop(['id','diagnosis'],axis=1)
Y = df.iloc[:,1]

### Global
xTrain, xTest, yTrain, yTest = sklearn.model_selection.train_test_split(X,Y, train_size=0.8)
(xTrain.shape, yTrain.shape)

####
def logistic_Regression_model():
    regressionModel = LogisticRegression(solver='newton-cg')
    lr_fit = regressionModel.fit(xTrain,yTrain)
    lr_score = regressionModel.score(xTrain,yTrain)

    predRegression =regressionModel.predict(xTest)

    # st.text("Actual breast cancer : ")
    # st.text(yTest.values)

    # st.text("\nPredicted breast cancer : ")
    # st.text(predRegression)

    lr_accuracy = (accuracy_score(yTest, predRegression) * 100)
    plt.figure(figsize=(16, 14))
    plt.subplot(3, 3, 1)
    heatmap_lr = sns.heatmap(sklearn.metrics.confusion_matrix(yTest, predRegression), annot=True).set(title='Logistic Regression')
    classification_report = st.text(sklearn.metrics.classification_report(yTest, predRegression))
    return lr_fit, lr_score, predRegression, lr_accuracy. heatmap_lr, classification_report


###
KNNModel = KNeighborsClassifier(n_neighbors=5,leaf_size=1, weights='uniform')
KNNModel.fit(xTrain,yTrain)
KNNModel.score(xTrain,yTrain)


###
predKNN =KNNModel.predict(xTest)

st.text("Actual breast cancer : ")
st.text(yTest.values)

st.text("\nPredicted breast cancer : ")
st.text(predKNN)

st.text("\nAccuracy score : %f" %(accuracy_score(yTest, predKNN) * 100))

###
NB=GaussianNB()
NB.fit(xTrain,yTrain)
NB.score(xTrain,yTrain)

###
predNB =NB.predict(xTest)


###
st.text("Actual breast cancer : ")
st.text(yTest.values)

st.text("\nPredicted breast cancer : ")
st.text(predNB)

st.text("\nAccuracy score : %f" %(accuracy_score(yTest, predNB) * 100))


###
randomFModel = RandomForestClassifier()
randomFModel.fit(xTrain, yTrain)
randomFModel.score(xTrain,yTrain)

###
predRandomF=randomFModel.predict(xTest)


###
st.text("Actual breast cancer : ")
st.text(yTest.values)

st.text("\nPredicted breast cancer : ")
st.text(predRandomF)

st.text("\nAccuracy score : %f" %(accuracy_score(yTest, predRandomF) * 100))


###

plt.subplot(3, 3, 2)
sns.heatmap(sklearn.metrics.confusion_matrix(yTest, predKNN), annot=True).set(title='KNN')
plt.subplot(3, 3, 3)
sns.heatmap(sklearn.metrics.confusion_matrix(yTest, predRandomF), annot=True).set(title='Random Forest')
plt.subplot(3, 3, 4)
sns.heatmap(sklearn.metrics.confusion_matrix(yTest, predNB), annot=True).set(title='Naive Byes')
st.pyplot(plt.show())

###
predVals = pd.DataFrame(data={'truth': yTest, 'knn': predKNN,'random-forest': predRandomF,'Naive Byes':predNB})


###
st.text("code-> predVals.head()")
st.text(predVals.head())

###
model = KNeighborsClassifier()


# Tunning Params
param_grid = {
    'n_neighbors': list(range(1, 30)),
    'leaf_size': list(range(1,30)),
    'weights': [ 'distance', 'uniform' ]
}


# Implement GridSearchCV
gsc = GridSearchCV(model, param_grid, cv=20)

# Model Fitting
gsc.fit(xTrain, yTrain)

st.text("\n Best Score is ")
st.text(gsc.best_score_)

st.text("\n Best Estinator is ")
st.text(gsc.best_estimator_)

st.text("\n Best Parametes are")
st.text(gsc.best_params_)


###
model = LogisticRegression()

# Tunning Params
param_grid = {
    'C': [1,10,100,1000],
}


# Implement GridSearchCV
gsc = GridSearchCV(model, param_grid, cv=20)

# Model Fitting
gsc.fit(xTrain, yTrain)

st.text("\n Best Score is ")
st.text(gsc.best_score_)

st.text("\n Best Estinator is ")
st.text(gsc.best_estimator_)

st.text("\n Best Parametes are")
st.text(gsc.best_params_)


###
model = GaussianNB()

# Tunning Params
param_grid = {
    'var_smoothing':np.logspace(0,-9,num=100)
}


# Implement GridSearchCV
gsc = GridSearchCV(model, param_grid, cv=20,verbose=2)

# Model Fitting
gsc.fit(xTrain, yTrain)

st.text("\n Best Score is ")
st.text(gsc.best_score_)

st.text("\n Best Estimator is ")
st.text(gsc.best_estimator_)

st.text("\n Best Parametes are")
st.text(gsc.best_params_)

###


###
st.text("knn report")
st.text(sklearn.metrics.classification_report(yTest, predKNN))

###
st.text("naive bayes report")
st.text(sklearn.metrics.classification_report(yTest, predNB))

###
st.text("random forest report")
st.text(sklearn.metrics.classification_report(yTest, predRandomF))