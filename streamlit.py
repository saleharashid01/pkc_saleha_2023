import pandas as pd
import streamlit as st
from sklearn.ensemble import RandomForestClassifier
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score, f1_score, recall_score
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
import joblib 


st. write("""
# Best Model Classifier App
This App will predict the Best Model for the data 
### by Saleha Rasid""")

st.sidebar.header("Models List")
option = st.sidebar.selectbox(
   '# Select the Best Model',
   ('Linear Regression', 'Random Forest Classifier', 'KNN Classifer', 'Logistc Regression', ' Decision Tree Classifier', 'SVMs', 'Naive Bayes','Gradient Boosting'))
st.write('You selected:', option)

st.sidebar.subheader("Data Points")
def user_input_features():
    
    people_fully_vaccinated=st.sidebar.slider('People Fully Vaccinated', 4000, 78000000, 4000)
    New_deaths=st.sidebar.slider('New Deaths', 0, 1400, 1)
    data= {
        'People Fully Vaccinated':people_fully_vaccinated,
          'New Deaths': New_deaths  }
    feature =pd.DataFrame(data, index=[0])
    return feature
df= user_input_features()

st.subheader('COVID-19 Parameters')
st.write(df)

covid = pd.read_excel('data_covid.xlsx')
st.subheader('COVID-19 Dataset')
st.write(covid)
data_sets = st.container()
with data_sets:
 
    st.subheader("New Deaths")
    st.bar_chart(covid['New_deaths'].value_counts())
    st.subheader('People Fully Vaccinated')
    st.bar_chart(covid['people_fully_vaccinated'].value_counts())
    
import streamlit as st
import pandas as pd

# Load Data

def load_data():
    df = pd.read_excel('data_covid.xlsx')
    return df
data = load_data()
# Dropdown Menu for Country
country = st.sidebar.selectbox(
    'Choose a Country',
    data['country'].unique())

# Filter Data
selected_country = data[data['country'] == country]
st.header('EDA ANALYSIS')
# Show New Deaths
st.subheader('New Deaths')
st.bar_chart(selected_country['New_deaths'].value_counts())
st.line_chart(selected_country['New_deaths'].value_counts())
# Show People Fully Vaccinated
st.subheader('people_fully_vaccinated')
st.bar_chart(selected_country['people_fully_vaccinated'].value_counts())
st.line_chart(selected_country['people_fully_vaccinated'].value_counts())

st.write('## After Normalizing the Data')
def load_data():
    new1 = pd.read_csv('covid_preprocessed_data.csv')
    return new1

# Show New Deaths
st.subheader('New Deaths')
st.bar_chart(selected_country['New_deaths'].value_counts())
st.line_chart(selected_country['New_deaths'].value_counts())
# Show People Fully Vaccinated
st.subheader('people_fully_vaccinated')
st.bar_chart(selected_country['people_fully_vaccinated'].value_counts())
st.line_chart(selected_country['people_fully_vaccinated'].value_counts())



