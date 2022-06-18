import pandas as pd
import numpy as np
import streamlit as st
from xgboost import XGBClassifier
import pickle
import sklearn

header = st.container()
dataset = st.container()
modeling = st.container()

with header : 
    st.title("The program is to predict whether a person earns less than 50 thousand in a year")
    st.markdown("* **First Section** is about to understanding the Dataset")
    st.markdown("* **Second Section** is about to give inputs and see the prediction result")
with dataset :
    st.header("The Income Dataset")
    df = pd.read_csv("X_test_Dataset")
    st.write(df.head())
    
    

    set_col , disp_col=st.columns(2)
    st.sidebar.header("Inputs Giving")
    Age = st.sidebar.slider("Please choose input for the Age variable" , min_value = 17 , max_value = 90 , value = 17 , step = 1)
    Workclass = st.sidebar.selectbox("Plese choose input for the Workclass variable" , options = ["State-gov" , "Self-emp-not-inc" , "Private" , "Private" , "Federal-gov" , "Local-gov" , "Self-emp-inc" , "Without-pay" , "Never-worked"] , index = 0)
    Education = st.sidebar.selectbox("Plese choose input for the Education variable" , options = ["Bachelors" , "HS-grad" , "11th" , "Masters" , "9th" , "Some-college" , "Assoc-acdm" , "Assoc-voc" , "7th-8th" , "Doctorate" , "Prof-school" , "5th-6th" , "10th" , "1st-4th" , "Preschool" , "12th"] , index = 0)
    Education_num = st.sidebar.slider("Please choose input for the Education number variable" , min_value = 1 , max_value = 16 , value = 1 , step = 1)
    Marital_status = st.sidebar.selectbox("Plese choose input for the Marital status variable" , options = ["Never-married" , "Married-civ-spouse" , "Divorced" , "Married-spouse-absent" , "Separated" , "Married-AF-spouse" , "Widowed"] , index = 0)
    Occupation = st.sidebar.selectbox("Plese choose input for the Occupation variable" , options = ["Adm-clerical" , "Exec-managerial" , "Handlers-cleaners" , "Prof-specialty" , "Other-service" , "Sales" , "Craft-repair" , "Transport-moving" , "Farming-fishing" , "Machine-op-inspct" , "Tech-support" , "Protective-serv" , "Armed-Forces" , "Priv-house-serv"] , index = 0)
    Relationship = st.sidebar.selectbox("Plese choose input for the Relationship variable" , options = ["Not-in-family" , "Husband" , "Wife" , "Own-child" , "Unmarried" , "Other-relative"] , index = 0)
    Race = st.sidebar.selectbox("Plese choose input for the Race variable" , options = ["White" , "Black" , "Asian-Pac-Islander" , "Amer-Indian-Eskimo" , "Other"] , index = 0)
    Sex = st.sidebar.radio("Plese choose input for the Sex variable : Male = 1 , Female = 0" , options = [0 , 1] , index = 0)
    Capital_gain = st.sidebar.slider("Please choose input for the Capital Gain variable" , min_value = 0 , max_value = 99999 , value = 0 , step = 1)
    Capital_loss = st.sidebar.slider("Please choose input for the Capital Loss variable" , min_value = 0 , max_value = 4356 , value = 0 , step = 1)
    Hours_per_week = st.sidebar.slider("Please choose input for the Hours_per_week variable" , min_value = 1 , max_value = 99 , value = 1 , step = 1)
    Native_country = st.sidebar.radio("Plese choose input for the native country variable" , options = ["United-States" , "Others"] , index = 0)

with modeling : 
    set_col.header("Inputs")
    set_col.markdown("* You have entered these inputs")
    input_data = pd.DataFrame(data ={"Age" : [Age] , "Workclass" : [Workclass] , "Education" : [Education] ,"Education-num" : [Education_num] ,  "Marital-status" : [Marital_status] , 
         "Occupation" :[Occupation] , "Relationship" : [Relationship] , "Race" : [Race] , "Sex" : [Sex] , "Capital-gain" : [Capital_gain],"Capital-loss" : [Capital_loss], "Hours-per-week" : [Hours_per_week], "Native-country" : [Native_country]})
    st.write(input_data)
    
    btn = st.sidebar.button("PREDICT")
    st.header("**The Prediction Result**")
    
    if btn : 
    
        model = pickle.load(open("MoneyProject" , "rb"))
        y_pred = model.predict(pd.DataFrame(data = {"Age" : [Age] , "Workclass" : [Workclass] , "Education" : [Education] ,"Education-num" : [Education_num] ,  "Marital-status" : [Marital_status] , 
         "Occupation" :[Occupation] , "Relationship" : [Relationship] , "Race" : [Race] , "Sex" : [Sex] , "Capital-gain" : [Capital_gain],"Capital-loss" : [Capital_loss], "Hours-per-week" : [Hours_per_week], "Native-country" : [Native_country]}))
        
        y_pred_proba = model.predict_proba(pd.DataFrame(data = {"Age" : [Age] , "Workclass" : [Workclass] , "Education" : [Education] ,"Education-num" : [Education_num] ,  "Marital-status" : [Marital_status] , 
         "Occupation" :[Occupation] , "Relationship" : [Relationship] , "Race" : [Race] , "Sex" : [Sex] , "Capital-gain" : [Capital_gain],"Capital-loss" : [Capital_loss], "Hours-per-week" : [Hours_per_week], "Native-country" : [Native_country]}))
        
        
        if y_pred == [1] : 
            st.markdown("This person earns **LESS** than 50 thousand in a year and probablity is {:.2%}".format(y_pred_proba[: , 1][0]))
        else : 
            st.markdown("This person earns **GREATER** than 50 thousand in a year and probablity is {:.2%}".format(y_pred_proba[:,0][0]))
