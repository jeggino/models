from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn import tree
import numpy as np
import streamlit as st


uploaded_file = st.file_uploader("Choose a file")
if uploaded_file is not None:
  dataframe = pd.read_csv(uploaded_file)
  
  dataframe

else:
  st.stop()

st.write(dataset.columns)

option = st.selectbox(
   "Chose the Label variable",
  list(dataset.columns),
   index=None,
   placeholder="Select a label...",
)

st.write('You selected:', option)


