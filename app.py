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
  dataset = pd.read_csv(uploaded_file)
  
  dataset

else:
  st.stop()


label = st.selectbox(
   "Chose the Label variable",
  list(dataset.columns),
   index=None,
   placeholder="Select a label...",
)

if label is not None:
  df_grouped_by = dataset.groupby([label])

else:
  st.stop()

# balance the dataset
df_grouped_by = dataset.groupby([label])
df_balanced = df_grouped_by.apply(lambda x: x.sample(df_grouped_by.size().min()).reset_index(drop=True))
df_balanced = df_balanced.droplevel([label])

le = LabelEncoder()
Y = le.fit_transform(df_balanced.pop(label))


df_cont = df_balanced.select_dtypes(exclude="object")
df_lab = df_balanced.select_dtypes(include="object")

encoder = OneHotEncoder()
standard = StandardScaler()


if df_lab is None and df_cont is not None:
    
    X = standard.fit_transform(df_cont) 
    
elif df_lab is not None and df_cont is None:
    
    X = encoder.fit_transform(df_lab).toarray()
    
else:
    
    df_cont = standard.fit_transform(df_cont)
    df_lab = encoder.fit_transform(df_lab).toarray()
    X = np.concatenate((df_cont,df_lab), axis=1)
    


# Create the models
dict_df = {}

dict_model = {"XGBClassifier":XGBClassifier(),
              "DecisionTreeClassifier":tree.DecisionTreeClassifier(),
              "HistGradientBoostingClassifier":HistGradientBoostingClassifier(max_iter=100),
              "GaussianNB":GaussianNB(),
              "RandomForestClassifier":RandomForestClassifier()}


for models in dict_model.keys():
    
    dict_df[models] = {}
    

    for test_size in [0.1,0.2,0.3]:
        
        ac = []
        
        for _ in range(5):

            # split data into train and test sets
            X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=test_size, random_state=None)
              

            # fit model no training data
            model = dict_model[models]
            model.fit(X_train, y_train)

            # make predictions for test data
            y_pred = model.predict(X_test)

            # evaluate predictions
            accuracy = accuracy_score(y_test, y_pred)
            
            ac.append(accuracy)

        dict_df[models][test_size] = ac

dict_of_df = {k: pd.DataFrame(v) for k,v in dict_df.items()}
df = pd.concat(dict_of_df, axis=1)
df_describe = df.describe().round(3)

df_describe

