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


  
uploaded_file = st.sidebar.file_uploader("Choose a file")

if uploaded_file is not None:

  dataset = pd.read_csv(uploaded_file)

else:
  st.stop()


label = st.sidebar.selectbox(
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


if (len(df_lab.columns) ==  0) and (len(df_cont.columns) != 0):
    
    X = standard.fit_transform(df_cont) 
    
if (len(df_lab.columns) !=  0) and (len(df_cont.columns) == 0):
    
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

"---"

import altair as alt

source = df.melt(var_name=["Model","Test size"],value_name="Accuracy")

source["Accuracy"] = source["Accuracy"]*100

# box = alt.Chart(source).mark_boxplot(extent='min-max',size=30,).encode(
#     x=alt.X("Test size:N",title=""),
#     y=alt.Y('Accuracy:Q',scale=alt.Scale(zero=False),title=""),
#     color=alt.Color("Model:N").legend(None),
#     column='Model:N'
# ).properties(width=150,height=300).configure_axis(
#     labelFontSize=10,
# )

# st.altair_chart(box, use_container_width=True)

"---"

# f"""
# Highest mean: {df_describe.idxmax(axis="columns")["mean"]} ({df_describe.loc["mean",df_describe.idxmax(axis="columns")["mean"]]}) 
# Highest min: {df_describe.idxmax(axis="columns")["min"]} ({df_describe.loc["min",df_describe.idxmax(axis="columns")["min"]]}) 
# Highest max: {df_describe.idxmax(axis="columns")["max"]} ({df_describe.loc["max",df_describe.idxmax(axis="columns")["max"]]})
# Lowest std: {df_describe.idxmin(axis="columns")["std"]} ({df_describe.loc["std",df_describe.idxmin(axis="columns")["std"]]}) 
# """

"""
ksjhdbckasjhbc
akdsclasc
cachas
"""
"---"

mod = st.sidebar.selectbox(
  "Chose model",
  list(dict_model.keys()),
  index=None,
  placeholder="Select a model...",
)

if mod is not None:
  model = dict_model[mod]

else:
  st.stop()
  
