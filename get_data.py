import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb
pd.set_option('display.max_columns', 500)
from sklearn.feature_selection import SelectKBest,chi2
df=pd.read_csv('./rawdata_new.csv')



# processed_df=SelectKBest(chi2,k=30).fit_transform()
length=len(df)
store_cancel=[]
for col in df.columns:
    if df[col].isnull().sum()/length >= 0.70:
        store_cancel.append(col)

df=df.drop(store_cancel,axis=1)

only_categorical=list(df.select_dtypes(include=['O']).columns)
for col in only_categorical:
    codes,uniques=pd.factorize(df[col])
    df[col]=codes
    
all_features = df.columns
names = [feat for feat in all_features if "net_name" in feat] # excluded for privacy reasons
useless = ["info_gew","info_resul","interviewtime","id","date"] # features that we expect are uninformative
drop_list = names + useless 

# Remove the questionnaire about agricultural practices until I can better understand it
practice_list = ["legum","conc","add","lact","breed","covman","comp","drag","cov","plow","solar","biog","ecodr"]
for feat in all_features:
    if any(x in feat for x in practice_list):
        drop_list.append(feat)


df = df.drop(columns=drop_list)
# print(df.head(2))
df.to_csv("data_processed.csv")