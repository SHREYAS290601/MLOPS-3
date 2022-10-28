import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb
import json
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble.RandomForestClassifier
from sklearn import preprocessing
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
pd.set_option('display.max_columns', 500)
df=pd.read_csv('data_processed.csv')
si=SimpleImputer(missing_values=np.nan,strategy='mean')
y=[1 if val >=4 else 0 for val in df.cons_general]
X = df.to_numpy()
X = preprocessing.scale(X)
si.fit(X)
X = si.transform(X)
clf = RandomForestClassifier()
y_pred = cross_val_predict(clf, X, y, cv=5)
acc=np.mean(y_pred==y)
tn, fp, fn, tp = confusion_matrix(y, y_pred).ravel()
specificity = tn / (tn+fp)
sensitivity = tp / (tp + fn)

with open("metrics.json", 'w') as outfile:
        json.dump({ "Acc": acc, "Spec": specificity, "Sens":sensitivity}, outfile)
score = y_pred == y
score_int = [int(s) for s in score]
df['pred_acc'] = score_int

# Bar plot by region
ax = sb.barplot(x="region", y="pred_acc", data=df, palette = "Blues")
ax.set(xlabel="Region", ylabel = "Model accuracy")
plt.savefig("by_region.png",dpi=80)
# print(df.groupby('region').pred_acc.value_counts().unstack())
