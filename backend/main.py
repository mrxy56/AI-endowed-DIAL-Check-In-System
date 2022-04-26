import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from NN import Net
from SVM import SVM_DUAL

train_df = pd.read_excel('covid.xlsx')
model = LogisticRegression(solver='liblinear')
valid_col = ['Fever', 'Cough', 'Fatigue', 'Breathing', 'Ache', 'Headache', 'LossTaste']
target_col = 'Pred'
tr = train_df[valid_col]
y = train_df[target_col]
print(tr[:30])
model.fit(tr[:30], y[:30])
X_train = tr[:30]
train_preds = model.predict(X_train)
print(train_preds)
#net = Net() #TBA
#svm = SVM_DUAL() #TBA