import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
from joblib import dump
import numpy as np

df = pd.read_csv('dataset.csv')
df = df.sample(frac=1,random_state=42).reset_index(drop=True)

cols = ['temp', 'humidity','sealevelpressure', 'cloudcover']
df = df[cols + ['precipprob']]

scaler = MinMaxScaler()
df[cols] = scaler.fit_transform(df[cols])

split_index = int(0.85 * len(df))
train_df = df[:split_index]

train_target = train_df.pop('precipprob')
train_X = train_df.values
train_y = train_target.values

test_df = df[split_index:]
test_target = test_df.pop('precipprob')
test_X = test_df.values
test_y = test_target.values

model = LogisticRegression(solver='liblinear')
model.fit(train_X, train_y)
dump(model, 'logistic_model.joblib')
dump(scaler, 'scaler.joblib')


out = model.predict(test_X)

correct = 0
true_pos = 0
true_neg = 0
false_pos = 0
false_neg = 0
total = len(test_y)

for i in range(len(test_y)):
    if out[i] == test_y[i]:
        correct += 1
    if out[i] == 1 and test_y[i] == 1:
        true_pos += 1
    elif out[i] == 0 and test_y[i] == 0:
        true_neg += 1
    elif out[i] == 1 and test_y[i] == 0:
        false_pos += 1
    elif out[i] == 0 and test_y[i] == 1:
        false_neg += 1
        
print(f'Accuracy: {correct/total:.2f} ({correct}/{total})')
print(f'True Positives: {true_pos}({true_pos/total:.2f}), True Negatives: {true_neg}({true_neg/total:.2f})')
print(f'False Positives: {false_pos}({false_pos/total:.2f}), False Negatives: {false_neg}({false_neg/total:.2f})')
