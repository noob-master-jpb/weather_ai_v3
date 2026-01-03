import argparse
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
from joblib import load
import numpy as np


model = load('logistic_model.joblib')
scaler = load('scaler.joblib')

parser = argparse.ArgumentParser(description='Predict precipitation probability.')
parser.add_argument(
    '-d', '--data',
    nargs='+',
    type=float,
    required=True,
    help='Space-separated list of feature values'
)
parser.add_argument(
    '-v', '--verbose',
    type=bool,
    help='Enable verbose output'
)
parser.add_argument(
    '-w', '--weights',
    type=bool,
    help='Show model weights'
)

args = parser.parse_args()
features = pd.DataFrame([args.data], columns=['temp', 'humidity', 'sealevelpressure', 'cloudcover'])
scaled = scaler.transform(features)
prediction = model.predict(scaled)[0]

print(f'Prediction: {"Rain" if prediction == 1 else "No Rain"}')
if args.verbose:
    print(f'Scaled Features: {scaled[0]}')
    print(f"probability: {model.predict_proba(scaled)[0]}")
if args.weights:
    print(f'Model Coefficients: {model.coef_}, Intercept: {model.intercept_}')
    