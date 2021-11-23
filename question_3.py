import pickle

from flask import Flask
from flask import request
from flask import jsonify

no_1 = 'model1.bin'

with open(no_1, 'rb') as f_in:
    model = pickle.load(f_in)

no_2 = 'dv.bin'

with open(no_2, 'rb') as f_in:
    dv = pickle.load(f_in)

customer = {"contract": "two_year", "tenure": 12, "monthlycharges": 19.7}
X = dv.transform([customer])
y_pred = model.predict_proba(X)[0, 1]
print(y_pred)



