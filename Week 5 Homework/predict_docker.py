import pickle

from flask import Flask
from flask import request
from flask import jsonify

no_1 = 'model2.bin'

with open(no_1, 'rb') as f_in:
    model = pickle.load(f_in)

no_2 = 'dv.bin'

with open(no_2, 'rb') as f_in:
    dv = pickle.load(f_in)

app = Flask('churn')


@app.route('/predict', methods=['POST'])
def predict():
    customer = request.get_json()

    X = dv.transform([customer])
    y_pred = model.predict_proba(X)[0, 1]

    result = {
        'churn_probability': float(y_pred)
    }

    return jsonify(result)


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=9696)
