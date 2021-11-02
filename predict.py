import pickle
import xgboost as xgb
from flask import Flask, jsonify, request

model_file = "model.bin"

with open(model_file, "rb") as f_in:
    dv, model = pickle.load(f_in)

app = Flask("diabeties")


@app.route("/predict", methods=['POST'])
def predict():
    person = request.get_json()

    X = dv.transform([person])
    dX = xgb.DMatrix(X, feature_names=dv.get_feature_names())
    y_pred = model.predict(dX)[0]
    heartdisease = y_pred >= 0.5

    result = {
        "heartdisease_probability": float(y_pred),
        "heartdisease": bool(heartdisease),
    }

    return jsonify(result)

@app.route('/test', methods = ['GET'])
def test():
    return '<h1>Hello to Diabetes Prediction Server</h1>'

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=9696)