from flask import Flask, request, jsonify
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression

app = Flask(__name__)

# Load and train model
X, y = load_iris(return_X_y=True)
model = LogisticRegression(max_iter=200)
model.fit(X, y)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json.get("features")
        if not data:
            return jsonify({"error": "No features provided"}), 400
        prediction = model.predict([data])[0]
        return jsonify({"prediction": int(prediction)})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
