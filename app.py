from flask import Flask, request, jsonify
import numpy as np
import joblib
from tensorflow.keras.models import load_model

app = Flask(__name__)

# Load the model and scaler once when the server starts
model = load_model('stacked_lstm_model.keras')
scaler = joblib.load('scaler.save')
seq_length = 60  # same sequence length used in training

@app.route('/predict', methods=['POST'])
def predict():
    """
    Expects JSON input:
    {
        "recent_prices": [list of 60 recent closing prices],
        "days": number of days to predict (optional, default 1)
    }
    Returns predicted closing prices for the next N days.
    """

    data = request.get_json(force=True)

    recent_prices = data.get('recent_prices')
    days = data.get('days', 1)  # default to 1 day if not specified

    # Validate input
    if recent_prices is None:
        return jsonify({"error": "Missing 'recent_prices' in request"}), 400

    if len(recent_prices) != seq_length:
        return jsonify({"error": f"'recent_prices' length must be {seq_length}"}), 400

    try:
        recent_prices = np.array(recent_prices).reshape(-1, 1).astype(float)
    except Exception as e:
        return jsonify({"error": "Invalid 'recent_prices' data. Must be list of numbers."}), 400

    # Scale the input prices
    input_scaled = scaler.transform(recent_prices).flatten().tolist()

    predictions_scaled = []

    for _ in range(days):
        # Prepare input for model [1, seq_length, 1]
        X_input = np.array(input_scaled[-seq_length:]).reshape(1, seq_length, 1)

        # Predict scaled price
        pred_scaled = model.predict(X_input)[0, 0]

        predictions_scaled.append(pred_scaled)

        # Append prediction to input for next step
        input_scaled.append(pred_scaled)

    # Inverse scale predictions to original price scale
    predictions_scaled = np.array(predictions_scaled).reshape(-1, 1)
    predictions = scaler.inverse_transform(predictions_scaled).flatten()

    # Convert to list for JSON serialization
    predicted_prices = predictions.tolist()

    return jsonify({
        "predicted_prices": predicted_prices
    })

if __name__ == '__main__':
    # Run the Flask app
    app.run(debug=True)
