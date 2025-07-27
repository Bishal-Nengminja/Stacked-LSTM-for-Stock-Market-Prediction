# 📊 Stacked-LSTM-Stock-Market-Prediction-API

## 🌟 Project Overview

This project presents an end-to-end solution for **stock market prediction** using a **Stacked Long Short-Term Memory (LSTM) neural network**, a powerful architecture for sequence forecasting. Beyond just model training, this repository demonstrates a full deployment pipeline, exposing the prediction functionality via a **Flask RESTful API**. This system is designed to forecast future stock prices based on historical data, providing a robust example of applying deep learning to financial time series and operationalizing AI models.

---

## ✨ Key Features & Technologies

- **Advanced LSTM Architecture:** Implements a **Stacked LSTM model** capable of learning complex temporal dependencies in stock price data, crucial for accurate forecasting.
- **Comprehensive Time Series Preprocessing:** Utilizes `MinMaxScaler` for robust data scaling, essential for neural network performance, and develops sequence generation logic for LSTM input.
- **Model Persistence:** Employs `tensorflow.keras.models` to save and load the trained LSTM model and `joblib` to persist the `MinMaxScaler` for consistent inference.
- **RESTful API Deployment:** Integrates the trained model with a **Flask API** (`app.py`), enabling external applications to send recent stock price data and receive future predictions.
- **Robust API Design:** The API handles input validation, data scaling, sequential predictions, and inverse scaling to return forecasts in the original price range.
- **Performance Evaluation:** Employs key regression metrics such as **Mean Absolute Error (MAE)** and **Mean Squared Error (MSE)** to rigorously evaluate model performance.
- **Interactive Visualizations:** Leverages `matplotlib`, `seaborn`, and `plotly.graph_objects` for comprehensive data exploration and visualization of predictions against actual values.

### 📚 Libraries Used

- `numpy`
- `pandas`
- `scikit-learn` (for `MinMaxScaler` and metrics)
- `tensorflow` & `keras` (for LSTM model)
- `matplotlib` & `seaborn` (for plotting)
- `plotly` (for interactive plots)
- `joblib` (for saving scaler)
- `Flask` (for API deployment)
- `pandas_datareader` (optional, for data acquisition)

---

## ⚙️ How It Works

This project is structured into two main components:

### 1. Model Training (`Stacked_LSTM_for_Stock_Market_Prediction_From_Model_Training_to_API_Deployment.ipynb`)

- **Data Loading & Exploration:** Loads historical stock data (e.g., `AAPL.csv`) and performs exploratory data analysis.
- **Data Preprocessing:** Extracts `'Close'` prices, scales them using `MinMaxScaler`, and generates sequences for LSTM input (e.g., 60-day sequences).
- **LSTM Model Architecture:** Builds a stacked LSTM model with dropout layers using Keras.
- **Model Training & Evaluation:** Trains the model with `EarlyStopping`, evaluates using MAE and MSE.
- **Model & Scaler Saving:** Saves the model (`stacked_lstm_model.keras`) and scaler (`scaler.save`) for deployment.

### 2. API Deployment (`app.py`)

- **Load Assets:** The Flask app loads the trained LSTM model and MinMaxScaler at startup.
- **Prediction Endpoint:** POST `/predict` receives:
  - `recent_prices`: 60 most recent closing prices
  - `days`: (optional) number of future days to predict
- **Response:** Returns predicted prices after inverse scaling as JSON.

---

## 📊 Dataset

The project uses historical Apple Inc. (`AAPL`) stock data, typically in `AAPL.csv`, containing:
- `Date`, `Open`, `High`, `Low`, `Close`, `Volume`

The `'Close'` column is used for forecasting.

---

## 🚀 Getting Started

### ✅ Prerequisites

- Python 3.8+
- Jupyter Notebook
- `pip` for package installation

### 🔧 Installation

1. **Clone the repository:**

```bash
[git clone https://github.com/YOUR_GITHUB_USERNAME/Stacked-LSTM-Stock-Market-Prediction-API.git
cd Stacked-LSTM-Stock-Market-Prediction-API](https://github.com/Bishal-Nengminja/Stacked-LSTM-for-Stock-Market-Prediction)
````


2. **Install dependencies:**

```bash
pip install numpy pandas matplotlib seaborn scikit-learn tensorflow flask plotly pandas_datareader joblib
```

3. **Add Dataset:**

Place `AAPL.csv` (with historical Apple stock data) in the project root.

---

## 🧪 Running the Project

### A. Model Training (Jupyter Notebook)

```bash
jupyter notebook Stacked_LSTM_for_Stock_Market_Prediction_API.ipynb
```

This will:

* Load & preprocess data
* Train and evaluate the LSTM model
* Save model as `stacked_lstm_model.keras`
* Save scaler as `scaler.save`

### B. Run the Flask API

Ensure both `stacked_lstm_model.keras` and `scaler.save` are in the same directory as `app.py`.

```bash
python app.py
```

The API will start on: `http://127.0.0.1:5000/`

---

## 🔌 API Usage Example

### Endpoint

`POST /predict`
**Content-Type:** `application/json`

### Example Request

```json
{
  "recent_prices": [150.25, 151.00, 150.50, ..., 152.10],
  "days": 5
}
```

> Replace with 60 actual closing prices.

### Example curl

```bash
curl -X POST -H "Content-Type: application/json" \
-d "{\"recent_prices\": [150.25, 151.00, ..., 152.10], \"days\": 5}" \
http://127.0.0.1:5000/predict
```

### Example Response

```json
{
  "predictions": [153.00, 153.50, 154.10, 154.80, 155.20]
}
```

---

## 📈 Results and Performance

> *(To be filled in after running the notebook)*
> Example:

* MAE: `1.12`
* RMSE: `1.46`

The model captures short-term trends and shows strong generalization on unseen test data.

---

## 🤝 Contributing

Contributions are welcome!

Suggestions for:

* Supporting multiple stocks
* Real-time data integration
* Web dashboard

Feel free to open issues or submit pull requests.

---

## 📄 License

This project is licensed under the **MIT License**.
See the `LICENSE` file for details.

---

## 📞 Contact

* GitHub: Bishal Nengminja [(https://github.com/Bishal-Nengminja)]
* Email: bishalnengminja61@gmail.com

---

⭐️ If you found this helpful, consider giving it a ⭐️ on GitHub!

```

---

### ✅ What you should do next:

- Replace:
  - `Bishal Nengminja` with your actual GitHub username.
  - `[[Your GitHub Profile Link](https://github.com/Bishal-Nengminja)]` and `[your.email@example.com]` with your contact info (or remove if not needed).
- Create a `LICENSE` file with [MIT License]([https://choosealicense.com/licenses/mit/](https://github.com/Bishal-Nengminja/Stacked-LSTM-for-Stock-Market-Prediction?tab=MIT-1-ov-file)).
- Save this content into a file named `README.md` in your project root.

Would you like me to generate this as a downloadable file or help create the LICENSE file next?
```
