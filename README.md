# 📈 AI Stock Analyst: Professional Price Prediction Pipeline

A comprehensive, end-to-end Streamlit application that combines **Quantitative Technical Analysis** with **Qualitative News Sentiment** to provide actionable stock market insights.

---

## 🌟 Overview

The **AI Stock Analyst** is a high-performance financial tool designed to analyze live market data, evaluate news sentiment using transformer models, and predict future price movements using a weighted ensemble of machine learning architectures. It features a "Bloomberg-style" dark UI with a real-time execution tracker.

## 🚀 Key Features

- **Live Market Data Integration**: Fetches real-time OHLCV data and company metadata directly from Yahoo Finance (`yfinance`).
- **Deep Learning Sentiment Analysis**: Utilizes a pre-trained **BERT-base** model (NLP Town) from Hugging Face to analyze the sentiment of the latest news articles.
- **Hybrid Machine Learning Pipeline**:
    - **Linear Regression**: Fast baseline for trend identification.
    - **Random Forest**: Captures non-linear dependencies and feature importance.
    - **LSTM (RNN)**: A 2-layer Long Short-Term Memory network for temporal sequence prediction.
- **Intelligent Ensemble Engine**: Combines model outputs using an **Inverse-MAE Weighted Average**, prioritizing models that historically performed better on the stock's recent volatility.
- **Actionable Trading Signals**: Generates **BUY**, **SELL**, or **HOLD** recommendations by cross-referencing quantitative forecasts with qualitative sentiment scores.
- **Dynamic UI/UX**: Includes a 10-step real-time pipeline tracker, professional metric cards, and interactive Plotly visualizations.

## 🛠️ Tech Stack

| Category | Tools/Libraries |
| :--- | :--- |
| **Frontend** | Streamlit, Custom CSS (Bloomberg Dark Mode) |
| **Data Fetching** | `yfinance`, `NewsAPI` |
| **Data Processing** | `Pandas`, `NumPy` |
| **Machine Learning** | `Scikit-Learn`, `TensorFlow/Keras` |
| **Natural Language Processing** | `Hugging Face Transformers` (BERT), `PyTorch` |
| **Visualization** | `Plotly`, Native Streamlit Charts |

## 📥 Installation

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd "Stock prediction CLG PROject"
   ```

2. **Create a virtual environment** (Recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

## 🔑 Configuration

To fetch real-time news, you will need a **NewsAPI Key**.
1. Get your free key at [newsapi.org](https://newsapi.org/).
2. Enter the key in the application sidebar when prompted.

## 🏃 Getting Started

Run the application using Streamlit:

```bash
streamlit run app.py
```

## 🧠 Application Workflow

The pipeline executes 10 sequential steps:

1. **Environment Setup**: Initializing the UI and styles.
2. **Data Acquisition**: Downloading 2 years of daily stock history.
3. **News Harvest**: Fetching the latest `N` articles via NewsAPI.
4. **Sentiment Scoring**: Running news snippets through the BERT model.
5. **Feature Engineering**: Computing technical indicators like RSI, Moving Averages (7, 21, 50), and Volatility.
6. **Linear Modeling**: Training and testing the LR baseline.
7. **Forest Regression**: Training the Random Forest regressor.
8. **RNN/LSTM Training**: Training the temporal neural network.
9. **Ensemble Logic**: Weighing inputs by inverse-error to find the "Consensus Forecast".
10. **Signal Generation**: Synthesizing all data into a final recommendation.

---

## ⚠️ Disclaimer

*This tool is for educational and research purposes only. Stock market investments carry inherent risks. Past performance (and machine learning predictions) are not indicative of future results. Always consult with a certified financial advisor before making investment decisions.*
