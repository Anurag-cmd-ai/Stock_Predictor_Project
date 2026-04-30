# 📈 Stock Price Predictor

A simple web app that predicts the future price trend of any stock using an LSTM deep learning model. Built with Streamlit and powered by Yahoo Finance live data.

---

## 🚀 Features

- Enter any stock symbol and get instant predictions
- Fetches live data automatically from Yahoo Finance
- Predicts next 30 days of stock price movement
- Shows whether price will go UP or DOWN
- Displays Week High, Week Low, Month Total stats
- Clean charts for historical and forecast prices
- Predicted price table for all 30 days

---

## 🛠️ Tech Stack

| Tool | Purpose |
|---|---|
| Python | Programming language |
| Streamlit | Web app interface |
| PyTorch | LSTM deep learning model |
| Yahoo Finance (yfinance) | Live stock data |
| Scikit-learn | Data scaling |
| Matplotlib | Charts and graphs |
| Pandas & Numpy | Data processing |

---

## 📁 Project Files

```
files1/
├── app.py            → Main application (everything is here)
└── requirements.txt  → Python libraries to install
```

---

## ⚙️ Installation

**Step 1 — Install Python**
Download from python.org (version 3.8 or above)

**Step 2 — Install required libraries**
Open Command Prompt, navigate to the project folder and run:
```
pip install -r requirements.txt
```

**Step 3 — Run the app**
```
streamlit run app.py
```

**Step 4 — Open in browser**
```
http://localhost:8501
```

---

## 📊 How to Use

1. Open the app in your browser
2. Type a stock symbol in the input box
3. Select how much historical data to use (1y / 2y / 3y / 5y)
4. Click the Predict button
5. Wait a few seconds — results appear automatically

---

## 🔤 Stock Symbol Examples

| Stock | Symbol |
|---|---|
| Apple | AAPL |
| Tesla | TSLA |
| Google | GOOGL |
| Microsoft | MSFT |
| Amazon | AMZN |
| Reliance Industries | RELIANCE.NS |
| TCS | TCS.NS |
| Infosys | INFY.NS |
| HDFC Bank | HDFCBANK.NS |
| Wipro | WIPRO.NS |

> For Indian stocks listed on NSE, always add .NS at the end of the symbol.

---

## 🌐 Hosting (Free Options)

### Streamlit Community Cloud (Recommended)
1. Push code to GitHub
2. Go to share.streamlit.io
3. Connect your GitHub repo
4. Click Deploy
5. Get a live link like https://yourapp.streamlit.app

### Other Options
- Hugging Face Spaces — great for ML projects
- Render — general web hosting
- Railway — fast deployment

---

## 🤖 How the Model Works

1. Downloads historical stock price data from Yahoo Finance
2. Scales the data using MinMaxScaler (0 to 1 range)
3. Creates sequences of 30 days to train the LSTM model
4. LSTM learns the pattern of price movements
5. Predicts the next 30 days iteratively
6. Scales predictions back to actual price values
7. Shows result as UP or DOWN with a forecast chart

---

## ⚠️ Disclaimer

This app is built for educational purposes only.
Stock predictions are not guaranteed to be accurate.
Do not use this for real financial or investment decisions.

---

## 👨‍💻 Author

Anurag Tiwari
Stock Price Predictor — ML Project
