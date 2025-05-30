import tensorflow as tf
import pandas as pd
import numpy as np
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from flask import Flask, jsonify
from flask_cors import CORS
import warnings
warnings.filterwarnings("ignore")

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Step 1: Fetch and Preprocess Data
def fetch_stock_data(ticker, start_date, end_date):
    df = yf.download(ticker, start=start_date, end=end_date, progress=False)
    if df.empty:
        raise ValueError(f"No data found for {ticker}")
    return df['Close'].values.reshape(-1, 1)

def prepare_lstm_data(data, time_steps=60):
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(data)
    X, y = [], []
    for i in range(time_steps, len(scaled_data)):
        X.append(scaled_data[i-time_steps:i, 0])
        y.append(scaled_data[i, 0])
    return np.array(X), np.array(y), scaler

# Step 2: Build and Train LSTM Model
def build_lstm_model(time_steps):
    model = tf.keras.Sequential([
        tf.keras.layers.LSTM(64, return_sequences=True, input_shape=(time_steps, 1), dropout=0.2, 
                            kernel_regularizer=tf.keras.regularizers.l2(0.01)),
        tf.keras.layers.LSTM(32, dropout=0.2, kernel_regularizer=tf.keras.regularizers.l2(0.01)),
        tf.keras.layers.Dense(16, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01)),
        tf.keras.layers.Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    return model

def predict_stock_price(model, data, scaler, time_steps, days_ahead=7):
    last_sequence = data[-time_steps:].reshape(1, time_steps, 1)
    prediction = model.predict(last_sequence, verbose=0)
    prediction = scaler.inverse_transform(prediction)
    return prediction[0][0]

# Step 3: Build and Train DNN Model for Portfolio Health Score
def build_health_score_model():
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(128, activation='relu', input_shape=(3,)),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='mse')
    return model

def train_health_score_model(model):
    X = np.random.rand(1000, 3)  # [sector_exposure, volatility, performance]
    y = 1 - (X[:, 0] * 0.4 + X[:, 1] * 0.4 + (1 - X[:, 2]) * 0.2)
    model.fit(X, y, epochs=10, batch_size=32, verbose=0)
    return model

# Step 4: Portfolio Analysis
def analyze_portfolio():
    sheet_url = 'https://docs.google.com/spreadsheets/d/1wZR0tdfgem_KKAAyY_Anu3c-1nE_DzuYqZ6i80JBuWA/export?format=csv'
    df = pd.read_csv(sheet_url)
    df['Units'] = df['Units'].astype(float)
    df['Purchase Price'] = df['Purchase Price'].astype(float)
    df['Current Value '] = df['Current Value '].replace('N/A', '0').astype(float)
    total_value = df['Current Value '].sum()
    if total_value == 0:
        raise ValueError("Total portfolio value is zero. Check if current prices are updated.")
    sector_exposure = df.groupby('Sector')['Current Value '].sum() / total_value * 100

    start_date = "2023-05-29"
    end_date = "2025-05-29"
    portfolio_volatility = 0
    portfolio_returns = pd.DataFrame()
    stock_volatilities = {}
    for _, row in df.iterrows():
        stock = row['Stock Symbol']
        weight = row['Current Value '] / total_value
        yfinance_symbol = stock.replace("NSE:", "").strip() + ".NS"
        try:
            stock_data_df = yf.download(yfinance_symbol, start=start_date, end=end_date, progress=False)
            if stock_data_df.empty:
                raise ValueError(f"No data found for {stock}")
            stock_data = stock_data_df['Close'].values.reshape(-1, 1)
            returns = stock_data_df['Close'].pct_change().dropna()
            if len(returns) < 2:
                print(f"Insufficient data to calculate volatility for {stock}: {len(returns)} returns")
                continue
            stock_volatility = np.std(returns.values) * np.sqrt(252)
            stock_volatilities[stock] = stock_volatility
            portfolio_volatility += weight * stock_volatility
            portfolio_returns[stock] = returns * weight
        except Exception as e:
            print(f"Error calculating volatility for {stock}: {e}")
            continue

    portfolio_daily_returns = portfolio_returns.sum(axis=1)
    annualized_return = portfolio_daily_returns.mean() * 252 if not portfolio_daily_returns.empty else 0
    risk_free_rate = 0.06
    sharpe_ratio = (annualized_return - risk_free_rate) / portfolio_volatility if portfolio_volatility != 0 else 0

    stock_contribution = (df['Current Value '] - df['Purchase Price']) / (total_value - df['Purchase Price'].sum()) * 100
    return df, sector_exposure, portfolio_volatility, sharpe_ratio, stock_contribution, stock_volatilities

# Step 5: Sector Benchmarking
def benchmark_sector_performance(df, start_date, end_date):
    sector_indices = {'Energy': 'NIFTY_ENERGY.NS'}
    sector_performance = {}
    for sector in df['Sector'].unique():
        sector_stocks = df[df['Sector'] == sector]
        stock_returns = ((sector_stocks['Current Value '] - sector_stocks['Purchase Price']) / sector_stocks['Purchase Price']).mean()
        if sector in sector_indices:
            try:
                index_data = fetch_stock_data(sector_indices[sector], start_date, end_date)
                index_return = (index_data[-1][0] - index_data[0][0]) / index_data[0][0]
                sector_performance[sector] = (stock_returns, index_return)
            except Exception as e:
                print(f"Error fetching index data for {sector} (ticker: {sector_indices[sector]}): {e}")
    return sector_performance

# Step 6: Risk Assessment and Scenario Analysis
def assess_risks(predictions, sector_exposure, portfolio_volatility, stock_volatilities):
    risks = {}
    risk_rewards = {}
    for stock, (pred_price, current_price) in predictions.items():
        volatility = stock_volatilities.get(stock, 0)
        expected_return = (pred_price - current_price) / current_price
        risks[stock] = "High" if volatility > 0.05 else "Moderate"
        risk_rewards[stock] = expected_return / volatility if volatility != 0 else 0
    portfolio_risk = "High" if sector_exposure.max() > 60 else "Moderate"
    return risks, portfolio_risk, risk_rewards

def scenario_analysis(predictions, portfolio_value, sector_exposure):
    market_drop = 0.05
    portfolio_impact = 0
    for sector, exposure in sector_exposure.items():
        sector_sensitivity = 0.8 if sector == 'Energy' else 0.6
        sector_impact = market_drop * sector_sensitivity * (exposure / 100)
        portfolio_impact += sector_impact
    return portfolio_impact * portfolio_value

# Step 7: Generate Recommendations with Trading Strategies
def generate_recommendations(predictions, risks, sector_exposure, risk_tolerance, df, technical_indicators, historical_data, sector_trends, risk_rewards):
    recommendations = []
    for stock, (pred_price, current_price) in predictions.items():
        rsi, macd = technical_indicators[stock]
        confidence = 0.9
        risk_reward = risk_rewards[stock]

        momentum_signal = "Neutral"
        if rsi > 70 and macd < 0:
            momentum_signal = "Sell (Overbought, Bearish Momentum)"
        elif rsi < 30 and macd > 0:
            momentum_signal = "Buy (Oversold, Bullish Momentum)"

        historical_prices = historical_data.get(stock, pd.Series()).values
        historical_mean = np.mean(historical_prices) if len(historical_prices) > 0 else current_price
        deviation = (current_price - historical_mean) / historical_mean if historical_mean != 0 else 0
        mean_reversion_signal = "Neutral"
        if deviation > 0.1:
            mean_reversion_signal = "Sell (Overvalued)"
        elif deviation < -0.1:
            mean_reversion_signal = "Buy (Undervalued)"

        predicted_change = (pred_price - current_price) / current_price * 100
        if momentum_signal.startswith("Buy") or mean_reversion_signal.startswith("Buy"):
            if predicted_change > 5 and confidence > 0.85:
                action = f"Buy more {stock}"
            else:
                action = f"Hold {stock}"
        elif momentum_signal.startswith("Sell") or mean_reversion_signal.startswith("Sell"):
            if predicted_change < -5:
                units = df[df['Stock Symbol'] == stock]['Units'].iloc[0]
                action = f"Sell {min(100, units)} shares of {stock}"
            else:
                action = f"Hold {stock}"
        else:
            action = f"Hold {stock}"

        rationale = f"Predicted change: {predicted_change:.1f}%, Risk: {risks[stock]}, Risk/Reward: {risk_reward:.2f}, Confidence: {confidence:.2f}, Momentum: {momentum_signal}, Mean Reversion: {mean_reversion_signal}"
        recommendations.append(f"{action} ({rationale})")

    top_sector = max(sector_trends, key=sector_trends.get)
    if sector_exposure[top_sector] < 30 and sector_trends[top_sector] > 10:
        recommendations.append(f"Sector Rotation: Increase exposure to {top_sector} (Expected change: {sector_trends[top_sector]:.1f}%)")

    for sector, exposure in sector_exposure.items():
        if exposure > 60 and risk_tolerance == "Moderate":
            recommendations.append(f"Rebalance: Reduce {sector} exposure (currently {exposure:.1f}%)")

    return recommendations

# New Features
def calculate_technical_indicators(data):
    delta = np.diff(data.flatten())
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = np.mean(gain[-14:])
    avg_loss = np.mean(loss[-14:])
    rs = avg_gain / avg_loss if avg_loss != 0 else 0
    rsi = 100 - (100 / (1 + rs)) if avg_loss != 0 else 100

    exp1 = pd.Series(data.flatten()).ewm(span=12, adjust=False).mean()
    exp2 = pd.Series(data.flatten()).ewm(span=26, adjust=False).mean()
    macd = exp1 - exp2
    signal = macd.ewm(span=9, adjust=False).mean()
    macd_value = macd.iloc[-1] - signal.iloc[-1]

    return rsi, macd_value

def calculate_diversification_score(sector_exposure):
    num_sectors = len(sector_exposure)
    ideal_exposure = 100 / num_sectors
    deviation = sum(abs(exposure - ideal_exposure) for exposure in sector_exposure.values)
    score = max(0, 100 - deviation)
    return score

# Utility to Convert NumPy Types to JSON-Serializable Types
def convert_to_json_serializable(obj):
    if isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: convert_to_json_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_json_serializable(item) for item in obj]
    return obj

# Data Preparation for API
portfolio_data = {}
def prepare_data():
    global portfolio_data
    start_date = "2023-05-29"
    end_date = "2025-05-29"
    time_steps = 60
    risk_tolerance = "Moderate"

    portfolio_df, sector_exposure, portfolio_volatility, sharpe_ratio, stock_contribution, stock_volatilities = analyze_portfolio()
    total_value = portfolio_df['Current Value '].sum()

    lstm_model = build_lstm_model(time_steps)

    predictions = {}
    sector_predictions = {}
    technical_indicators = {}
    historical_data = {}
    for _, row in portfolio_df.iterrows():
        stock = row['Stock Symbol']
        current_price = row['Current Value '] / row['Units']
        try:
            yfinance_symbol = stock.replace("NSE:", "").strip() + ".NS"
            print(f"Fetching data for {stock} (ticker: {yfinance_symbol})...")
            data = fetch_stock_data(yfinance_symbol, start_date, end_date)
            print(f"Data fetched for {stock}: {len(data)} days")
            X, y, scaler = prepare_lstm_data(data, time_steps)
            X = X.reshape(X.shape[0], X.shape[1], 1)

            train_size = int(len(X) * 0.8)
            X_train, X_val = X[:train_size], X[train_size:]
            y_train, y_val = y[:train_size], y[train_size:]

            lstm_model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=30, batch_size=32, verbose=0)

            pred_price = predict_stock_price(lstm_model, data, scaler, time_steps)
            predictions[stock] = (pred_price, current_price)

            rsi, macd = calculate_technical_indicators(data)
            technical_indicators[stock] = (rsi, macd)

            df_stock = yf.download(yfinance_symbol, start=start_date, end=end_date, progress=False)
            if not df_stock.empty:
                historical_data[stock] = df_stock['Close']
            else:
                print(f"No historical data fetched for {stock}")

            sector = portfolio_df[portfolio_df['Stock Symbol'] == stock]['Sector'].iloc[0]
            if sector not in sector_predictions:
                sector_predictions[sector] = []
            sector_predictions[sector].append((pred_price - current_price) / current_price)
        except Exception as e:
            print(f"Error processing {stock} (ticker: {yfinance_symbol}): {e}")

    sector_trends = {sector: np.mean(returns) * 100 for sector, returns in sector_predictions.items()}

    health_model = build_health_score_model()
    health_model = train_health_score_model(health_model)
    max_exposure = sector_exposure.max() / 100
    performance = (total_value - portfolio_df['Purchase Price'].sum()) / portfolio_df['Purchase Price'].sum()
    health_inputs = np.array([[max_exposure, portfolio_volatility, performance]])
    health_score = health_model.predict(health_inputs, verbose=0)[0][0] * 100

    sector_performance = benchmark_sector_performance(portfolio_df, start_date, end_date)

    if predictions:
        risks, portfolio_risk, risk_rewards = assess_risks(predictions, sector_exposure, portfolio_volatility, stock_volatilities)
        market_drop_impact = scenario_analysis(predictions, total_value, sector_exposure)
    else:
        risks, portfolio_risk, risk_rewards, market_drop_impact = {}, "N/A (no predictions)", {}, 0

    energy_sentiment, it_sentiment, energy_score, it_score = get_sector_sentiment()
    macro_insights = [
        f"RBI cuts repo rate to 6% on April 9, 2025, to stimulate growth (Inflation: 4.2% projected for 2025-26).",
        f"Global trade tensions rise due to U.S. tariffs, impacting Energy exports (Sentiment: {energy_sentiment}, {energy_score:.1f}).",
        f"IT sector benefits from digital payments growth (Sentiment: {it_sentiment}, {it_score:.1f})."
    ]

    diversification_score = calculate_diversification_score(sector_exposure)

    portfolio_value = pd.DataFrame()
    for stock, prices in historical_data.items():
        units = portfolio_df[portfolio_df['Stock Symbol'] == stock]['Units'].iloc[0]
        portfolio_value[stock] = prices * units
    portfolio_value['Total'] = portfolio_value.sum(axis=1)

    # Prepare portfolio data with JSON-serializable types
    portfolio_data = {
        "total_value": float(total_value),
        "portfolio_volatility": float(portfolio_volatility),
        "health_score": float(health_score),
        "sector_exposure": {k: float(v) for k, v in sector_exposure.items()},
        "stocks": [],
        "historical_data": {
            "dates": portfolio_value.index.strftime('%Y-%m-%d').tolist(),
            "values": portfolio_value['Total'].tolist()
        }
    }

    for _, row in portfolio_df.iterrows():
        stock = row['Stock Symbol']
        sector = row['Sector']
        current_price = float(row['Current Value '] / row['Units'])
        pred_price, _ = predictions.get(stock, (0, 0))
        pred_price = float(pred_price)
        change = (pred_price - current_price) / current_price * 100 if current_price != 0 else 0
        portfolio_data["stocks"].append({
            "name": stock,
            "sector": sector,
            "current_price": current_price,
            "predicted_price": pred_price,
            "change_percent": float(change),
            "value": float(row['Current Value '])
        })

    # Ensure all nested data is JSON-serializable
    portfolio_data = convert_to_json_serializable(portfolio_data)

# API Endpoints
@app.route('/api/portfolio')
def get_portfolio():
    return jsonify(portfolio_data)

# Sentiment Analysis (for completeness)
def get_sector_sentiment():
    sia = SentimentIntensityAnalyzer()
    energy_headlines = [
        "RBI rate cut to 6% may boost renewable energy investments despite global trade tensions.",
        "Energy sector faces pressure from U.S. tariffs and volatile oil prices."
    ]
    it_headlines = [
        "IT sector sees growth potential with strong domestic demand and digital payments surge.",
        "Geopolitical tensions may impact IT exports amid U.S. tariff hikes."
    ]
    
    energy_scores = [sia.polarity_scores(headline)['compound'] for headline in energy_headlines]
    it_scores = [sia.polarity_scores(headline)['compound'] for headline in it_headlines]
    
    energy_sentiment = "Positive" if sum(energy_scores) / len(energy_scores) > 0 else "Negative"
    it_sentiment = "Positive" if sum(it_scores) / len(it_scores) > 0 else "Negative"
    energy_score = sum(energy_scores) / len(energy_scores)
    it_score = sum(it_scores) / len(it_scores)
    
    return energy_sentiment, it_sentiment, energy_score, it_score

if __name__ == "__main__":
    print("Preparing portfolio data...")
    prepare_data()
    print("Starting Flask server on http://localhost:5001")
    app.run(debug=True, host='0.0.0.0', port=5001)