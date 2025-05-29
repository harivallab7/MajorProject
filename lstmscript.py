from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import json
from alpha_vantage.timeseries import TimeSeries
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt

# Alpha Vantage API Key (Replace with your own)
API_KEY = "M91K1W25QYRDLBL5"

# Fixed sentiment data file path
SENTIMENT_FILE_PATH = r"C:\\Major Project\\financial_sentiment_scores.json"
PREDICTION_DAYS = 5

# Function to download stock data using Alpha Vantage
def download_stock_data(ticker, start_date, end_date):
    print(f"Downloading stock data for {ticker} from Alpha Vantage...")

    ts = TimeSeries(key=API_KEY, output_format="pandas")
    stock_data, meta_data = ts.get_daily(symbol=ticker, outputsize="full")

    # Renaming columns
    stock_data.rename(columns={
        '1. open': 'Open',
        '2. high': 'High',
        '3. low': 'Low',
        '4. close': 'Close',
        '5. volume': 'Volume'
    }, inplace=True)

    stock_data.index = pd.to_datetime(stock_data.index)
    stock_data = stock_data.sort_index()  # Ensure chronological order

    # Print available date range
    print(f"Stock data available from {stock_data.index.min()} to {stock_data.index.max()}")

    # Filter data with proper handling
    stock_data = stock_data[(stock_data.index >= pd.to_datetime(start_date)) & (stock_data.index <= pd.to_datetime(end_date))]

    if stock_data.empty:
        print(f"No stock data found for {ticker} in the specified date range.")
        return None
    else:
        print(f"Stock data for {ticker} downloaded successfully.")
        print(stock_data.head())

    stock_data.to_csv(f"{ticker}_stock_data.csv", index=True)
    return stock_data

# Function to load and merge stock and sentiment data
def load_and_merge_data(stock_file, sentiment_file, stock_ticker):
    print(f"Loading and merging stock data and sentiment data...")
    stock_df = pd.read_csv(stock_file)
    stock_df['Date'] = pd.to_datetime(stock_df['date'])

    with open(sentiment_file, 'r', encoding='utf-8') as file:
        sentiment_data = json.load(file)

    sentiment_df = pd.DataFrame(sentiment_data).T
    sentiment_df.index = pd.to_datetime(sentiment_df.index)
    sentiment_df.columns = ['positive', 'neutral', 'negative']

    # Merging stock and sentiment data based on Date
    combined_df = pd.merge(stock_df, sentiment_df, left_on='Date', right_index=True, how='inner')

    if combined_df.empty:
        print("No data found after merging. Ensure that the stock dates and sentiment dates match.")
    else:
        print(f"Merged data sample: {combined_df.head()}")

    # Save the merged data to a file with the stock ticker name
    merged_file_path = f"{stock_ticker}_merged_data.csv"
    combined_df.to_csv(merged_file_path, index=False)
    print(f"Merged data saved to {merged_file_path}")

    return combined_df

# Function to preprocess data
def preprocess_data(df):
    feature_scaler = MinMaxScaler(feature_range=(0, 1))
    target_scaler = MinMaxScaler(feature_range=(0, 1))

    features = df[['Open', 'High', 'Low', 'Volume', 'positive', 'neutral', 'negative']].values
    target = df['Close'].values

    scaled_features = feature_scaler.fit_transform(features)
    scaled_target = target_scaler.fit_transform(target.reshape(-1, 1))

    return scaled_features, scaled_target, feature_scaler, target_scaler

# Function to create LSTM model
def create_lstm_model(input_shape):
    model = Sequential()
    model.add(LSTM(units=64, return_sequences=True, input_shape=input_shape))
    model.add(LSTM(units=32, return_sequences=True))
    model.add(LSTM(units=16, return_sequences=False))
    model.add(Dense(units=1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

# LSTM model and forecast prices
def train_lstm_and_forecast(stock_file, sentiment_file, stock_ticker):
    combined_df = load_and_merge_data(stock_file, sentiment_file, stock_ticker)
    X, y, feature_scaler, target_scaler = preprocess_data(combined_df)
    X = X.reshape((X.shape[0], 1, X.shape[1]))

    model = create_lstm_model((X.shape[1], X.shape[2]))
    model.fit(X, y, epochs=50, batch_size=32, verbose=1)

    # Save the model
    model_path = f"{stock_ticker}_lstm_model.h5"
    model.save(model_path)
    print(f"Model saved as {model_path}")

    predicted_stock_price = model.predict(X)
    predicted_stock_price = target_scaler.inverse_transform(predicted_stock_price)
    true_stock_price = target_scaler.inverse_transform(y.reshape(-1, 1))

    mae = mean_absolute_error(true_stock_price, predicted_stock_price)
    rmse = np.sqrt(mean_squared_error(true_stock_price, predicted_stock_price))
    mape = np.mean(np.abs((true_stock_price - predicted_stock_price) / true_stock_price)) * 100
    accuracy = 100 - mape

    # Predict future stock prices
    last_known_data = combined_df.iloc[-1]
    future_dates = [datetime.today() + timedelta(days=i) for i in range(1, PREDICTION_DAYS + 1)]

    future_predictions = []
    current_input = feature_scaler.transform(
        last_known_data[['Open', 'High', 'Low', 'Volume', 'positive', 'neutral', 'negative']].values.reshape(1, -1)
    ).reshape(1, 1, -1)

    for _ in range(PREDICTION_DAYS):
        predicted_price = model.predict(current_input)
        future_predictions.append(predicted_price[0][0])

        scaled_predicted_price = target_scaler.transform(np.array(predicted_price[0][0]).reshape(1, -1))
        current_input = np.roll(current_input, -1, axis=2)
        current_input[0, 0, -1] = scaled_predicted_price[0][0]

    future_predictions = target_scaler.inverse_transform(np.array(future_predictions).reshape(-1, 1))

    future_predictions_dates = pd.DataFrame({
        'Date': future_dates,
        'Predicted Price': future_predictions.flatten()
    })

    # Save predictions
    prediction_file = f"{stock_ticker}_future_predictions.csv"
    future_predictions_dates.to_csv(prediction_file, index=False)
    print(f"Predictions saved to {prediction_file}")

    return future_predictions_dates, accuracy, true_stock_price, predicted_stock_price

def plot_training_data(true_prices, predicted_prices, stock_ticker):
    plt.figure(figsize=(12, 6))
    plt.plot(true_prices, label="True Prices", color='green', alpha=0.7)
    plt.plot(predicted_prices, label="Predicted Prices", color='orange', alpha=0.7)
    plt.title("Training Data: True Prices vs Predicted Prices")
    plt.xlabel("Time (days)")
    plt.ylabel("Stock Price")
    plt.legend()
    plt.grid()
    plt.show()

if __name__ == '__main__':
    stock_ticker = input("Enter stock ticker (e.g., AAPL): ")
    start_date = input("Enter start date (YYYY-MM-DD): ")
    end_date = input("Enter end date (YYYY-MM-DD): ")

    try:
        download_stock_data(stock_ticker, start_date, end_date)
        predictions, accuracy, true_prices, predicted_prices = train_lstm_and_forecast(f"{stock_ticker}_stock_data.csv", SENTIMENT_FILE_PATH, stock_ticker)

        print(f"\nPredictions:\n{predictions}")
        print(f"\nModel Accuracy: {accuracy:.2f}%")

        plot_training_data(true_prices, predicted_prices, stock_ticker)

    except Exception as e:
        print(f"Error: {str(e)}")