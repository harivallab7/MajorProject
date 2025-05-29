from flask import Flask, render_template_string, request, jsonify
from datetime import datetime, timedelta
import pandas as pd
import json
import os
from lstmscript import download_stock_data, train_lstm_and_forecast
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'


app = Flask(__name__)

# HTML template with embedded CSS and JavaScript
HTML_TEMPLATE = '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Stock Price Prediction</title>
    <style>
    * {
        margin: 0;
        padding: 0;
        box-sizing: border-box;
    }

    body {
        font-family: Arial, sans-serif;
        line-height: 1.6;
        background-color: #f5f5f5;
        color: #333;
    }

    .container {
        max-width: 1200px;
        margin: 0 auto;
        padding: 2rem;
    }

    h1 {
        text-align: center;
        color: #2c3e50;
        margin-bottom: 2rem;
    }

    .input-section {
        background-color: white;
        padding: 2rem;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
        margin-bottom: 2rem;
        display: flex;
        flex-wrap: wrap;
        gap: 1rem;
        align-items: flex-end;
    }

    .form-group {
        flex: 1;
        min-width: 200px;
    }

    label {
        display: block;
        margin-bottom: 0.5rem;
        color: #666;
    }

    input {
        width: 100%;
        padding: 0.8rem;
        border: 1px solid #ddd;
        border-radius: 4px;
        font-size: 1rem;
    }

    button {
        background-color: #3498db;
        color: white;
        padding: 0.8rem 2rem;
        border: none;
        border-radius: 4px;
        cursor: pointer;
        font-size: 1rem;
        transition: background-color 0.3s;
    }

    button:hover {
        background-color: #2980b9;
    }

    .loading-spinner {
        text-align: center;
        margin: 2rem 0;
    }

    .spinner {
        border: 4px solid #f3f3f3;
        border-top: 4px solid #3498db;
        border-radius: 50%;
        width: 50px;
        height: 50px;
        animation: spin 1s linear infinite;
        margin: 0 auto 1rem;
    }

    @keyframes spin {
        0% { transform: rotate(0deg); }
        100% { transform: rotate(360deg); }
    }

    .accuracy-section {
        text-align: center;
        margin: 2rem 0;
        padding: 1rem;
        background-color: white;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
    }

    .prediction-table-container {
        background-color: white;
        padding: 2rem;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
        margin-top: 2rem;
    }

    .prediction-table-container h3 {
        text-align: center;
        margin-bottom: 1rem;
        color: #2c3e50;
    }

    .prediction-table {
        width: 100%;
        border-collapse: collapse;
        margin-top: 1rem;
    }

    .prediction-table th,
    .prediction-table td {
        padding: 0.8rem;
        text-align: left;
        border-bottom: 1px solid #ddd;
    }

    .prediction-table th {
        background-color: #f2f2f2;
        color: #2c3e50;
    }

    .prediction-table tr:hover {
        background-color: #f5f5f5;
    }

    .hidden {
        display: none;
    }

    @media (max-width: 768px) {
        .input-section {
            flex-direction: column;
        }
        
        .form-group {
            width: 100%;
        }
    }
    </style>
</head>
<body>
    <div class="container">
        <h1>Stock Price Prediction with LSTM</h1>
        
        <div class="input-section">
            <div class="form-group">
                <label for="ticker">Stock Ticker:</label>
                <input type="text" id="ticker" placeholder="e.g., AAPL" required>
            </div>
            
            <div class="form-group">
                <label for="startDate">Start Date:</label>
                <input type="date" id="startDate" required>
            </div>
            
            <div class="form-group">
                <label for="endDate">End Date:</label>
                <input type="date" id="endDate" required>
            </div>
            
            <button id="predictBtn" onclick="predict()">Predict</button>
        </div>

        <div id="loadingSpinner" class="loading-spinner hidden">
            <div class="spinner"></div>
            <p>Training model and generating predictions...</p>
        </div>

        <div id="results" class="hidden">
            <div class="accuracy-section">
                <h2>Model Accuracy: <span id="accuracyValue">-</span>%</h2>
            </div>
            
            <div class="prediction-table-container">
                <h3>Future Predictions</h3>
                <table class="prediction-table" id="predictionTable">
                    <thead>
                        <tr>
                            <th>Date</th>
                            <th>Predicted Price ($)</th>
                        </tr>
                    </thead>
                    <tbody id="predictionTableBody">
                        <!-- Table data will be inserted here dynamically -->
                    </tbody>
                </table>
            </div>
        </div>
    </div>
    
    <script>
    async function predict() {
        const ticker = document.getElementById('ticker').value;
        const startDate = document.getElementById('startDate').value;
        const endDate = document.getElementById('endDate').value;

        if (!ticker || !startDate || !endDate) {
            alert('Please fill in all fields');
            return;
        }

        // Show loading spinner
        document.getElementById('loadingSpinner').classList.remove('hidden');
        document.getElementById('results').classList.add('hidden');

        try {
            const response = await fetch('/predict', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    ticker,
                    startDate,
                    endDate
                })
            });

            const result = await response.json();

            if (!result.success) {
                throw new Error(result.error);
            }

            // Update accuracy
            document.getElementById('accuracyValue').textContent = result.data.accuracy.toFixed(2);

            // Update prediction table
            updatePredictionTable(result.data);

            // Show results
            document.getElementById('results').classList.remove('hidden');
        } catch (error) {
            alert('Error: ' + error.message);
        } finally {
            document.getElementById('loadingSpinner').classList.add('hidden');
        }
    }

    function updatePredictionTable(data) {
        const tableBody = document.getElementById('predictionTableBody');
        
        // Clear previous data
        tableBody.innerHTML = '';
        
        // Add new rows
        for (let i = 0; i < data.dates.length; i++) {
            const row = document.createElement('tr');
            
            const dateCell = document.createElement('td');
            dateCell.textContent = data.dates[i];
            
            const priceCell = document.createElement('td');
            priceCell.textContent = data.prices[i].toFixed(2);
            
            row.appendChild(dateCell);
            row.appendChild(priceCell);
            
            tableBody.appendChild(row);
        }
    }

    // Set default dates
    document.addEventListener('DOMContentLoaded', () => {
        const today = new Date();
        const oneYearAgo = new Date();
        oneYearAgo.setFullYear(today.getFullYear() - 1);
        
        document.getElementById('startDate').value = oneYearAgo.toISOString().split('T')[0];
        document.getElementById('endDate').value = today.toISOString().split('T')[0];
    });
    </script>
</body>
</html>
'''

# Configuration
SENTIMENT_FILE_PATH = 'C:\\Major Project\\financial_sentiment_scores.json'

@app.route('/')
def index():
    return render_template_string(HTML_TEMPLATE)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        ticker = data['ticker']
        start_date = data['startDate']
        end_date = data['endDate']

        # Download stock data
        stock_data = download_stock_data(ticker, start_date, end_date)
        
        # Train model and get predictions
        predictions, accuracy, true_prices, predicted_prices = train_lstm_and_forecast(
            f"{ticker}_stock_data.csv",
            SENTIMENT_FILE_PATH,
            ticker
        )

        # Convert predictions to JSON-serializable format
        predictions_dict = {
            'dates': predictions['Date'].dt.strftime('%Y-%m-%d').tolist(),
            'prices': predictions['Predicted Price'].tolist(),
            'accuracy': float(accuracy),
            'historical_true': true_prices.flatten().tolist(),
            'historical_predicted': predicted_prices.flatten().tolist()
        }

        return jsonify({'success': True, 'data': predictions_dict})

    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)