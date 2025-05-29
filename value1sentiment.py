from transformers import AutoTokenizer, AutoModelForSequenceClassification
import json
import torch

# Initialize FinBERT sentiment analysis model
model_name = "yiyanghkust/finbert-tone"  # Pre-trained FinBERT model for financial sentiment
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

# Move model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Function to calculate overall normalized sentiment from financial headlines
def calculate_overall_sentiment(headlines):
    # Extract headline texts
    texts = [headline['heading'] for headline in headlines if 'heading' in headline]
    
    if not texts:  # Check for empty headlines
        return {}

    # Tokenize the input texts
    inputs = tokenizer(
        texts,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=512,
        return_attention_mask=True
    )
    
    # Move inputs to the correct device
    inputs = {key: value.to(device) for key, value in inputs.items()}
    
    # Get model predictions
    outputs = model(**inputs)
    logits = outputs.logits  # Logits are raw model outputs
    
    # Convert logits to probabilities
    scores = logits.softmax(dim=1)  # Apply softmax to get probabilities
    
    # Calculate the average sentiment scores for the day
    avg_scores = scores.mean(dim=0)  # Average across all headlines for the day
    
    # Create sentiment labels
    sentiment_labels = ['positive', 'neutral', 'negative']
    
    # Map each sentiment's average probability
    sentiment_scores = {sentiment_labels[i]: avg_scores[i].item() for i in range(len(sentiment_labels))}
    
    return sentiment_scores

# Function to analyze and save sentiment scores
def analyze_and_save_sentiment(input_file, output_file):
    # Load headlines data from input JSON file
    with open(input_file, 'r', encoding='utf-8') as file:
        data = json.load(file)
    
    result = {}
    for date, headlines in data.items():
        # Skip if no valid headlines for the date
        if not headlines:
            print(f"{date} has no valid headlines. Skipping...")
            continue
        
        # Calculate overall sentiment for the day
        sentiment_scores = calculate_overall_sentiment(headlines)
        if sentiment_scores:  # Only store results if sentiment was calculated
            print(f"{date} > Overall Sentiment Scores: {sentiment_scores}")
            result[date] = sentiment_scores
        else:
            print(f"{date} > No valid sentiment data.")

    # Save the calculated sentiment scores to an output JSON file
    with open(output_file, 'w', encoding='utf-8') as output:
        json.dump(result, output, ensure_ascii=False, indent=2)

# Paths to input and output files
input_json_file = "C:\\Major Project\\headlines.json"  # Input JSON file with headlines
output_json_file = 'C:\\Major Project\\financial_sentiment_scores.json'  # Output JSON file for sentiment scores

# Run sentiment analysis
analyze_and_save_sentiment(input_json_file, output_json_file)