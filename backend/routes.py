from flask import Blueprint, request, jsonify, send_from_directory
import torch
import tweepy
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
import time
import os

# Create a Blueprint for routes
routes = Blueprint("routes", __name__)

# Load trained model
model_path = "models/bert_sentiment"
model = DistilBertForSequenceClassification.from_pretrained(model_path)
tokenizer = DistilBertTokenizer.from_pretrained(model_path)
model.eval()

# Label mapping
label_map = ["Normal", "Depression", "Suicidal", "Anxiety", "Stress", "Bi-Polar", "Personality Disorder"]

# Twitter API credentials (Replace with your actual keys)
TWITTER_BEARER_TOKEN = "AAAAAAAAAAAAAAAAAAAAAE3%2BzwEAAAAABuhnMkqJHx9ooBKRFx2KH9ohDUc%3DGvsOY7sJpudKeihfL669eYF3mp8BzjBgOnvimU3YzcTbIMq5Z8"

# Tweepy client
client = tweepy.Client(bearer_token=TWITTER_BEARER_TOKEN)

# Selenium setup for taking screenshots
CHROMEDRIVER_PATH = "chromedriver"  # Set the correct path for your system
SCREENSHOTS_DIR = "screenshots"  # Removed "backend/" for Flask compatibility

# Ensure the screenshots directory exists
os.makedirs(SCREENSHOTS_DIR, exist_ok=True)

@routes.route("/", methods=["GET"])
def home():
    return jsonify({"message": "Welcome to Sentiment Analysis API"})


@routes.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.json
        tweet_url = data["tweet_url"]

        # Extract Tweet ID from URL
        tweet_id = tweet_url.split("/")[-1]
        time.sleep(5)
        
        # Fetch Tweet using Twitter API
        tweet_response = client.get_tweet(tweet_id, tweet_fields=["text"])
        if not tweet_response.data:
            return jsonify({"error": "Tweet not found"}), 404

        tweet_text = tweet_response.data.text

        # Tokenize input text
        encoding = tokenizer(tweet_text, truncation=True, padding=True, max_length=128, return_tensors="pt")

        # Perform prediction
        with torch.no_grad():
            output = model(**encoding)

        # Get predicted label
        predicted_class = torch.argmax(output.logits).item()
        sentiment = label_map[predicted_class]

        # Take screenshot of the tweet
        screenshot_filename = f"tweet_{tweet_id}.png"
        screenshot_path = take_screenshot(tweet_url, screenshot_filename)

        return jsonify({
            "sentiment": sentiment,
            "tweet_text": tweet_text,
            "screenshot_url": f"/screenshots/{screenshot_filename}"  # Return Flask-accessible URL
        })
    
    except Exception as e:
        return jsonify({"error": str(e)})


def take_screenshot(tweet_url, screenshot_filename):
    """ Captures a screenshot of the tweet using Selenium """
    try:
        chrome_options = Options()
        chrome_options.add_argument("--headless")
        chrome_options.add_argument("--no-sandbox")
        chrome_options.add_argument("--disable-dev-shm-usage")

        service = Service(CHROMEDRIVER_PATH)  # Correct usage
        driver = webdriver.Chrome(service=service, options=chrome_options)

        driver.get(tweet_url)
        time.sleep(3)  # Wait for the page to load

        screenshot_path = os.path.join(SCREENSHOTS_DIR, screenshot_filename)
        driver.save_screenshot(screenshot_path)
        driver.quit()

        return screenshot_path
    except Exception as e:
        return str(e)


@routes.route('/screenshots/<filename>')
def get_screenshot(filename):
    return send_from_directory("screenshots", filename)  # Ensure this matches the folder name
