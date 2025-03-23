"""from flask import Flask
from flask_cors import CORS
from routes import routes  # Import routes

app = Flask(__name__)
CORS(app)  # Enable CORS for frontend

# Register routes
app.register_blueprint(routes)

if __name__ == "__main__":
    app.run(debug=True)""
"""
import os
import requests
from flask import Flask
from flask_cors import CORS
from routes import routes  # Import routes

app = Flask(__name__)
CORS(app)  # Enable CORS for frontend

# Define model path and URL
MODEL_PATH = "backend/models/bert_sentiment/model.safetensors"
MODEL_URL = "https://github.com/pilican142asuflex/tweet-sentiment-analysis/releases/download/v1.0.0/model.safetensors"

# Function to download model if missing
def download_model():
    if not os.path.exists(MODEL_PATH):
        print("Downloading BERT model...")
        os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)

        response = requests.get(MODEL_URL, stream=True)
        with open(MODEL_PATH, "wb") as file:
            for chunk in response.iter_content(chunk_size=8192):
                file.write(chunk)

        print("Model downloaded successfully!")
    else:
        print("Model already exists.")

# Download the model before starting the app
download_model()

# Register routes
app.register_blueprint(routes)

if __name__ == "__main__":
    #app.run(debug=True)
    app.run(host="0.0.0.0", port=5000)

