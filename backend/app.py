from flask import Flask
from flask_cors import CORS
from routes import routes  # Import routes

app = Flask(__name__)
CORS(app)  # Enable CORS for frontend

# Register routes
app.register_blueprint(routes)

if __name__ == "__main__":
    app.run(debug=True)
