from pymongo import MongoClient

# MongoDB connection
client = MongoClient("mongodb://localhost:27017/")
db = client["sentiment_analysis"]
users_collection = db["users"]
