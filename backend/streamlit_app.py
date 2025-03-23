import streamlit as st
import requests

API_URL = "http://127.0.0.1:5000/predict"

st.title("Tweet Sentiment Analyzer")

# Input field for Tweet URL
tweet_url = st.text_input("Paste the Tweet URL:")

if st.button("Analyze"):
    if tweet_url:
        response = requests.post(API_URL, json={"tweet_url": tweet_url})
        data = response.json()

        if "error" in data:
            st.error(f"Error: {data['error']}")
        else:
            st.success(f"**Sentiment:** {data['sentiment']}")
            st.write(f"**Tweet Text:** {data['tweet_text']}")

            # Get screenshot URL from API response
            screenshot_url = f"http://127.0.0.1:5000{data['screenshot_url']}"

            # Display the screenshot
            st.image(screenshot_url, caption="Tweet Screenshot", use_column_width=True)
    else:
        st.warning("Please enter a Tweet URL.")
