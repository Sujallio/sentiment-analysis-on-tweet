import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
import joblib
import re
from nltk.corpus import stopwords
import numpy as np
import os # Import os for better file path checking

# --- 1. Configuration and Asset Paths ---
MODEL_PATH = 'sentiment_model.h5'
TOKENIZER_PATH = 'tokenizer.pkl'

# Hyperparameters (MUST match training)
MAX_LEN = 50
# Initialize STOPWORDS outside the function to ensure it's available
try:
    # Ensure NLTK is ready
    import nltk
    nltk.download('stopwords', quiet=True)
    STOPWORDS = set(stopwords.words('english'))
except Exception:
    # Fallback if nltk or stopwords download fails
    STOPWORDS = set() 
    
# --- 2. Load Assets with Caching ---
# Streamlit's caching prevents the model and tokenizer from reloading 
# every time the user interacts with the app, which is CRUCIAL for performance.
@st.cache_resource
def load_assets():
    """Loads the model and tokenizer from disk."""
    if not os.path.exists(MODEL_PATH) or not os.path.exists(TOKENIZER_PATH):
        st.error(f"Asset files not found. Please ensure '{MODEL_PATH}' and '{TOKENIZER_PATH}' are in the same directory as this script.")
        st.stop()
        
    try:
        # We need to disable Keras compilation warnings due to saved optimizer state
        with st.spinner("Loading assets..."):
            model = tf.keras.models.load_model(MODEL_PATH)
            tokenizer = joblib.load(TOKENIZER_PATH)
            return model, tokenizer
    except Exception as e:
        st.error(f"Error loading assets: {e}")
        st.stop()

# Load the model and tokenizer at the start
model, tokenizer = load_assets()

# --- 3. Core Preprocessing and Prediction Logic ---

def clean_tweet(text):
    """
    Replicates the exact text cleaning performed during training.
    """
    text = str(text).lower()
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)  # Remove URLs
    text = re.sub(r'@\w+', '', text)                                       # Remove Mentions
    text = re.sub(r'[^\w\s]', '', text)                                      # Remove Punctuation/Symbols
    text = re.sub(r'#', '', text)                                           # Remove Hashtag symbol
    
    # Stopword removal
    text = ' '.join([word for word in text.split() if word not in STOPWORDS]) 
    text = ' '.join(text.split())                                          # Remove extra whitespace
    return text

def predict_sentiment(text, model, tokenizer):
    """
    Cleans, tokenizes, and predicts the sentiment of the input text.
    """
    # 1. Clean
    cleaned_text = clean_tweet(text)
    
    # 2. Tokenize and Convert to Sequence
    sequence = tokenizer.texts_to_sequences([cleaned_text])
    
    # 3. Pad
    padded_sequence = pad_sequences(
        sequence, 
        maxlen=MAX_LEN, 
        padding='post', 
        truncating='post'
    )
    
    # 4. Predict
    # Prediction returns a probability (value between 0 and 1)
    # The [0][0] extracts the single floating-point number
    prediction = model.predict(padded_sequence, verbose=0)[0][0]
    
    # 5. Determine Sentiment
    # Since 0=Negative and 1=Positive was used in training, a value >= 0.5 means Positive
    if prediction >= 0.5:
        sentiment = "POSITIVE"
        color = "green"
    else:
        sentiment = "NEGATIVE"
        color = "red"
        
    # Confidence is the probability of the predicted class
    confidence = float(prediction) if sentiment == "POSITIVE" else float(1.0 - prediction)
    
    return sentiment, confidence, color

# --- 4. Streamlit UI ---

# Set page configuration
st.set_page_config(page_title="Tweet Sentiment Predictor", layout="centered")

# Custom CSS for styling (Tailwind-like classes)
st.markdown(
    """
    <style>
    .big-font {
        font-size:30px !important;
        font-weight: 700;
        color: #1E40AF; /* Indigo 700 */
    }
    .main-card {
        padding: 30px;
        border-radius: 15px;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
        background-color: white;
    }
    .positive-box {
        background-color: #D1FAE5; /* Emerald 100 */
        border: 2px solid #10B981; /* Emerald 500 */
        color: #065F46; /* Emerald 900 */
        padding: 15px;
        border-radius: 10px;
        font-weight: bold;
    }
    .negative-box {
        background-color: #FEE2E2; /* Red 100 */
        border: 2px solid #EF4444; /* Red 500 */
        color: #7F1D1D; /* Red 900 */
        padding: 15px;
        border-radius: 10px;
        font-weight: bold;
    }
    </style>
    """, 
    unsafe_allow_html=True
)

st.markdown('<div class="main-card">', unsafe_allow_html=True)
st.markdown('<p class="big-font">Tweet Sentiment Analyzer</p>', unsafe_allow_html=True)
st.write("Enter any text or tweet below to get a real-time sentiment prediction (Positive or Negative).")
st.caption(f"Model loaded: Bi-LSTM trained on 1.6M tweets. Vocabulary size: {len(tokenizer.word_index)} words.")
# 

# Text Input Area
tweet_input = st.text_area(
    "Input Text", 
    placeholder="Example: I'm so excited about the new project, it looks amazing!",
    height=150
)

# Analysis Button
if st.button("Analyze Sentiment", use_container_width=True):
    if tweet_input:
        # Show spinner while predicting
        with st.spinner('Predicting sentiment...'):
            sentiment, confidence, color = predict_sentiment(tweet_input, model, tokenizer)
        
        # Display Result
        st.subheader("Analysis Result:")
        
        # Use HTML/Markdown for colored boxes
        if sentiment == "POSITIVE":
            st.markdown(f'<div class="positive-box">Sentiment: {sentiment} (Confidence: {confidence*100:.2f}%)</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="negative-box">Sentiment: {sentiment} (Confidence: {confidence*100:.2f}%)</div>', unsafe_allow_html=True)
    else:
        st.warning("Please enter some text to analyze.")

st.markdown('</div>', unsafe_allow_html=True)

# Footer/Technical Details
st.markdown(
    """
    ---
    <p style='font-size: small; color: gray;'>
    Technical note: The prediction relies on the exact cleaning, tokenization, and padding 
    used during the original Bi-LSTM training for an accuracy of ~77%.
    </p>
    """,
    unsafe_allow_html=True
)
