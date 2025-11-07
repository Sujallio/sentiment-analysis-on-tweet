# Tweet Sentiment Analyzer

A real-time sentiment analysis web application built with Streamlit and Deep Learning that predicts whether text or tweets express positive or negative sentiment.

## 🎯 Problem Statement

Develop an intelligent system using Deep Learning techniques that can accurately classify the sentiment of text/tweets as positive or negative. The system should:

- Provide real-time sentiment predictions with confidence scores
- Process text with the same preprocessing pipeline used during training
- Offer a simple and user-friendly web interface
- Deliver fast and accurate predictions for social media text analysis

## 🚀 Features

- **Real-time Sentiment Analysis**: Instantly classify text as positive or negative
- **Confidence Scoring**: Get probability scores for predictions
- **Bi-LSTM Deep Learning Model**: Trained on 1.6 million tweets for ~77% accuracy
- **Text Preprocessing**: Automatic cleaning (URL removal, mention removal, stopword filtering)
- **Interactive UI**: Built with Streamlit for easy interaction
- **Visual Feedback**: Color-coded results (green for positive, red for negative)

## 🏗️ Architecture

- **Model**: Bidirectional LSTM (Bi-LSTM) neural network
- **Training Data**: 1.6 million tweets
- **Text Processing**: NLTK for stopword removal, regex for text cleaning
- **Framework**: TensorFlow/Keras for deep learning
- **Web Interface**: Streamlit
- **Tokenization**: Custom tokenizer with vocabulary mapping

## 📋 Requirements

```
streamlit
tensorflow
nltk
joblib
numpy
```

## 🔧 Installation

1. Clone the repository or download the project files

2. Install the required dependencies:
```bash
pip install streamlit tensorflow nltk joblib numpy
```

3. Ensure you have the following files in your project directory:
   - `sentiment_app.py` (main application)
   - `sentiment_model.h5` (trained Bi-LSTM model)
   - `tokenizer.pkl` (fitted tokenizer)

## 🎮 Usage

1. Run the Streamlit application:
```bash
streamlit run sentiment_app.py
```

2. Open your web browser (it should open automatically)

3. Enter any text or tweet in the input area

4. Click "Analyze Sentiment" to get the prediction

5. View the sentiment result with confidence score

## 🧪 Example Inputs

**Positive Examples:**
- "I'm so excited about the new project, it looks amazing!"
- "What a beautiful day! Feeling grateful and happy."

**Negative Examples:**
- "This is terrible, I'm so disappointed with the service."
- "Worst experience ever, completely frustrated."

## 🛠️ Technical Details

### Text Preprocessing Pipeline

The application applies the following preprocessing steps (matching the training pipeline):

1. Convert text to lowercase
2. Remove URLs (http/https links)
3. Remove @mentions
4. Remove punctuation and special characters
5. Remove hashtag symbols
6. Filter out stopwords (using NLTK)
7. Remove extra whitespace

### Model Specifications

- **Architecture**: Bidirectional LSTM
- **Max Sequence Length**: 50 tokens
- **Padding**: Post-padding and post-truncation
- **Output**: Binary classification (0=Negative, 1=Positive)
- **Threshold**: 0.5 for sentiment classification

### Performance

- **Model Accuracy**: ~77% on test data
- **Training Dataset**: 1.6 million tweets
- **Prediction Speed**: Real-time (< 1 second)

## 📁 Project Structure

```
sentiment_app.py       # Main Streamlit application
sentiment_model.h5     # Trained Bi-LSTM model
tokenizer.pkl          # Fitted tokenizer with word index
README.md              # Project documentation
```

## 🎨 UI Features

- Custom CSS styling with Tailwind-inspired design
- Color-coded sentiment boxes (green for positive, red for negative)
- Responsive layout with centered design
- Loading spinners for better UX
- Technical notes and model information display

## 🔍 How It Works

1. **Input**: User enters text in the text area
2. **Preprocessing**: Text is cleaned using the same pipeline as training
3. **Tokenization**: Text is converted to sequences using the loaded tokenizer
4. **Padding**: Sequences are padded to fixed length (50 tokens)
5. **Prediction**: Model predicts probability score (0 to 1)
6. **Classification**: Score ≥ 0.5 → Positive, Score < 0.5 → Negative
7. **Output**: Display sentiment with confidence percentage

## ⚙️ Configuration

Key hyperparameters (must match training):
- `MAX_LEN = 50` - Maximum sequence length
- `MODEL_PATH = 'sentiment_model.h5'` - Model file path
- `TOKENIZER_PATH = 'tokenizer.pkl'` - Tokenizer file path

## 🐛 Troubleshooting

- **Asset files not found**: Ensure `sentiment_model.h5` and `tokenizer.pkl` are in the same directory as `sentiment_app.py`
- **NLTK stopwords error**: The app will automatically download stopwords on first run
- **Low confidence predictions**: This is normal for ambiguous or neutral text

## 📝 License

This project is open-source and available for educational and research purposes.

## 🤝 Contributing

Feel free to fork this project and submit pull requests for improvements or bug fixes.

## 📧 Contact

For questions or feedback, please open an issue in the repository.
