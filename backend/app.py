from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import joblib
import google.generativeai as genai

# Konfigurasi API key dari Google AI Studio
genai.configure(api_key="AIzaSyDb5d0IC3ugl7-RFzasXH-UcFVSuFBdVJw")  # Ganti dengan API key kamu

# Gunakan Gemini Flash
gemini_model = genai.GenerativeModel('models/gemini-1.5-flash-latest')

# Load model Logistic Regression dan TF-IDF Vectorizer
logreg_model = joblib.load('model/logistic_regression_model.joblib')
vectorizer = joblib.load('model/tfidf_vectorizer.joblib')

app = Flask(__name__)
CORS(app)

# Analisis dengan model Logistic Regression (NLP)
def analyze_with_model(text):
    vectorized = vectorizer.transform([text])
    prediction = logreg_model.predict(vectorized)[0]
    return "Positive" if prediction == 1 else "Negative"

# Analisis dengan Geminii (LLM)
def analyze_sentiment(text):
    prompt = f"Review: \"{text}\". Analyze the sentiment and answer with only one word: Positive, Negative, or Neutral."
    response = gemini_model.generate_content(prompt)
    return response.text.strip()

@app.route('/analyze', methods=['POST'])
def analyze_text():
    try:
        data = request.get_json()
        text = data.get('text', '')
        use_gemini = data.get('use_gemini', False)
        
        if use_gemini:
            sentiment = analyze_sentiment(text)
        else:
            sentiment = analyze_with_model(text)
        
        return jsonify({'sentiment': sentiment})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/', methods=['GET'])
def home():
    return send_from_directory('../frontend', 'index.html')

if __name__ == '__main__':
    app.run(debug=True)
