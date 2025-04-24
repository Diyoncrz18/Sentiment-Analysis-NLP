import google.generativeai as genai

# Konfigurasi API key dari Google AI Studio
genai.configure(api_key="AIzaSyDb5d0IC3ugl7-RFzasXH-UcFVSuFBdVJw")  # Ganti dengan API key kamu yang valid

# Gunakan Gemini Flash (Gemini 1.5 Flash)
model = genai.GenerativeModel('models/gemini-1.5-flash-latest')

def analyze_sentiment(text):
    prompt = f"Review: \"{text}\". Analyze the sentiment and answer with only one word: Positive, Negative, or Neutral."
    response = model.generate_content(prompt)
    sentiment = response.text.strip()
    return sentiment
