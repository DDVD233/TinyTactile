from flask import Flask
import random

from classifier import classifier

app = Flask(__name__)

@app.route("/")
def hello_world():
    return "Hello World"

# --- CLASSIFIER ---
@app.route("/model/classify", methods=['POST'])
def classify_text():   
    data = request.get_json()
    res = ""

    for c in data['text']:
        res += classifier[c]
    
    return res, 201

# --- CALIBRATION/FINE-TUNING ---

@app.route("/calibrate/letters", methods=['GET'])
def get_calibration_letter_prompts():
    return [
        "The quick brown fox jumped over the lazy dog.",
    ], 200

@app.route("/calibrate/letters", methods=['POST'])
def calibrate_letters():
    reference = "The quick brown fox jumped over the lazy dog.!"
    data = request.get_json()
    for r, c in zip(reference, data['data']):
        classifier[r.lower()] = c.lower()
        classifier[r.upper()] = c.upper()
    return "", 201

@app.route("/calibrate/modifiers", methods=['GET'])
def get_calibration_modifier_prompts():
    return [
        "Tap all the locations for Shift",
        "Tap all the locations for Space",
        "Tap all the locations for Enter",
        "Tap all the locations for Backspace"
        "Tap all the locations for Caps Lock"
    ], 200

@app.route("/calibrate/letters", methods=['POST'])
def post_calibration_letter_prompts():
    data = request.get_json()
    # TODO: Pass this JSON data to Model for finetuning

@app.route("/calibrate/modifiers", methods=['POST'])
def post_calibration_modifier_prompts():
    data = request.get_json()
    # TODO: Pass this JSON data to Model for finetuning

# --- TEXT EXCERPTS ---

@app.route("/excerpt", methods=['GET'])
def get_excerpt():
    excerpts = [
        "Generative AI is revolutionizing creative industries by producing realistic images, music, and text.",
        "AI safety and alignment research is gaining momentum",
        "LLMs are being tailored for specialized domains",
        "AI-powered robotics is advancing automation",
        "Ethical Ai development remains a hot topic",
        "AI is transforming drug discovery with faster, cost-effective solutions.",
        "Edge AI enables real-time processing on devices without internet dependency",
        "Multi-modal AI intergrates text, images, and speech for richer UX",
        "Green AI emphasizes sustainable practices, reducing energy use in training models.",
        "Explainable AI focuses on improving transparency and trust in automated decisions."
    ]

    return random.choice(excerpts)
if __name__ == "__main__":
    app.run(debug=True)

