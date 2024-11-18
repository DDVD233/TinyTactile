from flask import Flask
import random

app = Flask(__name__)

@app.route("/")
def hello_world():
    return "Hello World"

// --- CALIBRATION/FINE-TUNING ---

@app.route("/calibrate/letters", method=['GET'])
def get_calibration_letter_prompts():
    return [
        "The quick brown fox jumped over the lazy dog.",
        "", // We need more informed prompts?
    ]

@app.route("/calibrate/modifiers", method=['GET'])
def get_calibration_modifier_prompts():
    return [
        "Tap all the locations for Shift",
        "Tap all the locations for Space",
        "Tap all the locations for Enter",
        "Tap all the locations for Backspace"
        "Tap all the locations for Caps Lock"
    ]

@app.route("/calibrate/letters", method=['POST'])
def post_calibration_letter_prompts():
    data = request.get_json()
    // TODO: Pass this JSON data to Model for finetuning

@app.route("/calibrate/modifiers", method=['POST'])
def post_calibration_modifier_prompts():
    data = request.get_json()
    // TODO: Pass this JSON data to Model for finetuning

// --- TEXT EXCERPTS ---

@app.route("/excerpt", method=['GET'])
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

