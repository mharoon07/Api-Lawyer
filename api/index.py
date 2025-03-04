from flask import Flask, request, jsonify
import requests
from collections import Counter
import os
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)

API_URL = "https://api-inference.huggingface.co/models/facebook/bart-large-mnli"
API_KEY = os.getenv("HF_API_TOKEN") 
HEADERS = {"Authorization": f"Bearer {API_KEY}"}


CASE_DATASET = [
    {"text": "Killing someone intentionally or unintentionally.", "label": "Criminal Law: Murder"},
    {"text": "Stealing someone's property.", "label": "Criminal Law: Theft"},
    {"text": "Physically harming or threatening someone.", "label": "Criminal Law: Assault"},
    {"text": "Possessing, using, or selling illegal drugs.", "label": "Criminal Law: Drug Offenses"},
    {"text": "Acts causing fear, violence, or harm to the public.", "label": "Criminal Law: Terrorism"},
    {"text": "Crimes using technology, like hacking or fraud.", "label": "Criminal Law: Cybercrime"},
    {"text": "Offenses against religious beliefs.", "label": "Criminal Law: Blasphemy"},
    {"text": "Crimes committed by minors.", "label": "Criminal Law: Juvenile Crimes"},
    {"text": "Abuse within a family or relationship.", "label": "Criminal Law: Domestic Violence"},
    {"text": "Deceiving others for financial gain.", "label": "Criminal Law: Fraud"},
    {"text": "Unwanted sexual advances or behavior.", "label": "Criminal Law: Sexual Harassment"},
    {"text": "Illegal possession or use of firearms.", "label": "Criminal Law: Weapons Crimes"},
    {"text": "Forcing people into illegal labor or trade.", "label": "Criminal Law: Human Trafficking"},
    {"text": "Sexual violence against someone without consent.", "label": "Criminal Law: Rape"},
    {"text": "Causing riots or disturbing peace.", "label": "Criminal Law: Public Disturbance"},
    {"text": "Taking someone against their will.", "label": "Criminal Law: Kidnapping"},
    {"text": "Forcing someone to pay money through threats.", "label": "Criminal Law: Extortion"},
    {"text": "Hiding or disguising illegal money sources.", "label": "Criminal Law: Money Laundering"},
    {"text": "Offering or accepting illegal payments.", "label": "Criminal Law: Bribery & Corruption"},
    {"text": "Stealing someone's personal information.", "label": "Criminal Law: Identity Theft"},
    {"text": "Destroying or damaging property on purpose.", "label": "Criminal Law: Vandalism"},
    {"text": "Setting property on fire intentionally.", "label": "Criminal Law: Arson"},
    {"text": "Committing crimes based on hate or discrimination.", "label": "Criminal Law: Hate Crimes"},
    {"text": "Illegally transporting goods or people.", "label": "Criminal Law: Smuggling"},
    {"text": "Organized illegal activities by groups.", "label": "Criminal Law: Organized Crime"},
    {"text": "Fraud or embezzlement in business settings.", "label": "Criminal Law: White Collar Crimes"},
    {"text": "Illegal copying or distribution of media.", "label": "Criminal Law: Piracy"},
    {"text": "Entering someone's property without permission.", "label": "Criminal Law: Trespassing"},
    {"text": "Repeatedly harassing or following someone.", "label": "Criminal Law: Stalking"},
    {"text": "Lying under oath in legal matters.", "label": "Criminal Law: Perjury"},
    {"text": "Legal rules for weddings and relationships.", "label": "Family Law: Marriage"},
    {"text": "Ending a marriage legally.", "label": "Family Law: Divorce"},
]

LABEL_KEYWORDS = {
    "Criminal Law: Murder": ["kill", "murder", "death", "intentionally"],
    "Criminal Law: Theft": ["steal", "stole", "theft", "property"],
    "Criminal Law: Assault": ["harm", "threat", "assault", "hit"],
    "Criminal Law: Drug Offenses": ["drug", "possession", "selling"],
    "Criminal Law: Terrorism": ["terrorism", "fear", "violence", "public"],
    "Criminal Law: Cybercrime": ["hack", "cyber", "technology", "fraud"],
    "Criminal Law: Blasphemy": ["religion", "offense", "belief"],
    "Criminal Law: Juvenile Crimes": ["minor", "juvenile", "child"],
    "Criminal Law: Domestic Violence": ["abuse", "family", "relationship"],
    "Criminal Law: Fraud": ["deceive", "fraud", "scam", "financial"],
    "Criminal Law: Sexual Harassment": ["sexual", "harassment", "unwanted"],
    "Criminal Law: Weapons Crimes": ["firearm", "gun", "weapon"],
    "Criminal Law: Human Trafficking": ["traffic", "labor", "force"],
    "Criminal Law: Rape": ["rape", "sexual", "violence", "consent"],
    "Criminal Law: Public Disturbance": ["riot", "disturb", "peace"],
    "Criminal Law: Kidnapping": ["kidnap", "taken", "abduct"],
    "Criminal Law: Extortion": ["threat", "pay", "force"],
    "Criminal Law: Money Laundering": ["money", "launder", "hide"],
    "Criminal Law: Bribery & Corruption": ["bribe", "payment", "corrupt"],
    "Criminal Law: Identity Theft": ["identity", "steal", "personal"],
    "Criminal Law: Vandalism": ["damage", "destroy", "property"],
    "Criminal Law: Arson": ["fire", "burn", "arson"],
    "Criminal Law: Hate Crimes": ["hate", "discrimination", "crime"],
    "Criminal Law: Smuggling": ["smuggle", "transport", "illegal"],
    "Criminal Law: Organized Crime": ["organized", "group", "crime"],
    "Criminal Law: White Collar Crimes": ["fraud", "embezzle", "business"],
    "Criminal Law: Piracy": ["piracy", "copy", "media"],
    "Criminal Law: Trespassing": ["trespass", "enter", "property"],
    "Criminal Law: Stalking": ["stalk", "harass", "follow"],
    "Criminal Law: Perjury": ["lie", "oath", "legal"],
    "Family Law: Marriage": ["marriage", "wedding", "relationship"],
    "Family Law: Divorce": ["divorce", "marriage", "end"],
}

def select_relevant_labels(case_text, max_labels=9):
    """Dynamically select up to 9 relevant labels based on keywords, plus 'Other'."""
    words = case_text.lower().split()
    word_counts = Counter(words)
    
    label_scores = {}
    for label, keywords in LABEL_KEYWORDS.items():
        score = sum(word_counts.get(keyword, 0) for keyword in keywords)
        if score > 0:
            label_scores[label] = score
    
    sorted_labels = sorted(label_scores, key=label_scores.get, reverse=True)[:max_labels]
    return sorted_labels + ["Other"]

def classify_case(case_description):
    if not case_description or not isinstance(case_description, str):
        return {"error": "Case description must be a non-empty string"}

    candidate_labels = select_relevant_labels(case_description)
    
    payload = {
        "inputs": case_description,
        "parameters": {"candidate_labels": candidate_labels}
    }
    
    try:
        response = requests.post(API_URL, headers=HEADERS, json=payload, timeout=10)
        response.raise_for_status()
        result = response.json()
        
        if "labels" in result and "scores" in result:
            best_label = result["labels"][0]
            best_score = result["scores"][0]
            return {
                "case_type": best_label,
                "confidence": round(best_score, 4),
                "top_matches": [
                    {"label": label, "score": round(score, 4)}
                    for label, score in zip(result["labels"][:3], result["scores"][:3])
                ],
                "selected_labels": candidate_labels
            }
        else:
            return {"error": "Invalid response format from API"}
    
    except requests.exceptions.Timeout:
        return {"error": "Request to Hugging Face API timed out"}
    except requests.exceptions.RequestException as e:
        return {"error": f"API request failed: {str(e)}"}

@app.route('/')
def home():
    return jsonify({"message": "Welcome to the Case Study Analyzer API!", "status": "running"})

@app.route('/analyze-case', methods=['POST'])
def analyze_case():
    data = request.get_json(silent=True)
    if not data or 'case' not in data:
        return jsonify({"error": "Please provide a case study in the 'case' field"}), 400
    
    case_text = data['case']
    result = classify_case(case_text)
    
    if "error" in result:
        return jsonify(result), 500
    return jsonify(result)

if __name__ == "__main__":
    PORT = int(os.getenv("PORT", 5000))
    if os.name == "nt":
        from waitress import serve
        serve(app, host="0.0.0.0", port=PORT)
    else:
        app.run(host="0.0.0.0", port=PORT)