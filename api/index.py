from flask import Flask, request, jsonify
import requests
from collections import Counter
import os
from dotenv import load_dotenv
from nltk.stem import PorterStemmer
import nltk

# Download NLTK data (run once)
nltk.download('punkt')

# Load environment variables
load_dotenv()

app = Flask(__name__)

API_URL = "https://api-inference.huggingface.co/models/facebook/bart-large-mnli"
API_KEY = os.getenv("HF_API_TOKEN")
HEADERS = {"Authorization": f"Bearer {API_KEY}"}

# Initialize stemmer
stemmer = PorterStemmer()

# Full Case Study Dataset
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

# Expanded keywords with indirect terms, synonyms, and stemmed forms for each label
LABEL_KEYWORDS = {
    "Criminal Law: Murder": ["kill", "murder", "death", "slay", "homicide", "intent", "fatal", "die", "dead", "breath", "stop", "extinguish", "lifeless", "motionless", "unnatural", "end", "terminate", "assassinate"],
    "Criminal Law: Theft": ["steal", "theft", "rob", "burglary", "property", "take", "loot", "pilfer", "snatch", "embezzle", "pinch", "lift"],
    "Criminal Law: Assault": ["harm", "threat", "assault", "hit", "beat", "attack", "injure", "hurt", "strike", "punch", "wound", "maim"],
    "Criminal Law: Drug Offenses": ["drug", "possess", "sell", "narcotic", "deal", "substance", "use", "traffic", "smuggle", "addict", "chemical"],
    "Criminal Law: Terrorism": ["terror", "fear", "violence", "bomb", "public", "attack", "threaten", "destroy", "extremist", "sabotage", "panic"],
    "Criminal Law: Cybercrime": ["hack", "cyber", "tech", "fraud", "phish", "data", "online", "steal", "breach", "virus", "malware", "digital"],
    "Criminal Law: Blasphemy": ["religion", "offense", "belief", "sacrilege", "profane", "blaspheme", "insult", "disrespect", "holy", "desecrate"],
    "Criminal Law: Juvenile Crimes": ["minor", "juvenile", "child", "youth", "teen", "underage", "youngster", "adolescent", "kid", "delinquent"],
    "Criminal Law: Domestic Violence": ["abuse", "family", "relationship", "domestic", "violence", "fight", "beat", "hurt", "harass", "control", "threaten"],
    "Criminal Law: Fraud": ["deceive", "fraud", "scam", "cheat", "financial", "trick", "swindle", "defraud", "mislead", "con", "hoax", "forgery"],
    "Criminal Law: Sexual Harassment": ["sexual", "harass", "unwanted", "molest", "advance", "touch", "assault", "inappropriate", "abuse", "pressure"],
    "Criminal Law: Weapons Crimes": ["firearm", "gun", "weapon", "shoot", "arm", "knife", "blade", "bullet", "pistol", "rifle", "carry", "possess"],
    "Criminal Law: Human Trafficking": ["traffic", "labor", "force", "slave", "exploit", "smuggle", "trade", "abduct", "capture", "coerce"],
    "Criminal Law: Rape": ["rape", "sexual", "violence", "assault", "consent", "abuse", "force", "violate", "molest", "attack"],
    "Criminal Law: Public Disturbance": ["riot", "disturb", "peace", "uproar", "disorder", "chaos", "protest", "noise", "brawl", "trouble"],
    "Criminal Law: Kidnapping": ["kidnap", "abduct", "taken", "capture", "hostage", "seize", "hold", "detain", "snatch", "vanish"],
    "Criminal Law: Extortion": ["threat", "pay", "force", "blackmail", "extort", "coerce", "demand", "pressure", "ransom", "intimidate"],
    "Criminal Law: Money Laundering": ["money", "launder", "hide", "clean", "fund", "wash", "transfer", "disguise", "illegal", "cash"],
    "Criminal Law: Bribery & Corruption": ["bribe", "pay", "corrupt", "kickback", "influence", "buy", "grease", "fix", "solicit", "payoff"],
    "Criminal Law: Identity Theft": ["identity", "steal", "personal", "impersonate", "fraud", "clone", "hack", "usurp", "pretend", "misuse"],
    "Criminal Law: Vandalism": ["damage", "destroy", "vandal", "property", "wreck", "break", "ruin", "smash", "deface", "graffiti"],
    "Criminal Law: Arson": ["fire", "burn", "arson", "ignite", "torch", "blaze", "set", "flame", "incendiary", "smoke"],
    "Criminal Law: Hate Crimes": ["hate", "discriminate", "bias", "prejudice", "crime", "target", "attack", "bigotry", "racism", "intolerance"],
    "Criminal Law: Smuggling": ["smuggle", "transport", "illegal", "contraband", "move", "hide", "carry", "traffic", "ship", "sneak"],
    "Criminal Law: Organized Crime": ["organize", "group", "crime", "gang", "syndicate", "mafia", "network", "cartel", "racket", "crew"],
    "Criminal Law: White Collar Crimes": ["fraud", "embezzle", "business", "corporate", "financial", "cheat", "misconduct", "scheme", "deceive", "steal"],
    "Criminal Law: Piracy": ["piracy", "copy", "media", "steal", "illegal", "download", "duplicate", "bootleg", "counterfeit", "share"],
    "Criminal Law: Trespassing": ["trespass", "enter", "property", "intrude", "unauthorized", "break", "invade", "cross", "encroach", "violate"],
    "Criminal Law: Stalking": ["stalk", "harass", "follow", "pursue", "watch", "track", "shadow", "monitor", "obsess", "threaten"],
    "Criminal Law: Perjury": ["lie", "oath", "legal", "false", "swear", "perjure", "deceive", "testify", "mislead", "fabricate"],
    "Family Law: Marriage": ["marriage", "wedding", "spouse", "relationship", "union", "nuptial", "partner", "ceremony", "husband", "wife"],
    "Family Law: Divorce": ["divorce", "separate", "end", "marriage", "split", "dissolve", "breakup", "part", "terminate", "annul"],
}

# Key phrases for additional context (optional, can expand for all labels)
PHRASES = {
    "Criminal Law: Murder": ["stop breath", "life extinguish", "motionless", "unnatur caus", "found dead", "pass away"],
    "Criminal Law: Theft": ["take property", "gone missing", "disappear asset", "lost item"],
    "Criminal Law: Assault": ["physical attack", "hurt badly", "strike hard", "bodily harm"],
    "Criminal Law: Drug Offenses": ["found substance", "deal drugs", "narcotic use", "smuggle pills"],
    "Criminal Law: Terrorism": ["public panic", "bomb threat", "violent act public", "fear spread"],
    "Criminal Law: Cybercrime": ["data breach", "hack account", "online scam", "tech theft"],
    "Criminal Law: Blasphemy": ["offend faith", "religious insult", "sacrilege act", "belief attack"],
    "Criminal Law: Juvenile Crimes": ["youth crime", "teen offense", "minor act", "child wrongdoing"],
    "Criminal Law: Domestic Violence": ["family fight", "abuse home", "relationship harm", "spouse hurt"],
    "Criminal Law: Fraud": ["deceive money", "scam scheme", "financial trick", "cheat gain"],
    "Criminal Law: Sexual Harassment": ["unwanted touch", "inappropriate advance", "sexual pressure", "harass behavior"],
    "Criminal Law: Weapons Crimes": ["carry gun", "knife found", "weapon use", "shoot incident"],
    "Criminal Law: Human Trafficking": ["force labor", "trade people", "exploit worker", "smuggle human"],
    "Criminal Law: Rape": ["sexual force", "assault consent", "violent act", "abuse sexual"],
    "Criminal Law: Public Disturbance": ["riot street", "peace break", "uproar crowd", "disorder public"],
    "Criminal Law: Kidnapping": ["taken away", "hold captive", "abduct person", "seize individual"],
    "Criminal Law: Extortion": ["force payment", "threat money", "blackmail demand", "coerce funds"],
    "Criminal Law: Money Laundering": ["hide funds", "clean cash", "launder money", "disguise finance"],
    "Criminal Law: Bribery & Corruption": ["pay bribe", "corrupt deal", "kickback offer", "influence money"],
    "Criminal Law: Identity Theft": ["steal id", "impersonate person", "hack identity", "clone data"],
    "Criminal Law: Vandalism": ["destroy property", "damage building", "wreck site", "deface area"],
    "Criminal Law: Arson": ["set fire", "burn building", "ignite property", "torch site"],
    "Criminal Law: Hate Crimes": ["bias attack", "hate act", "discriminate harm", "prejudice crime"],
    "Criminal Law: Smuggling": ["move illegal", "hide goods", "transport contraband", "sneak items"],
    "Criminal Law: Organized Crime": ["gang act", "syndicate theft", "crime network", "racket operation"],
    "Criminal Law: White Collar Crimes": ["business fraud", "corporate cheat", "financial scheme", "embezzle funds"],
    "Criminal Law: Piracy": ["copy media", "steal content", "illegal download", "bootleg file"],
    "Criminal Law: Trespassing": ["enter unauthorized", "intrude property", "cross boundary", "break in"],
    "Criminal Law: Stalking": ["follow person", "harass target", "watch closely", "track individual"],
    "Criminal Law: Perjury": ["lie court", "false oath", "swear wrongly", "mislead justice"],
    "Family Law: Marriage": ["wedding issue", "spouse conflict", "relationship legal", "union dispute"],
    "Family Law: Divorce": ["end marriage", "split couple", "separate spouse", "dissolve union"],
}

def select_relevant_labels(case_text, max_labels=9):
    """Dynamically select up to 9 relevant labels based on stemmed keywords and phrases, plus 'Other'."""
    # Preprocess: Split into words and check for key phrases
    words = [stemmer.stem(word) for word in case_text.lower().split()]
    word_counts = Counter(words)
    
    # Check for specific phrases
    phrase_matches = {}
    for label, phrase_list in PHRASES.items():
        for phrase in phrase_list:
            stemmed_phrase = " ".join(stemmer.stem(w) for w in phrase.split())
            if stemmed_phrase in case_text.lower():
                phrase_matches[label] = phrase_matches.get(label, 0) + 1

    # Score labels based on keywords and phrases
    label_scores = {}
    for label, keywords in LABEL_KEYWORDS.items():
        # Keyword score
        keyword_score = sum(word_counts.get(stemmer.stem(kw), 0) for kw in keywords)
        # Phrase boost (if any)
        phrase_boost = phrase_matches.get(label, 0)
        total_score = keyword_score + phrase_boost * 2  # Give phrases more weight
        if total_score > 0:
            label_scores[label] = total_score
    
    # If no strong matches, use default labels
    if not label_scores:
        default_labels = [
            "Criminal Law: Murder", "Criminal Law: Theft", "Criminal Law: Assault",
            "Criminal Law: Fraud", "Criminal Law: Kidnapping", "Criminal Law: Domestic Violence",
            "Criminal Law: Rape", "Criminal Law: Weapons Crimes", "Criminal Law: Arson"
        ]
        return default_labels[:max_labels] + ["Other"]
    
    # Get top labels (up to 9)
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
    return jsonify({"message": "Welcome to the Case Analyzer API!", "status": "running"})

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
