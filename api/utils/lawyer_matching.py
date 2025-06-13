from collections import Counter
import math
from data.lawyer_data import LAWYER_DATA
from data.label_keywords import LABEL_KEYWORDS
from data.phrases import PHRASES
from utils.text_processing import simple_stem
def select_relevant_labels(case_description):
    """Select up to 5 relevant labels based on keyword and phrase matching."""
    case_words = set(simple_stem(word) for word in case_description.lower().split())
    label_scores = Counter()
    for label, keywords in LABEL_KEYWORDS.items():
        stemmed_keywords = set(simple_stem(kw) for kw in keywords)
        matches = len(case_words & stemmed_keywords)
        if matches > 0:
            label_scores[label] += matches

    for label, phrases in PHRASES.items():
        for phrase in phrases:
            if phrase.lower() in case_description.lower():
                label_scores[label] += 2

    relevant_labels = [label for label, _ in label_scores.most_common(5)]
    if not relevant_labels:
        relevant_labels = list(LABEL_KEYWORDS.keys())[:5]
    return relevant_labels
def get_nearest_lawyer(case_type, user_lat=None, user_lon=None, return_all=False):
    """Find the nearest lawyer or all lawyers matching the case type."""
    if not return_all and (user_lat is None or user_lon is None):
        return {"error": "User location not provided"}
    if not return_all:
        try:
            user_lat = float(user_lat)
            user_lon = float(user_lon)
        except (ValueError, TypeError):
            return {"error": "Invalid user coordinates"}
    def haversine(lat1, lon1, lat2, lon2):
        lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
        c = 2 * math.asin(math.sqrt(a))
        r = 6371
        return c * r
    LAWYER_SPECIALIZATION_MAP = {
        "Criminal Law: Murder": ["Criminal law", "Criminal Trials"],
        "Criminal Law: Theft": ["Criminal law"],  
    }
    relevant_specializations = LAWYER_SPECIALIZATION_MAP.get(case_type, ["Not Specified"])
    suitable_lawyers = [
        lawyer for lawyer in LAWYER_DATA
        if any(spec.lower() in lawyer["Specialization"].lower() for spec in relevant_specializations)
        and (return_all or (lawyer["Map"]["latitude"] != "null" and lawyer["Map"]["longitude"] != "null"))
    ]
    if not suitable_lawyers:
        return {"error": "No lawyers found for this case type"}
    if return_all:
        return [
            {
                "name": lawyer["Name"],
                "contact": lawyer["Contact"],
                "chamber": lawyer["Chamber"],
                "specialization": lawyer["Specialization"],
                "address": lawyer["Address"],
                "latitude": lawyer["Map"]["latitude"],
                "longitude": lawyer["Map"]["longitude"]
            }
            for lawyer in suitable_lawyers
        ]
    nearest_lawyer = None
    min_distance = float('inf')
    for lawyer in suitable_lawyers:
        try:
            lawyer_lat = float(lawyer["Map"]["latitude"])
            lawyer_lon = float(lawyer["Map"]["longitude"])
            distance = haversine(user_lat, user_lon, lawyer_lat, lawyer_lon)
            if distance < min_distance:
                min_distance = distance
                nearest_lawyer = lawyer
        except (ValueError, TypeError):
            continue
    if nearest_lawyer:
        return {
            "name": nearest_lawyer["Name"],
            "contact": nearest_lawyer["Contact"],
            "chamber": nearest_lawyer["Chamber"],
            "specialization": nearest_lawyer["Specialization"],
            "address": nearest_lawyer["Address"],
            "distance_km": round(min_distance, 2)
        }
    else:
        return {"error": "No lawyers with valid coordinates found"}