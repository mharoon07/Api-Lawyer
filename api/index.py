from flask import Flask, request, jsonify
from utils.case_classification import classify_case
app = Flask(__name__)
@app.route('/')
def home():
    return jsonify({"message": "Welcome to the Legal Cases Classification API"})
@app.route('/analyze-case', methods=['POST'])
def classify():
    data = request.get_json()
    if not data or 'case' not in data:
        return jsonify({"error": "Case description is required"}), 400

    case_description = data.get('case')
    user_lat = data.get('latitude')
    user_lon = data.get('longitude')

    result, status_code = classify_case(case_description, user_lat, user_lon)
    return jsonify(result), status_code

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
