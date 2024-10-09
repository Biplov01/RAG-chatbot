import os
import requests
from flask import Flask, request, jsonify
from dotenv import load_dotenv
from datetime import datetime, timedelta
import re

# Load environment variables from .env file
load_dotenv()

app = Flask(__name__)

LANGCHAIN_API_KEY = os.getenv("LANGCHAIN_API_KEY")
LANGCHAIN_ENDPOINT = os.getenv("LANGCHAIN_ENDPOINT", "https://api.smith.langchain.com")

@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json()
    if not data or "message" not in data:
        return jsonify({"error": "Invalid input"}), 400

    user_message = data["message"]
    headers = {"Authorization": f"Bearer {LANGCHAIN_API_KEY}", "Content-Type": "application/json"}
    payload = {"query": user_message}

    try:
        response = requests.post(LANGCHAIN_ENDPOINT, json=payload, headers=headers)
        response.raise_for_status()
        bot_response = response.json()
    except requests.exceptions.RequestException as e:
        return jsonify({"error": f"Error occurred: {e}"}), 500

    return jsonify({"response": bot_response.get("response", "No response from bot")})

@app.route("/book_appointment", methods=["POST"])
def book_appointment():
    data = request.get_json()
    if not data or not all(k in data for k in ("name", "phone", "email", "date")):
        return jsonify({"error": "Missing required fields"}), 400

    name = data["name"]
    phone = data["phone"]
    email = data["email"]
    date_str = data["date"]

    # Validate email
    if not re.match(r"[^@]+@[^@]+\.[^@]+", email):
        return jsonify({"error": "Invalid email address"}), 400

    # Validate phone number
    if not re.match(r"^\+?\d{10,15}$", phone):
        return jsonify({"error": "Invalid phone number"}), 400

    # Parse and validate date
    try:
        appointment_date = parse_date(date_str)
        if appointment_date < datetime.now():
            return jsonify({"error": "Date cannot be in the past"}), 400
    except ValueError:
        return jsonify({"error": "Invalid date format"}), 400

    # Normally, you'd save the appointment details to a database here

    return jsonify({"message": "Appointment booked successfully"})

def parse_date(date_str):
    today = datetime.now()
    if date_str.lower() == "next monday":
        days_ahead = 0 - today.weekday() + 7
        if days_ahead <= 0:
            days_ahead += 7
        return today + timedelta(days_ahead)
    try:
        return datetime.strptime(date_str, "%Y-%m-%d")
    except ValueError:
        raise ValueError("Invalid date format. Please use YYYY-MM-DD.")

if __name__ == "__main__":
    app.run(debug=True)
