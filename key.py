import os
from dotenv import load_dotenv

load_dotenv()  # Load variables from .env file

api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    print("❌ GEMINI_API_KEY not found. Please check your .env file.")
else:
    print("✅ Your API key is:", api_key)
