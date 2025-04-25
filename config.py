import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Gemini API Configuration
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY not found in environment variables. Please set it in your .env file.")

# YouTube API Configuration
YOUTUBE_API_KEY = os.getenv('YOUTUBE_API_KEY')
if not YOUTUBE_API_KEY:
    raise ValueError("YOUTUBE_API_KEY not found in environment variables. Please set it in your .env file.")

# Database Configuration
DB_PATH = "isl_database"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# Website Configuration
ISL_WEBSITE = "https://indiansignlanguage.org"

# Vector Search Configuration
SIMILARITY_THRESHOLD = 0.85  # Minimum similarity score for vector matching 