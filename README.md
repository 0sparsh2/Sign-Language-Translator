# English to Indian Sign Language Translator

🤖 AI-powered English to Indian Sign Language (ISL) translator using Gemini AI and YouTube API integration.

![Sign Language Translation Sample](sign%20language%20translated%20gif.gof)

## About
This project translates English text into Indian Sign Language by converting English grammar to ISL grammar and finding corresponding sign language videos. It uses Google's Gemini AI for intelligent word matching and grammar conversion, and integrates with the ISL Dictionary YouTube channel for video demonstrations.

### Key Features
- 🔄 Real-time English to ISL grammar conversion
- 🎥 Automatic video lookup from ISL Dictionary
- 🧠 Smart word matching with Gemini AI
- 📝 Comprehensive logging system
- 🌐 User-friendly Streamlit interface
- 🔐 Secure API key management

### Tech Stack
- Python
- Streamlit
- Google Gemini AI
- YouTube Data API
- Environment-based configuration

## Credits and Data Sources
This project uses data and resources from:
- [Indian Sign Language Dictionary](https://indiansignlanguage.org/) - For ISL word mappings and grammar rules
- [ISL Dictionary YouTube Channel](https://www.youtube.com/@isldictionary) - For sign language video demonstrations

## Disclaimer
This project is for educational and non-commercial purposes only. All sign language videos and content are sourced from the ISL Dictionary YouTube channel and are used with the intent of promoting accessibility and learning. This project is not affiliated with or endorsed by indiansignlanguage.org or the ISL Dictionary YouTube channel.

## Prerequisites

- Python 3.9 or higher
- Google Gemini API key
- YouTube Data API key

## Setup

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd Indian-Sign-Language
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Create a `.env` file in the project root with your API keys:
   ```
   GEMINI_API_KEY=your_gemini_api_key_here
   YOUTUBE_API_KEY=your_youtube_api_key_here
   ```

4. (Optional) Install Watchdog for better performance:
   ```bash
   xcode-select --install  # macOS only
   pip install watchdog
   ```

## Usage

1. Start the Streamlit app:
   ```bash
   streamlit run app.py
   ```

2. Open your browser and navigate to the provided URL (typically http://localhost:8501)

3. Enter English text in the input field

4. Click "Translate" to:
   - Convert the text to ISL grammar
   - Find matching sign videos
   - Display the translation and videos

## How it Works

1. **Text Input**: User enters English text
2. **Grammar Conversion**: Gemini AI converts text to ISL grammar
3. **Word Matching**:
   - First tries exact match in mapping.json
   - If not found, uses Gemini to find similar words
   - Falls back to original word if no match found
4. **Video Lookup**:
   - Searches ISL Dictionary YouTube channel
   - Returns first matching video
   - Handles errors gracefully with logging
5. **Display**: Shows translation and videos in sequence

## Configuration

The application uses several configuration options in `config.py`:
- `DB_PATH`: Path to the ISL database
- `ISL_WEBSITE`: Base URL for ISL website
- `SIMILARITY_THRESHOLD`: Minimum similarity score for word matching

## Logging

The application maintains detailed logs in `output.log`:
- Translation process
- Word matching attempts
- Video search results
- Error handling

## Note

This is a production-ready version with:
- Proper error handling
- Environment-based configuration
- Comprehensive logging
- API key security
- Performance optimizations

## Contributing

Feel free to submit issues and enhancement requests! 
