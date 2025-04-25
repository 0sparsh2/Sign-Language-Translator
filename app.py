import streamlit as st
import google.generativeai as genai
from config import GEMINI_API_KEY, YOUTUBE_API_KEY
import os
import logging
import json
import string
from typing import Optional, List, Dict, Tuple
import googleapiclient.discovery
from googleapiclient.errors import HttpError

# Set up logging with both file and console handlers
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Create formatters
file_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
console_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

# File handler
file_handler = logging.FileHandler('output.log')
file_handler.setLevel(logging.INFO)
file_handler.setFormatter(file_formatter)

# Console handler
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
console_handler.setFormatter(console_formatter)

# Add handlers to logger
logger.addHandler(file_handler)
logger.addHandler(console_handler)

def load_word_list() -> List[str]:
    """Load word list from mapping.json and save to words.txt"""
    try:
        with open('isl_database/mapping.json', 'r') as f:
            mapping = json.load(f)
            # Convert all words to uppercase for case-insensitive matching
            words = [item['word'].upper() for item in mapping]
            logger.info(f"✅ Loaded {len(words)} words from mapping.json")
            
            # # Save words to a separate file
            # with open('words.txt', 'w') as wf:
            #     wf.write('\n'.join(words))
            # logger.info("✅ Saved word list to words.txt")
            
            return words
    except Exception as e:
        logger.error(f"❌ Error loading mapping.json: {str(e)}")
        return []

# Initialize word list
available_words = load_word_list()

# Initialize Gemini
genai.configure(api_key=GEMINI_API_KEY)
model = genai.GenerativeModel('gemini-2.0-flash')

# Initialize YouTube API
youtube = googleapiclient.discovery.build('youtube', 'v3', developerKey=YOUTUBE_API_KEY)

def clean_word(word: str) -> str:
    """Remove punctuation and whitespace from a word"""
    # Remove punctuation
    word = word.translate(str.maketrans('', '', string.punctuation))
    # Remove whitespace
    word = word.strip()
    return word

def find_best_match(word: str, context: str = "") -> Optional[str]:
    """Find the best matching word using mapping.json and Gemini"""
    # Clean the word first
    clean_word_str = clean_word(word)
    logger.info(f"🔍 Starting word matching process for: '{word}' (cleaned: '{clean_word_str}') (Context: '{context}')")
    
    # First try exact case-insensitive match in available words
    word_upper = clean_word_str.upper()
    logger.info(f"Attempting exact match for: '{word_upper}'")
    
    if word_upper in available_words:
        logger.info(f"✅ Found exact match in mapping.json: '{word_upper}'")
        return word_upper
        
    # If no exact match, use Gemini to find similar word from available words
    prompt = f"""You are a word similarity expert for Indian Sign Language (ISL) dictionary. 
    Your task is to find the most appropriate word to search for the sign of: "{clean_word_str}"
    The word appears in this context: "{context}"

    Available words in the dictionary: {', '.join(available_words[:10])}... (and {len(available_words)-10} more)

    Rules:
    1. Return ONLY a word from the available dictionary words, nothing else
    2. Consider emotional context and synonyms
    3. If the word is a name or proper noun, return it as is
    4. If the word has multiple meanings, choose the most common one
    5. If the word is complex, break it down into simpler words
    6. For compound words, you may split them if individual signs would be clearer
    7. IGNORE past tense and future tense - use present tense only
    8. For pronouns (I, me, my) always use 'ME'
    
    Return format: word"""
    
    try:
        # Get response from Gemini
        logger.info(f"🤖 Consulting Gemini for semantic match of: '{clean_word_str}'")
        response = model.generate_content(prompt)
        similar_word = response.text.strip().upper()
        
        if similar_word in available_words:
            logger.info(f"✨ Gemini found matching word: '{similar_word}' (original: '{clean_word_str}')")
            return similar_word
        else:
            logger.warning(f"⚠️ No matching word found for: '{clean_word_str}'")
            return None
        
    except Exception as e:
        logger.error(f"❌ Error using Gemini for word '{clean_word_str}': {str(e)}")
        return None

def search_youtube_videos(word: str) -> Optional[str]:
    """Search for videos on the ISL YouTube channel using YouTube API"""
    try:
        logger.info(f"🎥 Starting YouTube search for word: '{word}'")
        
        # First get the channel ID for @isldictionary
        logger.info("🔍 Looking up ISL Dictionary channel ID")
        channel_request = youtube.search().list(
            part='snippet',
            q='@isldictionary',
            type='channel',
            maxResults=1
        )
        channel_response = channel_request.execute()
        
        if not channel_response['items']:
            logger.warning("⚠️ Could not find ISL Dictionary channel")
            return None
            
        channel_id = channel_response['items'][0]['id']['channelId']
        logger.info(f"✅ Found channel ID: {channel_id}")
        
        # Search for videos in the channel
        logger.info(f"🔍 Searching for videos matching: '{word}'")
        request = youtube.search().list(
            part='snippet',
            channelId=channel_id,
            q=word,
            type='video',
            maxResults=1,
            order='relevance'
        )
        response = request.execute()
        
        if response['items']:
            video_id = response['items'][0]['id']['videoId']
            video_title = response['items'][0]['snippet']['title']
            logger.info(f"✅ Found video for '{word}': {video_id} (Title: {video_title})")
            return f'https://www.youtube.com/embed/{video_id}'
        
        logger.warning(f"⚠️ No videos found for: '{word}'")
        return None
        
    except HttpError as e:
        logger.error(f"❌ YouTube API error for word '{word}': {str(e)}")
        return None
    except Exception as e:
        logger.error(f"❌ Error searching YouTube for '{word}': {str(e)}")
        return None

def get_youtube_url(word: str, context: str = "") -> Optional[str]:
    """Get YouTube URL for a word using YouTube search"""
    try:
        logger.info(f"🔍 Starting video search process for word: '{word}'")
        
        # First try finding a better match using mapping.json and Gemini
        search_word = find_best_match(word, context)
        if not search_word:
            logger.warning(f"⚠️ No matching word found for: '{word}'")
            return None
            
        logger.info(f"✨ Using search word: '{search_word}' (original: '{word}')")
        
        # Try searching YouTube with the matched word
        url = search_youtube_videos(search_word)
        if url:
            logger.info(f"✅ Found direct video match for '{word}'")
            return url
            
        logger.warning(f"⚠️ No video found for word: '{word}'")
        return None
        
    except Exception as e:
        logger.error(f"❌ Error getting video for '{word}': {str(e)}")
        return None

def translate_to_isl(english_text):
    """Use Gemini to translate English to ISL grammar"""
    logger.info(f"🌐 Starting translation for text: '{english_text}'")
    prompt = f"""
    Convert this English sentence to Indian Sign Language (ISL) grammar:
    "{english_text}"
    
    Rules:
    1. Return only the words in ISL order, separated by spaces
    2. IGNORE past tense and future tense - use present tense only
    3. For pronouns (I, me, my) always use 'ME'
    4. Do not include any explanations or additional text
    
    Return format: words separated by spaces"""
    
    try:
        logger.info("🤖 Consulting Gemini for translation")
        response = model.generate_content(prompt)
        words = response.text.strip().split()
        logger.info(f"✨ Translation result: {words}")
        return words
    except Exception as e:
        logger.error(f"❌ Error in translation: {str(e)}")
        return None

def main():
    st.title("English to Indian Sign Language Translator")
    
    # Input text
    english_text = st.text_input("Enter English text:")
    
    if st.button("Translate"):
        if english_text:
            with st.spinner("Translating..."):
                # Translate to ISL grammar
                isl_words = translate_to_isl(english_text)
                
                if isl_words:
                    st.write("ISL Translation:", " ".join(isl_words))
                    
                    # Display videos
                    for word in isl_words:
                        st.write(f"Sign for: {word}")
                        
                        # Get video URL
                        url = get_youtube_url(word, english_text)
                        
                        if url:
                            st.components.v1.iframe(
                                url,
                                height=315,
                                scrolling=False
                            )
                        else:
                            st.warning(f"No video found for: {word}")
                else:
                    st.error("Translation failed. Please try again.")
        else:
            st.warning("Please enter some text to translate.")

if __name__ == "__main__":
    main() 