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
import tempfile
import cv2
from Video_generator.action_generator import HandTracker, download_video
import moviepy.editor as mpe
import shutil
from dotenv import load_dotenv
import requests
from bs4 import BeautifulSoup
from googleapiclient.discovery import build
import datetime

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

# Add this after the imports
COMMON_WORD_VIDEOS = {
    'FAMILY': 'QwTqm5nM-mU'
}

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
    """Search YouTube for videos matching the word"""
    try:
        logger.info(f"🔍 Searching YouTube for: '{word}'")
        
        # Build the service
        youtube = build('youtube', 'v3', developerKey=YOUTUBE_API_KEY)
        
        # First get the channel ID for @isldictionary
        logger.info("🔍 Looking up ISL Dictionary channel ID")
        channel_request = youtube.search().list(
            part='snippet',
            q='@isldictionary',
            type='channel',
            maxResults=1
        )
        channel_response = channel_request.execute()
        
        if not channel_response.get('items'):
            logger.warning("⚠️ Could not find ISL Dictionary channel")
            return None
            
        channel_id = channel_response['items'][0]['id']['channelId']
        logger.info(f"✅ Found ISL Dictionary channel ID: {channel_id}")
        
        # Search request
        request = youtube.search().list(
            part="snippet",
            q=f"{word} Indian Sign Language",
            type="video",
            channelId=channel_id,
            maxResults=1,
            videoDuration="short",
            relevanceLanguage="en"
        )
        
        # Execute the request
        response = request.execute()
        
        if not response.get('items'):
            logger.warning(f"⚠️ No videos found for: '{word}'")
            return None
            
        video_id = response['items'][0]['id']['videoId']
        video_title = response['items'][0]['snippet']['title']
        video_url = f"https://www.youtube.com/watch?v={video_id}"
        logger.info(f"✅ Found video: '{video_title}'")
        logger.info(f"📺 Video URL: {video_url}")
        logger.info(f"🆔 Video ID: {video_id}")
        
        return video_id
        
    except HttpError as e:
        logger.error(f"❌ YouTube API error: {str(e)}")
        return None
    except Exception as e:
        logger.error(f"❌ Error searching YouTube: {str(e)}")
        return None

def get_youtube_url(word: str, context: str = "") -> Optional[str]:
    """Get YouTube URL for a word using YouTube search or fallback to common words"""
    try:
        logger.info(f"🔍 Starting video search process for word: '{word}'")
        
        # First try finding a better match using mapping.json and Gemini
        search_word = find_best_match(word, context)
        if not search_word:
            logger.warning(f"⚠️ No matching word found for: '{word}'")
            return None
            
        logger.info(f"✨ Using search word: '{search_word}' (original: '{word}')")
        
        # Check if word exists in common words dictionary
        if search_word in COMMON_WORD_VIDEOS:
            video_id = COMMON_WORD_VIDEOS[search_word]
            logger.info(f"✅ Found video ID in common words dictionary: {video_id}")
            return video_id
        
        # Try searching YouTube with the matched word
        try:
            url = search_youtube_videos(search_word)
            if url:
                logger.info(f"✅ Found direct video match for '{word}'")
                return url
        except HttpError as e:
            if "quota" in str(e).lower():
                logger.warning("⚠️ YouTube API quota exceeded, falling back to common words")
                # If quota exceeded and word exists in common words, use that
                if search_word in COMMON_WORD_VIDEOS:
                    video_id = COMMON_WORD_VIDEOS[search_word]
                    logger.info(f"✅ Using fallback video ID: {video_id}")
                    return video_id
            else:
                raise e
            
        logger.warning(f"⚠️ No video found for word: '{word}'")
        return None
        
    except Exception as e:
        logger.error(f"❌ Error getting video for '{word}': {str(e)}")
        return None

def process_video(url):
    try:
        # Create a temporary directory for the output
        with tempfile.TemporaryDirectory() as temp_dir:
            # Download the video first
            video_path = download_video(url, temp_dir)
            if not video_path:
                logger.error("Failed to download video")
                return None
                
            # Initialize HandTracker
            tracker = HandTracker()
            
            # Generate unique output filename using timestamp
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            output_filename = f"animated_sign_language_{timestamp}.mp4"
            output_path = os.path.join(temp_dir, output_filename)
            
            # Process the video
            result_path = tracker.process_video(video_path, output_path)
            if not result_path:
                logger.error("Failed to process video")
                return None
                
            # Copy the result to a permanent location
            final_path = os.path.join("animated_videos", output_filename)
            os.makedirs("animated_videos", exist_ok=True)
            shutil.copy2(result_path, final_path)
            
            # Convert video to a format that's more widely supported
            try:
                video = mpe.VideoFileClip(final_path)
                compatible_path = os.path.join("animated_videos", f"compatible_{output_filename}")
                video.write_videofile(compatible_path, codec='libx264', audio=False)
                video.close()
                return compatible_path
            except Exception as e:
                logger.error(f"Error converting video: {str(e)}")
                return final_path
            
    except Exception as e:
        logger.error(f"Error processing video: {str(e)}")
        return None

def get_isl_translation(text: str) -> Optional[List[str]]:
    """Translate English text to a list of ISL words using Gemini."""
    try:
        logger.info(f"🔤 Starting translation for text: '{text}'")
        
        # Create prompt for Gemini
        prompt = f"""Translate the following English text into individual Indian Sign Language (ISL) words.
        Break down the text into individual words that can be signed.
        Return ONLY the words separated by spaces, nothing else.
        
        Text: "{text}"
        
        Rules:
        1. Return ONLY words that exist in the ISL dictionary
        2. Break down complex words into simpler ones
        3. Remove articles (a, an, the) and prepositions
        4. Convert verbs to their base form
        5. For proper nouns, return them as is
        6. For numbers, write them as words (e.g., "one" instead of "1")
        7. For punctuation, ignore it unless it's part of the word
        8. Return words in the order they should be signed
        
        Example:
        Input: "Hello, how are you today?"
        Output: "HELLO HOW YOU TODAY"
        
        Your translation:"""
        
        # Get response from Gemini
        response = model.generate_content(prompt)
        translation = response.text.strip().upper()
        
        # Split into words and clean
        words = [word.strip() for word in translation.split() if word.strip()]
        
        if not words:
            logger.warning(f"⚠️ No words found in translation for: '{text}'")
            return None
            
        logger.info(f"✅ Translated to ISL words: {words}")
        return words
        
    except Exception as e:
        logger.error(f"❌ Error in translation: {str(e)}")
        return None

def translate_to_isl(text: str) -> Tuple[Optional[str], Optional[List[str]]]:
    """Translate English text to ISL with video generation."""
    try:
        logger.info(f"🌐 Starting translation for text: '{text}'")
        
        # Get ISL translation
        isl_words = get_isl_translation(text)
        if not isl_words:
            return None, None
            
        # Create temporary directory for videos
        with tempfile.TemporaryDirectory() as temp_dir:
            # Process each word's video
            video_paths = []
            words = []  # Keep track of words for each video
            for word in isl_words:
                # Get URL from mapping.json
                url = get_youtube_url(word, text)
                if url:
                    output_path = os.path.join(temp_dir, f"{word}.mp4")
                    video_path = process_video(url)
                    if video_path:
                        video_paths.append(video_path)
                        words.append(word)
            
            if not video_paths:
                return None, None
                
            # Combine all videos into one
            try:
                # Create VideoFileClip objects for each video
                clips = []
                for path, word in zip(video_paths, words):
                    # Load the video clip
                    video = mpe.VideoFileClip(path)
                    
                    # Create text clip
                    txt_clip = mpe.TextClip(
                        word, 
                        font='Arial-Bold',
                        fontsize=48,
                        color='white',
                        stroke_color='black',
                        stroke_width=2
                    )
                    
                    # Position text above the character (adjust y position as needed)
                    txt_clip = txt_clip.set_position(('center', 50))
                    txt_clip = txt_clip.set_duration(video.duration)
                    
                    # Combine video and text
                    final_clip = mpe.CompositeVideoClip([video, txt_clip])
                    clips.append(final_clip)
                
                # Concatenate all clips
                final_clip = mpe.concatenate_videoclips(clips)
                
                # Generate unique output filename
                timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                final_path = os.path.join("animated_videos", f"combined_{timestamp}.mp4")
                os.makedirs("animated_videos", exist_ok=True)
                
                # Write the final video
                final_clip.write_videofile(final_path, codec='libx264', audio=False)
                
                # Close all clips to free resources
                for clip in clips:
                    clip.close()
                final_clip.close()
                
                return final_path, isl_words
                
            except Exception as e:
                logger.error(f"Error combining videos: {str(e)}")
                # If combination fails, return the first video
                return video_paths[0], isl_words
                
    except Exception as e:
        logger.error(f"Error in translation process: {e}")
        return None, None

def main():
    st.title("English to Indian Sign Language Translator")
    
    # Input text
    text = st.text_input("Enter English text:")
    
    if st.button("Translate"):
        if text:
            # Show a loading spinner while processing
            with st.spinner('Processing...'):
                # Translate and get video
                video_path, _ = translate_to_isl(text)
                
                if video_path:
                    # Display only the combined video
                    st.video(video_path, format='video/mp4')
                else:
                    st.error("Translation failed. Please try again.")
        else:
            st.warning("Please enter some text to translate.")

if __name__ == "__main__":
    main() 
