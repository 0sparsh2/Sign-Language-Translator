import cv2
import mediapipe as mp
import numpy as np
import yt_dlp
import os
import random
import logging
import subprocess
import json
from pathlib import Path

logger = logging.getLogger(__name__)

def get_random_user_agent():
    user_agents = [
        'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
        'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
        'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:89.0) Gecko/20100101 Firefox/89.0',
        'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/14.1.1 Safari/605.1.15'
    ]
    return random.choice(user_agents)

def get_yt_dlp_path():
    """Get the full path to yt-dlp executable."""
    import site
    return str(Path(site.USER_BASE) / "bin" / "yt-dlp")

def get_video_formats(yt_dlp_path, url):
    """Get available video formats."""
    try:
        cmd = [
            yt_dlp_path,
            url,
            "--dump-json",
            "--no-playlist",
            "--user-agent", "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            logger.error(f"Failed to get video formats: {result.stderr}")
            return None
            
        video_info = json.loads(result.stdout)
        formats = video_info.get('formats', [])
        
        # Filter for video formats
        video_formats = [f for f in formats if f.get('vcodec', 'none') != 'none']
        
        if not video_formats:
            logger.error("No video formats found")
            return None
            
        # Sort by quality (resolution)
        video_formats.sort(key=lambda x: int(x.get('height', 0)), reverse=True)
        
        return video_formats[0]['format_id']  # Return the highest quality format
        
    except Exception as e:
        logger.error(f"Error getting video formats: {str(e)}")
        return None

def download_video(url, output_path="."):
    """Download a YouTube video using yt-dlp command line."""
    try:
        logger.info(f"Starting download of: {url}")
        
        # Get yt-dlp path
        yt_dlp_path = get_yt_dlp_path()
        if not Path(yt_dlp_path).exists():
            logger.error(f"yt-dlp not found at {yt_dlp_path}")
            return None
            
        # Get best video format
        format_id = get_video_formats(yt_dlp_path, url)
        if not format_id:
            logger.error("Could not determine video format")
            return None
            
        # Ensure output directory exists
        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Construct the yt-dlp command
        cmd = [
            yt_dlp_path,
            url,
            "-f", format_id,
            "--output", str(output_path / "%(title)s.%(ext)s"),
            "--no-playlist",
            "--user-agent", "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        ]
        
        # Execute the command
        logger.info("Executing yt-dlp command...")
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True
        )
        
        # Stream the output in real-time
        while True:
            output = process.stdout.readline()
            if output == '' and process.poll() is not None:
                break
            if output:
                logger.info(output.strip())
        
        # Get the return code and any error output
        return_code = process.poll()
        errors = process.stderr.read()
        
        if return_code != 0:
            logger.error(f"yt-dlp failed with return code {return_code}")
            if errors:
                logger.error(f"Error output: {errors}")
            return None
            
        # Find the downloaded file (most recent video file in the directory)
        video_files = list(output_path.glob("*.*"))
        if not video_files:
            logger.error("No video file found after download")
            return None
            
        downloaded_file = max(video_files, key=lambda x: x.stat().st_mtime)
        
        if not downloaded_file.exists() or downloaded_file.stat().st_size == 0:
            logger.error("Downloaded file is empty or does not exist")
            if downloaded_file.exists():
                downloaded_file.unlink()  # Delete empty file
            return None
            
        logger.info(f"Successfully downloaded: {downloaded_file}")
        return str(downloaded_file)
        
    except Exception as e:
        logger.error(f"Error downloading video: {str(e)}")
        return None

class HandTracker:
    def __init__(self):
        # Initialize MediaPipe Hands, Pose, and Face
        self.mp_hands = mp.solutions.hands
        self.mp_pose = mp.solutions.pose
        self.mp_face = mp.solutions.face_mesh
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5
        )
        self.pose = self.mp_pose.Pose(
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5
        )
        self.face = self.mp_face.FaceMesh(
            max_num_faces=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5
        )
        self.mp_draw = mp.solutions.drawing_utils

    def process_frame(self, frame):
        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process the frame for hands, pose, and face
        self.hand_results = self.hands.process(rgb_frame)
        self.pose_results = self.pose.process(rgb_frame)
        self.face_results = self.face.process(rgb_frame)
        
        return self.hand_results, self.pose_results, self.face_results

    def create_animated_frame(self, frame_shape):
        # Create a light blue background
        h, w = frame_shape[:2]
        animated_frame = np.zeros((h, w, 3), dtype=np.uint8)
        animated_frame[:, :] = [200, 230, 255]  # Light blue background
            
        # Draw pose if detected
        if self.pose_results.pose_landmarks:
            landmarks = self.pose_results.pose_landmarks.landmark
            
            # Calculate body proportions
            shoulder_width = abs(landmarks[11].x - landmarks[12].x) * w
            body_height = abs(landmarks[0].y - landmarks[28].y) * h
            
            # Draw head with skin color
            nose = landmarks[0]
            nose_pos = (int(nose.x * w), int(nose.y * h))
            head_radius = int(shoulder_width * 0.4)
            cv2.circle(animated_frame, nose_pos, head_radius, (255, 220, 200), -1)
            
            # Draw neck
            neck_start = (int(landmarks[0].x * w), int(landmarks[0].y * h + head_radius))
            neck_end = (int(landmarks[0].x * w), int(landmarks[0].y * h + head_radius * 1.5))
            cv2.line(animated_frame, neck_start, neck_end, (255, 220, 200), 3)
            
            # Draw torso with shirt color
            torso_points = [
                (int(landmarks[11].x * w), int(landmarks[11].y * h)),
                (int(landmarks[12].x * w), int(landmarks[12].y * h)),
                (int(landmarks[24].x * w), int(landmarks[24].y * h)),
                (int(landmarks[23].x * w), int(landmarks[23].y * h))
            ]
            cv2.fillPoly(animated_frame, [np.array(torso_points)], (100, 150, 255))  # Blue shirt
            
            # Draw arms with skin color and natural curves
            arm_connections = [
                (11, 13, 15),  # Left arm
                (12, 14, 16)   # Right arm
            ]
            
            for connection in arm_connections:
                start = landmarks[connection[0]]
                mid = landmarks[connection[1]]
                end = landmarks[connection[2]]
                
                # Create curved lines for arms using Bezier curves
                points = []
                for t in np.linspace(0, 1, 10):
                    # Quadratic Bezier curve
                    x = int((1-t)**2 * start.x * w + 2*(1-t)*t * mid.x * w + t**2 * end.x * w)
                    y = int((1-t)**2 * start.y * h + 2*(1-t)*t * mid.y * h + t**2 * end.y * h)
                    points.append([x, y])
                
                # Draw the curved arm with natural thickness
                points = np.array(points, np.int32)
                cv2.polylines(animated_frame, [points], False, (255, 220, 200), 5)
            
            # Draw legs with pants color and natural curves
            leg_connections = [
                (23, 25, 27),  # Left leg
                (24, 26, 28)   # Right leg
            ]
            
            for connection in leg_connections:
                start = landmarks[connection[0]]
                mid = landmarks[connection[1]]
                end = landmarks[connection[2]]
                
                # Create curved lines for legs using Bezier curves
                points = []
                for t in np.linspace(0, 1, 10):
                    # Quadratic Bezier curve
                    x = int((1-t)**2 * start.x * w + 2*(1-t)*t * mid.x * w + t**2 * end.x * w)
                    y = int((1-t)**2 * start.y * h + 2*(1-t)*t * mid.y * h + t**2 * end.y * h)
                    points.append([x, y])
                
                # Draw the curved leg with natural thickness
                points = np.array(points, np.int32)
                cv2.polylines(animated_frame, [points], False, (50, 50, 150), 5)  # Dark blue pants

        # Draw face if detected
        if self.face_results.multi_face_landmarks:
            for face_landmarks in self.face_results.multi_face_landmarks:
                # Draw eyes
                left_eye = face_landmarks.landmark[33]
                right_eye = face_landmarks.landmark[263]
                eye_radius = int(head_radius * 0.1)
                
                cv2.circle(animated_frame, 
                          (int(left_eye.x * w), int(left_eye.y * h)), 
                          eye_radius, (0, 0, 0), -1)  # Black eyes
                cv2.circle(animated_frame, 
                          (int(right_eye.x * w), int(right_eye.y * h)), 
                          eye_radius, (0, 0, 0), -1)
                
                # Draw mouth with natural curve
                mouth_top = face_landmarks.landmark[13]
                mouth_bottom = face_landmarks.landmark[14]
                mouth_width = int(head_radius * 0.4)
                mouth_height = int(head_radius * 0.2)
                
                # Create curved mouth using Bezier curve
                mouth_points = []
                for t in np.linspace(0, 1, 10):
                    # Quadratic Bezier curve for smile
                    x = int(mouth_top.x * w + mouth_width * (t - 0.5))
                    y = int(mouth_top.y * h + mouth_height * 4 * t * (1-t))
                    mouth_points.append([x, y])
                
                # Draw the curved mouth
                mouth_points = np.array(mouth_points, np.int32)
                cv2.polylines(animated_frame, [mouth_points], False, (0, 0, 0), 2)

        # Draw hands with skin color and natural curves
        if self.hand_results.multi_hand_landmarks:
            for hand_landmarks in self.hand_results.multi_hand_landmarks:
                # Draw palm
                palm_points = []
                for i in [0, 5, 9, 13, 17]:
                    palm_points.append([int(hand_landmarks.landmark[i].x * w), 
                                      int(hand_landmarks.landmark[i].y * h)])
                cv2.fillPoly(animated_frame, [np.array(palm_points)], (255, 220, 200))
                
                # Draw fingers with natural curves
                for finger in [[0, 1, 2, 3, 4], [0, 5, 6, 7, 8], 
                             [0, 9, 10, 11, 12], [0, 13, 14, 15, 16], 
                             [0, 17, 18, 19, 20]]:
                    points = []
                    for i in range(len(finger)-1):
                        start = hand_landmarks.landmark[finger[i]]
                        end = hand_landmarks.landmark[finger[i+1]]
                        
                        # Create curved lines for fingers
                        for t in np.linspace(0, 1, 5):
                            x = int(start.x * w + t * (end.x * w - start.x * w))
                            y = int(start.y * h + t * (end.y * h - start.y * h))
                            points.append([x, y])
                    
                    # Draw the curved finger
                    points = np.array(points, np.int32)
                    cv2.polylines(animated_frame, [points], False, (255, 220, 200), 2)

        # Apply some post-processing effects
        animated_frame = cv2.GaussianBlur(animated_frame, (3, 3), 0)
        
        return animated_frame

    def process_video(self, video_path, output_path="animated_sign_language.mp4"):
        """Process a video file and create an animated version."""
        try:
            logger.info(f"Starting video processing for file: {video_path}")
            
            try:
                # Initialize video capture
                logger.info("Initializing video capture")
                cap = cv2.VideoCapture(video_path)
                if not cap.isOpened():
                    logger.error("Failed to open video file")
                    return None
                
                # Get video properties
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                fps = int(cap.get(cv2.CAP_PROP_FPS))
                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                half_frames = total_frames // 2
                
                if width == 0 or height == 0 or fps == 0:
                    logger.error("Invalid video properties")
                    cap.release()
                    return None
                
                logger.info(f"Video properties - Width: {width}, Height: {height}, FPS: {fps}, Total frames: {total_frames}")
                logger.info(f"Processing first {half_frames} frames")
                
                # Create output directory if it doesn't exist
                os.makedirs(os.path.dirname(output_path), exist_ok=True)
                
                # Create video writer
                logger.info(f"Creating output video at: {output_path}")
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
                if not out.isOpened():
                    logger.error("Failed to create output video writer")
                    cap.release()
                    return None
                
                frame_count = 0
                while frame_count < half_frames:
                    success, frame = cap.read()
                    if not success:
                        logger.info("End of video reached")
                        break
                    
                    # Process frame
                    hand_results, pose_results, face_results = self.process_frame(frame)
                    
                    # Create animated frame
                    animation = self.create_animated_frame(frame.shape)
                    
                    # Write the animated frame
                    out.write(animation)
                    
                    frame_count += 1
                    if frame_count % 10 == 0:  # Log progress every 10 frames
                        logger.info(f"Processed {frame_count}/{half_frames} frames")
                
                # Cleanup
                logger.info("Cleaning up resources")
                cap.release()
                out.release()
                
                # Verify output file
                if not os.path.exists(output_path) or os.path.getsize(output_path) == 0:
                    logger.error("Failed to create output video file")
                    return None
                
                logger.info(f"Animation saved as {output_path}")
                logger.info(f"Processed {frame_count} frames out of {total_frames} total frames")
                
                return output_path
                
            except Exception as e:
                logger.error(f"Error processing video frames: {str(e)}")
                return None
            
        except Exception as e:
            logger.error(f"Error in process_video: {str(e)}")
            return None

def main():
    # Get YouTube URL from user
    youtube_url = input("Enter YouTube video URL: ")
    
    # Initialize tracker
    tracker = HandTracker()
    
    # Process the video
    output_path = tracker.process_video(youtube_url)
    if not output_path:
        print("Failed to process video")
        return
    
    print(f"Animation saved as {output_path}")

if __name__ == "__main__":
    main() 
