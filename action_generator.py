import cv2
import mediapipe as mp
import numpy as np
import yt_dlp
import os
import urllib.request

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
        
        # Load character image
        self.character_img = self.load_character_image()
        
    def load_character_image(self):
        # Create a simple character with transparent background
        img = np.zeros((200, 200, 4), dtype=np.uint8)
        
        # Draw character parts with transparency
        # Head
        cv2.circle(img, (100, 50), 30, (255, 220, 200, 255), -1)
        
        # Body (shirt)
        cv2.rectangle(img, (70, 80), (130, 150), (100, 150, 255, 255), -1)
        
        # Arms
        cv2.rectangle(img, (40, 80), (70, 120), (255, 220, 200, 255), -1)  # Left arm
        cv2.rectangle(img, (130, 80), (160, 120), (255, 220, 200, 255), -1)  # Right arm
        
        # Legs (pants)
        cv2.rectangle(img, (70, 150), (90, 200), (50, 50, 150, 255), -1)  # Left leg
        cv2.rectangle(img, (110, 150), (130, 200), (50, 50, 150, 255), -1)  # Right leg
        
        # Face features
        cv2.circle(img, (80, 40), 5, (0, 0, 0, 255), -1)  # Left eye
        cv2.circle(img, (120, 40), 5, (0, 0, 0, 255), -1)  # Right eye
        cv2.ellipse(img, (100, 60), (15, 5), 0, 0, 180, (0, 0, 0, 255), 2)  # Mouth
        
        # Make background transparent
        img[:, :, 3] = 255  # Set alpha channel to fully opaque
        
        return img

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

def download_youtube_video(url):
    try:
        ydl_opts = {
            'format': 'best[ext=mp4]',
            'outtmpl': '%(id)s.%(ext)s',
        }
        
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=True)
            video_id = info['id']
            video_path = f"{video_id}.mp4"
            print(f"Downloaded: {info['title']}")
            return video_path
    except Exception as e:
        print(f"Error downloading video: {e}")
        return None

def main():
    # Get YouTube URL from user
    youtube_url = input("Enter YouTube video URL: ")
    
    # Download the video
    video_path = download_youtube_video(youtube_url)
    if not video_path:
        print("Failed to download video")
        return
    
    # Initialize video capture
    cap = cv2.VideoCapture(video_path)
    
    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    half_frames = total_frames // 2
    
    print(f"Total frames: {total_frames}")
    print(f"Processing first {half_frames} frames")
    
    # Create video writer - only need half width since we're not showing original video
    output_path = "animated_sign_language.mp4"
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    # Initialize tracker
    tracker = HandTracker()
    
    frame_count = 0
    while frame_count < half_frames:
        success, frame = cap.read()
        if not success:
            print("End of video")
            break
        
        # Process frame
        hand_results, pose_results, face_results = tracker.process_frame(frame)
        
        # Create animated frame
        animation = tracker.create_animated_frame(frame.shape)
        
        # Write only the animated frame to video file
        out.write(animation)
        
        # Resize for display if needed
        if animation.shape[1] > 1920:
            scale = 1920 / animation.shape[1]
            display_frame = cv2.resize(animation, None, fx=scale, fy=scale)
        else:
            display_frame = animation
        
        # Add instructions and progress
        progress = (frame_count / half_frames) * 100
        cv2.putText(display_frame, f"Progress: {progress:.1f}%", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(display_frame, "Press 'q' to quit", (10, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # Show frame
        cv2.imshow("Sign Language Animation", display_frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
        frame_count += 1
    
    # Cleanup
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    
    print(f"Animation saved as {output_path}")
    print(f"Processed {frame_count} frames out of {total_frames} total frames")
    
    # Remove downloaded video
    if os.path.exists(video_path):
        os.remove(video_path)
        print("Temporary video file removed")

if __name__ == "__main__":
    main() 