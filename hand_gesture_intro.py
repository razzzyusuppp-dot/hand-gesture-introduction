import cv2
import mediapipe as mp
import numpy as np
from collections import deque

# Initialize MediaPipe
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5
)

# Introduction text based on finger count
introductions = {
    1: "Perkenalkan nama saya Ezza Raditya Endrian Putra",
    2: "Saya dari kelas 1B",
    3: "Dari kelompok 1 ANCHORAE",
    4: "Motivasi dan alasan saya ingin mengembangkan minat dan bakat saya di ilmu komputer",
    5: "TERIMA KASIHH"
}

# Colors
colors = {
    1: (255, 0, 0),    # Blue
    2: (0, 255, 0),    # Green
    3: (0, 0, 255),    # Red
    4: (255, 255, 0),  # Cyan
    5: (255, 0, 255)   # Magenta
}

def count_fingers(hand_landmarks, handedness):
    """
    Count the number of fingers raised
    Returns the count of raised fingers
    """
    fingers = []
    
    # Thumb (comparing x coordinates)
    if handedness == "Right":
        if hand_landmarks[4].x < hand_landmarks[3].x:
            fingers.append(1)
        else:
            fingers.append(0)
    else:  # Left hand
        if hand_landmarks[4].x > hand_landmarks[3].x:
            fingers.append(1)
        else:
            fingers.append(0)
    
    # Other 4 fingers (comparing y coordinates)
    finger_tips = [8, 12, 16, 20]
    finger_pips = [6, 10, 14, 18]
    
    for tip, pip in zip(finger_tips, finger_pips):
        if hand_landmarks[tip].y < hand_landmarks[pip].y:
            fingers.append(1)
        else:
            fingers.append(0)
    
    return sum(fingers)

def main():
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: Cannot open camera")
        return
    
    # Smooth finger count
    finger_count_buffer = deque(maxlen=5)
    current_intro = ""
    current_color = (255, 255, 255)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Flip the frame for selfie view
        frame = cv2.flip(frame, 1)
        h, w, c = frame.shape
        
        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process hand detection
        results = hands.process(rgb_frame)
        
        if results.multi_hand_landmarks and results.multi_handedness:
            hand_landmarks = results.multi_hand_landmarks[0]
            handedness = results.multi_handedness[0].classification[0].label
            
            # Draw hand landmarks
            mp_drawing.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS
            )
            
            # Count fingers
            finger_count = count_fingers(hand_landmarks.landmark, handedness)
            finger_count_buffer.append(finger_count)
            
            # Get most common finger count
            if len(finger_count_buffer) > 0:
                most_common = max(set(finger_count_buffer), key=finger_count_buffer.count)
                
                if most_common in introductions:
                    current_intro = introductions[most_common]
                    current_color = colors[most_common]
        
        # Display current introduction
        if current_intro:
            # Create a semi-transparent background for text
            text_size = cv2.getTextSize(current_intro, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)[0]
            text_x = (w - text_size[0]) // 2
            text_y = 60
            
            # Draw background rectangle
            cv2.rectangle(frame, 
                         (text_x - 10, text_y - text_size[1] - 10),
                         (text_x + text_size[0] + 10, text_y + 10),
                         current_color, -1)
            
            # Draw text
            cv2.putText(frame, current_intro,
                       (text_x, text_y),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
        
        # Display instructions at bottom
        instructions = "1 Jari: Nama | 2 Jari: Kelas | 3 Jari: Kelompok | 4 Jari: Motivasi | 5 Jari: Terima Kasih"
        cv2.putText(frame, instructions,
                   (10, h - 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        # Display finger count in corner
        if results.multi_hand_landmarks:
            finger_count = max(set(finger_count_buffer), key=finger_count_buffer.count) if len(finger_count_buffer) > 0 else 0
            cv2.putText(frame, f"Jari: {finger_count}",
                       (w - 150, 40),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2)
        
        # Show frame
        cv2.imshow('Hand Gesture Introduction', frame)
        
        # Press 'q' to exit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
