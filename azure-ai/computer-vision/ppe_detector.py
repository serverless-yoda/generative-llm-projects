from dotenv import load_dotenv
import os
import cv2
import time
import os
from azure.cognitiveservices.vision.computervision import ComputerVisionClient
from msrest.authentication import CognitiveServicesCredentials
from PIL import Image
import io
import numpy as np


_ = load_dotenv()


class PPEDetector:
    def __init__(self, subscription_key, endpoint):
        """Initialize the PPE detector with Azure credentials."""
        self.client = ComputerVisionClient(
            endpoint,
            CognitiveServicesCredentials(subscription_key)
        )
        
    def analyze_frame(self, frame):
        """Analyze a single frame for PPE detection."""
        # Convert frame to bytes
        is_success, buffer = cv2.imencode(".jpg", frame)
        io_buf = io.BytesIO(buffer)
        
        # Detect PPE
        detect_result = self.client.detect_objects_in_stream(io_buf)
        
        # Process results and draw bounding boxes
        annotated_frame = frame.copy()
        if detect_result.objects:
            for object in detect_result.objects:
                # Get bounding box coordinates
                x = object.rectangle.x
                y = object.rectangle.y
                w = object.rectangle.w
                h = object.rectangle.h
                
                # Draw bounding box
                cv2.rectangle(
                    annotated_frame,
                    (x, y),
                    (x + w, y + h),
                    (0, 255, 0),
                    2
                )
                
                # Add label
                cv2.putText(
                    annotated_frame,
                    object.object_property,
                    (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 0),
                    2
                )
        
        return annotated_frame

def main():
    # Azure credentials
    subscription_key = os.getenv('AZURE_SUBSCRIPTION_KEY')
    endpoint = os.getenv('AZURE_ENDPOINT')

    
    # Initialize detector
    detector = PPEDetector(subscription_key, endpoint)
    
    # Ask user for input source
    print("Select input source:")
    print("1. Webcam")
    print("2. Video file")
    choice = input("Enter choice (1 or 2): ")
    
    if choice == "1":
        cap = cv2.VideoCapture(0)
    else:
        video_path = input("Enter video file path: ")
        cap = cv2.VideoCapture(video_path)
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Analyze frame
        annotated_frame = detector.analyze_frame(frame)
        
        # Display result
        cv2.imshow('PPE Detection', annotated_frame)
        
        # Process every 1 second to avoid Azure API rate limits
        time.sleep(1)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
