import cv2
import gradio as gr
import time
import os
from azure.cognitiveservices.vision.computervision import ComputerVisionClient
from msrest.authentication import CognitiveServicesCredentials
from PIL import Image
import io
import numpy as np
from dotenv import load_dotenv
import threading
from datetime import datetime
import logging
import tempfile

logging.basicConfig(level=logging.INFO)

# Load environment variables
load_dotenv()

class PPEDetector:
    def __init__(self):
        self.ppe_list = [
            "Glasses",
            "Helmet",
            "Safety glasses", 
            "Face shield", 
            "Hard hat", 
            "Earplugs",
            "Earmuffs",
            "Gloves",
            "Steel-toed boots",
            "Protective clothing",
            "Masks",
            "Reflective vests"
        ]
                  
        """Initialize the PPE detector with Azure credentials."""
        self.subscription_key = os.getenv('AZURE_SUBSCRIPTION_KEY')
        self.endpoint = os.getenv('AZURE_ENDPOINT')
        
        if not self.subscription_key or not self.endpoint:
            raise ValueError("Azure credentials not found in environment variables")
        
        self.client = ComputerVisionClient(
            self.endpoint,
            CognitiveServicesCredentials(self.subscription_key)
        )
        self.selected_ppe = []
        self.alerts = []
        self.alert_callback = None
        self.is_detecting = False
        self.cap = None
        self.out = None
        self.processed_video_path = None

    def set_alert_callback(self, callback):
        """Set callback for alerts."""
        self.alert_callback = callback

    def add_alert(self, msg):
        """Add alert message."""
        timestamp = datetime.now().strftime("%H:%M:%S")
        alert = f"[{timestamp}] {msg}"
        self.alerts.append(alert)
        return "\n".join(self.alerts[-5:])  # Show last 5 alerts

    def set_ppe_items(self, items):
        """Set the PPE items to detect."""
        self.selected_ppe = items
    
    def update_ppe_items(self, items):
        """Update PPE items to detect."""
        self.set_ppe_items([item.lower() for item in items])
        return f"Now detecting: {', '.join(items)}"

    def analyze_frame(self, frame):
        """Analyze a single frame for PPE detection."""
        if frame is None or not self.is_detecting:
            return frame
        
        # Convert frame to bytes
        is_success, buffer = cv2.imencode(".jpg", frame)
        io_buf = io.BytesIO(buffer)
        
        try:
            # Detect PPE
            detect_result = self.client.detect_objects_in_stream(io_buf)
            
            # Process results and draw bounding boxes
            annotated_frame = frame.copy()
            detected_items = []
            
            if detect_result.objects:
                for obj in detect_result.objects:
                    object_name = obj.object_property.lower()
                    logging.info(f'object detected: {object_name}')
                    
                    # Skip if not in selected items
                    if self.selected_ppe and object_name not in self.selected_ppe:
                        continue
                        
                    detected_items.append(object_name)
                    
                    # Draw bounding box
                    x = obj.rectangle.x
                    y = obj.rectangle.y
                    w = obj.rectangle.w
                    h = obj.rectangle.h
                    
                    cv2.rectangle(
                        annotated_frame,
                        (x, y),
                        (x + w, y + h),
                        (0, 255, 0),
                        2
                    )
                    
                    # Add label with confidence score
                    label = f"{object_name}: {obj.confidence:.2f}"
                    cv2.putText(
                        annotated_frame,
                        label,
                        (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0, 255, 0),
                        2
                    )
            
            # Check for missing PPE and trigger alert
            if self.selected_ppe and self.alert_callback:
                missing_ppe = [item for item in self.selected_ppe if item not in detected_items]
                if missing_ppe:
                    alert_msg = f"⚠️ Missing PPE: {', '.join(missing_ppe)}"
                    self.alert_callback(alert_msg)
                    
                    # Add alert text to frame
                    cv2.putText(
                        annotated_frame,
                        alert_msg,
                        (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (0, 0, 255),
                        2
                    )
            
            return annotated_frame
            
        except Exception as e:
            print(f"Error analyzing frame: {str(e)}")
            return frame

    def process_video(self, video_path):
        """Process uploaded video file for PPE detection."""
        if not video_path:
            return None
            
        try:
            # Create temporary file for output
            temp_output = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
            self.processed_video_path = temp_output.name
            
            # Open video file
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                raise ValueError("Could not open video file")
                
            # Get video properties
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = int(cap.get(cv2.CAP_PROP_FPS))
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            # Initialize video writer
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(self.processed_video_path, fourcc, fps, (width, height))
            
            # Process each frame
            frame_count = 0
            self.is_detecting = True
            
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                    
                # Process frame
                processed_frame = self.analyze_frame(frame)
                out.write(processed_frame)
                
                # Update progress
                frame_count += 1
                progress = (frame_count / total_frames) * 100
                if self.alert_callback:
                    self.alert_callback(f"Processing video: {progress:.1f}%")
                    
            # Clean up
            cap.release()
            out.release()
            self.is_detecting = False
            
            if self.alert_callback:
                self.alert_callback("Video processing complete")
                
            return self.processed_video_path
            
        except Exception as e:
            if self.alert_callback:
                self.alert_callback(f"Error processing video: {str(e)}")
            return None

    def start_detection(self):
        self.is_detecting = True
        try:
            self.cap = cv2.VideoCapture(0)
            if not self.cap.isOpened():
                return "Error: Could not open webcam", gr.update(interactive=True), gr.update(interactive=False)
            
            frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            self.out = cv2.VideoWriter('recording.avi', 
                                     cv2.VideoWriter_fourcc(*'XVID'), 
                                     20.0, 
                                     (frame_width, frame_height))
            return "Detection started", gr.update(interactive=False), gr.update(interactive=True)
        except Exception as e:
            return f"Error: {str(e)}", gr.update(interactive=True), gr.update(interactive=False)

    def stop_detection(self):
        self.is_detecting = False
        try:
            if self.cap:
                self.cap.release()
            if self.out:
                self.out.release()
            cv2.destroyAllWindows()
            return "Detection stopped", gr.update(interactive=True), gr.update(interactive=False)
        except Exception as e:
            return f"Error while stopping: {str(e)}", gr.update(interactive=True), gr.update(interactive=False)

# Create PPE Detector instance
detector = PPEDetector()

def create_interface():
    with gr.Blocks(title="Workplace PPE Detector") as demo:
        gr.Markdown("""
            # PPE Detection System
            This system detects the following Personal Protective Equipment:            
            """)
        
        # PPE Selection
        ppe_items = gr.CheckboxGroup(
            choices=detector.ppe_list,
            label="Select PPE Items to Detect",
            value=["Glasses"]
        )

        # Alert component
        alert = gr.Textbox(label="Status", interactive=False)
        
        

        # Input and Output Video Components
        with gr.Tabs():
            with gr.TabItem("Webcam"):
                with gr.Row():
                    webcam_input = gr.Image(sources=["webcam"], streaming=True, height=400, width=180)
                    webcam_output = gr.Image(label="Detection Output", streaming=True, height=400, width=180)

            # Control Buttons
            with gr.Row():
                detect_btn = gr.Button("Start Webcam", variant="primary")
                stop_btn = gr.Button("Stop Webcam", variant="stop")
            
            with gr.TabItem("Video Upload"):
                with gr.Row():
                    video_input = gr.Video(label="Upload Video")
                    video_output = gr.Video(label="Processed Video")
                process_btn = gr.Button("Process Video", variant="primary")
        
        # Button Interactions
        def start_detection():
            detection_result = detector.start_detection()
            return [detection_result[0], *detection_result[1:]]

        def stop_detection():
            stop_result = detector.stop_detection()
            return [stop_result[0], *stop_result[1:]]

        def process_video_file(video_path):
            if not video_path:
                return [None, "Please upload a video first"]
            processed_path = detector.process_video(video_path)
            return [processed_path, "Video processing complete"]

        # Define click events
        detect_btn.click(
            fn=start_detection,
            outputs=[alert, detect_btn, stop_btn]
        )

        stop_btn.click(
            fn=stop_detection,
            outputs=[alert, detect_btn, stop_btn]
        )

        process_btn.click(
            fn=process_video_file,
            inputs=[video_input],
            outputs=[video_output, alert]
        )

        # Set initial state
        detect_btn.interactive = True
        stop_btn.interactive = False
        
        # PPE selection change handler
        ppe_items.change(detector.update_ppe_items, inputs=[ppe_items], outputs=alert)
        
        # Set up alert callback
        detector.set_alert_callback(lambda msg: detector.add_alert(msg))
           
        # Stream processing for webcam
        webcam_input.stream(
            detector.analyze_frame,                
            inputs=webcam_input,
            outputs=webcam_output,
            stream_every=1
        )

    return demo

# Launch the interface
if __name__ == "__main__":
    demo = create_interface()
    demo.launch()