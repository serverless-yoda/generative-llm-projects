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

logging.basicConfig(level=logging.INFO)

# Load environment variables
load_dotenv()

class PPEDetector:
    def __init__(self):
        """Initialize the PPE detector with Azure credentials."""
        self.subscription_key = os.getenv('AZURE_SUBSCRIPTION_KEY')
        self.endpoint = os.getenv('AZURE_ENDPOINT')
        
        if not self.subscription_key or not self.endpoint:
            raise ValueError("Azure credentials not found in environment variables")
        
        self.client = ComputerVisionClient(
            self.endpoint,
            CognitiveServicesCredentials(self.subscription_key)
        )
        self.is_running = False
        self.selected_ppe = []
        self.alert_callback = None
        
    def set_ppe_items(self, items):
        """Set the PPE items to detect."""
        self.selected_ppe = items
        
    def set_alert_callback(self, callback):
        """Set callback for alerts."""
        self.alert_callback = callback
        
    def analyze_frame(self, frame):
        """Analyze a single frame for PPE detection."""
        
        if frame is None or not self.is_running:
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
                    
                    # Get bounding box coordinates
                    x = obj.rectangle.x
                    y = obj.rectangle.y
                    w = obj.rectangle.w
                    h = obj.rectangle.h
                    
                    # Draw bounding box
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

def process_webcam(frame):
    """Process webcam frame and return annotated frame."""
    detector = PPEDetector()
    return detector.analyze_frame(frame)

class PPEInterface:
    def __init__(self):
        self.detector = PPEDetector()
        self.alerts = []
        
    def start_detection(self):
        """Start PPE detection."""
        self.detector.is_running = True
        return "Detection Started"
        
    def stop_detection(self):
        """Stop PPE detection."""
        self.detector.is_running = False
        return "Detection Stopped"
        
    def update_ppe_items(self, items):
        """Update PPE items to detect."""
        self.detector.set_ppe_items([item.lower() for item in items])
        return f"Now detecting: {', '.join(items)}"
        
    def add_alert(self, msg):
        """Add alert message."""
        timestamp = datetime.now().strftime("%H:%M:%S")
        alert = f"[{timestamp}] {msg}"
        self.alerts.append(alert)
        return "\n".join(self.alerts[-5:])  # Show last 5 alerts
        
    def process_webcam(self, frame):
        """Process webcam frame."""
        return self.detector.analyze_frame(frame)
    
    def create_interface(self):
         

        """Create Gradio interface."""
        with gr.Blocks(title="Workplace PPE Detector") as interface:
            gr.Markdown("""
            # PPE Detection System
            This system detects the following Personal Protective Equipment:
            - Hard Hat (Head Protection)
            - Face Mask
            - Safety Vest
            """)
            
            with gr.Row():
                with gr.Column(scale=2):
                    # PPE Selection
                    ppe_items = gr.CheckboxGroup(
                        #choices=["Helmet", "Safety glasses", "Face shield", "Welding mask", "Hard hat", "Earplugs","Earmuffs","Gloves","Steel-toed boots","Protective clothing","Masks","Reflective vests"],
                        choices=["Helmet", "glasses"],
                        label="Select PPE Items to Detect",
                        value=["glasses"]
                    )
                    
                    # Control buttons
                    with gr.Row():
                        start_btn = gr.Button("Start Detection", variant="primary")
                        stop_btn = gr.Button("Stop Detection", variant="secondary")
                    
                    # Alert area
                    alerts = gr.Textbox(
                        label="Alerts",
                        placeholder="Alert messages will appear here...",
                        lines=5
                    )
                
                with gr.Column(scale=3):
                    # Camera feed
                    webcam = gr.Image(sources=["webcam"], streaming=True)
                    output = gr.Image(label="Detection Output", streaming=True)
            
            # Event handlers
            start_btn.click(self.start_detection, outputs=alerts)
            stop_btn.click(self.stop_detection, outputs=alerts)
            ppe_items.change(self.update_ppe_items, inputs=[ppe_items], outputs=alerts)
            
            # Set up alert callback
            self.detector.set_alert_callback(lambda msg: self.add_alert(msg))
            
            # Stream processing
            webcam.stream(
                self.process_webcam,                
                inputs=webcam,
                outputs=output,
                stream_every=1
            )
            
           
        
        return interface

if __name__ == "__main__":
    # Create and launch the interface
    interface = PPEInterface()
    app = interface.create_interface()
    app.launch(share=True)
