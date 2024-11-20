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
        self.ppe_list = [
            "Glasses",
            "Helmet",
            "Safety glasses", 
            "Face shield", 
            #"Welding mask", 
            "Hard hat", 
            "Earplugs",
            "Earmuffs",
            "Gloves",
            "Steel-toed boots",
            "Protective clothing",
            "Masks",
            "Reflective vests"]
                  
         

        """Initialize the PPE detector with Azure credentials."""
        self.subscription_key = os.getenv('AZURE_SUBSCRIPTION_KEY')
        self.endpoint = os.getenv('AZURE_ENDPOINT')
        
        if not self.subscription_key or not self.endpoint:
            raise ValueError("Azure credentials not found in environment variables")
        
        self.client = ComputerVisionClient(
            self.endpoint,
            CognitiveServicesCredentials(self.subscription_key)
        )
        #self.is_running = False
        self.selected_ppe = []
        self.alerts = []
        self.alert_callback = None
        self.is_detecting = False
        self.cap = None
        self.out = None

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

    def process_frame(self):
        if self.cap and self.out:
            self.out.write(frame)
        return frame

    def analyze_frame(self, frame):
        """Analyze a single frame for PPE detection."""
        #if self.cap and self.out:

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

    def detect_ppe(self, video_input, ppe_type):
        """
        Simulated PPE detection method
        
        Args:
            video_input (np.ndarray): Input video frame
            ppe_type (str): Type of PPE to detect
        
        Returns:
            np.ndarray: Output frame with detection results
        """
        if not self.is_detecting or not ppe_type:
            return video_input

        # Simulated detection with simple rectangle overlay
        output = video_input.copy()
        height, width = output.shape[:2]
        
        # Draw detection rectangle (simulating detection)
        cv2.rectangle(
            output, 
            (width//4, height//4), 
            (width*3//4, height*3//4), 
            (0, 255, 0), 
            2
        )
        
        # Add text for detected PPE
        cv2.putText(
            output, 
            f"Detected: {ppe_type}", 
            (10, 30), 
            cv2.FONT_HERSHEY_SIMPLEX, 
            1, 
            (0, 255, 0), 
            2
        )
        
        return output

    

    def start_detection(self):
        self.is_detecting = True
        # Start webcam
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
            self.is_running = True
            self.prev_frame = None
            return "Detection started", gr.update(interactive=False), gr.update(interactive=True)
        except Exception as e:
            return f"Error: {str(e)}", gr.update(interactive=True), gr.update(interactive=False)

    def stop_detection(self):
        self.is_detecting = False
        # Stop recording and release webcam
        try:
            self.is_running = False
            if self.cap:
                self.cap.release()
            if self.out:
                self.out.release()
            self.prev_frame = None
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
                    #choices=["Helmet", "Safety glasses", "Face shield", "Welding mask", "Hard hat", "Earplugs","Earmuffs","Gloves","Steel-toed boots","Protective clothing","Masks","Reflective vests"],
                    choices=detector.ppe_list,
                    label="Select PPE Items to Detect",
                    value=["Glasses"]
        )

        # Alert component
        alert = gr.Textbox(label="Status", interactive=False)
        
        # Control Buttons
        with gr.Row():
            detect_btn = gr.Button("Detect", variant="primary")
            stop_btn = gr.Button("Stop", variant="stop")
            

        # Input and Output Video Components
        with gr.Tabs():
            with gr.TabItem("Webcam"):
                with gr.Row():
                    # Camera feed
                    webcam_input = gr.Image(sources=["webcam"], streaming=True, height=400, width=180)
                    webcam_output = gr.Image(label="Detection Output", streaming=True,height=400, width=180)
            
            with gr.TabItem("Video"):
                video_input = gr.Video(label="Video Input")
                video_output = gr.Video(label="Detection Output")
        
        
        # Button Interactions
        def start_detection():
            detection_result = detector.start_detection()
            return [detection_result, gr.update(interactive=False), gr.update(interactive=True)]

        def stop_detection():
            stop_result = detector.stop_detection()
            return [stop_result, gr.update(interactive=True), gr.update(interactive=False)]

        # Define the click events
        detect_btn.click(
            fn=start_detection,
            outputs=[alert, detect_btn, stop_btn]
        )

        stop_btn.click(
            fn=stop_detection,
            outputs=[alert, detect_btn, stop_btn]
        )

        # Set initial state - detect button enabled, stop button disabled
        detect_btn.interactive = True
        stop_btn.interactive = False
        
        ppe_items.change(detector.update_ppe_items, inputs=[ppe_items], outputs=alert)
        
        # Set up alert callback
        detector.set_alert_callback(lambda msg: detector.add_alert(msg))
           

        # Stream processing
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
