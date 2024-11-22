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
import asyncio
from concurrent.futures import ThreadPoolExecutor
import base64
from pathlib import Path

logging.basicConfig(level=logging.INFO)
load_dotenv()

class VideoPlayer:
    def __init__(self):
        self.cap = None
        self.is_playing = False
        self.current_frame_number = 0
        self.total_frames = 0
        self.fps = 0
        
    def load_video(self, video_path):
        """Load a video file and get its properties."""
        if self.cap is not None:
            self.cap.release()
            
        self.cap = cv2.VideoCapture(video_path)
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps = int(self.cap.get(cv2.CAP_PROP_FPS))
        self.current_frame_number = 0
        return self.total_frames, self.fps
        
    def get_frame(self, frame_number=None):
        """Get a specific frame or the next frame."""
        if self.cap is None:
            return None
            
        if frame_number is not None:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
            self.current_frame_number = frame_number
        
        ret, frame = self.cap.read()
        if ret:
            self.current_frame_number += 1
            return frame
        return None
        
    def release(self):
        """Release video resources."""
        if self.cap is not None:
            self.cap.release()
            self.cap = None
            self.is_playing = False

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
        self.video_player = VideoPlayer()
        self.current_video_path = None
        self.is_playing = False
        self.processing_thread = None
        
    def set_alert_callback(self, callback):
        self.alert_callback = callback

    def add_alert(self, msg):
        timestamp = datetime.now().strftime("%H:%M:%S")
        alert = f"[{timestamp}] {msg}"
        self.alerts.append(alert)
        return "\n".join(self.alerts[-5:])

    def update_ppe_items(self, items):
        self.selected_ppe = [item.lower() for item in items]
        return f"Now detecting: {', '.join(items)}"

    def analyze_frame(self, frame):
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
            
            # Check for missing PPE
            if self.selected_ppe:
                missing_ppe = [item for item in self.selected_ppe if item not in detected_items]
                if missing_ppe:
                    alert_msg = f"⚠️ Missing PPE: {', '.join(missing_ppe)}"
                    if self.alert_callback:
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
            logging.error(f"Error analyzing frame: {str(e)}")
            return frame

    async def play_video_with_detection(self, video_path, progress=gr.Progress()):
        """Play video with real-time PPE detection."""
        if not video_path:
            return None, "No video selected"
            
        try:
            # Load video
            self.current_video_path = video_path
            total_frames, fps = self.video_player.load_video(video_path)
            self.is_detecting = True
            self.is_playing = True
            
            # Create temporary file for output
            temp_output = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
            output_path = temp_output.name
            
            # Get video properties
            width = int(self.video_player.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(self.video_player.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            # Initialize video writer
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
            
            frame_count = 0
            loop = asyncio.get_event_loop()
            
            # Process frames
            while self.is_playing and frame_count < total_frames:
                frame = self.video_player.get_frame()
                if frame is None:
                    break
                    
                # Process frame in thread pool
                processed_frame = await loop.run_in_executor(
                    ThreadPoolExecutor(),
                    self.analyze_frame,
                    frame
                )
                
                out.write(processed_frame)
                frame_count += 1
                
                # Update progress
                progress((frame_count / total_frames), desc="Processing video")
                
                # Control playback speed
                await asyncio.sleep(1/fps)
            
            # Cleanup
            out.release()
            self.video_player.release()
            self.is_detecting = False
            self.is_playing = False
            
            if frame_count == 0:
                return None, "Error processing video"
                
            return output_path, "Video processing complete"
            
        except Exception as e:
            self.is_detecting = False
            self.is_playing = False
            error_msg = f"Error processing video: {str(e)}"
            logging.error(error_msg)
            return None, error_msg

    def stop_video(self):
        """Stop video playback and processing."""
        self.is_playing = False
        self.is_detecting = False
        self.video_player.release()
        return None, "Video stopped"

    def start_detection(self):
        """Start webcam detection."""
        self.is_detecting = True
        try:
            self.cap = cv2.VideoCapture(0)
            if not self.cap.isOpened():
                return "Error: Could not open webcam", gr.update(interactive=True), gr.update(interactive=False)
            
            return "Detection started", gr.update(interactive=False), gr.update(interactive=True)
        except Exception as e:
            return f"Error: {str(e)}", gr.update(interactive=True), gr.update(interactive=False)

    def stop_detection(self):
        """Stop webcam detection."""
        self.is_detecting = False
        try:
            if self.cap:
                self.cap.release()
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
            This system detects Personal Protective Equipment in real-time.
            """)
        
        # PPE Selection
        ppe_items = gr.CheckboxGroup(
            choices=detector.ppe_list,
            label="Select PPE Items to Detect",
            value=["Glasses"]
        )

        # Alert component
        alert = gr.Textbox(label="Status", interactive=False)
        
        # Control Buttons for Webcam
        with gr.Row():
            detect_btn = gr.Button("Start Webcam", variant="primary")
            stop_btn = gr.Button("Stop Webcam", variant="stop")

        # Input and Output Video Components
        with gr.Tabs():
            with gr.TabItem("Webcam"):
                with gr.Row():
                    webcam_input = gr.Image(sources=["webcam"], streaming=True, height=400, width=180)
                    webcam_output = gr.Image(label="Detection Output", streaming=True, height=400, width=180)
            
            with gr.TabItem("Video Upload"):
                with gr.Row():
                    video_input = gr.Video(label="Upload Video")
                    video_output = gr.Video(label="Processed Video")
                with gr.Row():
                    play_btn = gr.Button("Play with Detection", variant="primary")
                    stop_video_btn = gr.Button("Stop Video", variant="stop")
        
        # Button Interactions
        def start_detection():
            detection_result = detector.start_detection()
            return [detection_result[0], *detection_result[1:]]

        def stop_detection():
            stop_result = detector.stop_detection()
            return [stop_result[0], *stop_result[1:]]

        # Define click events
        detect_btn.click(
            fn=start_detection,
            outputs=[alert, detect_btn, stop_btn]
        )

        stop_btn.click(
            fn=stop_detection,
            outputs=[alert, detect_btn, stop_btn]
        )

        play_btn.click(
            fn=detector.play_video_with_detection,
            inputs=[video_input],
            outputs=[video_output, alert]
        )

        stop_video_btn.click(
            fn=detector.stop_video,
            outputs=[video_output, alert]
        )

        # Set initial states
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