import os
import cv2
import roboflow
import numpy as np
import supervision as sv
from typing import List, Dict
from dotenv import load_dotenv
from supervision import BoxAnnotator, LabelAnnotator,MaskAnnotator
import gradio as gr

class VehicleDamageDetector:
    def __init__(self, api_key: str, project_id: str, version: int = 1):
        self.rf = roboflow.Roboflow(api_key=api_key)
        try:
            project = self.rf.workspace().project(project_id)
            self.model = project.version(version).model
        except Exception as e:
            print(f"Error loading Roboflow model: {e}")
            raise

    def detect_damages(self, image_path: str) -> Dict:
        try:
            predictions = self.model.predict(image_path, confidence=40)
            detections = []
            for prediction in predictions.json()['predictions']:
                detections.append({
                    'class': prediction['class'],
                    'confidence': prediction['confidence'],
                    'bbox': {
                        'x': prediction['x'],
                        'y': prediction['y'],
                        'width': prediction['width'],
                        'height': prediction['height']
                    }
                })
            return {
                'total_damages': len(detections),
                'detections': detections
            }
        except Exception as e:
            print(f"Detection error: {e}")
            return {'error': str(e)}

    def visualize_damages(self, image_path: str) -> np.ndarray:
        image = cv2.imread(image_path)
        results = self.detect_damages(image_path)
        if results.get('detections'):
            detections = sv.Detections(
                xyxy=np.array([
                    [
                        det['bbox']['x'] - det['bbox']['width']/2,
                        det['bbox']['y'] - det['bbox']['height']/2,
                        det['bbox']['x'] + det['bbox']['width']/2,
                        det['bbox']['y'] + det['bbox']['height']/2
                    ] for det in results['detections']
                ]),
                confidence=np.array([det['confidence'] for det in results['detections']]),
                class_id=np.array([0] * len(results['detections']))
            )
            labels = [item["class"] for item in results['detections']]
            label_annotator = sv.LabelAnnotator()
            #mask_annotator = sv.MaskAnnotator()
            box_annotator = sv.BoxAnnotator()
            
            # working
            annotated_image = box_annotator.annotate(scene=image, detections=detections)
            annotated_image = label_annotator.annotate(scene=annotated_image, detections=detections, labels=labels)


            #annotated_image = mask_annotator.annotate(scene=image, detections=detections)
            #annotated_image = label_annotator.annotate(scene=annotated_image, detections=detections)

            return annotated_image
        return image

def scan_image(image_path):
    API_KEY = os.environ['ROBOFLOW_API']
    PROJECT_ID = os.environ['CAR_DAMAGE_DETECTION']
    detector = VehicleDamageDetector(API_KEY, PROJECT_ID)
    annotated_image = detector.visualize_damages(image_path)
    return annotated_image

def main():
    load_dotenv()

    with gr.Blocks(title="🚗 AI Vehicle Damage detector", theme="soft")  as demo:
        gr.Markdown("""
                # AI Vehicle Damage detector 
                This is a Python-based application that uses computer vision and machine learning to detect and visualize vehicle damages
            """)

        with gr.Accordion("Overview", open=False):
            gr.Markdown("""
                - Roboflow for AI model
                - OpenCV for image processing
                - Gradio for web interface
                - Supervision library for annotation
            """)

            image = gr.Image(value="architecture.png", label="Architecture")
       
        with gr.Accordion("Technical Implementations", open=False):
            gr.Markdown("""
                - Image is uploaded via Gradio interface and initializes the Roboflow detector
                - Roboflow model analyzes image for damages and give back detection results
                - Supervision library annotates image and give us back with detection results with a bounding boxes
                
            """)

        with gr.Row():
            with gr.Column():
                image_input = gr.Image(type="filepath", label="Upload Image")
                scan_button = gr.Button("Scan")
                
            with gr.Column():
                image_output = gr.Image(label="Annotated Image")
                cancel_button = gr.Button("Cancel")

        scan_button.click(fn=scan_image, inputs=image_input, outputs=image_output)
        cancel_button.click(fn=lambda: None, inputs=None, outputs=image_output)

        
        
        with gr.Accordion("Limitations", open=False):
            gr.Markdown("""
                - Detection accuracy depends on model quality
                - Currently set to detect at 40% confidence
                - Relies on pre-trained Roboflow model
            """)

    demo.launch()

if __name__ == '__main__':
    main()