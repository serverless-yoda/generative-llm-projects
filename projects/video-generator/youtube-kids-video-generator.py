import os
import uuid
from typing import List, Dict
from azure.core.credentials import AzureKeyCredential
from azure.cognitiveservices.speech import AudioDataStream, SpeechConfig, SpeechSynthesizer, ResultReason
from azure.cognitiveservices.vision.computervision import ComputerVisionClient
from azure.cognitiveservices.vision.computervision.models import VisualFeatureTypes
from azure.ai.textanalytics import TextAnalyticsClient
from azure.identity import DefaultAzureCredential
from moviepy.editor import ImageClip, AudioFileClip, CompositeVideoClip
import random
import json

from dotenv import load_dotenv

class KidsVideoGenerator:
    def __init__(self, 
                 azure_speech_key: str, 
                 azure_speech_endpoint: str,
                 azure_speech_region: str, 
                 azure_cv_endpoint: str, 
                 azure_cv_key: str):
        """
        Initialize the Kids Video Generator with Azure service credentials
        
        Args:
            azure_speech_key (str): Azure Speech Service key
            azure_speech_region (str): Azure Speech Service region
            azure_cv_endpoint (str): Azure Computer Vision endpoint
            azure_cv_key (str): Azure Computer Vision key
        """
        print(azure_speech_key)
        print(azure_speech_endpoint)
        print(azure_speech_region)

        # Speech synthesis configuration
        speech_config = SpeechConfig(subscription=azure_speech_key, region=azure_speech_region)
        
        self.speech_synthesizer = SpeechSynthesizer(speech_config=speech_config)
        
        credential = AzureKeyCredential(azure_cv_key)
        # cv_client = ImageAnalysisClient(
        #     endpoint=self.cognitive_service_url,
        #     credential=credential
        # )

        # Computer Vision client
        self.cv_client = ComputerVisionClient(
            endpoint=azure_cv_endpoint, 
            credentials=DefaultAzureCredential()
        )
        
        # Ensure output directories exist
        os.makedirs('output/audio', exist_ok=True)
        os.makedirs('output/images', exist_ok=True)
        os.makedirs('output/videos', exist_ok=True)

    def generate_stem_story(self) -> Dict:
        """
        Generate a STEM-based educational story for children
        
        Returns:
            Dict containing story content
        """
        stem_topics = [
            {
                "title": "The Amazing World of Robotics",
                "content": "In a small town, a young girl named Emma loved building robots. She learned that robots can help people in many ways, like assisting doctors or exploring dangerous places. Emma's first robot could help elderly people pick up objects they dropped.",
                "moral": "Technology can be used to help and support others."
            },
            # Add more STEM stories here
        ]
        return random.choice(stem_topics)

    def text_to_speech(self, text: str, filename: str) -> str:
        """
        Convert text to speech using Azure Speech Service
        
        Args:
            text (str): Text to convert
            filename (str): Output audio filename
        
        Returns:
            str: Path to generated audio file
        """
        audio_path = f'output/audio/{filename}.wav'
        result = self.speech_synthesizer.speak_text_async(text).get()
        
        if result.reason == ResultReason.SynthesizingAudioCompleted:
            stream = AudioDataStream(result)
            stream.save_to_wav_file(audio_path)
        
        return audio_path

    def generate_educational_images(self, story_content: str) -> List[str]:
        """
        Generate or select educational images related to the story
        
        Args:
            story_content (str): Story text to generate images for
        
        Returns:
            List of image paths
        """
        # Placeholder for image generation logic
        # Could use Azure Computer Vision or external image generation services
        placeholder_images = [
            'resources/stem_image1.jpg',
            'resources/stem_image2.jpg',
            'resources/stem_image3.jpg'
        ]
        return placeholder_images[:3]  # Return first 3 images

    def create_video(self, images: List[str], audio_path: str, output_filename: str) -> str:
        """
        Create video from images and audio
        
        Args:
            images (List[str]): List of image paths
            audio_path (str): Path to audio file
            output_filename (str): Output video filename
        
        Returns:
            str: Path to generated video
        """
        clips = [
            ImageClip(img).set_duration(5)  # 5 seconds per image
            for img in images
        ]
        
        audio_clip = AudioFileClip(audio_path)
        final_clip = CompositeVideoClip(clips)
        final_clip = final_clip.set_audio(audio_clip)
        
        video_path = f'output/videos/{output_filename}.mp4'
        final_clip.write_videofile(video_path, fps=24)
        
        return video_path

    def generate_video(self) -> Dict:
        """
        Generate a complete educational video for kids
        
        Returns:
            Dict with video generation details
        """
        # Generate story
        story = self.generate_stem_story()
        
        # Convert text to speech
        unique_id = str(uuid.uuid4())
        audio_path = self.text_to_speech(
            f"{story['title']}. {story['content']} {story['moral']}", 
            f"story_{unique_id}"
        )
        
        # Generate images
        images = self.generate_educational_images(story['content'])
        
        # Create video
        video_path = self.create_video(
            images, 
            audio_path, 
            f"kids_video_{unique_id}"
        )
        
        return {
            "title": story['title'],
            "video_path": video_path,
            "moral": story['moral']
        }

def main():
    # Load Azure credentials from environment or config
    generator = KidsVideoGenerator(
        azure_speech_key=os.getenv('azure_speech_key'),
        azure_speech_endpoint=os.getenv('azure_speech_endpoint'),
        azure_speech_region=os.getenv('location_region'),
        azure_cv_endpoint=os.getenv('azure_vision_endpoint'),
        azure_cv_key=os.getenv('azure_vision_key')
    )
    
    # Generate video
    video_details = generator.generate_video()
    print(f"Generated Video: {video_details['title']}")
    print(f"Moral of the Story: {video_details['moral']}")

if __name__ == "__main__":
    load_dotenv()
    main()
