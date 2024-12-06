import os
import random
import uuid
import json
import requests
from dotenv import load_dotenv
import azure.cognitiveservices.speech as speechsdk
from azure.cognitiveservices.speech import AudioDataStream, SpeechConfig, SpeechSynthesizer
from azure.cognitiveservices.vision.computervision import ComputerVisionClient
from msrest.authentication import CognitiveServicesCredentials
from openai import OpenAI
#import moviepy.editor as mp
from PIL import Image, ImageDraw, ImageFont
import numpy as np
from scipy.io import wavfile
from openai import AzureOpenAI
from langchain_together import ChatTogether

class AzureKidsStoryVideoGenerator:
    def __init__(self, 
                 azure_speech_key, 
                 azure_speech_region, 
                 azure_vision_key, 
                 azure_vision_endpoint,
                 azure_openai_api_key,
                 azure_openai_api_endpoint,
                 azure_openai_deployment,
                 together_api_key):
                         
        # Azure Speech Service Configuration
        self.speech_config = SpeechConfig(
            subscription=azure_speech_key, 
            region=azure_speech_region
        )
        
        # Azure Neural Voices for different characters
        self.character_voices = {
            "child": [
                "en-US-AidenNeural",  # Boy voice
                "en-US-AriaNeural",   # Girl voice
            ],
            "animal": [
                "en-US-CoraNeural",   # Playful voice
            ],
            "adult": [
                "en-US-GuyNeural",    # Adult male
                "en-US-JennyNeural",  # Adult female
            ]
        }
        
        # Azure Computer Vision Configuration
        self.vision_credentials = CognitiveServicesCredentials(azure_vision_key)
        self.vision_client = ComputerVisionClient(
            azure_vision_endpoint, 
            self.vision_credentials
        )
        
        # OpenAI configuration
        #self.openai_client = OpenAI(api_key=openai_api_key)
        # self.azure_openai_deployment = azure_openai_deployment
        # self.openai_client = AzureOpenAI(
        #     api_key = azure_openai_api_key,  
        #     api_version = "2024-05-01-preview",
        #     azure_endpoint = azure_openai_api_endpoint
        # )
        self.llm = ChatTogether(model="meta-llama/Llama-3-70b-chat-hf", temperature=0)
        # Output directory setup
        os.makedirs('outputs/voices', exist_ok=True)
        os.makedirs('outputs/sounds', exist_ok=True)
        os.makedirs('outputs/illustrations', exist_ok=True)
        os.makedirs('outputs/stories', exist_ok=True)
        
    
    def generate_character_voices(self, story):
        """Generate character voices using Azure Neural TTS"""
        character_voice_files = {}
        
        for character in story['characters']:
            # Select appropriate voice type
            voice_type = character.get('voice_type', 'child')
            voice_options = self.character_voices.get(voice_type, self.character_voices['child'])
            selected_voice = random.choice(voice_options)
            
            self.speech_config.speech_synthesis_voice_name = selected_voice
            
            # Create audio output configuration
            voice_file = f"outputs/voices/{character['name']}_voice.wav"
            audio_config = speechsdk.audio.AudioOutputConfig(filename=voice_file)
            
            # Create speech synthesizer
            synthesizer = SpeechSynthesizer(
                speech_config=self.speech_config, 
                audio_config=audio_config
            )
            
            # Synthesize character dialogue
            dialogue = character.get('intro_dialogue', f"Hi, I'm {character['name']}!")
            result = synthesizer.speak_text_async(dialogue).get()
            
            if result.reason == speechsdk.ResultReason.SynthesizingAudioCompleted:
                print(f"Voice generated for {character['name']}")
                character_voice_files[character['name']] = voice_file
        
        return character_voice_files
    
    def generate_enhanced_story(self):
        """Generate a more complex, interactive story with multiple characters"""
        prompt = """Create an interactive children's story with:
        - Multiple unique characters with distinct personalities
        - Clear moral lesson about teamwork or empathy
        - Interactive elements where viewer can make choices
        - Sound effect and action suggestions
        - Character dialogue
        - Appropriate for ages 5-8
        - Include clear STEM or social-emotional learning objectives
        - Use visual storytelling techniques
        - Incorporate interactive problem-solving elements
        - Maintain age-appropriate complexity
        - Multiple unique characters with distinct personalities
        - Clear moral lesson about teamwork or empathy
        - Sound effect and action suggestions
        - Character dialogue
        
        Story format:
        {
            "title": "Story Title",
            "moral": "Lesson learned",
            "characters": [
                {
                    "name": "Character Name",
                    "personality": "Description",
                    "voice_type": "child/adult/animal",
                    "intro_dialogue": "Character's first line"
                }
            ],
            "scenes": [
                {
                    'setting': 'forest',
                    'mood': 'peaceful',
                    'characters': ['Alice', 'Bob'],
                    'primary_action': 'walking',
                    'description': 'Alice and Bob stroll through a quiet forest'
                    "dialogue": "Character dialogue",
                    "sound_effects": ["effect1", "effect2"],
                    "interactive_choices": ["Choice 1", "Choice 2"]
                }
            ]
        }"""
        
        messages = [
                (
                    "system", 
                    "IMPORTANT INSTRUCTIONS: \n"
                    "1. You MUST respond ONLY in valid JSON format\n"
                    "2. Strictly follow this JSON structure:\n"
                    "3. No additional text, comments, or explanations\n"
                    "4. Ensure valid JSON syntax\n"
                    "5. Fill out ALL fields completely\n\n"
                    "JSON SCHEMA: " + 
                    json.dumps({
                        "title": "Story Title (string)",
                        "moral": "Lesson learned (string)",
                        "characters": [
                            {
                                "name": "Character Name (string)",
                                "personality": "Character Description (string)",
                                "voice_type": "child/adult/animal (string)",
                                "intro_dialogue": "First line of dialogue (string)"
                            }
                        ],
                        "scenes": [
                            {
                                'setting': 'forest/city/indoor/dramatic (string)',
                                'mood': 'peaceful/tense/suspenseful (string)',
                                'characters': ['character1', 'character2'],
                                'primary_action': 'character action (string)',
                                "description": "Scene setting (string)",
                                "dialogue": "Key dialogue (string)",
                                "sound_effects": ["sound effect 1", "sound effect 2"],
                                "interactive_choices": ["Choice 1", "Choice 2"]
                            }
                        ]
                    }, indent=2)
                ),
                (
                    "assistant", 
                    "I understand. I will generate the story strictly in the specified JSON format."
                ),
                (
                    "human", 
                    f'{prompt}'
                )
            ]

                
        result = self.llm.invoke(messages)
        
        # try:
        #     story_json = json.loads(result.content)
        # except json.JSONDecodeError:
        #     print("Failed to parse JSON. Attempting manual extraction...")
            
        self.save_story_to_json(result)

        # print(file_contents)
        #return json.loads(story)

    def log_synthesis_error(self, result):
        """
        Log detailed error information for sound effect synthesis.
        
        Args:
            result: Speech synthesis result object
        """
        if result.reason == speechsdk.ResultReason.Canceled:
            cancellation = result.cancellation_details
            print(f"Speech synthesis canceled: {cancellation.reason}")
            print(f"Error Code: {cancellation.error_code}")
            print(f"Error Details: {cancellation.error_details}")

    def generate_sound_effect_prompt(self, scene):
        """
        Generate a descriptive prompt for sound effect synthesis based on scene context.
        
        Args:
            scene (dict): A scene dictionary from the story
        
        Returns:
            str: A description to generate an appropriate sound effect
        """
        # Extract scene details
        setting = scene.get('setting', 'default')
        mood = scene.get('mood', 'neutral')
        characters = scene.get('characters', [])
        action = scene.get('primary_action', 'generic movement')
        
        # Create a sound effect description
        sound_prompts = {
            'forest': [
                'soft rustling leaves and distant bird calls',
                'gentle wind through pine trees',
                'creek bubbling in the background'
            ],
            'city': [
                'distant traffic and urban ambiance',
                'street noise with occasional car horns',
                'cafe background chatter'
            ],
            'indoor': [
                'soft background music',
                'gentle ambient room sounds',
                'subtle electronic hum'
            ],
            'dramatic': [
                'tense orchestral underscore',
                'building suspenseful background noise',
                'ominous low-frequency rumble'
            ],
            'default':[
                'a very peaceful day, only birds chirp and wind can be heard'
            ]
        }
        
        # Select an appropriate sound prompt based on context
        base_prompt = ''
        if mood == 'tense' or mood == 'suspenseful':
            base_prompt = random.choice(sound_prompts['dramatic'])
        elif setting in sound_prompts:
            base_prompt = random.choice(sound_prompts.get(setting, sound_prompts['default']))
        else:
             base_prompt = 'neutral background ambient sound'
        
        # # Enhance prompt with scene-specific details
        if characters:
            base_prompt += f" with subtle hints of {' and '.join(characters)}"
        
        base_prompt += f" representing {action}"
        
        return base_prompt
        
    def generate_scene_sound_effects(self, story):
        """
        Generate sound effects for each scene in the story.
        
        Args:
            story (dict): A dictionary containing story scenes
        
        Returns:
            dict: A mapping of scene IDs to their generated sound effect files
        """
        scene_sound_effects = {}
        
        for scene_index, scene in enumerate(story['scenes']):
            try:
                # Generate a unique identifier for the scene
                scene_id = f"scene_{scene_index}_{uuid.uuid4()}"
                
                # Determine the sound effect based on scene description or context
                scene_description = scene.get('description', 'generic scene')
                
                # Use Azure Speech SDK to create scene ambient sound
                self.speech_config.speech_synthesis_voice_name = "en-US-GuyNeural"  # If available
                
                # Create audio output configuration
                scene_sound_file = f"outputs/scene_sounds/{scene_id}.wav"
                
                # Ensure the output directory exists
                os.makedirs(os.path.dirname(scene_sound_file), exist_ok=True)
                
                audio_config = speechsdk.audio.AudioOutputConfig(filename=scene_sound_file)
                
                #Create speech synthesizer
                synthesizer = SpeechSynthesizer(
                    speech_config=self.speech_config, 
                    audio_config=audio_config
                )
                
                # Generate sound effect description based on scene context
                sound_prompt = self.generate_sound_effect_prompt(scene)
                

                # Synthesize the sound effect
                result = synthesizer.speak_text_async(sound_prompt).get()
                
                if result.reason == speechsdk.ResultReason.SynthesizingAudioCompleted:
                    scene_sound_effects[scene_id] = {
                        'file': scene_sound_file,
                        'description': sound_prompt,
                        'voice': self.speech_config.speech_synthesis_voice_name
                    }
                    print(f"Generated sound effect for scene {scene_id}")
                else:
                    print(f"Sound effect generation failed for scene: {scene_id}")
                    self.log_synthesis_error(result)
            
            except Exception as e:
                print(f"Error generating sound effect for scene {scene_id}: {e}")
        
        return scene_sound_effects

    def _generate_sound_effects(self, story):
        """Generate sound effects using Azure Speech Synthesis with different voice configurations"""
        sound_effects = {}
        
        # Predefined sound effect types
        effect_types = {
            "magic": "A sparkling, enchanting magical sound",
            "happy": "A cheerful, bouncy melody that lifts the spirits",
            "sad": "A soft, melancholic tone that evokes sorrow",
            "adventure": "An exciting, heroic musical flourish full of energy",
            "leaves": "The crisp sound of dry leaves crunching underfoot",
            "robotbeep": "A series of mechanical, electronic beeps from a robot",
            "dig": "The gritty sound of soil being dug up",
            "waterdrip": "The rhythmic dripping of water droplets",
            "bee": "The constant, buzzing hum of a busy bee",
            "flowers": "The gentle, almost silent blooming of flowers",
            "watersplash": "The refreshing sound of water splashing",
            "cheering": "A happy and loud children cheering, full of joy"
        }
                
        for scene in story['scenes']:
            for effect in scene.get('sound_effects', []):
                if effect in effect_types:
                    try:
                        # Use a playful child voice for sound effects
                        self.speech_config.speech_synthesis_voice_name = "en-US-JennyNeural"
                        
                        # Create audio output configuration
                        effect_file = f"outputs/sounds/{effect}_{uuid.uuid4()}.wav"
                        
                        # Ensure the directory exists
                        os.makedirs(os.path.dirname(effect_file), exist_ok=True)
                        
                        audio_config = speechsdk.audio.AudioOutputConfig(filename=effect_file)
                        
                        # Create speech synthesizer
                        synthesizer = SpeechSynthesizer(
                            speech_config=self.speech_config, 
                            audio_config=audio_config
                        )
                        
                        # Synthesize sound effect description
                        result = synthesizer.speak_text_async(effect_types[effect]).get()
                        
                        if result.reason == speechsdk.ResultReason.SynthesizingAudioCompleted:
                            sound_effects[effect] = effect_file
                        else:
                            print(f"Speech synthesis failed for effect: {effect}")
                            print(f"Reason: {result.reason}")
                            
                            # If result is an error, print additional error details
                            if result.reason == speechsdk.ResultReason.Canceled:
                                cancellation = result.cancellation_details
                                print(f"Error details: {cancellation.reason}")
                                print(f"Error code: {cancellation.error_code}")
                                print(f"Error message: {cancellation.error_details}")
                    
                    except Exception as e:
                        print(f"An error occurred while generating sound effect for {effect}: {e}")
    
        return sound_effects
    
    def generate_ai_illustrations(self, story):
        """Use Azure OpenAI or DALL-E to generate scene illustrations"""
        illustrations = {}
        
        for i, scene in enumerate(story['scenes']):
            # Generate detailed prompt for image generation
            image_prompt = f"Illustration for children's story: {scene['description']}. " \
                           f"Colorful, cartoon-style, suitable for ages 5-8"
            
            response = self.openai_client.images.generate(
                model="dall-e-3",
                prompt=image_prompt,
                size="1024x1024",
                quality="standard",
                n=1
            )
            
            image_url = response.data[0].url
            image_file = f"outputs/illustrations/scene_{i}.png"
            
            # Download and save image
            with open(image_file, 'wb') as handler:
                handler.write(requests.get(image_url).content)
            
            illustrations[f"scene_{i}"] = image_file
        
        return illustrations
    
    def create_interactive_video(self, story, character_voices, sound_effects, illustrations):
        """Create an interactive video with multiple scenes and potential branching"""
        video_clips = []
        
        for scene in story['scenes']:
            # Select illustration
            illustration = illustrations[f"scene_{scene['index']}"]
            image_clip = mp.ImageClip(illustration).set_duration(10)  # 10 seconds per scene
            
            # Add character voice
            character_name = scene.get('active_character', story['characters'][0]['name'])
            voice_file = character_voices[character_name]
            voice_clip = mp.AudioFileClip(voice_file)
            
            # Add sound effects
            scene_effects = []
            for effect_name in scene.get('sound_effects', []):
                effect_clip = mp.AudioFileClip(sound_effects[effect_name])
                scene_effects.append(effect_clip)
            
            # Combine audio
            combined_audio = mp.CompositeAudioClip([voice_clip] + scene_effects)
            
            # Create interactive overlay for choices
            if scene.get('interactive_choices'):
                # Create text overlay with choices
                interactive_text = "\n".join([f"{i+1}. {choice}" for i, choice in enumerate(scene['interactive_choices'])])
                txt_clip = mp.TextClip(
                    interactive_text, 
                    fontsize=30, 
                    color='white', 
                    font='Arial', 
                    method='label'
                ).set_position(('center', 'bottom')).set_duration(10)
                
                # Overlay choices on image
                scene_clip = mp.CompositeVideoClip([image_clip, txt_clip])
            else:
                scene_clip = image_clip
            
            # Set audio
            scene_clip = scene_clip.set_audio(combined_audio)
            video_clips.append(scene_clip)
        
        # Concatenate scenes
        final_video = mp.concatenate_videoclips(video_clips)
        
        # Export video
        output_path = f"outputs/interactive_story_{uuid.uuid4()}.mp4"
        final_video.write_videofile(output_path, fps=24)
        
        return output_path

    def save_story_to_json(self,result, output_file='outputs/stories/story.json'):
        """
        Save the generated story result to a JSON file with the specified structure.
        
        :param result: The generated story result from the language model
        :param output_file: Path to the output JSON file (default: 'story.json')
        """
        try:
            # Check if result is an AIMessage object and extract its content
            if hasattr(result, 'content'):
                result_text = result.content
            elif isinstance(result, str):
                result_text = result
            else:
                raise ValueError("Unsupported result type")
            
            # Attempt to parse the result as a JSON object
            story_data = json.loads(result_text)
            
            # Validate the structure matches the expected format
            required_keys = ['title', 'moral', 'characters', 'scenes']
            for key in required_keys:
                if key not in story_data:
                    raise KeyError(f"Missing required key: {key}")
            
            # Write the validated JSON to file
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(story_data, f, indent=4, ensure_ascii=False)
            
            print(f"Story saved successfully to {output_file}")
            return story_data
        
        except json.JSONDecodeError:
            print("Error: Result is not a valid JSON")
            # Optionally, save the raw text result
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(result_text)
            print(f"Raw result saved to {output_file}")
        
        except KeyError as e:
            print(f"Validation Error: {e}")
        
        except Exception as e:
            print(f"An unexpected error occurred: {e}")
            # Optionally, print the actual result for debugging
            print("Result:", result)

    def generate_interactive_story_video(self):
        """Main method to generate entire interactive story video"""
        # Generate story structure
        #story = self.generate_enhanced_story()

        with open('outputs/stories/story.json', 'r') as file:
            data = json.load(file)

        #Generate character voices
        #character_voices = self.generate_character_voices(data)
        
        # Generate sound effects
        sound_effects = self.generate_scene_sound_effects(data)
        
        # Generate AI illustrations
        #illustrations = self.generate_ai_illustrations(story)
        
        # Create interactive video
        # video_path = self.create_interactive_video(
        #     story, 
        #     character_voices, 
        #     sound_effects, 
        #     illustrations
        # )
        
        #return video_path, story
        return None,None

def main():
    # Replace with your actual Azure and OpenAI credentials
    video_generator = AzureKidsStoryVideoGenerator(
        azure_speech_key=os.environ['AZURE_SPEECH_KEY'],
        azure_speech_region=os.environ['AZURE_SPEECH_REGION'],
        azure_vision_key=os.environ['AZURE_CV_KEY'],
        azure_vision_endpoint=os.environ['AZURE_CV_ENDPOINT'],
        azure_openai_api_key=os.environ['AZURE_OPENAI_KEY'],
        azure_openai_api_endpoint=os.environ['AZURE_OPENAI_ENDPOINT'],
        azure_openai_deployment=os.environ['AZURE_OPENAI_DEPLOYMENT'],
        together_api_key=os.environ['TOGETHER_API_KEY']
    )
    
    # Generate interactive story video
    video_path, story_details = video_generator.generate_interactive_story_video()
    # print(f"Interactive Video Generated: {video_path}")
    # print(f"Story Title: {story_details['title']}")
    # print(f"Moral Lesson: {story_details['moral']}")

if __name__ == "__main__":
    load_dotenv()
    main()

# Required Dependencies:
# pip install azure-cognitiveservices-speech azure-cognitiveservices-vision-computervision 
# pip install openai moviepy pillow scipy requests
