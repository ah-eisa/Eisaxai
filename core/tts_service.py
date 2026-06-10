import os
from elevenlabs import VoiceSettings
from elevenlabs.client import ElevenLabs
import re

class TTSService:
    def __init__(self):
        self.client = ElevenLabs(api_key=os.getenv("ELEVENLABS_API_KEY", ""))
        
        # Voice IDs - Professional natural voices
        self.voices = {
            "en": "JBFqnCBsd6RMkjVDRZzb",  # George - Professional male English
            "ar": "L10lEremDiJfPicq5CPh",  # Yasmine - Egyptian female Arabic
        }
    
    def clean_text_for_speech(self, text: str) -> str:
        """Remove markdown formatting for natural speech"""
        
        # Remove markdown headers
        text = re.sub(r'#{1,6}\s+', '', text)
        
        # Remove bold/italic markers
        text = re.sub(r'[*_]{1,2}([^*_]+)[*_]{1,2}', r'\1', text)
        
        # Remove bullet points
        text = re.sub(r'^\s*[-*+]\s+', '', text, flags=re.MULTILINE)
        
        # Remove numbered lists
        text = re.sub(r'^\s*\d+\.\s+', '', text, flags=re.MULTILINE)
        
        # Remove extra whitespace and newlines
        text = re.sub(r'\n\s*\n', '. ', text)
        text = re.sub(r'\s+', ' ', text)
        
        # Remove URLs
        text = re.sub(r'http[s]?://\S+', '', text)
        
        return text.strip()
    
    def generate_speech(self, text: str, language: str = "en") -> bytes:
        """Generate realistic speech audio from text"""
        
        # Clean text for natural speech
        clean_text = self.clean_text_for_speech(text)
        
        # Limit text length (10K chars/month free tier)
        if len(clean_text) > 500:
            clean_text = clean_text[:500] + "..."
        
        # Select voice based on language
        voice_id = self.voices.get(language, self.voices["en"])
        
        # Generate audio with ElevenLabs
        audio = self.client.text_to_speech.convert(
            voice_id=voice_id,
            optimize_streaming_latency="0",
            output_format="mp3_44100_128",
            text=clean_text,
            model_id="eleven_turbo_v2_5",  # Fast & high quality
            voice_settings=VoiceSettings(
                stability=0.5,
                similarity_boost=0.75,
                style=0.0,
                use_speaker_boost=True,
            ),
        )
        
        # Collect audio bytes
        audio_bytes = b""
        for chunk in audio:
            if chunk:
                audio_bytes += chunk
        
        return audio_bytes
