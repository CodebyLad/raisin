import asyncio
import tempfile
import wave
import numpy as np
import sounddevice as sd
import openai
import os
import platform
from typing import Optional, Dict, Any
import threading
from queue import Queue
import time

class CustomRealtimeModel:
    """Custom realtime model that mimics OpenAI's RealtimeModel interface"""
    
    def __init__(self, model: str = "gpt-4o", voice: str = "alloy", **kwargs):
        self.model = model
        self.voice = voice
        self.openai_client = openai.OpenAI()
        
        # Audio settings
        self.sample_rate = 16000
        self.channels = 1
        self.chunk_duration = 0.1
        self.buffer_duration = 3
        self.audio_buffer = []
        self.is_listening = False
        self.is_processing = False
        
        # Session state
        self.session_active = False
        self.conversation_history = []
        
        print(f"🤖 Custom Realtime Model initialized with {model} and {voice} voice")
    
    async def connect(self):
        """Connect to the model (mimics RealtimeModel interface)"""
        print("🔌 Connecting to Custom Realtime Model...")
        self.session_active = True
        
        # Start audio processing
        await self.start_audio_processing()
        return self
    
    async def disconnect(self):
        """Disconnect from the model"""
        print("🔌 Disconnecting from Custom Realtime Model...")
        self.session_active = False
        self.is_listening = False
    
    def audio_callback(self, indata, frames, time, status):
        """Audio input callback"""
        if status:
            print(f"Audio status: {status}")
        
        if self.is_listening and self.session_active:
            self.audio_buffer.append(indata.copy())
    
    async def start_audio_processing(self):
        """Start the audio processing loop"""
        self.is_listening = True
        
        # Start audio stream
        self.audio_stream = sd.InputStream(
            samplerate=self.sample_rate,
            channels=self.channels,
            callback=self.audio_callback,
            blocksize=int(self.sample_rate * self.chunk_duration),
            dtype=np.float32
        )
        
        self.audio_stream.start()
        
        # Start processing task
        asyncio.create_task(self.process_audio_continuously())
        
        print("🎤 Audio processing started - listening continuously...")
    
    async def process_audio_continuously(self):
        """Continuously process audio buffer"""
        while self.session_active:
            try:
                if (len(self.audio_buffer) >= int(self.buffer_duration / self.chunk_duration) 
                    and not self.is_processing):
                    await self.process_audio_buffer()
                
                await asyncio.sleep(0.1)
                
            except Exception as e:
                print(f"❌ Audio processing error: {e}")
                await asyncio.sleep(1)
    
    async def process_audio_buffer(self):
        """Process accumulated audio"""
        if self.is_processing or not self.audio_buffer:
            return
        
        self.is_processing = True
        
        try:
            # Combine audio chunks
            combined_audio = np.concatenate(self.audio_buffer)
            self.audio_buffer.clear()
            
            # Check for meaningful audio
            if np.max(np.abs(combined_audio)) > 0.01:
                print("🎤 Processing speech...")
                
                # Transcribe
                transcript = await self.transcribe_audio(combined_audio)
                
                if transcript and transcript.strip():
                    print(f"📝 User: {transcript}")
                    
                    # Generate response
                    response = await self.generate_response(transcript)
                    
                    if response:
                        print(f"💬 AI: {response}")
                        await self.speak_response(response)
        
        except Exception as e:
            print(f"❌ Processing error: {e}")
        finally:
            self.is_processing = False
    
    async def transcribe_audio(self, audio_data) -> str:
        """Transcribe audio using Whisper"""
        try:
            # Convert to WAV
            audio_int16 = (audio_data * 32767).astype(np.int16)
            temp_file = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
            
            with wave.open(temp_file.name, 'wb') as wav_file:
                wav_file.setnchannels(self.channels)
                wav_file.setsampwidth(2)
                wav_file.setframerate(self.sample_rate)
                wav_file.writeframes(audio_int16.tobytes())
            
            # Transcribe
            with open(temp_file.name, 'rb') as audio_file:
                response = await self.openai_client.audio.transcriptions.create(
                    model="whisper-1",
                    file=audio_file,
                    response_format="text"
                )
            
            # Cleanup
            os.unlink(temp_file.name)
            return response.strip()
            
        except Exception as e:
            print(f"❌ Transcription error: {e}")
            return ""
    
    async def generate_response(self, text: str) -> str:
        """Generate AI response"""
        try:
            # Add to conversation history
            self.conversation_history.append({"role": "user", "content": text})
            
            # Keep conversation history manageable
            if len(self.conversation_history) > 10:
                self.conversation_history = self.conversation_history[-8:]
            
            # Generate response
            messages = [
                {
                    "role": "system", 
                    "content": "You are a helpful voice assistant. Keep responses concise and conversational."
                }
            ] + self.conversation_history
            
            response = await self.openai_client.chat.completions.create(
                model=self.model,
                messages=messages,
                max_tokens=150,
                temperature=0.7
            )
            
            ai_response = response.choices[0].message.content
            
            # Add to conversation history
            self.conversation_history.append({"role": "assistant", "content": ai_response})
            
            return ai_response
            
        except Exception as e:
            print(f"❌ Response generation error: {e}")
            return "I'm having trouble processing that right now."
    
    async def speak_response(self, text: str):
        """Convert text to speech and play"""
        try:
            # Generate speech
            speech = await self.openai_client.audio.speech.create(
                model="tts-1",
                voice=self.voice,
                input=text,
                response_format="mp3"
            )
            
            # Save audio
            with open("response.mp3", "wb") as f:
                f.write(speech.content)
            
            # Play audio
            await self.play_audio("response.mp3")
            
        except Exception as e:
            print(f"❌ TTS error: {e}")
    
    async def play_audio(self, audio_file: str):
        """Play audio file"""
        try:
            system = platform.system()
            if system == "Windows":
                os.system(f"start {audio_file}")
            elif system == "Darwin":
                os.system(f"afplay {audio_file}")
            elif system == "Linux":
                os.system(f"mpg123 {audio_file}")
            
            # Small delay
            await asyncio.sleep(1)
            
        except Exception as e:
            print(f"❌ Audio playback error: {e}")
    
    # Additional methods to match RealtimeModel interface
    async def send_message(self, message: str):
        """Send a message (for compatibility)"""
        return await self.generate_response(message)
    
    def set_instructions(self, instructions: str):
        """Set system instructions"""
        if self.conversation_history and self.conversation_history[0]["role"] == "system":
            self.conversation_history[0]["content"] = instructions
        else:
            self.conversation_history.insert(0, {"role": "system", "content": instructions})
    
    # Context manager support
    async def __aenter__(self):
        await self.connect()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.disconnect()

# Enhanced version with screenshot capability
class CustomRealtimeModelWithScreenshots(CustomRealtimeModel):
    """Extended version that can handle screenshots"""
    
    def __init__(self, model: str = "gpt-4o", voice: str = "alloy", screenshot_interval: int = 5, **kwargs):
        super().__init__(model, voice, **kwargs)
        self.screenshot_interval = screenshot_interval
        self.current_screenshot = None
        self.last_screenshot_time = 0
        
    async def start_audio_processing(self):
        """Start both audio and screenshot processing"""
        await super().start_audio_processing()
        
        # Start screenshot capture
        asyncio.create_task(self.screenshot_capture_loop())
        print("📸 Screenshot capture started - taking screenshots every 5 seconds...")
    
    async def screenshot_capture_loop(self):
        """Capture screenshots periodically"""
        import pyautogui
        import base64
        import io
        from PIL import Image
        
        while self.session_active:
            try:
                # Take screenshot
                screenshot = pyautogui.screenshot()
                
                # Resize for efficiency
                if screenshot.width > 1024:
                    ratio = 1024 / screenshot.width
                    new_height = int(screenshot.height * ratio)
                    screenshot = screenshot.resize((1024, new_height), Image.Resampling.LANCZOS)
                
                # Convert to base64
                buffer = io.BytesIO()
                screenshot.save(buffer, format='JPEG', quality=85)
                image_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
                
                self.current_screenshot = f"data:image/jpeg;base64,{image_base64}"
                self.last_screenshot_time = time.time()
                
                print("📸 Screenshot captured")
                await asyncio.sleep(self.screenshot_interval)
                
            except Exception as e:
                print(f"❌ Screenshot error: {e}")
                await asyncio.sleep(1)
    
    async def generate_response(self, text: str) -> str:
        """Generate response with screenshot context"""
        try:
            # Add to conversation history
            self.conversation_history.append({"role": "user", "content": text})
            
            # Prepare messages
            messages = [
                {
                    "role": "system", 
                    "content": "You are a helpful voice assistant that can see the user's screen. Respond naturally to what you see and hear. Keep responses concise."
                }
            ] + self.conversation_history[:-1]  # All except the last user message
            
            # Add the last user message with screenshot if available
            if self.current_screenshot and time.time() - self.last_screenshot_time < 10:
                messages.append({
                    "role": "user",
                    "content": [
                        {"type": "text", "text": text},
                        {"type": "image_url", "image_url": {"url": self.current_screenshot}}
                    ]
                })
            else:
                messages.append({"role": "user", "content": text})
            
            response = await self.openai_client.chat.completions.create(
                model=self.model,
                messages=messages,
                max_tokens=150,
                temperature=0.7
            )
            
            ai_response = response.choices[0].message.content
            self.conversation_history.append({"role": "assistant", "content": ai_response})
            
            return ai_response
            
        except Exception as e:
            print(f"❌ Response generation error: {e}")
            return "I'm having trouble processing that right now."
