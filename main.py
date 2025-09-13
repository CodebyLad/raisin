import argparse
import sys
from functools import partial
from typing import Optional
import asyncio
import time
import base64
import io
import pyautogui
from PIL import Image
import openai
import os
import concurrent.futures

from dotenv import load_dotenv
from livekit.agents import (
    AutoSubscribe,
    JobContext,
    WorkerOptions,
    WorkerType,
    cli,
)
from livekit.agents.voice import Agent, AgentSession
from livekit.plugins import bey, openai as openai_plugin

class ScreenshotContextManager:
    """Manages screenshot context for the realtime conversation"""
    
    def __init__(self, interval: int = 5):
        self.interval = interval
        self.current_screenshot_description = ""
        self.running = False
        # Initialize OpenAI client (synchronous version)
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable not set!")
        self.openai_client = openai.OpenAI(api_key=api_key)
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=2)
        pyautogui.FAILSAFE = False
        print("✅ Screenshot context manager initialized")
    
    async def start(self):
        """Start screenshot analysis"""
        self.running = True
        asyncio.create_task(self.screenshot_analysis_loop())
        print("📸 Screenshot context manager started")
    
    async def stop(self):
        """Stop screenshot analysis"""
        self.running = False
        self.executor.shutdown(wait=False)
    
    async def screenshot_analysis_loop(self):
        """Analyze screenshots and update context"""
        while self.running:
            try:
                print("📸 Taking and analyzing screenshot...")
                
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
                screenshot_url = f"data:image/jpeg;base64,{image_base64}"
                
                # Analyze screenshot with GPT-4V (run in thread pool)
                description = await self.analyze_screenshot(screenshot_url)
                self.current_screenshot_description = description
                
                print(f"🖼️ Screenshot analyzed: {description[:100]}...")
                
                await asyncio.sleep(self.interval)
                
            except Exception as e:
                print(f"❌ Screenshot analysis error: {e}")
                await asyncio.sleep(1)
    
    def _analyze_screenshot_sync(self, screenshot_url: str) -> str:
        """Synchronous screenshot analysis"""
        try:
            response = self.openai_client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": "Describe what you see on this screen in 2-3 sentences. Focus on the main content, applications, or activities visible."
                            },
                            {
                                "type": "image_url",
                                "image_url": {"url": screenshot_url}
                            }
                        ]
                    }
                ],
                max_tokens=150
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            print(f"❌ Sync screenshot analysis error: {e}")
            return "Unable to analyze current screen content."
    
    async def analyze_screenshot(self, screenshot_url: str) -> str:
        """Analyze screenshot and return description (async wrapper)"""
        try:
            # Run the synchronous OpenAI call in a thread pool
            loop = asyncio.get_event_loop()
            description = await loop.run_in_executor(
                self.executor, 
                self._analyze_screenshot_sync, 
                screenshot_url
            )
            return description
            
        except Exception as e:
            print(f"❌ Screenshot analysis error: {e}")
            return "Unable to analyze current screen content."
    
    def get_current_context(self) -> str:
        """Get current screenshot context for conversation"""
        if self.current_screenshot_description:
            return f"Current screen context: {self.current_screenshot_description}"
        return "No current screen context available."

class ContextAwareAgent(Agent):
    """Agent that includes screenshot context in conversations"""
    
    def __init__(self, instructions: str, screenshot_manager: ScreenshotContextManager):
        # Include screenshot context in instructions
        enhanced_instructions = f"""{instructions}

IMPORTANT: You can see what's on the user's screen. Here's what I can currently see:
{screenshot_manager.get_current_context()}

When responding, you can reference what you see on their screen. This context is updated every 5 seconds.
"""
        super().__init__(instructions=enhanced_instructions)
        self.screenshot_manager = screenshot_manager
        self.last_context_update = 0
    
    async def update_context_if_needed(self):
        """Update agent context with latest screenshot info"""
        current_time = time.time()
        if current_time - self.last_context_update > 20:  # Update every 5 seconds
            new_context = self.screenshot_manager.get_current_context()
            # Update the agent's understanding of current screen
            self.last_context_update = current_time
            print(f"🔄 Updated agent context: {new_context[:50]}...")

async def entrypoint(ctx: JobContext, avatar_id: Optional[str]) -> None:
    print("🚀 Starting realtime agent with screenshot context...")
    
    await ctx.connect(auto_subscribe=AutoSubscribe.AUDIO_ONLY)
    print("✅ Connected to room")

    # Create and start screenshot context manager
    screenshot_manager = ScreenshotContextManager()
    await screenshot_manager.start()

    # Use OpenAI's RealtimeModel for smooth voice interaction
    local_agent_session = AgentSession(
        llm=openai_plugin.realtime.RealtimeModel(
            model="gpt-4o-realtime-preview",
            voice="cedar"
        )
    )
    print("✅ Realtime agent session created")

    if avatar_id is not None: 
        bey_avatar_session = bey.AvatarSession(avatar_id=avatar_id)
    else:
        bey_avatar_session = bey.AvatarSession()
    
    await bey_avatar_session.start(local_agent_session, room=ctx.room)
    print("✅ Bey avatar session started")

    # Create context-aware agent
    agent = ContextAwareAgent(
        instructions="""You are a helpful AI assistant that can see the user's screen and hear their voice in real-time. 
        You have access to periodic screenshots of what the user is currently viewing.
        
        Respond naturally and conversationally. You can:
        - Comment on what you see on their screen
        - Help with applications or content they're viewing
        - Answer questions about what's displayed
        - Provide assistance based on their current context
        
        Keep responses concise and helpful.""",
        screenshot_manager=screenshot_manager
    )

    await local_agent_session.start(
        agent=agent,
        room=ctx.room,
    )
    print("✅ Context-aware agent started - I can see your screen and hear you in real-time!")

if __name__ == "__main__":
    # Load environment variables FIRST
    load_dotenv()
    
    # Check if API key is available
    if not os.getenv("OPENAI_API_KEY"):
        print("❌ OPENAI_API_KEY environment variable not found!")
        print("Please set it with: export OPENAI_API_KEY='your-api-key-here'")
        sys.exit(1)
    else:
        print("✅ OPENAI_API_KEY found")

    parser = argparse.ArgumentParser(description="Run realtime agent with screenshot context.")
    parser.add_argument("--avatar-id", type=str, help="Avatar ID to use.")
    args = parser.parse_args()

    sys.argv = [sys.argv[0], "dev"]
    cli.run_app(
        WorkerOptions(
            entrypoint_fnc=partial(entrypoint, avatar_id=args.avatar_id),
            worker_type=WorkerType.ROOM,
        )
    )
