import argparse
import sys
from functools import partial
from typing import Optional

from dotenv import load_dotenv
from livekit.agents import (
    AutoSubscribe,
    JobContext,
    WorkerOptions,
    WorkerType,
    cli,
)
from livekit.agents.voice import Agent, AgentSession
from livekit.plugins import bey

# Import our custom model
from agent import CustomRealtimeModel, CustomRealtimeModelWithScreenshots

async def entrypoint(ctx: JobContext, avatar_id: Optional[str]) -> None:
    await ctx.connect(auto_subscribe=AutoSubscribe.AUDIO_ONLY)

    # Use our custom model instead of OpenAI's
    local_agent_session = AgentSession(
        llm=CustomRealtimeModelWithScreenshots(  # or CustomRealtimeModel for voice-only
            model="gpt-4o",
            voice="cedar",
            screenshot_interval=5
        )
    )

    if avatar_id is not None: 
        bey_avatar_session = bey.AvatarSession(avatar_id=avatar_id)
    else:
        bey_avatar_session = bey.AvatarSession()
    
    await bey_avatar_session.start(local_agent_session, room=ctx.room)

    await local_agent_session.start(
        agent=Agent(instructions="Talk to me! I can see your screen and hear your voice."),
        room=ctx.room,
    )

if __name__ == "__main__":
    load_dotenv()

    parser = argparse.ArgumentParser(description="Run a LiveKit agent with custom realtime model.")
    parser.add_argument("--avatar-id", type=str, help="Avatar ID to use.")
    args = parser.parse_args()

    sys.argv = [sys.argv[0], "dev"]
    cli.run_app(
        WorkerOptions(
            entrypoint_fnc=partial(entrypoint, avatar_id=args.avatar_id),
            worker_type=WorkerType.ROOM,
        )
    )
