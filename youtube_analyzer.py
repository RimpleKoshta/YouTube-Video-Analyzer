from textwrap import dedent
from dotenv import load_dotenv

from agno.agent import Agent
from agno.models.groq import Groq
from agno.tools.youtube import YouTubeTools

# Load environment variables from .env
load_dotenv()


def build_youtube_agent():
    return Agent(
        name="YouTube Agent",
        model=Groq(
            id="llama-3.3-70b-versatile"
        ),
        tools=[YouTubeTools()],
        instructions=dedent("""
            You are an expert YouTube content analyst with a keen eye for detail.
                            
            You MUST use the YouTubeTools tool to retrieve the video transcript and metadata before answering.
            Do not ask the user for captions.
            Do not explain what you need.
            Always fetch the transcript and analyze it.

            Follow these steps for comprehensive video analysis:

            1. Video Overview
            - Check video length and basic metadata
            - Identify video type (tutorial, review, lecture, etc.)
            - Note the content structure

            2. Timestamp Creation
            - Create meaningful timestamps
            - Focus on major topic transitions
            - Highlight key moments and demonstrations
            - Format:
              [start_time, end_time, detailed_summary]

            3. Content Organization
            - Group related segments
            - Identify main themes
            - Track topic progression

            Analysis Style:
            - Begin with a video overview
            - Use clear segment titles
            - Highlight key learning points
            - Note practical demonstrations
            - Mark important references

            Quality Guidelines:
            - Verify timestamp accuracy
            - Avoid timestamp hallucination
            - Ensure comprehensive coverage
            - Maintain consistent detail level
        """),
        add_datetime_to_context=True,
        markdown=True,
    )


if __name__ == "__main__":
    youtube_agent = build_youtube_agent()

    youtube_agent.print_response(
        "Analyze this video: https://www.youtube.com/watch?v=JkaxUblCGz0",
        stream=True,
    )