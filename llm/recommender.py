"""
Handles communication with the LLM and returns
structured business recommendations.
"""

import os
import json
from dotenv import load_dotenv
from openai import OpenAI

from llm.prompts import SYSTEM_PROMPT

load_dotenv()


class ChurnRecommender:

    def __init__(self):

        self.client = OpenAI(
            api_key=os.getenv("OPENAI_API_KEY")
        )

        self.model = os.getenv("OPENAI_MODEL", "gpt-5-mini")

    def generate_report(self, prompt: str):

        try:

            response = self.client.responses.create(

                model=self.model,

                input=[
                    {
                        "role": "system",
                        "content": SYSTEM_PROMPT
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ]

            )

            return response.output_text

        except Exception as e:

            return f"Error: {e}"