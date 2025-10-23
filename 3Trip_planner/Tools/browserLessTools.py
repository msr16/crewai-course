import json
import requests
import streamlit as st
from crewai.tools import BaseTool
from pydantic import BaseModel, Field
from unstructured.partition.html import partition_html
from crewai import Agent, Task
from crewai import LLM

from openai import OpenAI
import os


class WebsiteInput(BaseModel):
    website: str = Field(..., description="The website URL to scrape")

class BrowserTools(BaseTool):
    name: str = "Scrape website content"
    description: str = "Useful to scrape and summarize a website content"
    args_schema: type[BaseModel] = WebsiteInput

    def _run(self, website: str) -> str:
        try:
            # url = f"https://chrome.browserless.io/content?token={st.secrets['BROWSERLESS_API_KEY']}"
            url = f"https://production-sfo.browserless.io/content?token={os.getenv('BROWSERLESS_API_KEY')}"
            payload = json.dumps({"url": website})
            headers = {'cache-control': 'no-cache', 'content-type': 'application/json'}
            response = requests.request("POST", url, headers=headers, data=payload)

            if response.status_code != 200:
                return f"Error: Failed to fetch website content. Status code: {response.status_code}"
            
            elements = partition_html(text=response.text)
            content = "\n\n".join([str(el) for el in elements])
            content = [content[i:i + 8000] for i in range(0, len(content), 8000)]
            summaries = []
            
            client = OpenAI()
            system_prompt = (
                "You are a Principal Researcher at a major company. "
                "Your goal is to create amazing research and summaries based on the content provided. "
                "Analyze the content thoroughly and extract the most relevant information."
            )

            for chunk in content:
                user_prompt = (
                    f"Analyze and summarize the content below. Make sure to include the "
                    f"most relevant information in the summary. Return ONLY the summary, nothing else.\n\n"
                    f"CONTENT\n----------\n{chunk}"
                )

                response = client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    temperature=0.7  # Adjust for creativity vs. consistency
                )
                
                # 5. Extract the summary from the response
                summary = response.choices[0].message.content
                summaries.append(summary.strip())

            return "\n\n".join(summaries)
        except Exception as e:
            return f"Error while processing website: {str(e)}"

    async def _arun(self, website: str) -> str:
        raise NotImplementedError("Async not implemented")