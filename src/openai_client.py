import os
import openai
from dotenv import load_dotenv
from openai import OpenAI
from typing import Optional, Dict, List


class OpenAIClient:
    context: str = ""

    def __init__(self, api_key: Optional[str] = None, client_context: str = None, stream: bool = False):
        # Load environment variables from .env file
        load_dotenv()

        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("No API key provided.")

        self.client = OpenAI(api_key=self.api_key)
        self.model = 'gpt-3.5-turbo'
        self.stream = stream
        if client_context:
            self.add_context(client_context)

    def add_context(self, context: str) -> None:
        self.context += context

    def get_context(self) -> Dict[str, str]:
        message = {"role": "system", "content": self.context}
        return message

    def chat_completions(self, messages: List[Dict[str, str]]) -> Optional[str]:
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                stream=self.stream
            )
        except openai.APIConnectionError as e:
            raise Exception(f"Failed to connect to OpenAI API: {e}")
        except openai.RateLimitError as e:
            raise Exception(f"OpenAI API request exceeded rate limit: {e}")
        except openai.APIError as e:
            raise Exception(f"OpenAI API returned an error: {e}")
        except Exception as e:
            raise Exception(f"Error: {e}")

        if self.stream:
            try:
                for chunk in response:
                    answer_chunk = chunk.choices[0].delta.content
                    if answer_chunk:
                        print(chunk.choices[0].delta.content, end="", flush=True)
                print()
            except Exception as e:
                raise Exception(f"Error during streaming: {e}")
        else:
            return response.choices[0].message.content

    def query(self, message: str) -> Optional[str]:
        messages = [self.get_context(), {"role": "user", "content": message}]
        answer = self.chat_completions(messages)
        if not self.stream:
            return answer
