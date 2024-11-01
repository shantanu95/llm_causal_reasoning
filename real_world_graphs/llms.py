from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

from dataclasses_json import dataclass_json
from dotenv import load_dotenv
from openai import OpenAI

from real_world_graphs.utils.progress_utils import progress_iter

load_dotenv()


class LLM(ABC):

    @abstractmethod
    def generate(self, prompt: str) -> str:
        pass

    def generate_batched(
        self, prompts: list[str], show_progress: bool = False
    ) -> list[str]:
        return [self.generate(prompt=p) for p in progress_iter(prompts, show_progress)]


@dataclass_json
@dataclass
class OpenAIConfig:
    model_name: str = "gpt-4o"
    temperature: float = 0
    max_tokens: int = 10
    max_workers: int = 1


class OpenAILLM(LLM):

    def __init__(self, config: OpenAIConfig) -> None:
        super().__init__()
        # assumes that OPENAI_API_KEY is in `.env`.
        self.client = OpenAI()
        self.config = config

    def generate(self, prompt: str) -> str:
        chat_completion = self.client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens,
            model=self.config.model_name,
        )
        return chat_completion.choices[0].message.content

    def generate_batched(
        self, prompts: list[str], show_progress: bool = False
    ) -> list[str]:
        with ThreadPoolExecutor(max_workers=self.config.max_workers) as executor:
            return list(
                r
                for r in progress_iter(
                    executor.map(self.generate, prompts), show_progress
                )
            )
