from abc import ABC, abstractmethod
from typing import Any


class Task(ABC):

    @abstractmethod
    def generate_prompt_data(self, show_progress: bool = False) -> list[Any]:
        pass

    @abstractmethod
    def prompts_to_response(
        self, prompt_data: list[Any], show_progress: bool = False
    ) -> list[Any]:
        pass

    @abstractmethod
    def evaluate(self, response_data: list[Any]) -> list[Any]:
        pass
