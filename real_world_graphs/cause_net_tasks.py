from typing import Any, Optional
from task import Task
from dataclasses import dataclass

from dataclasses_json import dataclass_json
import networkx as nx
from real_world_graphs.cause_net_graph import CauseNetGraph
from itertools import combinations
from real_world_graphs.utils.progress_utils import progress_iter

from real_world_graphs.llms import LLM

import json

import random


@dataclass_json
@dataclass
class ChainEstimationPrompt:
    graph: nx.DiGraph
    chain: list[str]
    cause: str
    effect: str
    narrative: str

    def get_distance(self) -> int:
        return abs(self.chain.index(self.effect) - self.chain.index(self.cause))

    def is_causal(self) -> bool:
        return self.chain.index(self.cause) < self.chain.index(self.effect)

    def get_prompt_text(self) -> str:
        return f"""Consider the following hypothetical narrative.

{self.narrative}

According to the hypothetical narrative, does {self.cause.replace("_", " ")} have a (direct or indirect) causal effect on {self.effect.replace("_", " ")}?
Answer in Yes/No.""".strip()

    def get_prompt_text_cot(self) -> str:
        return f"""Consider the following hypothetical narrative.

{self.narrative}

According to the hypothetical narrative, does {self.cause.replace("_", " ")} have a (direct or indirect) causal effect on {self.effect.replace("_", " ")}?
Think step by step and end your answer with <answer>Yes/No</answer>.
""".strip()

    def get_prompt_text_with_graph(self, estimated_chain: list[str]) -> str:
        return f"""Consider the following hypothetical narrative.

{self.narrative}

The causal chain graph for this narrative is: [{" -> ".join(estimated_chain)}].

According to the hypothetical narrative and the causal chain graph, does {self.cause.replace("_", " ")} have a (direct or indirect) causal effect on {self.effect.replace("_", " ")}?
Answer in Yes/No.
""".strip()


@dataclass_json
@dataclass
class ChainEstimationResponse:
    prompt: ChainEstimationPrompt
    response_text: str

    def is_yes(self) -> bool:
        return "yes" in self.response_text.lower()

    def is_correct(self) -> bool:
        is_yes = self.is_yes()
        is_causal = self.prompt.is_causal()
        return (is_yes and is_causal) or ((not is_yes) and (not is_causal))

    def is_correct_cot(self) -> bool:
        is_yes = "<answer>yes</answer>" in self.response_text.lower()
        is_causal = self.prompt.is_causal()
        return (is_yes and is_causal) or ((not is_yes) and (not is_causal))


class CausalChainEstimationTask(Task):

    def __init__(
        self,
        graph_path: str,
        llm: LLM,
        narrative_path: str,
        min_chain_length: int = 1,
        max_narratives: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.cause_net_graph = CauseNetGraph(graph_path=graph_path)
        self.llm = llm
        self.causal_chains: list[list[str]] = []
        self.narrative_path = narrative_path
        self.min_chain_length = min_chain_length
        self.max_narratives = max_narratives

    def generate_prompt_data(
        self, show_progress: bool = False
    ) -> list[ChainEstimationPrompt]:
        prompts: list[ChainEstimationPrompt] = []
        num_narratives = 0
        with open(self.narrative_path, "r") as file:
            for line in progress_iter(file, show_progress):
                narrative_data = json.loads(line)
                if len(narrative_data["nodes"]) < self.min_chain_length:
                    continue

                if (
                    self.max_narratives is not None
                    and num_narratives == self.max_narratives
                ):
                    return prompts

                chain = narrative_data["nodes"]
                if "sentences" in narrative_data:
                    narrative = " ".join(narrative_data["sentences"])
                elif "narrative" in narrative_data:
                    narrative = narrative_data["narrative"]
                else:
                    raise ValueError("incorrect narrative data.")

                num_narratives += 1
                for u, v in progress_iter(combinations(chain, 2), show_progress):
                    prompts.append(
                        ChainEstimationPrompt(
                            graph=None,
                            chain=chain,
                            cause=u,
                            effect=v,
                            narrative=narrative,
                        )
                    )
                    prompts.append(
                        ChainEstimationPrompt(
                            graph=None,
                            chain=chain,
                            cause=v,
                            effect=u,
                            narrative=narrative,
                        )
                    )
        return prompts

    def prompts_to_response(
        self,
        prompt_data: list[ChainEstimationPrompt],
        show_progress: bool = False,
        with_cot: bool = False,
    ) -> list[ChainEstimationResponse]:
        responses = self.llm.generate_batched(
            prompts=[
                p.get_prompt_text_cot() if with_cot else p.get_prompt_text()
                for p in prompt_data
            ],
            show_progress=show_progress,
        )
        return [
            ChainEstimationResponse(prompt=p, response_text=r)
            for p, r in zip(prompt_data, responses)
        ]

    def evaluate(self, response_data: list[ChainEstimationResponse]) -> list[Any]:
        raise NotImplementedError()


@dataclass_json
@dataclass
class GraphEstimationPrompt:
    graph: nx.DiGraph
    chain: list[str]
    narrative: str

    def get_prompt_text(self, shuffle_chain: bool = True) -> str:
        chain = (
            random.sample(self.chain, k=len(self.chain))
            if shuffle_chain
            else list(self.chain)
        )
        return f"""Consider the following hypothetical narrative.

{self.narrative}

According to the hypothetical narrative, construct a causal chain graph using the following nodes: {str(chain)}.
Ensure that the graph contains all the given nodes and only output a single chain graph of the form <graph>node1 -> node2 -> node3</graph>.
Only output the graph between the <graph></graph> tags.""".strip()


@dataclass_json
@dataclass
class GraphEstimationResponse:
    prompt: GraphEstimationPrompt
    response_text: str

    def get_chain_graph(self) -> list[str]:
        return [
            n.strip()
            for n in self.response_text.split("<graph>")[-1]
            .split("</graph>")[0]
            .split("->")
        ]


class GraphEstimationTask(Task):

    def __init__(
        self, graph_path: str, llm: LLM, narrative_path: str, min_chain_length: int = 0
    ) -> None:
        super().__init__()
        self.cause_net_graph = CauseNetGraph(graph_path=graph_path)
        self.llm = llm
        self.causal_chains: list[list[str]] = []
        self.narrative_path = narrative_path
        self.min_chain_length = min_chain_length

    def generate_prompt_data(
        self, show_progress: bool = False
    ) -> list[GraphEstimationPrompt]:
        prompts: list[GraphEstimationPrompt] = []
        with open(self.narrative_path, "r") as file:
            for line in progress_iter(file, show_progress):
                narrative_data = json.loads(line)
                if len(narrative_data["nodes"]) < self.min_chain_length:
                    continue

                chain = narrative_data["nodes"]
                if "sentences" in narrative_data:
                    narrative = " ".join(narrative_data["sentences"])
                elif "narrative" in narrative_data:
                    narrative = narrative_data["narrative"]
                else:
                    raise ValueError("incorrect narrative data.")

                prompts.append(
                    GraphEstimationPrompt(
                        graph=None,
                        chain=list(chain),
                        narrative=narrative,
                    )
                )
        return prompts

    def prompts_to_response(
        self, prompt_data: list[GraphEstimationPrompt], show_progress: bool = False
    ) -> list[GraphEstimationResponse]:
        responses = self.llm.generate_batched(
            prompts=[p.get_prompt_text() for p in prompt_data],
            show_progress=show_progress,
        )
        return [
            GraphEstimationResponse(prompt=p, response_text=r)
            for p, r in zip(prompt_data, responses)
        ]

    def evaluate(self, response_data: list[Any]) -> list[Any]:
        raise NotImplementedError()
