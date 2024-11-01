import pickle
import networkx as nx
import random

from typing import Optional


class CauseNetGraph:

    def __init__(self, graph_path: Optional[str] = None) -> None:
        self.graph: Optional[nx.DiGraph] = (
            pickle.load(open(graph_path, "rb")) if graph_path is not None else None
        )

    def path_to_narrative(self, path: list[str]) -> str:
        narrative = ""
        for i in range(len(path) - 1):
            sentence = random.choice(
                self.graph.edges[path[i], path[i + 1]]["sentences"]
            )
            narrative += f""" {sentence.strip()}"""

        return narrative.strip()
