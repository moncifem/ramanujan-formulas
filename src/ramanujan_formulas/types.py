"""
Type definitions for the Ramanujan-Swarm system.
"""

from typing import TypedDict, List, Set
from typing_extensions import Annotated
import operator


class Candidate(TypedDict):
    """Represents a mathematical candidate formula."""
    expression: str
    value_str: str
    error: float
    score: float
    source: str


class ProposerInput(TypedDict):
    """Input state for proposer agents."""
    best_candidates: List[Candidate]
    iteration: int


class State(TypedDict):
    """
    Global state for the LangGraph system.
    
    Annotated fields use reducer functions to merge updates:
    - operator.add: Append new items to list
    - operator.or_: Union of sets
    """
    proposed_expressions: Annotated[List[str], operator.add]
    best_candidates: List[Candidate]
    discoveries: Annotated[List[dict], operator.add]
    tested_hashes: Annotated[Set[int], operator.or_]
    tested_values: Annotated[Set[str], operator.or_]
    iteration: int

