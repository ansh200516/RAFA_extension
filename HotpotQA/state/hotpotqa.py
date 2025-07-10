from dataclasses import dataclass,field
from typing import List,Optional
from langchain.agents.react.base import DocstoreExplorer

from HotpotQA.typedefs import State

@dataclass(frozen=True)
class StateHotpotQA(State):

    # The initial question to solve
    puzzle: str

    # Current state towards solving the puzzle
    current_state: str

    # Steps taken towards solving the puzzle
    steps: List[str]

    # The true answer to the question
    answer: str

    # Docstore the current state is using
    docstore: DocstoreExplorer

    # A random number associated with the state
    randomness: Optional[int]
    
    # A summary of the all reflections over previous trials
    summary: str

    # A list to store all reflections made during the process
    reflections: List[str] = field(default_factory=list)

    # Value for this state. None means the value has not been computed yet.
    value: Optional[float] = None

    # parent state
    parent: Optional['StateHotpotQA'] = None


    def serialize(self) -> dict:
        """
        Returns a dictionary representation of the state.
        """
        return {
            "current_state": self.current_state,
            "steps": " -> ".join(self.steps),
            "reflections": self.reflections,
            "value": self.value,
            "puzzle": self.puzzle,
            "answer": self.answer,
            "summary": self.summary,
        }
    
    def clone(self, randomness: int = None, new_reflection: Optional[str] = None, reset_reflections: bool = False, reset_trajectory: bool = False) -> "StateHotpotQA":
        """
        Returns a new instance of StateHotpotQA with an optional new randomness value,
        updated reflections, and optionally reset current_state and steps.
        """
        current_reflections = [] if reset_reflections else list(self.reflections)
        if new_reflection:
            current_reflections.append(new_reflection)
        
        new_current_state = "" if reset_trajectory else self.current_state
        new_steps = [] if reset_trajectory else list(self.steps)

        return StateHotpotQA(
            puzzle=self.puzzle,
            current_state=new_current_state,
            steps=new_steps,
            answer=self.answer,
            docstore=self.docstore,
            randomness=randomness or self.randomness,
            reflections=current_reflections,
            value=self.value,
            parent=self.parent,
            summary=self.summary
        )

    def get_seed(self) -> int:
        """
        Returns the randomness value associated with the state.
        """
        if self.randomness is None:
            raise ValueError("Randomness has not been set for this state. Call clone() with a randomness value.")
        return self.randomness
    
    def __hash__(self) -> int:
        """
        Returns a hash of the current state.
        """
        return hash(str(self.serialize()))
    

