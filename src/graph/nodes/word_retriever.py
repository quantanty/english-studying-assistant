from src.schemas.state import State

import logging

logger = logging.getLogger(__name__)

class WordRetriever:
    def __init__(self, llm):
        self.llm = llm
    
    # TODO: retriever logic
    def __call__(self, state: State) -> State:
        pass