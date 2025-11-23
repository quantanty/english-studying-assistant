import logging

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from src.schemas.state import State
from src.prompts import QUIZ_GENERATOR_SYSTEM, QUIZ_GENERATOR_USER

logger = logging.getLogger(__name__)


class QuizGenerator:
    def __init__(self, llm):
        self.llm = llm
        self.system = QUIZ_GENERATOR_SYSTEM
        self.user = QUIZ_GENERATOR_USER

    # TODO: quiz generator logic
    async def __call__(self, state: State):
        qgen_messages = state.get("qgen_messages", [])
        instruction = state.get("instruction")
        words = state.get("words", [])
        retrieved_information = state.get("collected_data")

        if qgen_messages:
            self.construct_feed_next(qgen_messages, instruction, words, retrieved_information)
        else:
            qgen_messages = self.construct_feed(qgen_messages, instruction, words, retrieved_information)

        response = await self.llm.ainvoke(qgen_messages)
        qgen_messages.append(response)

        return {"qgen_messages": qgen_messages}
        
        
    def construct_feed(self, instruction, words, retrieved_information):
        feed = [
            SystemMessage(content=self.system),
            HumanMessage(content=self.user.format(words=words, instruction=instruction, retrieved_information=retrieved_information))
        ]
        return feed
    
    def construct_feed_next(self, qgen_messages, instruction, words, retrieved_information):
        qgen_messages.append(HumanMessage(content=self.user.format(words=words, instruction=instruction, retrieved_information=retrieved_information)))