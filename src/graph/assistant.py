import logging
from typing import Any, AsyncGenerator, Dict, List, Optional

from langgraph.graph import END, START
from langgraph.graph.state import CompiledStateGraph, StateGraph

from src.graph.edges.decision import after_orchestrator, after_preparation, after_intent_classifier
from src.graph.nodes.intent_classifier import IntentClassifier
from src.graph.nodes.orchestrator import Orchestrator
from src.graph.nodes.planner import Planner
from src.graph.nodes.preparing import preparing
from src.graph.nodes.presenter import Presenter
from src.graph.nodes.quiz_generator import QuizGenerator
from src.graph.nodes.word_entry_generator import WordEntryGenerator
from src.graph.nodes.word_extractor import WordExtractor
from src.graph.nodes.word_retriever import WordRetriever
from src.schemas.chat import Message
from src.schemas.models import Action
from src.schemas.state import State
from src.services.llm_client import create_llm
from src.utils.graph import dump_messages
from langgraph.checkpoint.memory import InMemorySaver


class Assistant:
    def __init__(self, llm):
        self.llm = llm
        self._graph: Optional[CompiledStateGraph] = None

    async def create_graph(self, checkpointer=None):
        if self._graph is None:
            llm = self.llm

            graph_builder = StateGraph(State)

            orchestrator = Orchestrator(llm)
            graph_builder.add_node("preparing", preparing(orchestrator))

            graph_builder.add_node("intent_classifier", IntentClassifier(llm))
            graph_builder.add_node("planner", Planner(llm))
            graph_builder.add_node("orchestrator", orchestrator)
            graph_builder.add_node("word_entry_generator", WordEntryGenerator(llm))
            graph_builder.add_node("presenter", Presenter(llm))
            graph_builder.add_node("word_extractor", WordExtractor(llm))
            graph_builder.add_node("word_retriever", WordRetriever(llm, []))
            graph_builder.add_node("quiz_generator", QuizGenerator(llm))
            
            graph_builder.add_edge(START, "preparing")
            graph_builder.add_edge("preparing", "intent_classifier")
            # graph_builder.add_edge("intent_classifier", "planner")
            graph_builder.add_edge("planner", "orchestrator")
            graph_builder.add_edge("word_extractor", "orchestrator")
            graph_builder.add_edge("word_retriever", "word_entry_generator")
            graph_builder.add_edge("word_entry_generator", "orchestrator")
            graph_builder.add_edge("quiz_generator", "orchestrator")
            graph_builder.add_edge("presenter", END)
            
            graph_builder.add_conditional_edges(
                "intent_classifier",
                after_intent_classifier,
                {
                    "learn_words": "planner",
                    "general_chat": "presenter"
                }
            )

            graph_builder.add_conditional_edges(
                "orchestrator",
                after_orchestrator,
                {
                    "extract_words": "word_extractor",
                    "define_words": "word_retriever",
                    "generate_quizzes": "quiz_generator",
                    "present": "presenter",
                    "end": END
                }
            )

            self._graph = graph_builder.compile(checkpointer)
        
        return self._graph
    
    async def get_stream_response(
        self, messages: List[Message], session_id: str
    ) -> AsyncGenerator[str, None]:
        config = {
            "configurable": {"thread_id": session_id}
        }
        
        if self._graph is None:
            self._graph = await self.create_graph(checkpointer=InMemorySaver())
        
        try:
            async for chunk in self._graph.astream(
                {"messages": dump_messages(messages)}, config=config, stream_mode="custom"
            ):
                try:
                    if "presenter" in chunk:
                        yield chunk["presenter"]
                except Exception as chunk_error:
                    continue
        except Exception as stream_error:
            raise stream_error

    async def get_chat_history(
        self, session_id: str
    ):
        ...

    async def get_chat_history(self, session_id: str) -> ...:
        ...

    async def clear_chat_history(self, session_id: str):
        ...