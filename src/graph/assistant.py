import logging
from typing import Any, AsyncGenerator, Dict, List, Literal, Optional
from urllib.parse import quote_plus

from asgiref.sync import sync_to_async
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from langgraph.graph import END, START
from langgraph.graph.state import CompiledStateGraph, StateGraph
from langgraph.types import StateSnapshot
from psycopg_pool import AsyncConnectionPool

from src.config import settings
from src.graph.edges.decision import (
    after_intent_classifier,
    after_orchestrator,
    after_preparation,
    after_quiz_generator,
)
from src.graph.nodes.intent_classifier import IntentClassifier
from src.graph.nodes.orchestrator import Orchestrator
from src.graph.nodes.planner import Planner
from src.graph.nodes.preparing import preparing
from src.graph.nodes.presenter import Presenter
from src.graph.nodes.quiz_generator import QuizGenerator
from src.graph.nodes.quiz_tool_node import QuizToolNode
from src.graph.nodes.word_entry_generator import WordEntryGenerator
from src.graph.nodes.word_retriever import WordRetriever
from src.schemas.chat import Message
from src.schemas.llms import (
    CustomGoogleModelName,
    GoogleModelName,
    ModelSelection,
    ModelType,
)
from src.schemas.models import Action
from src.schemas.state import SelectedModel, State
from src.tools.quiz import (
    create_fill_in_blank_question,
    create_multiple_choice_question,
    create_true_false_question,
)
from src.utils.graph import dump_messages


class Assistant:
    def __init__(self, default_model: SelectedModel):
        self.default_model=default_model
        self._graph: Optional[CompiledStateGraph] = None
        self._connection_pool: Optional[AsyncConnectionPool] = None

        # TODO: validate default models

    async def _get_connection_pool(self) -> AsyncConnectionPool:
        if self._connection_pool is None:
            try:
                max_size = settings.POSTGRES_POOL_SIZE

                connection_url = (
                    "postgresql://"
                    f"{quote_plus(settings.POSTGRES_USER)}:{quote_plus(settings.POSTGRES_PASSWORD)}"
                    f"@{settings.POSTGRES_HOST}:{settings.POSTGRES_PORT}/{settings.POSTGRES_DB}"
                )

                self._connection_pool = AsyncConnectionPool(
                    connection_url,
                    open=False,
                    max_size=max_size,
                    kwargs={
                        "autocommit": True,
                        "connect_timeout": 5,
                        "prepare_threshold": None,
                    }
                )
                await self._connection_pool.open()
            except Exception as e:
                raise e
        return self._connection_pool

    async def create_graph(self, checkpointer=None):
        if self._graph is None:
            quiz_tools = [create_multiple_choice_question, create_fill_in_blank_question, create_true_false_question]

            graph_builder = StateGraph(State)

            graph_builder.add_node("preparing", preparing())

            graph_builder.add_node("intent_classifier", IntentClassifier(self.default_model["intent_classifier"]))
            graph_builder.add_node("planner", Planner(self.default_model["planner"]))
            graph_builder.add_node("orchestrator", Orchestrator(self.default_model["orchestrator"]))
            graph_builder.add_node("word_entry_generator", WordEntryGenerator(self.default_model["word_entry_generator"]))
            graph_builder.add_node("presenter", Presenter(self.default_model["presenter"]))
            graph_builder.add_node("word_retriever", WordRetriever(self.default_model["planner"], []))
            graph_builder.add_node("quiz_generator", QuizGenerator(self.default_model["quiz_generator"], quiz_tools))
            graph_builder.add_node("quiz_tools", QuizToolNode(quiz_tools))
            
            graph_builder.add_edge(START, "preparing")
            graph_builder.add_edge("preparing", "intent_classifier")
            # graph_builder.add_edge("intent_classifier", "planner")
            graph_builder.add_edge("planner", "orchestrator")
            graph_builder.add_edge("word_retriever", "word_entry_generator")
            graph_builder.add_edge("word_entry_generator", "orchestrator")
            graph_builder.add_edge("quiz_tools", "quiz_generator")
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
                    "define_words": "word_retriever",
                    "generate_quizzes": "quiz_generator",
                    "present": "presenter",
                    "end": END
                }
            )

            graph_builder.add_conditional_edges(
                "quiz_generator",
                after_quiz_generator,
                {
                    "tool_call": "quiz_tools",
                    "finish": "orchestrator"
                }
            )

            if checkpointer is None:
                connection_pool = await self._get_connection_pool()
                if connection_pool:
                    checkpointer = AsyncPostgresSaver(connection_pool)
                    await checkpointer.setup()
                else:
                    raise Exception("Connection pool initialization failed")
            self._graph = graph_builder.compile(checkpointer)
        
        return self._graph
    
    async def get_stream_response(
        self,
        message: Message,
        session_id: str,
        user_id: Optional[str] = None,
        selected_model: SelectedModel | None = None,
    ) -> AsyncGenerator[str, None]:
        config = {
            "configurable": {"thread_id": session_id}
        }
        
        if self._graph is None:
            self._graph = await self.create_graph()
        
        try:
            async for chunk in self._graph.astream(
                {"messages": dump_messages([message]), "selected_model": selected_model}, config=config, stream_mode="custom" # type: ignore
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
        if self._graph is None:
            self._graph = await self.create_graph()
            
            state: StateSnapshot = await sync_to_async(self._graph.get_state)(
                config={"configurable": {"thread_id": session_id}}
            )
            return state.values["messages"] if state.values else []

    async def clear_chat_history(self, session_id: str):
        try:
            conn_pool = await self._get_connection_pool()
            
            async with conn_pool.connection() as conn:
                for table in settings.CHECKPOINT_TABLES:
                    try:
                        await conn.execute(f"DELETE FROM {table} WHERE thread_id = %s", (session_id,))
                    except Exception as e:
                        raise
        except Exception as e:
            raise