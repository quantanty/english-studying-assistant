from src.utils.prompt_loader import PromptLoader
from src.services.llm_client import create_llm
from src.schemas.state import State
from src.graph.nodes.orchestrator import Orchestrator
from src.graph.nodes.word_entry_generator import WordEntryGenerator
from src.graph.nodes.presenter import Presenter
from src.graph.edges.decision import after_orchestrator

from langgraph.graph import StateGraph, START, END
import logging

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

def create_graph():
    orchestrator_prompt = PromptLoader.get("simple_prompt.system_prompt.orchestrator")
    word_entry_generator_prompt = PromptLoader.get("simple_prompt.system_prompt.dictionary")
    presenter_prompt = PromptLoader.get("simple_prompt.system_prompt.presenter")

    llm = create_llm()

    graph_builder = StateGraph(State)

    graph_builder.add_node("orchestrator", Orchestrator(llm, orchestrator_prompt))
    graph_builder.add_node("word_entry_generator", WordEntryGenerator(llm, word_entry_generator_prompt))
    graph_builder.add_node("presenter", Presenter(llm, presenter_prompt))
    
    graph_builder.add_edge(START, "orchestrator")
    graph_builder.add_edge("word_entry_generator", "presenter")
    graph_builder.add_edge("presenter", END)

    graph_builder.add_conditional_edges(
        "orchestrator",
        after_orchestrator,
        {
            "define_word": "word_entry_generator",
            "general_chat": "presenter"
        }
    )

    return graph_builder.compile()
