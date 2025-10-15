from src.utils.prompt_loader import PromptLoader
from src.services.llm_client import create_llm
from src.schemas.state import State
from src.schemas.models import Action

from src.graph.nodes.preparing import preparing
from src.graph.nodes.orchestrator import Orchestrator
from src.graph.nodes.word_entry_generator import WordEntryGenerator
from src.graph.nodes.presenter import Presenter
from src.graph.nodes.intent_classifier import IntentClassifier
from src.graph.nodes.word_extractor import WordExtractor
from src.graph.nodes.word_retriever import WordRetriever
from src.graph.nodes.quiz_generator import QuizGenerator
from src.graph.edges.decision import after_orchestrator, after_preparation

from langgraph.graph import StateGraph, START, END
import logging

def create_graph(llm=None, checkpointer=None):
    intent_classifier_system_prompt = PromptLoader.get("intent_classifier.system")
    orchestrator_system_prompt = PromptLoader.get("orchestrator.system")
    orchestrator_user_template = PromptLoader.get("orchestrator.user")
    orchestrator_observation_template = PromptLoader.get("orchestrator.observation")
    word_entry_generator_prompt = PromptLoader.get("simple_prompt.system_prompt.dictionary")
    presenter_prompt = PromptLoader.get("simple_prompt.system_prompt.presenter")

    llm = llm or create_llm()

    graph_builder = StateGraph(State)

    graph_builder.add_node("preparing", preparing(orchestrator_system_prompt, orchestrator_user_template))
    graph_builder.add_node("orchestrator", Orchestrator(llm, orchestrator_observation_template))
    graph_builder.add_node("word_entry_generator", WordEntryGenerator(llm, word_entry_generator_prompt))
    graph_builder.add_node("presenter", Presenter(llm, presenter_prompt))
    #TODO: write prompt for IntentClassifier, WordExtractor, WordRetriever, QuizGenerator
    graph_builder.add_node("intent_classifier", IntentClassifier(llm, intent_classifier_system_prompt))
    graph_builder.add_node("word_extractor", WordExtractor(llm))
    graph_builder.add_node("word_retriever", WordRetriever(llm))
    graph_builder.add_node("quiz_generator", QuizGenerator(llm))
    
    graph_builder.add_edge(START, "preparing")
    graph_builder.add_conditional_edges(
        "preparing",
        after_preparation,
        {
            "success": "intent_classifier",
            "error": END
        }
    )

    graph_builder.add_edge("intent_classifier", "orchestrator")
    graph_builder.add_edge("word_extractor", "orchestrator")
    graph_builder.add_edge("word_retriever", "word_entry_generator")
    graph_builder.add_edge("word_entry_generator", "orchestrator")
    graph_builder.add_edge("quiz_generator", "orchestrator")
    graph_builder.add_edge("presenter", END)

    graph_builder.add_conditional_edges(
        "orchestrator",
        after_orchestrator,
        {
            "extract_words": "word_extractor",
            "define_words": "word_retriever",
            "generate_quizzes": "quiz_generator",
            "general_chat": "presenter",
            "end_session": END
        }
    )

    return graph_builder.compile(checkpointer)
