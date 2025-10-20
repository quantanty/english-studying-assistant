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
from src.graph.nodes.planner import Planner
from src.graph.edges.decision import after_orchestrator, after_preparation

from langgraph.graph import StateGraph, START, END
import logging

def create_graph(llm=None, checkpointer=None):
    intent_classifier_system = PromptLoader.get("intent_classifier.system")
    intent_classifier_user = PromptLoader.get("intent_classifier.user")

    planner_system_1 = PromptLoader.get("planner.system.phase_1")
    planner_system_2 = PromptLoader.get("planner.system.phase_2")

    orchestrator_system = PromptLoader.get("orchestrator.system")
    orchestrator_user_1 = PromptLoader.get("orchestrator.user.first")
    orchestrator_user_2 = PromptLoader.get("orchestrator.user.after")

    word_entry_generator_prompt = PromptLoader.get("simple_prompt.system_prompt.dictionary")
    presenter_prompt = PromptLoader.get("simple_prompt.system_prompt.presenter")

    llm = llm or create_llm()

    graph_builder = StateGraph(State)

    # graph_builder.add_node("preparing", preparing(orchestrator_system_prompt, orchestrator_user_template))
    graph_builder.add_node("intent_classifier", IntentClassifier(llm, intent_classifier_system, intent_classifier_user))
    graph_builder.add_node("planner", Planner(llm, planner_system_1, planner_system_2))
    graph_builder.add_node("orchestrator", Orchestrator(llm, orchestrator_system, orchestrator_user_1, orchestrator_user_2))
    graph_builder.add_node("word_entry_generator", WordEntryGenerator(llm, word_entry_generator_prompt))
    graph_builder.add_node("presenter", Presenter(llm, presenter_prompt))
    #TODO: write prompt for Planner, WordExtractor, WordRetriever, QuizGenerator
    graph_builder.add_node("word_extractor", WordExtractor(llm))
    graph_builder.add_node("word_retriever", WordRetriever(llm))
    graph_builder.add_node("quiz_generator", QuizGenerator(llm))
    
    graph_builder.add_edge(START, "intent_classifier")

    graph_builder.add_edge("intent_classifier", "planner")
    graph_builder.add_edge("planner", "orchestrator")
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
            "present": "presenter",
            "end": END
        }
    )

    return graph_builder.compile(checkpointer)
