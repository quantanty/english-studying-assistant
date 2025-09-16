from src.utils.prompt_loader import PromptLoader
from src.graph.builder import create_graph

from langchain_core.runnables.graph_mermaid import draw_mermaid_png
from pprint import pprint
from pathlib import Path

if __name__ == "__main__":
    graph = create_graph()
    
    # mermaid_syntax = graph.get_graph().draw_mermaid()
    # output_file_path = "outputs/graph.png"
    # draw_mermaid_png(mermaid_syntax, output_file_path=output_file_path)

    dir = Path(__file__).resolve().parent / "prompts"
    prompt_1 = PromptLoader.get("tricky_prompts.tricky_4.prompt_1", dir=dir)
    prompt_2 = PromptLoader.get("tricky_prompts.tricky_4.prompt_2", dir=dir)
    stream = graph.stream(
        {"query": prompt_1},
        stream_mode="values"
    )
    for event in stream:
        pprint(event, width=160)

    stream = graph.stream(
        {"query": prompt_2},
        stream_mode="values"
    )
    for event in stream:
        pprint(event, width=160)