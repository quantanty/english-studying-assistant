from src.utils.prompt_loader import PromptLoader
from src.graph.builder import create_graph

from langchain_core.runnables.graph_mermaid import draw_mermaid_png
from pprint import pprint

if __name__ == "__main__":
    graph = create_graph()
    
    mermaid_syntax = graph.get_graph().draw_mermaid()
    output_file_path = "outputs/graph.png"
    draw_mermaid_png(mermaid_syntax, output_file_path=output_file_path)

    hard_prompt_1 = PromptLoader.get("simple_prompt.user_prompt.hard_1")
    # hard_prompt_2 = PromptLoader.get("simple_prompt.user_prompt.hard_2")
    stream = graph.stream(
        {"query": hard_prompt_1},
        stream_mode="values"
    )
    for event in stream:
        pprint(event, width=160)

    # stream = graph.stream(
    #     {"query": hard_prompt_2},
    #     stream_mode="values"
    # )
    # for event in stream:
    #     pprint(event, width=160)    