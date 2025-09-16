from src.schemas.state import State, WordEntryState

def get_word_defining_results(word_entries: WordEntryState) -> str:
    result_string = ""
    for word, entry in word_entries.items():
        result = entry.get("result", "error")
        result_string += f"'{word}': {result}\n"
    return result_string