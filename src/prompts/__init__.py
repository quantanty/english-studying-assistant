import os
import yaml
from pathlib import Path
TEMPLATE_KEY = "template"

def _get(full_name: str, dir:Path = None) -> str:
    parts = full_name.split(".")
    
    file_name, key_path = parts[0], parts[1:]
    data = _load_file(file_name + ".yaml")
    
    cur = data
    for k in key_path:
        if not isinstance(cur, dict) or k not in cur:
            raise KeyError(f"Prompt path '{full_name}' not found (stuck at {k})")
        cur = cur[k]

    if isinstance(cur, dict):
        return cur[TEMPLATE_KEY]
    return cur

def _load_file(file: str) -> dict:
    with open(os.path.join(os.path.dirname(__file__), file), "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    return data

INTENT_CLASSIFIER_SYSTEM = _get("intent_classifier.system")
INTENT_CLASSIFIER_USER = _get("intent_classifier.user")

PLANNER_SYSTEM_PHASE_1 = _get("planner.system.phase_1")
PLANNER_SYSTEM_PHASE_2 = _get("planner.system.phase_2")
PLANNER_USER_PHASE_1 = _get("planner.user.phase_1")
PLANNER_USER_PHASE_2 = _get("planner.user.phase_2")

ORCHESTRATOR_SYSTEM = _get("orchestrator.system")
ORCHESTRATOR_USER_START = _get("orchestrator.user.start")
ORCHESTRATOR_USER_NEXT = _get("orchestrator.user.next")

WORD_EXTRACTOR_SYSTEM = _get("word_extractor.system")
WORD_EXTRACTOR_USER = _get("word_extractor.user")

WORD_ENTRY_GENERATOR_SYSTEM = _get("word_entry_generator.system")
WORD_ENTRY_GENERATOR_USER = _get("word_entry_generator.user")

QUIZ_GENERATOR_SYSTEM = _get("quiz_generator.system")
QUIZ_GENERATOR_USER = _get("quiz_generator.user")

PRESENTER_SYSTEM = _get("presenter.system")
PRESENTER_USER = _get("presenter.user")
