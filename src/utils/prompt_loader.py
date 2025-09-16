import yaml
from src.config import PROMPT_DIR
from functools import lru_cache
from typing import Any, Dict
from pathlib import Path

TEMPLATE_KEY = "template"

class PromptLoader:
    """Flexible prompt loader.
    - full_name: 'file.key[.subkey]' will prioritize file.yaml (file=first segment) if exists.
    """
    @classmethod
    def get(cls, full_name: str, dir:Path = None) -> str:
        parts = full_name.split(".")
        
        file_name, key_path = parts[0], parts[1:]
        data = cls._load_file(file_name, dir)
        
        cur = data
        for k in key_path:
            if not isinstance(cur, dict) or k not in cur:
                raise KeyError(f"Prompt path '{full_name}' not found (stuck at {k})")
            cur = cur[k]

        if isinstance(cur, dict):
            return cur[TEMPLATE_KEY]
        return cur
        

    @classmethod
    def render(cls, full_name: str, **kwargs) -> str:
        prompt = cls.get(full_name)
        return prompt.format(**kwargs)
    
    @staticmethod
    @lru_cache(maxsize=None)
    def _load_file(file: str, dir: Path = None) -> dict:
        if dir:
            path = dir / f"{file}.yaml"
        else:
            path = PROMPT_DIR / f"{file}.yaml"
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")
        
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)

        return data
    
if __name__ == "__main__":
    print(PromptLoader.get("simple_prompt.simple_prompt"))