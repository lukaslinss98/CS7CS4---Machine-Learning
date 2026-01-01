from typing import List
import re

def word_tokenizer(data: str) -> List[str]:
    return re.split(r'(\s+)', data)
