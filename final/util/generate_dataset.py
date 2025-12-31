import random
from enum import Enum
from typing import List

random.seed(0)


class Operation(Enum):
    ADD = '+'
    SUB = '-'
    MULT = '*'
    DIV = '/'


def generate_dataset(numbers: List[int], op: Operation, test_split: float, extension_factor=1, padded=False):
    expressions = []
    for i in numbers:
        for j in numbers:
            i, j = int(i), int(j)

            result = None
            if op == Operation.ADD:
                result = i + j
            elif op == Operation.SUB:
                result = i - j
            elif op == Operation.MULT:
                result = i * j
            elif op == Operation.DIV:
                result = int(i / j)

            num1 = f'{i:04}' if padded else str(i)
            num2 = f'{j:04}' if padded else str(j)
            result = f'{result:04}' if padded else str(result)

            expression = f'{num1}{op.value}{num2}={result}'
            expressions.append(expression)

    random.shuffle(expressions)
    split_index = int(len(expressions) * test_split)
    return expressions[:split_index] * extension_factor, expressions[split_index:]

def write_to_file(path: str, data: List[str]):
    with open(path, 'w') as f:
        random.shuffle(data)
        file_data = '\n'.join(data)
        f.write(file_data)
