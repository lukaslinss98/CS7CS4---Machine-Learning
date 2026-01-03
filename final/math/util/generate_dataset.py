import random
from enum import Enum
from typing import List

random.seed(0)


class Operation(Enum):
    ADD = '+'
    SUB = '-'
    MULT = '*'
    DIV = '/'


def generate_dataset(numbers: List[int], op: Operation, test_split = 0.9, extension_factor=1, padded=False, reversed=False):
    expressions = []
    for num1 in numbers:
        for num2 in numbers:
            num1, num2 = int(num1), int(num2)

            result = None
            if op == Operation.ADD:
                result = num1 + num2
            elif op == Operation.SUB:
                result = num1 - num2
            elif op == Operation.MULT:
                result = num1 * num2
            elif op == Operation.DIV:
                result = int(num1 / num2)

            if padded:
                num1 = f'{num1:04}'
                num2 = f'{num2:04}'
                result = f'{result:04}'

            if reversed:
                result = str(result)[::-1]
                if result[-1] == '-':
                    result = '-' + result[:-1]

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
