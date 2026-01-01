import random
from enum import Enum
from typing import List


def write_to_file(path: str, data: List[str]):
    with open(path, 'w') as f:
        random.shuffle(data)
        file_data = '\n'.join(data)
        f.write(file_data)


class BooleanOperator(Enum):
    AND = '&&'
    OR = 'OR'
    XOR = 'XOR'
    NOT = 'NOT'


boolean_literal = ['TRUE', 'FALSE']
binary_operators = ['AND', 'OR', 'XOR']

mapping = {
    'TRUE': 'True',
    'FALSE': 'False',
    'AND': 'and',
    'OR': 'or',
    'XOR': '!=',
    'NOT': 'not',
    '(': '(',
    ')': ')'
}


def generate_boolean_expression(depth=1, percentage_unary=0.2, symmetrical=True, is_first=True):
    if depth == 0:
        return [random.choice(boolean_literal)]

    if not symmetrical and not is_first and random.random() < 0.2:
        return [random.choice(boolean_literal)]

    if random.random() < percentage_unary:
        return ['('] + ['NOT'] + generate_boolean_expression(depth=depth - 1, percentage_unary=0.0) + [')']

    left = generate_boolean_expression(depth=depth - 1, is_first=False, percentage_unary=percentage_unary)
    right = generate_boolean_expression(depth=depth - 1, is_first=False, percentage_unary=percentage_unary)
    operation = random.choice(binary_operators)

    return ['('] + left + [operation] + right + [')']


def evaluate_expression(expression: List[str]):
    mapped = [mapping[part] for part in expression]
    print(mapped)
    return eval(' '.join(mapped))


def generate_boolean_expressions(number_of_expressions, max_depth=1, symmetrical=False, percentage_unary=0.0, test_split=0.9):
    expressions = []
    for _ in range(number_of_expressions):
        expression = generate_boolean_expression(depth=max_depth, symmetrical=symmetrical, percentage_unary=percentage_unary)[1:-1]
        print(expression)
        result = evaluate_expression(expression)
        result = str(result).upper()
        expression.append('=')
        expression.append(result)
        expressions.append(' '.join(expression))

    split_index = int(len(expressions) * test_split)

    return expressions[:split_index], expressions[split_index:]

