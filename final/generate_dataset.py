import random
from enum import Enum
from typing import List

import numpy as np

random.seed(0)


class Operation(Enum):
    ADD = '+'
    SUB = '-'
    MULT = '*'
    DIV = '/'


def generate_dataset(numbers: List[int], op: Operation, test_split: float, extension_factor=1):
    expressions = []
    for i in numbers:
        for j in numbers:
            i, j = int(i), int(j)

            # if j > i:
            #     continue

            result = None
            if op == Operation.ADD:
                result = i + j
            elif op == Operation.SUB:
                result = i - j
            elif op == Operation.MULT:
                result = i * j
            elif op == Operation.DIV:
                result = int(i / j)

            expression = f'{i:03}{op.value}{j:03}={result:03}'
            expressions.append(expression)

    random.shuffle(expressions)
    split_index = int(len(expressions) * test_split)
    return expressions[:split_index] * extension_factor, expressions[split_index:] * extension_factor


train_addition_single, test_addition_single = generate_dataset(
    np.linspace(0, 9, 10).tolist(),
    Operation.ADD,
    0.9,
    extension_factor=100
)

train_addition_double, test_addition_double = generate_dataset(
    np.linspace(0, 49, 50).tolist(),
    Operation.ADD,
    0.9,
    extension_factor=4
)

train_sub_single, test_sub_single = generate_dataset(
    np.linspace(0, 9, 10).tolist(),
    Operation.SUB,
    0.9,
    extension_factor=100
)

train_sub_double, test_sub_double = generate_dataset(
    np.linspace(0, 49, 50).tolist(),
    Operation.SUB,
    0.9,
    extension_factor=8
)

train_mult_single, test_mult_single = generate_dataset(
    np.linspace(0, 9, 10).tolist(),
    Operation.MULT,
    0.9,
    extension_factor=100
)

train_mult_double, test_mult_double = generate_dataset(
    np.linspace(0, 19, 20).tolist(),
    Operation.MULT,
    0.9
)

train_div_single, test_div_single = generate_dataset(
    np.linspace(1, 9, 9).tolist(),
    Operation.DIV,
    0.9,
    extension_factor=100
)

with open('training.txt', 'w') as f:
    training = [
        *train_addition_single,
        *train_sub_single,
        *train_mult_single,
        *train_div_single,
        *train_addition_double,
        *train_sub_double,
    ]
    random.shuffle(training)
    data = '\n'.join(training)
    f.write(data)

with open('test.txt', 'w') as f:
    test = [
        *test_addition_single,
        *test_sub_single,
        *test_mult_single,
        *test_div_single,
        *test_addition_double,
        *test_sub_double
    ]
    random.shuffle(test)
    data = '\n'.join(test)
    f.write(data)
