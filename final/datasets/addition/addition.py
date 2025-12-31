import random

import numpy as np

from final.util.generate_dataset import Operation, generate_dataset, write_to_file

train_add_single, test_add_single = generate_dataset(
    list(range(0, 9)),
    Operation.ADD,
    0.9,
    extension_factor=100,
)

train_add_double, test_add_double = generate_dataset(
    np.linspace(0, 49, 50).tolist(),
    Operation.ADD,
    0.9,
    extension_factor=4,
)

write_to_file(
    'training_addition_double_single.txt',
    data=[
        *train_add_single,
        *train_add_double,
    ]
)

write_to_file(
    'test_addition_double_single.txt',
    data=[
        *test_add_single,
        *test_add_double,
    ]
)
