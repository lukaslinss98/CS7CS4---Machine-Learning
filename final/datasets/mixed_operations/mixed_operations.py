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

train_sub_single, test_sub_single = generate_dataset(
    np.linspace(0, 9, 10).tolist(),
    Operation.SUB,
    0.9,
    extension_factor=100,
)

train_sub_double, test_sub_double = generate_dataset(
    np.linspace(0, 49, 50).tolist(),
    Operation.SUB,
    0.9,
    extension_factor=4,
)

train_mult_single, test_mult_single = generate_dataset(
    np.linspace(0, 9, 10).tolist(),
    Operation.MULT,
    0.9,
    extension_factor=100,
)

train_mult_double, test_mult_double = generate_dataset(
    np.linspace(0, 19, 20).tolist(),
    Operation.MULT,
    0.9,
)

train_div_single, test_div_single = generate_dataset(
    np.linspace(1, 9, 9).tolist(),
    Operation.DIV,
    0.9,
    extension_factor=100,
)

write_to_file(
    'training_mixed_double_single.txt',
    data=[
        *train_add_single,
        *train_add_double,
        *train_sub_single,
        *train_sub_double,
        *train_mult_single,
        *train_mult_double,
        *train_div_single,
    ]
)

write_to_file(
    'test_mixed_double_single.txt',
    data=[
        *test_add_single,
        *test_add_double,
        *test_sub_single,
        *test_sub_double,
        *test_mult_single,
        *test_mult_double,
        *test_div_single,
    ]
)
