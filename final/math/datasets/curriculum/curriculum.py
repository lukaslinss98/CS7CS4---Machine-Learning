import random

import numpy as np

from final.math.util.generate_dataset import Operation, generate_dataset, write_to_file

train_add_double, test_add_double = generate_dataset(
    np.linspace(0, 50, 51).tolist(),
    Operation.ADD,
    0.9,
    extension_factor=2,
)

train_sub_double, test_sub_double = generate_dataset(
    np.linspace(0, 50, 51).tolist(),
    Operation.SUB,
    0.9,
    extension_factor=2,
)

train_mult_single, test_mult_single = generate_dataset(
    np.linspace(0, 9, 10).tolist(),
    Operation.MULT,
    0.9,
    extension_factor=100,
)

train_mult_double, test_mult_double = generate_dataset(
    np.linspace(0, 50, 51).tolist(),
    Operation.MULT,
    0.9,
    extension_factor=4,
)

train_div_single, test_div_single = generate_dataset(
    np.linspace(1, 10, 10).tolist(),
    Operation.DIV,
    0.9,
    extension_factor=100,
)

train_div_double, test_div_double = generate_dataset(
    np.linspace(1, 50, 50).tolist(),
    Operation.DIV,
    0.9,
    extension_factor=4,
)

print(
    f'add double: {len(test_add_double)}\nsub double: {len(test_sub_double)}\nmult double: {len(test_mult_double)}\ndiv double: {len(test_div_double)}')

curriculum_part1 = [
    *train_add_double,
    *train_sub_double,
]

random.shuffle(curriculum_part1)
curriculum_part1 = '\n'.join(curriculum_part1)

curriculum_part2 = [
    *train_mult_single,
    *train_div_single,
    *train_add_double,
    *train_sub_double,
]
random.shuffle(curriculum_part2)
curriculum_part2 = '\n'.join(curriculum_part2)

curriculum_part3 = [
    *train_mult_single*4,
    *train_div_single*4,
    *train_add_double,
    *train_sub_double,
    *train_mult_double*4,
    *train_div_double*4,
]
random.shuffle(curriculum_part3)
curriculum_part3 = '\n'.join(curriculum_part3)

curriculum = [curriculum_part1, curriculum_part2, curriculum_part3]

write_to_file(
    'training_curriculum_add_sub.txt',
    data=[
        *train_add_double,
        *train_sub_double,
    ]
)

write_to_file(
    'training_curriculum_mult_div_single.txt',
    data=[
        *train_mult_single,
        *train_div_single
    ]
)

write_to_file(
    'training_curriculum_mult_div_double.txt',
    data=[
        *train_mult_double,
        *train_div_double
    ]
)

write_to_file(
    'test_curriculum_mixed.txt',
    data=[
        *test_add_double,
        *test_sub_double,
        *test_mult_single,
        *test_div_single,
        *test_mult_double,
        *test_div_double
    ]
)
