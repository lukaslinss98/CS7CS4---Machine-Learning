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


train_mult_double, test_mult_double = generate_dataset(
    np.linspace(0, 50, 51).tolist(),
    Operation.MULT,
    0.9,
    extension_factor=4,
)


train_div_double, test_div_double = generate_dataset(
    np.linspace(1, 50, 50).tolist(),
    Operation.DIV,
    0.9,
    extension_factor=4,
)

print(
    f'add double: {len(test_add_double)}\nsub double: {len(test_sub_double)}\nmult double: {len(test_mult_double)}\ndiv double: {len(test_div_double)}')

write_to_file(
    'training_mixed_double_single.txt',
    data=[
        *train_add_double,
        *train_sub_double,
        *train_mult_double,
        *train_div_double
    ]
)

write_to_file(
    'test_mixed_double_single.txt',
    data=[
        *test_add_double,
        *test_sub_double,
        *test_mult_double,
        *test_div_double
    ]
)

