from final.boolean.util.generate_dataset import generate_boolean_expressions, write_to_file


def split_dataset(data, test_split):
    split_index = int(len(data) * test_split)
    return data[:split_index], data[split_index:]


depth1, _ = generate_boolean_expressions(10000, max_depth=1, percentage_unary=0.25, symmetrical=False, test_split=1.0)
depth2, _ = generate_boolean_expressions(10000, max_depth=2, percentage_unary=0.25, symmetrical=False, test_split=1.0)
depth3, _ = generate_boolean_expressions(5000, max_depth=3, percentage_unary=0.25, symmetrical=False, test_split=1.0)

depth1 = list(set(depth1))
depth2 = list(set(depth2))
depth3 = list(set(depth3))

train_depth1, test_depth1 = split_dataset(depth1, test_split=0.7)
train_depth2, test_depth2 = split_dataset(depth2, test_split=0.9)
train_depth3, test_depth3 = split_dataset(depth3, test_split=0.9)


print(len(train_depth1), len(train_depth2), len(train_depth3))
print(len(test_depth1), len(test_depth2), len(test_depth3))

write_to_file(
    'train_mix_depth_all_unsymmetric.txt',
    [
        *train_depth1*100,
        *train_depth2*10,
        *train_depth3
    ]
)
write_to_file(
    'test_mix_depth_all_unsymmetric.txt',
    [
        *test_depth1,
        *test_depth2,
        *test_depth3
    ]
)
