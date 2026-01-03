from final.boolean.util.generate_dataset import generate_boolean_expressions, write_to_file

train_depth2, test_depth2 = generate_boolean_expressions(5000, max_depth=2, percentage_unary=0.25, symmetrical=False)
train_depth1, test_depth_1 = generate_boolean_expressions(5000, max_depth=1, percentage_unary=0.25, symmetrical=False)

write_to_file('train_mix_depth_all_unsymmetric.txt', [*train_depth1,*train_depth2])
write_to_file('test_mix_depth_all_unsymmetric.txt',[ *train_depth1, *test_depth2])
