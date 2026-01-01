from final.boolean.util.generate_dataset import generate_boolean_expressions, write_to_file

train, test = generate_boolean_expressions(40000, max_depth=2, percentage_unary=0.25)

write_to_file('train_depth2_all.txt', train)
write_to_file('test_depth2_all.txt', test)
