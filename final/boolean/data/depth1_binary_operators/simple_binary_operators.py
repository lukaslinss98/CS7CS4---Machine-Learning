from final.boolean.util.generate_dataset import generate_boolean_expressions, write_to_file

train, test = generate_boolean_expressions(10000, percentage_unary=0.0)

write_to_file('train_simple_binary.txt', train)
write_to_file('test_simple_binary.txt', test)