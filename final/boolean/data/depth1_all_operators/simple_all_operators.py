from final.boolean.util.generate_dataset import generate_boolean_expressions, write_to_file

train, test = generate_boolean_expressions(30000, percentage_unary=0.3)

write_to_file('train_simple_all.txt', train)
write_to_file('test_simple_all.txt', test)