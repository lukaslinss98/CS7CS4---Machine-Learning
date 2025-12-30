import random
import torch
import re

from math_gpt import MathGPT

device = 'mps' if torch.backends.mps.is_available() else 'cpu'

model = MathGPT()
state_dictionary = torch.load('model_weights_part1.pth', map_location=device)
model.load_state_dict(state_dict=state_dictionary)
model.to(device)
model.eval()

with open('test.txt', 'r', encoding='utf-8') as f:
    text = f.read()

vocab = torch.load('math_vocab.pt')
stoi = {ch: i for i, ch in enumerate(vocab)}
itos = {i: ch for i, ch in enumerate(vocab)}
encode = lambda s: [stoi[c] for c in s]
decode = lambda l: ''.join([itos[i] for i in l])


def get_operation(expression: str) -> str:
    match = re.search(r'[+\-*/]', expression)
    return match.group()


number_sucess = 0
success_by_operation = {}
number_by_operation = {}

for line in text.split('\n')[:-1]:
    expression, expected_answer = line.split('=')
    operation = get_operation(expression)
    number_by_operation[operation] = number_by_operation.get(operation, 0) + 1

    prompt = torch.tensor([encode(expression + '=')], dtype=torch.long, device=device)
    output = decode(model.generate(prompt, max_new_tokens=4)[0].tolist())
    output = output[len(expression) + 1:]

    digits = []
    operants = ['+', '-']
    for c in output:
        if c.isdigit() or c in operants:
            digits.append(c)
        else:
            break

    digits = ''.join(digits)

    isSucess = False
    try:
        isSucess = int(expected_answer) == int(digits)
    except:
        print(f'could not parse {expected_answer=} or {digits=}')

    if isSucess:
        number_sucess += 1
        success_by_operation[operation] = success_by_operation.get(operation, 0) + 1

    print(f'{expression=}, expected: {expected_answer}, actual: {digits} | {isSucess}')

success_rate = number_sucess / len(text.split('\n'))
print(f"Success Rate: {success_rate}")
print(f"Number of operations: {number_by_operation}")
print(f"Sucess by operations: {success_by_operation}")
