import re

import torch

from final.decoder_encoder import create_encoder_decoder
from final.math.util.tokenizer import NumberTokenizer
from math_gpt import MathGPT

device = 'mps' if torch.backends.mps.is_available() else 'cpu'

model = MathGPT()
state_dictionary = torch.load('model_weights_part1.pth', map_location=device)
model.load_state_dict(state_dict=state_dictionary)
model.to(device)
model.eval()
with open('/Users/lukas/dev/machine_learning/final/math/datasets/mixed_operations/test_mixed_double_single.txt', 'r', encoding='utf-8') as f:
    text = f.read()

vocab = torch.load('math_vocab.pt')

encode, decode = create_encoder_decoder(vocabulary=vocab)


def get_operation(expression: str) -> str:
    match = re.search(r'[+\-*/]', expression)
    return match.group()


number_sucess = 0
success_by_operation = {}
number_by_operation = {}

for line in text.split('\n')[:-1]:
    expression, expected = line.split('=')
    operation = get_operation(expression)
    number_by_operation[operation] = number_by_operation.get(operation, 0) + 1

    tokenizer =  NumberTokenizer()
    tokenized = tokenizer.tokenize(expression + '=', padding=True)

    prompt = torch.tensor([encode(tokenized)], dtype=torch.long, device=device)
    output = decode(model.generate(prompt, max_new_tokens=6)[0].tolist())
    output = tokenizer.token_to_string(output)
    # output = output[len(expression) + 1:]

    result = []
    operants = ['+', '-']
    for c in output:
        if c.isdigit() or c in operants:
            result.append(c)
        else:
            break

    result = ''.join(result)

    isSucess = False
    try:
        isSucess = int(expected) == int(result)
    except:
        print(f'could not parse {expected=} or {result=}')

    if isSucess:
        number_sucess += 1
        success_by_operation[operation] = success_by_operation.get(operation, 0) + 1

    print(f'{expression=}, {expected=}, {result=} | {isSucess=}')
    print()

for op, nu in number_by_operation.items():
    print(
        f'operation={op} success: {success_by_operation[op]}, total: {nu}, success rate: {success_by_operation[op] / nu}')

success_rate = number_sucess / len(text.split('\n')[:-1])
print(f"Success Rate: {success_rate}")
