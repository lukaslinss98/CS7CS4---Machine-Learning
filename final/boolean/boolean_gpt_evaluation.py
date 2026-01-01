import re

import torch
from typing import List
from final.boolean.boolean_gpt import BooleanGPT
from final.boolean.util.tokenizer import word_tokenizer
from final.decoder_encoder import create_encoder_decoder

device = 'mps' if torch.backends.mps.is_available() else 'cpu'

model = BooleanGPT()
state_dictionary = torch.load('model_weights_part2.pth', map_location=device)
model.load_state_dict(state_dict=state_dictionary)
model.to(device)
model.eval()
with open('data/depth2_all_operators_unsymmetric/test_mix_depth_all_unsymmetric.txt', 'r', encoding='utf-8') as f:
    text = f.read()

vocabulary = torch.load('boolean_vocab.pt')

encode, decode = create_encoder_decoder(vocabulary=vocabulary)

number_sucess = 0


def get_operator(expression: str) -> List[str]:
    pattern = r'\b(?:AND|OR|XOR|NOT)\b'
    return re.findall(pattern, expression)


success_by_operator = {}
number_by_operator = {}
for line in text.split('\n')[:-1]:
    expression, expected_answer = line.split('=')
    expression = expression.strip()

    operators = get_operator(expression)
    for op in operators:
        number_by_operator[op] = number_by_operator.get(op, 0) + 1

    tokenized_expression = word_tokenizer(expression)
    tokenized_expression.append(' ')
    tokenized_expression.append('=')

    prompt = torch.tensor([encode(tokenized_expression)], dtype=torch.long, device=device)
    output = decode(model.generate(prompt, max_new_tokens=2)[0].tolist())
    output = output.split(' ')[-1]

    isSucess = expected_answer.strip() == output.strip()

    if isSucess:
        number_sucess += 1
        for op in operators:
            success_by_operator[op] = success_by_operator.get(op, 0) + 1

    print(f'{expression=}, expected: {expected_answer.strip()}, actual: {output.strip()} | {isSucess}')

print('|-----------------------------result--------------------------------|')
for op, nu in number_by_operator.items():
    print(
        f'operation={op} success: {success_by_operator[op]}, total: {nu}, success rate: {success_by_operator[op] / nu}')

success_rate = number_sucess / len(text.split('\n')[:-1])
print(f"Success Rate: {success_rate}")
