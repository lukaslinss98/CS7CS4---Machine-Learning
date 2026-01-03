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
with open('./data/split_datasets/test_mix_depth_all_unsymmetric.txt', 'r', encoding='utf-8') as f:
    text = f.read()

vocabulary = torch.load('boolean_vocab.pt')

encode, decode = create_encoder_decoder(vocabulary=vocabulary)

number_sucess = 0


def get_depth(expression: str) -> int:
    stack = []
    max_depth = 0
    for c in expression:
        if c == '(':
            stack.append(c)
            max_depth = max(max_depth, len(stack))
        elif c == ')':
            stack.pop()

    return max_depth + 1


success_by_depth = {}
number_by_depth = {}
for line in text.split('\n')[:-1]:
    expression, expected_answer = line.split('=')
    expression = expression.strip()
    depth = get_depth(expression)
    number_by_depth[depth] = number_by_depth.get(depth, 0) + 1

    tokenized_expression = word_tokenizer(expression)
    tokenized_expression.append(' ')
    tokenized_expression.append('=')

    prompt = torch.tensor([encode(tokenized_expression)], dtype=torch.long, device=device)
    output = decode(model.generate(prompt, max_new_tokens=2)[0].tolist())
    isSucess = expected_answer.strip() == output.strip()

    if isSucess:
        number_sucess += 1
        success_by_depth[depth] = success_by_depth.get(depth, 0) + 1

    print(f'{expression=}, expected: {expected_answer.strip()}, actual: {output.strip()} | {isSucess}')

print('|-----------------------------result--------------------------------|')
for depth, nu in number_by_depth.items():
    print(
        f'{depth=} success: {success_by_depth[depth]}, total: {nu}, success rate: {success_by_depth[depth] / nu}')

success_rate = number_sucess / len(text.split('\n')[:-1])
print(f"Success Rate: {success_rate}")
