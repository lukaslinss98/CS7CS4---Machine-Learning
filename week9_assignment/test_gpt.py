import random

import torch

from week9_assignment.gpt import GPTLanguageModel

device = 'mps' if torch.backends.mps.is_available() else 'cpu'
iterations = 1000
eval_interval = 100
batch_size = 64
block_size = 256

model = GPTLanguageModel()
state_dictionary = torch.load('child_speech_model_a.pt', map_location=device)
model.load_state_dict(state_dictionary)
model.to(device)
model.eval()
shuffle_input = True

with open('input_childSpeech_trainingSet.txt', 'r', encoding='utf-8') as f:
    text = f.read()
    if shuffle_input:
        text = ''.join(random.sample(text, len(text)))

vocab = torch.load('vocab.pt')
stoi = {ch: i for i, ch in enumerate(vocab)}
itos = {i: ch for i, ch in enumerate(vocab)}
encode = lambda s: [stoi[c] for c in s]
decode = lambda l: ''.join([itos[i] for i in l])

text = ''.join([c for c in text if c in stoi])
data = torch.tensor(encode(text), dtype=torch.long)


def get_batch():
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i + block_size] for i in ix])
    y = torch.stack([data[i + 1:i + block_size + 1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y


losses = []
with torch.no_grad():
    for i in range(iterations):
        X, Y = get_batch()
        _, loss = model(X, Y)
        losses.append(loss.item())
        if i % eval_interval == 0 or i == iterations - 1:
            print(f'step {i}: loss {loss:.4f}')

losses = torch.tensor(losses)
mean_loss = losses.mean().item()
std_loss = losses.std().item()
print(f'{mean_loss=} {std_loss=}')
