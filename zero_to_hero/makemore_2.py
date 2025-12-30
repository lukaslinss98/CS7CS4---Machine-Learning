#!/usr/bin/env python
# coding: utf-8

# In[130]:


import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from fontTools.unicodedata import block
get_ipython().run_line_magic('matplotlib', 'inline')


# In[131]:


words = open('data/name.txt', 'r').read().splitlines()
words[:8]


# In[132]:


len(words)


# In[133]:


chars = sorted(set(''.join(words)))
print(len(chars))
stoi = {c: i+1 for i,c in enumerate(chars)}
stoi['.'] = 0
itos = {i: c for c, i in stoi.items()}


# In[198]:


block_size = 5
X, Y = [], []
for word in words:
    context = [0] * block_size
    for c in word + '.':
        X.append(context)
        ix = stoi[c]
        Y.append(ix)
        context = context[1:] + [ix]

X = torch.tensor(X)
Y = torch.tensor(Y)


# In[201]:


C = torch.randn((27,10), requires_grad=True)
W1 = torch.randn((50,200), requires_grad=True)
b1 = torch.randn(200, requires_grad=True)
W2 = torch.randn((200, 27), requires_grad=True)
b2 = torch.randn(27, requires_grad=True)

paramerter = [C, W1, b1, W2, b2]


# In[202]:


for _ in range(50000):
    ix = torch.randint(0, X.shape[0], (64,))

    embedded = C[X[ix]]
    h = torch.tanh(embedded.view(-1, 50) @ W1 + b1)
    logits = h @ W2 + b2
    loss = F.cross_entropy(logits, Y[ix])
    print(loss.item())

    for p in paramerter:
        p.grad = None

    loss.backward()

    for p in paramerter:
        p.data += -0.1 * p.grad



# In[206]:


g = torch.Generator()
for _ in range(20):
    out = []
    context = [0] * block_size
    while True:
        emb = C[torch.tensor([context])]
        h = torch.tanh(emb.view(1, -1) @ W1 + b1)
        logits = h @ W2 + b2
        probs = F.softmax(logits, dim=1)
        ix = torch.multinomial(probs, num_samples=1,generator=g).item()
        context  = context[1:] + [ix]
        out.append(ix)
        if ix == 0:
            break

    print(''.join([itos[i] for i in out]))


