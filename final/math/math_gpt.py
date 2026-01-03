import torch
import torch.nn as nn
from torch.nn import functional as F

from final.decoder_encoder import create_encoder_decoder
from final.math.util.model_config import xl_model as config
from final.math.util.tokenizer import NumberTokenizer

# hyperparameters
batch_size = config['batch_size']
block_size = config['block_size']
n_embd = config['n_embd']
n_head = config['n_head']
n_layer = config['n_layer']

device = "mps" if torch.backends.mps.is_available() else "cpu"
max_iters = 5000
eval_interval = 500
learning_rate = 3e-4
eval_iters = 200
dropout = 0.2
# ------------

torch.manual_seed(1337)
print(device)

with open('/Users/lukas/dev/machine_learning/final/math/datasets/mixed_operations/training_mixed_double_single.txt',
          'r', encoding='utf-8') as f:
    text = f.read()

tokenized = NumberTokenizer().tokenize(text, padding=True)
vocabulary = sorted(list(set(tokenized)))
torch.save(vocabulary, 'math_vocab.pt')
vocab_size = len(vocabulary)
print(f'Vocabulary Size: {vocab_size}\nText length: {len(text)}\nVocabulary: {vocabulary}')

encode, decode = create_encoder_decoder(vocabulary=vocabulary)

# Train and test splits
data = torch.tensor(encode(tokenized), dtype=torch.long)
split_index = int(0.9 * len(data))
train_data = data[:split_index]
val_data = data[split_index:]


# data loading
def get_batch(split):
    # generate a small batch of data of inputs x and targets y
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i + block_size] for i in ix])
    y = torch.stack([data[i + 1:i + block_size + 1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y


class Head(nn.Module):
    """ one head of self-attention """

    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # input of size (batch, time-step, channels)
        # output of size (batch, time-step, head size)
        _, T, _ = x.shape
        keys = self.key(x)  # (B,T,hs)
        queries = self.query(x)  # (B,T,hs)
        # compute attention scores ("affinities")
        wei = queries @ keys.transpose(-2, -1) * keys.shape[-1] ** -0.5  # (B, T, hs) @ (B, hs, T) -> (B, T, T)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))  # (B, T, T)
        wei = F.softmax(wei, dim=-1)  # (B, T, T)
        wei = self.dropout(wei)
        # perform the weighted aggregation of the values
        values = self.value(x)  # (B,T,hs)
        out = wei @ values  # (B, T, T) @ (B, T, hs) -> (B, T, hs)
        return out


class MultiHeadAttention(nn.Module):
    """ multiple heads of self-attention in parallel """

    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(head_size * num_heads, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out


class FeedFoward(nn.Module):
    """ a simple linear layer followed by a non-linearity """

    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class Block(nn.Module):
    """ Transformer block: communication followed by computation """

    def __init__(self, n_embd, n_head):
        # n_embd: embedding dimension, n_head: the number of heads we'd like
        super().__init__()
        head_size = n_embd // n_head
        self.self_attention = MultiHeadAttention(n_head, head_size)
        self.feedforward = FeedFoward(n_embd)
        self.layer_norm_1 = nn.LayerNorm(n_embd)
        self.layer_norm_2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.self_attention(self.layer_norm_1(x))
        x = x + self.feedforward(self.layer_norm_2(x))
        return x


class MathGPT(nn.Module):

    def __init__(self):
        super().__init__()
        # each token directly reads off the logits for the next token from a lookup table
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[Block(n_embd, n_head=n_head) for _ in range(n_layer)])
        self.layer_norm = nn.LayerNorm(n_embd)
        self.dense_layer = nn.Linear(n_embd, vocab_size)

        # better init, not covered in the original GPT video, but important, will cover in followup video
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        B, T = idx.shape

        # idx and targets are both (B,T) tensor of integers
        tok_emb = self.token_embedding_table(idx)  # (B,T,C)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device))  # (T,C)
        x = tok_emb + pos_emb  # (B,T,C)
        x = self.blocks(x)  # (B,T,C)
        x = self.layer_norm(x)  # (B,T,C)
        logits = self.dense_layer(x)  # (B,T,vocab_size)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B * T, C)
            targets = targets.view(B * T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        # idx is (B, T) array of indices in the current context
        generated = []
        for _ in range(max_new_tokens):
            # crop idx to the last block_size tokens
            idx_cond = idx[:, -block_size:]
            # get the predictions
            logits, _ = self(idx_cond)
            # focus only on the last time step
            logits = logits[:, -1, :]  # becomes (B, C)
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1)  # (B, C)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)  # (B, 1)
            generated.append(idx_next)
            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1)  # (B, T+1)
        return torch.cat(generated, dim=1)


def main():
    model = MathGPT()

    m = model.to(device)
    # print the number of parameters in the model
    print(sum(p.numel() for p in m.parameters()) / 1e6, 'M parameters')

    # create a PyTorch optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-1)

    @torch.no_grad()
    def estimate_loss():
        losses_by_split = {}
        model.eval()
        for split in ['train', 'val']:
            losses = torch.zeros(eval_iters)
            for iteration in range(eval_iters):
                X, Y = get_batch(split)
                _, loss = model(X, Y)
                losses[iteration] = loss.item()
            losses_by_split[split] = losses.mean()
        model.train()
        return losses_by_split

    for iteration in range(max_iters):
        try:
            if iteration % eval_interval == 0 or iteration == max_iters - 1:
                losses = estimate_loss()
                print(f"step {iteration}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

            x_batch, y_batch = get_batch('train')

            logits, loss = model(x_batch, y_batch)

            optimizer.zero_grad(set_to_none=True)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        except KeyboardInterrupt:
            print('Training was stopped')
            break

    print('saving model weights')
    torch.save(model.state_dict(), './model_weights_part1.pth')

    # output = m.generate(prompt, max_new_tokens=6)[0].tolist()
    # print(decode(output))


if __name__ == '__main__':
    main()
