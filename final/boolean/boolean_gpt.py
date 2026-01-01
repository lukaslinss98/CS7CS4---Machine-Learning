import torch
import torch.nn as nn
from torch.nn import functional as F

from final.boolean.util.tokenizer import word_tokenizer
from final.decoder_encoder import create_encoder_decoder
from final.boolean.util.model_config import large_model as config

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

with open('data/depth2_all_operators_unsymmetric/train_mix_depth_all_unsymmetric.txt', 'r', encoding='utf-8') as f:
    text = f.read()

vocabulary = ['\n', '=', 'TRUE', 'FALSE', 'AND', 'OR', 'XOR', 'NOT', '(', ')', ' ']
vocab_size = len(vocabulary)
torch.save(vocabulary, 'boolean_vocab.pt')
print(f'Vocabulary Size: {vocab_size}\nText length: {len(text)}\nVocabulary: {vocabulary}')

encode, decode = create_encoder_decoder(vocabulary=vocabulary)

# Train and test splits
tokenized_text = word_tokenizer(text)

data = torch.tensor(encode(tokenized_text), dtype=torch.long)
split_index = int(0.9 * len(data))
train_data = data[:split_index]
val_data = data[split_index:]


# data loading
def get_batch(split):
    # generate a small batch of data of inputs x and targets y
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i : i + block_size] for i in ix])
    y = torch.stack([data[i + 1 : i + block_size + 1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y


class Head(nn.Module):
    """ one head of self-attention """

    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, kead_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # input of size (batch, time-step, channels)
        # output of size (batch, time-step, head size)
        _, T, _ = x.shape
        k = self.key(x)  # (B,T,hs)
        q = self.query(x)  # (B,T,hs)
        # compute attention scores ("affinities")
        wei = q @ k.transpose(-2, -1) * k.shape[-1] ** -0.5  # (B, T, hs) @ (B, hs, T) -> (B, T, T)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))  # (B, T, T)
        wei = F.softmax(wei, dim=-1)  # (B, T, T)
        wei = self.dropout(wei)
        # perform the weighted aggregation of the values
        v = self.value(x)  # (B,T,hs)
        out = wei @ v  # (B, T, T) @ (B, T, hs) -> (B, T, hs)
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
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedFoward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x


class BooleanGPT(nn.Module):

    def __init__(self):
        super().__init__()
        # each token directly reads off the logits for the next token from a lookup table
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[Block(n_embd, n_head=n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)  # final layer norm
        self.lm_head = nn.Linear(n_embd, vocab_size)

        # better init, not covered in the original GPT video, but important, will cover in followup video
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, input_batch, targets=None):
        B, T = input_batch.shape

        # idx and targets are both (B,T) tensor of integers
        tok_emb = self.token_embedding_table(input_batch)  # (B,T,C)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device))  # (T,C)
        x = tok_emb + pos_emb  # (B,T,C)
        x = self.blocks(x)  # (B,T,C)
        x = self.ln_f(x)  # (B,T,C)
        logits = self.lm_head(x)  # (B,T,vocab_size)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B * T, C)
            targets = targets.view(B * T)
            loss = F.cross_entropy(logits, targets)


        return logits, loss

    def generate(self, input_batch, max_new_tokens):
        # idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            # crop idx to the last block_size tokens
            input_batch_cropped = input_batch[:, -block_size:]
            # get the predictions
            logits, _ = self(input_batch_cropped)
            # focus only on the last time step
            logits = logits[:, -1, :]  # becomes (B, C)
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1)  # (B, C)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)  # (B, 1)
            # append sampled index to the running sequence
            input_batch = torch.cat((input_batch, idx_next), dim=1)  # (B, T+1)
        return input_batch


def main():
    model = BooleanGPT()

    m = model.to(device)
    # print the number of parameters in the model
    print(sum(p.numel() for p in m.parameters()) / 1e6, 'M parameters')

    # create a PyTorch optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-1)

    @torch.no_grad()
    def estimate_loss():
        out = {}
        model.eval()
        for split in ['train', 'val']:
            losses = torch.zeros(eval_iters)
            for k in range(eval_iters):
                X, Y = get_batch(split)
                _, loss = model(X, Y)
                losses[k] = loss.item()
            out[split] = losses.mean()
        model.train()
        return out

    for iter in range(max_iters):
        try:
            if iter % eval_interval == 0 or iter == max_iters - 1:
                losses = estimate_loss()
                print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

            x_batch, y_batch = get_batch('train')
            B, T = x_batch.shape

            logits, _ = model(x_batch, y_batch)
            optimizer.zero_grad(set_to_none=True)

            equals_id = encode(['='])[0]
            true_id = encode(['TRUE'])[0]
            false_id = encode(['FALSE'])[0]

            prev_is_eq = torch.zeros_like(y_batch, dtype=torch.bool)
            prev_is_eq[:, 1:] = (x_batch[:, :-1] == equals_id)

            is_truth_token = (y_batch == true_id) | (y_batch == false_id)

            answer_pos = prev_is_eq & is_truth_token

            # # ---- DEBUG PRINT EVERY 200 ITERS ----
            # if iter % 200 == 0:
            #     b = 0  # look at first example in batch
            #     print("---- DEBUG SAMPLE ----")
            #     print("x:", " ".join(decode([int(t)]) for t in x_batch[b]))
            #     print("y:", " ".join(decode([int(t)]) for t in y_batch[b]))
            #     print("answer flags:", answer_pos[b].tolist())
            #     for t in range(T):
            #         if answer_pos[b, t]:
            #             prev_tok = decode([int(x_batch[b, t - 1])]) if t > 0 else "<NONE>"
            #             cur_tok = decode([int(y_batch[b, t])])
            #             print(f"ANSWER POS t={t}: prev='{prev_tok}' cur='{cur_tok}'")

            logits_flat = logits.view(B * T, -1)
            targets_flat = y_batch.view(B * T)

            weights = torch.ones_like(y_batch, dtype=torch.float32)
            weights[answer_pos] = 5.0  # or 3–10

            weights_flat = weights.view(B * T)
            per_token_ce = F.cross_entropy(logits_flat, targets_flat, reduction='none')
            loss = (per_token_ce * weights_flat).sum() / weights_flat.sum()

            loss.backward()
            optimizer.step()
        except KeyboardInterrupt:
            print('Training was stopped')
            break

    print('saving model weights')
    torch.save(model.state_dict(), 'model_weights_part2.pth')

    # output = m.generate(prompt, max_new_tokens=6)[0].tolist()
    # print(decode(output))


if __name__ == '__main__':
    main()
