import torch
import torch.nn as nn
from torch.nn import functional as F
import tiktoken
import sys

SEED = 1337
DATASET_PATH = "dataset.txt"
WIKIBOT_INFO_PATH = "wikibot.txt"
MODEL_PATH = "wikibot.pkl"
TIKTOKEN_MODEL = "gpt2"
DATASET_OFFSET = 0
DATASET_SIZE = 10_000_000
BLOCK_SIZE = 128
BATCH_SIZE = 16
EVAL_INTERVAL = 100
MAX_ITERS = 5000
RESPONSE_LIMIT = 128
LEARNING_RATE = 3e-4
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
EVAL_ITERS = 200
N_EMBD = 384
N_HEAD = 6
N_LAYER = 6
DROPOUT = 0.2
TRAINING_SPLIT = 0.8

with open(DATASET_PATH) as file:
    with open(WIKIBOT_INFO_PATH) as wikibot_file:
        file.seek(DATASET_OFFSET)
        text = file.read(DATASET_SIZE) + '\n\n' + wikibot_file.read()

# Bigram Language Model
class Model(nn.Module):
    def __init__(self, tokenizer):
        super().__init__()
        self.token_embedding_table = nn.Embedding(tokenizer.n_vocab, N_EMBD)
        self.position_embedding_table = nn.Embedding(BLOCK_SIZE, N_EMBD)
        self.blocks = nn.Sequential(*[Block(N_EMBD, n_head=N_HEAD) for _ in range(N_LAYER)])
        self.ln_f = nn.LayerNorm(N_EMBD)
        self.lm_head = nn.Linear(N_EMBD, tokenizer.n_vocab)
        self.tokenizer = tokenizer

    def forward(self, idx, targets=None):
        B, T = idx.shape

        tok_emb = self.token_embedding_table(idx)
        pos_emb = self.position_embedding_table(torch.arange(T, device=DEVICE))
        x = tok_emb + pos_emb
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)
        
        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T,C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens=RESPONSE_LIMIT):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -BLOCK_SIZE:]
            logits, loss = self(idx_cond)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx

    def response(self, prompt):
        context = torch.zeros((1, 1), dtype=torch.long, device=DEVICE)
        return self.tokenizer.decode(self.generate(context, max_new_tokens=200)[0].tolist())

class Block(nn.Module):
    def __init__(self, n_embd, n_head):
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedForward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x

class FeedForward(nn.Module):
    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(DROPOUT),
        )

    def forward(self, x):
        return self.net(x)

# self-attention. A token can choose what to focus on in the tokens before it
class Head(nn.Module):
    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(N_EMBD, head_size, bias=False)
        self.query = nn.Linear(N_EMBD, head_size, bias=False)
        self.value = nn.Linear(N_EMBD, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(BLOCK_SIZE, BLOCK_SIZE)))

        self.dropout = nn.Dropout(DROPOUT)

    def forward(self, x):
        B,T,C = x.shape
        k = self.key(x)
        q = self.query(x)
        wei = q @ k.transpose(-2,-1) * C**-0.5
        wei = wei.masked_fill(self.tril[:T,:T] == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)
        v = self.value(x)
        out = wei @ v
        return out

class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(N_EMBD, N_EMBD)
        self.dropout = nn.Dropout(DROPOUT)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.proj(out)
        out = self.dropout(out)
        return out


if __name__ == "__main__":
    if len(sys.argv) == 2 and sys.argv[1] == "train":
        tokenizer = tiktoken.get_encoding(TIKTOKEN_MODEL)
        data = torch.tensor(tokenizer.encode(text), dtype=torch.long)

        # Training/Validation split
        n = int(TRAINING_SPLIT*len(data))
        train_data = data[:n]
        val_data = data[n:]

        torch.manual_seed(SEED)

        def get_batch(split):
            # generate a small batch of data of inputs x and targets y
            data = train_data if split == 'train' else val_data
            ix = torch.randint(len(data) - BLOCK_SIZE, (BATCH_SIZE,))
            x = torch.stack([data[i:i+BLOCK_SIZE] for i in ix])
            y = torch.stack([data[i+1:i+BLOCK_SIZE+1] for i in ix])
            x, y = x.to(DEVICE), y.to(DEVICE)
            return x, y

        @torch.no_grad()
        def estimate_loss():
            out = {}
            # model.eval()
            for split in ['train', 'val']:
                losses = torch.zeros(EVAL_ITERS)
                for k in range(EVAL_ITERS):
                    X, Y = get_batch(split)
                    logits, loss = model(X,Y)
                    losses[k] = loss.item()
                out[split] = losses.mean()
            # model.train()
            return out

        model = Model(tokenizer)
        m = model.to(DEVICE)

        # Training
        optimizer = torch.optim.AdamW(m.parameters(), lr=LEARNING_RATE)
        for iter in range(MAX_ITERS):
            if iter % EVAL_INTERVAL == 0:
                losses = estimate_loss()
                print(f"#{iter} train loss: {losses['train']:.4}, validation loss: {losses['val']:.4}")
                if iter > EVAL_INTERVAL:
                    print("Saving model at ./temp-" + MODEL_PATH)
                    torch.save(m.state_dict(), "temp-" + MODEL_PATH)
            xb, yb = get_batch('train')
            logits,loss = m(xb, yb)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

        print("Saving model at ./" + MODEL_PATH)
        torch.save(m.state_dict(), MODEL_PATH)
    elif len(sys.argv) <= 2:
        model = Model(tiktoken.get_encoding(TIKTOKEN_MODEL))
        if len(sys.argv) == 2:
            model.load_state_dict(torch.load(sys.argv[1], weights_only=False))
        else:
            model.load_state_dict(torch.load(MODEL_PATH, weights_only=False))
        m = model.to(DEVICE)
        print(m.response("A new user just started you. How do you respond?"))

        while True:
            print(m.response(input("> ")))
