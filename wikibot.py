import torch
import torch.nn as nn
from torch.nn import functional as F
from dataclasses import dataclass
import tiktoken
import warnings
import inspect
import array
import time
import math
import sys
import os

warnings.filterwarnings("ignore", "Attempting to use hipBLASLt")

TOKEN_DIR = "tokens"
TEMP_MODEL_PATH = "./temp-wikibot.pkl"
MODEL_PATH = "./wikibot.pkl"
TOKEN_ENCODING = "gpt2"

class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.c_attn = nn.Linear(config.n_embd, config.n_embd * 3)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        self.c_proj.SCALE_INIT = 1
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
                            .view(1, 1, config.block_size, config.block_size))

    def forward(self, x):
        B, T, C = x.size()
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.c_proj(y)
        return y

class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, config.n_embd * 4)
        self.gelu = nn.GELU(approximate='tanh')
        self.c_proj = nn.Linear(config.n_embd * 4, config.n_embd)
        self.c_proj.SCALE_INIT = 1

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return x

class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x

@dataclass
class WikibotConfig:
    block_size: int = 1024 # context window
    vocab_size: int = 50257 # number of tokens
    n_layer: int = 12 # layers
    n_head: int = 12 # heads
    n_embd: int = 768 # embedding dimension

class Wikibot(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            wpe = nn.Embedding(config.block_size, config.n_embd),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = nn.LayerNorm(config.n_embd),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.transformer.wte.weight = self.lm_head.weight
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            std = 0.02
            if hasattr(module, "SCALE_INIT"):
                std *= (2 * module.SCALE_INIT) ** -0.5
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        B, T = idx.size()
        assert T <= self.config.block_size, "Cannot forward, model block size is exhausted."
        pos = torch.arange(T, dtype=torch.long, device=idx.device)
        pos_emb = self.transformer.wpe(pos)
        tok_emb = self.transformer.wte(idx)
        x = tok_emb + pos_emb
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss

    def configure_optimizers(self, weight_decay, learning_rate, device):
        param_dict = {k: p for k, p in self.named_parameters() if p.requires_grad}
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {"params": decay_params, "weight_decay": weight_decay},
            {"params": nodecay_params, "weight_decay": 0.0},
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params} parameters")
        print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params} parameters")
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device != 'cpu'
        print(f"using fused adam: {use_fused}")
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=(0.9, 0.95), eps=1e-8)
        return optimizer

    def respond(self, prompt, max_length=64):
        encoder = tiktoken.get_encoding(TOKEN_ENCODING)
        x = encoder.encode(prompt)
        x.append(tokenizer._special_tokens['<|endoftext|>'])
        prompt_len = len(x)
        x = torch.tensor(x, dtype=torch.long).unsqueeze(0).to(device)

        while x.size(1) < max_length:
            with torch.no_grad():
                logits, _ = self(x)
                logits = logits[:, -1, :]
                probs = F.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                x = torch.cat([x, next_token], dim=1)
                if next_token == tokenizer._special_tokens['<|endoftext|>']:
                    break

        return encoder.decode(x[0, prompt_len:max_length].tolist())

class DataLoader:
    def __init__(self, B, T, split):
        self.B = B
        self.T = T
        self.shards = sorted(list(map(int, os.listdir(TOKEN_DIR))))
        split_num = int(len(self.shards) * 0.15)
        assert split in {'train', 'val'}
        if split == 'train':
            self.shards = self.shards[:-split_num]
        elif split == 'val':
            self.shards = self.shards[-split_num:]
        self.reset()

    def reset(self):
        self.file = self.shards[0]
        self.load_tokens()

    def load_tokens(self):
        path = os.path.join(TOKEN_DIR, str(self.file))
        print(f"Loading tokens from ./{path} ... ", end='', flush=True)
        with open(path, 'rb') as f:
            self.tokens = array.array('H')
            self.tokens.frombytes(f.read())
            self.tokens = torch.tensor(self.tokens)
        self.file += 1
        if self.file not in self.shards:
            self.file = self.shards[0]
        self.position = 0
        print(f"DONE ({len(self.tokens):,} tokens)")
    
    def __next__(self):
        B, T = self.B, self.T
        buf = self.tokens[self.position:self.position + B*T + 1]
        x = buf[:-1].view(B, T)
        y = buf[1:].view(B, T)
        self.position += B*T
        if self.position + B*T + 1 >= len(self.tokens):
            self.load_tokens()
        return x, y


if __name__ == '__main__':
    device = 'cpu'
    torch.manual_seed(42)
    if torch.cuda.is_available():
        device = 'cuda'
        torch.cuda.manual_seed(42)
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = 'mps'
    # print(f"using device: {device}")

    if len(sys.argv) == 2 and sys.argv[1] == 'train':
        total_batch_size = 524288
        B = 4
        T = 256
        assert total_batch_size % (B * T) == 0
        grad_acc_steps = total_batch_size // (B * T)
        print(f"total desired batch size: {total_batch_size}, gradient accumulation steps: {grad_acc_steps}")

        path = os.path.join(TOKEN_DIR, "4")
        with open(path, 'rb') as f:
            tokens = array.array('H')
            tokens.frombytes(f.read())

        train_loader = DataLoader(B=B, T=T, split='train')
        val_loader = DataLoader(B=B, T=T, split='val')

        wikibot = Wikibot(WikibotConfig())
        wikibot.to(device)
        # wikibot = torch.compile(wikibot)
        
        max_lr = 6e-4
        min_lr = max_lr * 0.1
        warmup_steps = int(375e6) // total_batch_size
        max_steps = int(3e9) // total_batch_size

        def get_lr(it):
            if it < warmup_steps:
                return max_lr * (it+1) / warmup_steps
            if it > max_steps:
                return min_lr
            decay_ratio = (it - warmup_steps) / (max_steps - warmup_steps)
            assert 0 <= decay_ratio <= 1
            coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
            return min_lr + coeff * (max_lr - min_lr)

        optimizer = wikibot.configure_optimizers(weight_decay=0.1, learning_rate=get_lr(0), device=device)
        for step in range(max_steps):
            t0 = time.time()

            if step % 100 == 0:
                wikibot.eval()
                val_loader.reset()
                with torch.no_grad():
                    val_loss_accum = 0.0
                    val_loss_steps = 20
                    for _ in range(val_loss_steps):
                        x, y = next(val_loader)
                        x, y = x.to(device), y.to(device)
                        # with torch.autocast(device_type=device, dtype=torch.bfloat16):
                        logits, loss = wikibot(x, y)
                        loss = loss / val_loss_steps
                        val_loss_accum += loss.detach()

                print(f"validation loss: {val_loss_accum.item():.4f}")

            optimizer.zero_grad()
            loss_accum = 0
            for _ in range(grad_acc_steps):
                x, y = next(train_loader)
                x, y = x.to(device), y.to(device)
                # with torch.autocast(device_type=device, dtype=torch.bfloat16):
                logits, loss = wikibot(x, y)
                loss = loss / grad_acc_steps
                loss_accum += loss.detach()
                loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(wikibot.parameters(), 1.0)
            lr = get_lr(step)
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
            optimizer.step()
            torch.cuda.synchronize()
            t1 = time.time()
            dt = (t1 - t0)*1000
            tokens_processed = train_loader.B * train_loader.T * grad_acc_steps
            tokens_per_second = tokens_processed / dt
            print(f"step {step:4d} | loss: {loss_accum:.6f} | lr: {lr:.4e} | norm: {norm:.4f} dt: {dt:.2f}ms | tok/sec: {tokens_per_second}")
            print("Saving wikibot at", TEMP_MODEL_PATH)
            torch.save(wikibot.state_dict(), TEMP_MODEL_PATH)
        print("Saving wikibot at", MODEL_PATH)
        torch.save(wikibot.state_dict(), MODEL_PATH)
    elif len(sys.argv) == 2 and sys.argv[1] == 'tokenize':
        tokenizer = tiktoken.get_encoding(TOKEN_ENCODING)
        if not os.path.exists(TOKENIZED_DIR):
            os.makedirs(TOKENIZED_DIR)
        with open(DATASET_PATH, encoding="utf-8") as dataset_file:
            i = 0
            print(f"Tokenizing dataset in chunks of {TOKENIZED_CHUNK_SIZE:,} bytes")
            while True:
                print(f"Tokenizing chunk #{i}")
                eot = tokenizer._special_tokens['<|endoftext|>']
                text = dataset_file.read(TOKENIZED_CHUNK_SIZE)
                if not text:
                    break
                data = [eot]
                if i == 0:
                    with open(WIKIBOT_INFO_PATH, encoding="utf-8") as wikibot_file:
                        data.extend(tokenizer.encode(wikibot_file.read() + "\n\n"))
                data.append(eot)
                data.extend(tokenizer.encode(text))
                tokenized_path = os.path.join(TOKENIZED_DIR, str(i))
                with open(tokenized_path, 'wb') as file:
                    print(f"Saving ./{tokenized_path}")
                    array.array('H', data).tofile(file)

                del data, text
                gc.collect()

                i += 1
    else:
        tokenizer = tiktoken.get_encoding(TOKEN_ENCODING)
        wikibot = Wikibot(WikibotConfig())
        if len(sys.argv) > 1:
            wikibot.load_state_dict(torch.load(sys.argv[1], weights_only=True))
        else:
            wikibot.load_state_dict(torch.load(MODEL_PATH, weights_only=True))
        wikibot.to(device)
        print(wikibot.respond("Hello, wikibot. I'm a human."))

        while True:
            print(wikibot.respond(input("> ")))

