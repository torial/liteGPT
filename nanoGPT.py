import math
import torch
import torch.nn as nn
from torch.nn import functional as F
from time import time
#import bitsandbytes as bnb
from hyperparameters import HyperParameters, hp_nano, hp_gpt3_small, hp_gpt3_medium, hp_gpt3_large, hp_gpt3_xl, \
    hp_gpt3_27B, hp_gpt3_67B, hp_gpt3_130B, hp_gpt3_1750B



print(torch.version.__version__)
torch.backends.cudnn.benchmark = True # on local tests, gives about 10% perf boost.

torch.manual_seed(1337)

with open("tears_of_tuon.txt", "r", encoding='utf-8') as f:
    text = f.read()

chars = sorted(list(set(text)))
vocab_size = len(chars)
print(f"Total Text Letter Count: {len(text)}")
print(''.join(chars))
print(f" Vocab Size: {vocab_size}")
print(f" Default Loss: {-math.log(1/vocab_size)}")

stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}
encode = lambda s: [stoi[c] for c in s]
decode = lambda s: "".join([itos[c] for c in s])


data = torch.tensor(encode(text), dtype=torch.long)
print(data.shape, data.dtype)
print(data[:1000])

n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]


def get_batch(split, hp: HyperParameters):
    # generate a small batch of data of inputs x and targets y
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - hp.block_size, (hp.batch_size,))
    x = torch.stack([data[i:i + hp.block_size] for i in ix])
    y = torch.stack([data[i + 1:i + hp.block_size + 1] for i in ix])
    return x.to(hp.device), y.to(hp.device)


def get_batch_iter(split, hp: HyperParameters):
    min_batch = 4
    while hp.batch_size > min_batch:
        try:
            return get_batch(split, hp)
        except torch.cuda.OutOfMemoryError:
            print(f"Got OOM for batch: {hp.batch_size}.  Cutting it in half")
            hp.batch_size = hp.batch_size // 2
    return get_batch(split, hp)

def checkpoint(model, filename):
    torch.save(model.state_dict(), filename)


def resume(model, filename):
    model.load_state_dict(torch.load(filename))


@torch.no_grad()
def estimate_loss(hp: HyperParameters):
    out = {}
    m.eval() # evaluation mode
    for split in ['train', 'val']:
        losses = torch.zeros(hp.eval_iters)
        for k in range(hp.eval_iters):
            X, Y = get_batch(split, hp)
            logits, loss = m(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    m.train() # training mode
    return out



class Head(nn.Module):
    """ one head of self-attention """

    def __init__(self, hp:HyperParameters):
        super().__init__()
        self.key = nn.Linear(hp.n_embed, hp.head_size, bias=False)
        self.query = nn.Linear(hp.n_embed, hp.head_size, bias=False)
        self.value = nn.Linear(hp.n_embed, hp.head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(hp.block_size, hp.block_size)))

        self.dropout = nn.Dropout(hp.dropout)

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x) # (B, T, C)
        q = self.query(x) # (B, T, C)
        # compute attention scores ("affininities")
        wei = q @ k.transpose(-2, -1) * C ** -0.5 # (B, T, C) @ (B, C, T) --> (B, T, T)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf')) # (B, T, T)
        wei = F.softmax(wei, dim=-1) # (B, T, T)
        wei = self.dropout(wei)
        # perform the weighted aggregation of the values
        v = self.value(x) # (B, T, C)
        out = wei @ v # (B, T, C) @ (B, T, C) -> (B, T, C)
        return out


class MultiHeadAttention(nn.Module):
    """ multiple heads of self-attention in parallel"""

    def __init__(self, hp:HyperParameters):
        super().__init__()
        self.heads = nn.ModuleList([Head(hp) for _ in range(hp.n_head)])
        self.proj = nn.Linear(hp.n_embed, hp.n_embed) # projection
        self.dropout = nn.Dropout(hp.dropout)
        self.hp = hp

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1) # concat over the channel dimension
        #print(f"{out.shape} vs ({self.hp.n_embed, self.hp.n_embed})")
        out = self.proj(out)
        out = self.dropout(out)
        return out

class FeedForward(nn.Module):
    """a simple linear layer followed by a non-linearity"""

    def __init__(self, hp:HyperParameters):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(hp.n_embed, 4 * hp.n_embed),
                                 nn.ReLU(),
                                 nn.Linear(4* hp.n_embed, hp.n_embed), # projection layer
                                 nn.Dropout(hp.dropout)
                                 )

    def forward(self, x):
        return self.net(x)


class Block(nn.Module):
    """ Transformer block: communication followed by computation """

    def __init__(self, hp:HyperParameters):
        # n_embd: embedding dimension, n_head: the number of heads we'd like
        super().__init__()
        self.sa = MultiHeadAttention(hp)
        self.ffwd = FeedForward(hp)
        self.ln1 = nn.LayerNorm(hp.n_embed)
        self.ln2 = nn.LayerNorm(hp.n_embed)

    def forward(self, x):
        x_proj = self.ln1(x)
        x = x + self.sa(x_proj)
        x_proj = self.ln2(x)
        x = x + self.ffwd(x_proj)
        return x


class UnigramLanguageModel(nn.Module):

    def __init__(self, hp: HyperParameters):
        super().__init__()
        # each token directly reads off the logits for the next token from a lookup table
        self.token_embedding_table = nn.Embedding(vocab_size, hp.n_embed)
        self.position_embedding_table = nn.Embedding(hp.block_size, hp.n_embed)
        self.blocks = nn.Sequential(*[Block(hp) for _ in range(hp.n_layer)])
        self.ln_final = nn.LayerNorm(hp.n_embed)
        self.lm_head = nn.Linear(hp.n_embed, vocab_size) # language model head
        self.hp = hp

    def forward(self, idx, targets=None):
        with torch.cuda.amp.autocast():
            B, T = idx.shape

            # idx and targets are both (B,T) tensors of integers
            token_emb = self.token_embedding_table(idx)  # (B, T, C) [Batch / Time / Channel ]  channel
            pos_emb = self.position_embedding_table(torch.arange(T, device=self.hp.device)) # (T, C)
            x = token_emb + pos_emb # (B, T, C)
            x = self.blocks(x) # (B, T, C)
            x = self.ln_final(x)
            logits = self.lm_head(x) # (B, T, vocab_size)

            if targets is None:
                loss = None
            else:
                B, T, C = logits.shape
                logits = logits.view(B*T, C)
                targets = targets.view(B*T) # or -1 which allows PyTorch to figure it out
                loss = F.cross_entropy(logits, targets) # negative log likelihood

            return logits, loss

    def generate(self, idx, max_new_tokens):
        # idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            # crop idx to the last block_size tokens
            idx_cond = idx[:, -self.hp.block_size:]
            # get the predictions
            logits, loss = self(idx_cond)
            #focus only on the last time step
            logits = logits[:, -1, :] # becomes (B, C)
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1) # (B, C)
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
            idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)

        return idx


def get_hyper_parameters_gpt3():
    return [hp_gpt3_small, hp_gpt3_medium, hp_gpt3_large, hp_gpt3_xl, hp_gpt3_27B, hp_gpt3_67B, hp_gpt3_130B, hp_gpt3_1750B]


def get_hyper_parameters_search_phase1():
    """
MODEL                   n_params    n_layers    n_embed     n_heads     head_size       batch_size      learning_rate
GPT-3 Small             125M        12          768         12          64              0.5M            6.0e-4
GPT-3 Medium            350M        24          1024        16          64              0.5M            3.0e-4
GPT-3 Large             760M        24          1536        16          96              0.5M            2.5e-4
GPT-3 XL                1.3B        24          2048        24          128             1M              2.0e-4
GPT-3 2.7B              2.7B        32          2560        32          80              1M              1.6e-4
GPT-3 6.7B              6.7B        32          4096        32          128             2M              1.2e-4
GPT-3 13B               13.0B       40          5140        40          128             2M              1.0e-4
GPT-3 175B              175.0B      96          12288       96          128             3.2M            0.6e-4

    """
    for lr in [6e-4, 3e-4, 2.5e-4, 2e-4, 1.6e-4, 1e-4, 0.6e-4]:
        for n_embed in [64, 128, 192, 256, 384, 512, 768]:  # must be multiples of 64
            for batch_size in [32, 48, 64]:
                for block_size in [32, 64, 128]:
                    for n_layer in [2, 4, 6]:
                        if n_layer == 6 and block_size == 128 and n_embed == 768 and batch_size == 64:
                            continue # this maximal combination causes the OOM w/ GPUT at 8GB
                        n_heads = n_embed // 64 # IMPORTANT: This must be a fixed ratio or matmul errors will occur
                        hp_nano.learning_rate = lr
                        hp_nano.n_head = n_heads
                        hp_nano.n_embed = n_embed
                        hp_nano.block_size = block_size
                        hp_nano.batch_size = batch_size
                        hp_nano.n_layer = n_layer
                        hp_nano.eval_interval = 100  # this is for quick testing
                        hp_nano.max_iters = 500  # for quick testing... exploring the idea of hyper-parameter tuning that quick fails... then could
                        # continue w/ the training if the combination made the cut ?

                        hp_nano.name = f"Baseline LR({lr}) Heads({n_heads}) Embeddings({n_embed}) Block Size({block_size}) Batch Size({batch_size}) Layers({n_layer})"
                        yield hp_nano


def get_hyper_parameters_search_phase2():
    """
MODEL                   n_params    n_layers    n_embed     n_heads     head_size       batch_size      learning_rate
GPT-3 Small             125M        12          768         12          64              0.5M            6.0e-4
GPT-3 Medium            350M        24          1024        16          64              0.5M            3.0e-4
GPT-3 Large             760M        24          1536        16          96              0.5M            2.5e-4
GPT-3 XL                1.3B        24          2048        24          128             1M              2.0e-4
GPT-3 2.7B              2.7B        32          2560        32          80              1M              1.6e-4
GPT-3 6.7B              6.7B        32          4096        32          128             2M              1.2e-4
GPT-3 13B               13.0B       40          5140        40          128             2M              1.0e-4
GPT-3 175B              175.0B      96          12288       96          128             3.2M            0.6e-4

    """
    for lr in [6e-4, 3e-4, 2.5e-4, 1.6e-4]:
        for n_embed in [384, 512, 768]:  # must be multiples of 64
            for batch_size in [32, 48, 64]:
                for block_size in [32, 64, 128]:
                    for n_layer in [2, 4, 6]:
                        if n_layer == 6 and block_size == 128 and n_embed == 768 and batch_size == 64:
                            continue # this maximal combination causes the OOM w/ GPUT at 8GB
                        n_heads = n_embed // 64 # IMPORTANT: This must be a fixed ratio or matmul errors will occur
                        hp_nano.learning_rate = lr
                        hp_nano.n_head = n_heads
                        hp_nano.n_embed = n_embed
                        hp_nano.block_size = block_size
                        hp_nano.batch_size = batch_size
                        hp_nano.n_layer = n_layer
                        hp_nano.eval_interval = 100  # this is for quick testing
                        hp_nano.max_iters = 1500  # for quick testing... exploring the idea of hyper-parameter tuning that quick fails... then could
                        # continue w/ the training if the combination made the cut ?

                        hp_nano.name = f"Baseline LR({lr}) Heads({n_heads}) Embeddings({n_embed}) Block Size({block_size}) Batch Size({batch_size}) Layers({n_layer})"
                        yield hp_nano


def get_hyper_parameters_search_phase3():
    """
MODEL                   n_params    n_layers    n_embed     n_heads     head_size       batch_size      learning_rate
GPT-3 Small             125M        12          768         12          64              0.5M            6.0e-4
GPT-3 Medium            350M        24          1024        16          64              0.5M            3.0e-4
GPT-3 Large             760M        24          1536        16          96              0.5M            2.5e-4
GPT-3 XL                1.3B        24          2048        24          128             1M              2.0e-4
GPT-3 2.7B              2.7B        32          2560        32          80              1M              1.6e-4
GPT-3 6.7B              6.7B        32          4096        32          128             2M              1.2e-4
GPT-3 13B               13.0B       40          5140        40          128             2M              1.0e-4
GPT-3 175B              175.0B      96          12288       96          128             3.2M            0.6e-4

    """
    for lr in [6e-4, 3e-4, 2.5e-4]:
        for n_embed in [512, 768]:  # must be multiples of 64
            for batch_size in [48, 64]:
                for block_size in [64, 128]:
                    for n_layer in [4, 6]:
                        if n_layer == 6 and block_size == 128 and n_embed == 768 and batch_size == 64:
                            continue # this maximal combination causes the OOM w/ GPUT at 8GB
                        n_heads = n_embed // 64 # IMPORTANT: This must be a fixed ratio or matmul errors will occur
                        hp_nano.learning_rate = lr
                        hp_nano.n_head = n_heads
                        hp_nano.n_embed = n_embed
                        hp_nano.block_size = block_size
                        hp_nano.batch_size = batch_size
                        hp_nano.n_layer = n_layer
                        hp_nano.eval_interval = 100  # this is for quick testing
                        hp_nano.max_iters = 5000  # for quick testing... exploring the idea of hyper-parameter tuning that quick fails... then could
                        # continue w/ the training if the combination made the cut ?

                        hp_nano.name = f"Baseline LR({lr}) Heads({n_heads}) Embeddings({n_embed}) Block Size({block_size}) Batch Size({batch_size}) Layers({n_layer})"
                        yield hp_nano


def get_hyper_parameters_search_phase4():
    """
MODEL                   n_params    n_layers    n_embed     n_heads     head_size       batch_size      learning_rate
GPT-3 Small             125M        12          768         12          64              0.5M            6.0e-4
GPT-3 Medium            350M        24          1024        16          64              0.5M            3.0e-4
GPT-3 Large             760M        24          1536        16          96              0.5M            2.5e-4
GPT-3 XL                1.3B        24          2048        24          128             1M              2.0e-4
GPT-3 2.7B              2.7B        32          2560        32          80              1M              1.6e-4
GPT-3 6.7B              6.7B        32          4096        32          128             2M              1.2e-4
GPT-3 13B               13.0B       40          5140        40          128             2M              1.0e-4
GPT-3 175B              175.0B      96          12288       96          128             3.2M            0.6e-4

    """
    for lr, n_embed, batch_size, block_size, n_layer in ((0.0006, 768, 48, 128, 6),
                                                         (0.0006, 768, 64, 128, 4),
                                                         (0.0003, 768, 48, 128, 6),
                                                         (0.0003, 768, 64, 128, 4),
                                                         (0.0006, 512, 64, 128, 6),
                                                         (0.00025, 768, 48, 128, 6),
                                                         (0.0006, 768, 48, 128, 4),
                                                         (0.0006, 512, 48, 128, 6)):
        n_heads = n_embed // 64 # IMPORTANT: This must be a fixed ratio or matmul errors will occur
        hp_nano.learning_rate = lr
        hp_nano.n_head = n_heads
        hp_nano.n_embed = n_embed
        hp_nano.block_size = block_size
        hp_nano.batch_size = batch_size
        hp_nano.n_layer = n_layer
        hp_nano.eval_interval = 100  # this is for quick testing
        hp_nano.max_iters = 10000  # for quick testing... exploring the idea of hyper-parameter tuning that quick fails... then could
        # continue w/ the training if the combination made the cut ?

        hp_nano.name = f"Baseline LR({lr}) Heads({n_heads}) Embeddings({n_embed}) Block Size({block_size}) Batch Size({batch_size}) Layers({n_layer})"
        yield hp_nano


def print_total_parameters(m: nn.Module):
    total_params = sum(p.numel() for p in m.parameters())
    print(f"Total Parameters: {total_params:,}")

if __name__ == "__main__":
    from contextlib import redirect_stdout
    from tqdm import tqdm
    total_hyperparameters_phase1 = 7 * 7 * 3 * 3 * 3 - 7 # 1316 possible experiments, 500 iterations
    total_hyperparameters_phase2 = 4 * 3 * 3 * 3 * 3 - 4 # 320 possible experiments, 1500 iterations
    total_hyperparameters_phase3 = 3 * 2 * 2 * 2 * 2 - 3 # 45 possible experiments, 5000 iterations
    total_hyperparameters_phase4 = 8 # 10_000 iterations
    with open('phase4/output.txt', 'w') as f:
        #with redirect_stdout(f):
            for hp in tqdm(get_hyper_parameters_search_phase4(), total=total_hyperparameters_phase4):
            #for hp in get_hyper_parameters_search_phase1():
                print(f"BEGINNING ({time()}): {hp.name}")
                if hp.batch_size > 256:
                    hp.batch_size = 128
                    hp.block_size = 256
                    hp.device = "cpu" # try running on CPU to get around the OOM issues of GPU
                    hp.eval_interval = 10

                xb, yb = get_batch_iter('train', hp)

                # default los expected: = -ln (1/vocab_size)
                m = UnigramLanguageModel(hp)
                print_total_parameters(m)
                m = m.to(hp.device)
                #m = torch.compile(m) NOT supported in Windows

                logits, loss = m(xb, yb)
                print(logits.shape)
                print(loss)

                # THESE statements are great for showing untrained output but
                # when doing a hyper parameter search add a lot of noise.
                context = torch.zeros((1, 1), dtype=torch.long, device=hp.device)
                #print(decode(m.generate(context, max_new_tokens=100)[0].tolist()))

                #continue
                optimizer = torch.optim.AdamW(m.parameters(), lr=hp.learning_rate)
                #optimizer = bnb.optim.AdamW(m.parameters(), lr=hp.learning_rate)
                start_time = time()

                scaler = torch.cuda.amp.GradScaler()

                for iter in range(hp.max_iters):
                    if iter % hp.eval_interval == 0:
                        losses = estimate_loss(hp)
                        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f} [{time()-start_time} sec]")

                    xb, yb = get_batch('train', hp)

                    # evaluate the loss
                    logits, loss = m(xb, yb)
                    optimizer.zero_grad(set_to_none=True)
                    #loss.backward()
                    #optimizer.step()
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                    #checkpoint(m, f"epoch-{iter}.pth")

                print(loss.item())
                print(f"Total Training Time: {time()-start_time} seconds")

                print(decode(m.generate(context, max_new_tokens=100)[0].tolist()))
