from dataclasses import dataclass
import torch
import torch.nn as nn
from torch.nn import functional as F
import math
import inspect
import time
import os
import tiktoken
from hellaswag import render_example, iterate_examples

class CausalSelfAttention(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        self.c_proj.NANOGPT_SCALE_INIT = 1 # flag for this module to use residual init
        # regularization
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        # not really a bias, but more of a mask, but following openai/hf naming though
        self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
                                     .view(1, 1, config.block_size, config.block_size))

    def forward(self, x):
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)
        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        # nh is "number of heads", hs is "head size", and C (number of channels) = nh * hs
        # e.g. in GPT-2 (124M), n_head=12, hs=64, so nh*hs=C=768 channels in the Transformer
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        # attention (materializes the large (T,T) matrix for all the queries and keys)
        # Flash attention ---------------------------------------------------
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        # att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        # att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
        # att = F.softmax(att, dim=-1)
        # y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        # -------------------------------------------------------------------
        y = y.transpose(1,2).contiguous().view(B, T, C) # re-assemble all head outputs side by side
        # output projection
        y = self.c_proj(y)
        return y


class MLP(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.c_fc   = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.gelu   = nn.GELU(approximate='tanh')
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd)
        self.c_proj.NANOGPT_SCALE_INIT = 1 # flag for this module to use residual init

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
        x = x + self.attn(self.ln_1(x)) # communication
        x = x + self.mlp(self.ln_2(x)) # computation

        return x

@dataclass
class GPTConfig:
    block_size: int = 1024 # max sequence length
    vocab_size: int = 50257 # number of tokens: 50,000 BPE merges + 256 bytes tokens + 1 <|endoftext|> token
    n_layer: int = 12 # number of layers
    n_head: int = 12 # number of heads
    n_embd: int = 768 # embedding dimension

class GPT(nn.Module):

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

        # weight sharing scheme
        # 30% parameters being saved by sharing the embedding and the classifier weights
        self.transformer.wte.weight = self.lm_head.weight

        # init params
        self.apply(self._init_weights) # iterate over all modules and apply _init_weights

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            # typically, std is 1/sqrt(fan_in) for linear layers but in gpt-2 they use 0.02
            # as 1/(768^0.5) = 0.036 for 124M model so 0.02 is in the vicinity
            std = 0.02
            if hasattr(module, 'NANOGPT_SCALE_INIT'):
                # Using the scaling factor of 1/sqrt(number of layers(n)) = n**-0.5 for residual init
                # 2 times-Every layer has 2 blocks that add to residual pathway(attention and MLP)
                std *= (2 * self.config.n_layer) ** -0.5 # residual init
            torch.nn.init.normal_(module.weight, mean=0.0, std=std) 
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias) # by default bias is initialized with uniform
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        # idx is of shape (B, T)
        B, T = idx.size()
        assert T <= self.config.block_size, f"Cannot forward sequence of length {T}, block size is only {self.config.block_size}"
        # forward the token and posisition embeddings
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device) # shape (T)
        pos_emb = self.transformer.wpe(pos) # position embeddings of shape (T, n_embd)
        tok_emb = self.transformer.wte(idx) # token embeddings of shape (B, T, n_embd)
        x = tok_emb + pos_emb
        # forward the blocks of the transformer
        for block in self.transformer.h:
            x = block(x)
        # forward the final layernorm and the classifier
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x) # (B, T, vocab_size)
        loss = None
        if targets is not None:
            # compute the loss
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
        return logits, loss

    @classmethod
    def from_pretrained(cls, model_type):
        """Loads pretrained GPT-2 model weights from huggingface"""
        assert model_type in {'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'}
        from transformers import GPT2LMHeadModel
        print("loading weights from pretrained gpt: %s" % model_type)

        # n_layer, n_head and n_embd are determined from model_type
        config_args = {
            'gpt2':         dict(n_layer=12, n_head=12, n_embd=768),  # 124M params
            'gpt2-medium':  dict(n_layer=24, n_head=16, n_embd=1024), # 350M params
            'gpt2-large':   dict(n_layer=36, n_head=20, n_embd=1280), # 774M params
            'gpt2-xl':      dict(n_layer=48, n_head=25, n_embd=1600), # 1558M params
        }[model_type]
        config_args['vocab_size'] = 50257 # always 50257 for GPT model checkpoints
        config_args['block_size'] = 1024 # always 1024 for GPT model checkpoints
        # create a from-scratch initialized minGPT model
        config = GPTConfig(**config_args)
        model = GPT(config)
        sd = model.state_dict()
        sd_keys = sd.keys()
        sd_keys = [k for k in sd_keys if not k.endswith('.attn.bias')] # discard this mask / buffer, not a param

        # init a huggingface/transformers model
        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()

        # copy while ensuring all of the parameters are aligned and match in names and shapes
        sd_keys_hf = sd_hf.keys()
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.masked_bias')] # ignore these, just a buffer
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.bias')] # same, just the mask (buffer)
        transposed = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']
        # basically the openai checkpoints use a "Conv1D" module, but we only want to use a vanilla Linear
        # this means that we have to transpose these weights when we import them
        assert len(sd_keys_hf) == len(sd_keys), f"mismatched keys: {len(sd_keys_hf)} != {len(sd_keys)}"
        for k in sd_keys_hf:
            if any(k.endswith(w) for w in transposed):
                # special treatment for the Conv1D weights we need to transpose
                assert sd_hf[k].shape[::-1] == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k].t())
            else:
                # vanilla copy over the other parameters
                assert sd_hf[k].shape == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])

        return model

    def configure_optimizers(self, weight_decay, learning_rate, device):
        # start with all of the candidate parameters (that require grad)
        param_dict = {pn: p for pn, p in self.named_parameters()}
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
        # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        if master_process:
            print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
            print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
        # Create AdamW optimizer and use the fused version if it is available
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and "cuda" in device
        if master_process:
            print(f"using fused AdamW: {use_fused}")
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=(0.9, 0.95), eps=1e-8, fused=use_fused)
        return optimizer

# -----------------------------------------------------------------------------------------------
# get a single data batch
# import tiktoken
import numpy as np

def load_tokens(filename):
    npt = np.load(filename) # load the uint16 numpy file
    ptt = torch.tensor(npt, dtype=torch.long) # convert to torch tensor of type long (int64)
    return ptt

class DataLoaderLite:

    def __init__(self, B, T, process_rank, num_processes, split):
        self.B = B
        self.T = T
        self.process_rank = process_rank
        self.num_processes = num_processes

        # # at init load tokens from disk and store them in memory
        # with open('input.txt', 'r') as f:
        #     text = f.read() # read the entire file into a string
        # enc = tiktoken.get_encoding("gpt2")
        # tokens = enc.encode(text)
        # self.tokens = torch.tensor(tokens) # tokenizing the file
        # print(f"loaded {len(self.tokens)} tokens from input.txt")
        # print(f"1 epoch = {len(self.tokens) // (B*T)} batches") # unique batches per epoch
        assert split in {'train', 'val'}

        # get the shard filenames
        data_root = "edu_fineweb10B"
        shards = os.listdir(data_root)
        shards = [s for s in shards if split in s]
        shards = sorted(shards)
        shards = [os.path.join(data_root, s) for s in shards]
        self.shards = shards
        assert len(shards) > 0, f"no shards found for split {split}"
        if master_process:
            # print(f"loaded {len(self.tokens)} tokens")
            print(f"found {len(shards)} shards for split {split}")
        self.reset()
        # # state, init at shard zero
        # self.current_shard = 0
        # self.tokens = load_tokens(self.shards[self.current_shard])
        # self.current_position = self.B * self.T * self.process_rank # each process starts at a different position in the tensor

    def reset(self):
        # state, init at shard zero
        self.current_shard = 0
        self.tokens = load_tokens(self.shards[self.current_shard])
        self.current_position = self.B * self.T * self.process_rank # each process starts at a different position in the tensor

    def next_batch(self):
        B, T = self.B, self.T
        buf = self.tokens[self.current_position : self.current_position + B*T + 1] # +1 because we need ground truth for the last token
        x = (buf[:-1].view(B, T)) # (B, T) inputs
        y = (buf[1:].view(B, T)) # (B, T) targets
        # advance the position in the tensor
        self.current_position += B*T*self.num_processes # each process will advance by B*T*num_processes tokens
        # if loading the next batch is out of bounds, advance to next shard
        if self.current_position + (B*T*self.num_processes + 1) >= len(self.tokens):
            # self.current_position =  self.B * self.T * self.process_rank
            self.current_shard = (self.current_shard + 1) % len(self.shards)
            self.tokens = load_tokens(self.shards[self.current_shard])
            self.current_position = B * T * self.process_rank
        return x, y

# -----------------------------------------------------------------------------
# helper function for HellaSwag eval
# takes tokens, mask, and logits, returns the index of the completion with the lowest loss

def get_most_likely_row(tokens, mask, logits):
    # evaluate the autoregressive loss at all positions
    shift_logits = (logits[..., :-1, :]).contiguous()
    shift_tokens = (tokens[..., 1:]).contiguous()
    flat_shift_logits = shift_logits.view(-1, shift_logits.size(-1))
    flat_shift_tokens = shift_tokens.view(-1)
    shift_losses = F.cross_entropy(flat_shift_logits, flat_shift_tokens, reduction='none')
    shift_losses = shift_losses.view(tokens.size(0), -1)
    # now get the average loss just for the completion region (where mask == 1), in each row
    shift_mask = (mask[..., 1:]).contiguous() # we must shift mask, so we start at the last prompt token
    masked_shift_losses = shift_losses * shift_mask
    # sum and divide by the number of 1s in the mask
    sum_loss = masked_shift_losses.sum(dim=1)
    avg_loss = sum_loss / shift_mask.sum(dim=1)
    # now we have a loss for each of the 4 completions
    # the one with the lowest loss should be the most likely
    pred_norm = avg_loss.argmin().item()
    return pred_norm
    
# --------------------------------------------------------------------------------------------------

# simple launch:
# python train_gpt2.py
# DDP launch for e.g. 8 GPUs:
# torchrun --standalone --nproc_per_node=8 train_gpt2.py

# run the training loop
from torch.distributed import init_process_group, destroy_process_group
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist

# set up a DDP(distributed data parallel)
# torchrun command sets the env variables RANK, LOCAL_RANK, WORLD_SIZE
ddp = int(os.environ.get('RANK', -1)) != -1 # Check if ddp is running
if ddp:
    # use of ddp atm demands CUDA, we set the device appropriately according to RANK
    assert torch.cuda.is_available(), "for now I think we need CUDA for DDP"
    init_process_group(backend='nccl') # initialize the distributed backend
    ddp_rank = int(os.environ['RANK']) # each process(same code, different data) has a unique rank, 0 to WORLD_SIZE-1
    ddp_local_rank = int(os.environ['LOCAL_RANK']) # used in multi-node setting, we only have 1 node
    ddp_world_size = int(os.environ['WORLD_SIZE']) # total number of processes(GPUs)
    device = f"cuda:{ddp_local_rank}" # set the device based on local rank
    torch.cuda.set_device(device) # set the device for this process
    master_process = ddp_rank == 0 # this process will do logging, checkpointing etc.
else:
    # vanilla, non-ddp run
    ddp_rank = 0
    ddp_local_rank = 0
    ddp_world_size = 1
    master_process = True
    # attempt to autodetect device
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda:0" # Added :0 for 1660 ti
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = "mps"
    print(f"using device: {device}")

# ---------------------------------------------------------------------------------------------------
# For reproducibility
torch.manual_seed(1337) # set the random seed for reproducibility
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(1337) # set the random seed for all GPUs

# ----------------------------------------------------------------------------------------------------
# Gradient Accumulation
total_batch_size = 524288 # 2^19 , approx 0.5M , in number of tokens
B = 1 # micro-batch size (cant increase much due to 1660ti 6GB VRAM), in number of sequences
T = 1024 # sequence length
assert total_batch_size % (B*T*ddp_world_size) == 0,"total_batch_size must be divisible by B * T * ddp_world_size"
grad_accum_steps = total_batch_size // (B * T * ddp_world_size) # number of micro-batches to accumulate gradients over
if master_process:
    print(f"total desired batch size: {total_batch_size}")
    print(f"==> calculated gradient accumulation steps: {grad_accum_steps}")

# Config as per 1660ti, use 2^x for values of B otherwise it runs inefficiently
# train_loader = DataLoaderLite(B=1, T=1024) # 1 example per batch, sequence length 1024
train_loader = DataLoaderLite(B=B, T=T, process_rank=ddp_rank, num_processes=ddp_world_size, split="train") # 1 example per batch, sequence length 1024
val_loader = DataLoaderLite(B=B, T=T, process_rank=ddp_rank, num_processes=ddp_world_size, split="val")

# enable TF32, works only with TF32 supported GPU architectures
torch.set_float32_matmul_precision('high')

# --------------------------------------------------------------------------------------------------------
# create a model
# model = GPT.from_pretrained('gpt2') # initialize from gpt2 model weights from huggingface
model = GPT(GPTConfig(vocab_size=50304)) # initialize a new model with the same config as GPT-2 (random model)
model.to(device)
# model = torch.compile(model) # compile the model for faster training
use_compile = False # torch.compile interferes with HellaSwag eval and Generation. TODO fix
if use_compile:
    model = torch.compile(model)
if ddp:
    # wrap the model with DDP container
    model = DDP(model, device_ids=[ddp_local_rank])
raw_model = model.module if ddp else model # unwrap the DDP container to get the raw model

# ---------------------------------------------------------------------------------------------------
# Learning rate scheduler - Cosine decay learning schedule with warmup

# Exact hyperparamters of GPT-3 
max_lr = 6e-4 # maximum learning rate
min_lr = max_lr * 0.1 # minimum learning rate (10% of max_lr)
warmup_steps = 715  # GPT-3 paper uses 375 million tokens so 375e6 / 2^^19 = 715 approx steps
max_steps = 19073 # 10e9(unique tokens) / 2^^19(total batch size) = 19073 approx steps

def get_lr(it):
    # 1. Linear warmup for warmup_iter steps
    if it < warmup_steps:
        return max_lr * (it + 1) / warmup_steps
    # 2. If it > lr_decay_iters, return min learning rate
    if it > max_steps:
        return min_lr
    # 3. In between, use cosine decay down to min learning rate
    decay_ratio = (it - warmup_steps) / (max_steps - warmup_steps)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # coeff starts at 1 and ends at 0
    return min_lr + coeff * (max_lr - min_lr) # scale the learning rate

# ---------------------------------------------------------------------------------------------------
# optimization

enc = tiktoken.get_encoding("gpt2")
# alternative to SGD
# optimizer = torch.optim.AdamW(model.parameters(), lr=6e-4, betas=(0.9, 0.95), eps=1e-8)
# optimizer = model.configure_optimizers(weight_decay=0.1, learning_rate=6e-4, device=device)
optimizer = raw_model.configure_optimizers(weight_decay=0.1, learning_rate=6e-4, device=device)

# create the log directory we will write checkpoints to and log to
log_dir = "log"
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, f"log.txt")
with open(log_file, "w") as f: # open for writing to clear the file
    pass

for step in range(max_steps):
    t0 = time.time()
    last_step = (step == max_steps - 1)

    # once in a while evaluate our validation loss
    # if step % 100 == 0:
    if step % 250 == 0 or last_step:
        model.eval()
        val_loader.reset()
        with torch.no_grad():
            val_loss_accum = 0.0
            val_loss_steps = 20
            for _ in range(val_loss_steps):
                x, y = val_loader.next_batch()
                x, y = x.to(device), y.to(device)
                with torch.autocast(device_type=device, dtype=torch.bfloat16):
                    logits, loss = model(x, y)
                loss = loss / val_loss_steps
                val_loss_accum += loss.detach()
        if ddp:
            dist.all_reduce(val_loss_accum, op=dist.ReduceOp.AVG)
        if master_process:
            print(f"validation loss: {val_loss_accum.item():.4f}")
            with open(log_file, "a") as f:
                f.write(f"{step} val {val_loss_accum.item():.4f}\n")

    # once in a while evaluate hellaswag
    if (step % 250 == 0 or last_step) and (not use_compile):
        num_correct_norm = 0
        num_total = 0
        for i, example in enumerate(iterate_examples("val")):
            # only process examples where i % ddp_world_size == ddp_rank
            if i % ddp_world_size != ddp_rank:
                continue
            # render the example into tokens and labels
            _, tokens, mask, label = render_example(example)
            tokens = tokens.to(device)
            mask = mask.to(device)
            # get the logits
            with torch.no_grad():
                with torch.autocast(device_type=device, dtype=torch.bfloat16):
                    logits, loss = model(tokens)
                pred_norm = get_most_likely_row(tokens, mask, logits)
            num_total += 1
            num_correct_norm += int(pred_norm == label)
        # reduce the stats across all processes
        if ddp:
            num_total = torch.tensor(num_total, dtype=torch.long, device=device)
            num_correct_norm = torch.tensor(num_correct_norm, dtype=torch.long, device=device)
            dist.all_reduce(num_total, op=dist.ReduceOp.SUM)
            dist.all_reduce(num_correct_norm, op=dist.ReduceOp.SUM)
            num_total = num_total.item()
            num_correct_norm = num_correct_norm.item()
        acc_norm = num_correct_norm / num_total
        if master_process:
            print(f"HellaSwag accuracy: {num_correct_norm}/{num_total}={acc_norm:.4f}")
            with open(log_file, "a") as f:
                f.write(f"{step} hella {acc_norm:.4f}\n")

    # once in a while generate from the model (except step 0, which is noise)
    # disabled because torch.compile throws a scary error i can't solve rn
    # if you disable torch.compile, this code works fine
    # if step > 0 and step % 100 == 0: # disabled torch.compile
    if ((step > 0 and step % 250 == 0) or last_step) and (not use_compile):
        model.eval()
        num_return_sequences = 4
        max_length = 32
        tokens = enc.encode("Hello, I'm a language model,")
        tokens = torch.tensor(tokens, dtype=torch.long)
        tokens = tokens.unsqueeze(0).repeat(num_return_sequences, 1)
        xgen = tokens.to(device)
        sample_rng = torch.Generator(device=device)
        sample_rng.manual_seed(42 + ddp_rank)
        while xgen.size(1) < max_length:
            # forward the model to get the logits
            with torch.no_grad():
                # logits, loss = model(xgen) # (B, T, vocab_size)
                with torch.autocast(device_type=device, dtype=torch.bfloat16):
                    logits, loss = model(xgen) # (B, T, vocab_size)
                # take the logits at the last position
                logits = logits[:, -1, :] # (B, vocab_size)
                # get the probabilities
                probs = F.softmax(logits, dim=-1)
                # do top-k sampling of 50 (huggingface pipeline default)
                # topk_probs here becomes (5, 50), topk_indices is (5, 50)
                topk_probs, topk_indices = torch.topk(probs, 50, dim=-1)
                # select a token from the top-k probabilities
                # note: multinomial does not demand the input to sum to 1
                ix = torch.multinomial(topk_probs, 1, generator=sample_rng) # (B, 1)
                # gather the corresponding indices
                xcol = torch.gather(topk_indices, -1, ix) # (B, 1)
                # append to the sequence
                xgen = torch.cat((xgen, xcol), dim=1)
        # print the generated text
        for i in range(num_return_sequences):
            tokens = xgen[i, :max_length].tolist()
            decoded = enc.decode(tokens)
            print(f"rank {ddp_rank} sample {i}: {decoded}")

    # training loop / do one step of the optimization
    model.train()
    optimizer.zero_grad() # start with 0 gradients
    loss_accum = 0.0
    for micro_step in range(grad_accum_steps):
        x, y = train_loader.next_batch()
        x, y = x.to(device), y.to(device)
        with torch.autocast(device_type=device, dtype=torch.bfloat16): # enable bf16 for faster training
            logits, loss = model(x, y)
        # we have to scale the loss to account for gradient accumulation
        # because gradients just add on each successive backward()
        # addition of gradients correspond to a SUM in the objective, but
        # instead of a SUM we want MEAN. Scale the loss here so it comes out of 
        loss = loss / grad_accum_steps # scale the loss to account for gradient accumulation
        loss_accum += loss.detach()
        if ddp:
            # only sync gradients on the last micro-step
            model.require_backward_grad_sync = (micro_step == grad_accum_steps - 1)
        loss.backward() # adds to gradient for each parameter based on the loss
    if ddp:
        dist.all_reduce(loss_accum, op=dist.ReduceOp.AVG) 
    # clip the gradient to prevent exploding gradients
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    # determine and set the learning rate for this iteration
    lr = get_lr(step)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    optimizer.step() # update the parameters and decrease the loss
    torch.cuda.synchronize() # wait for the GPU to finish before measuring time
    t1 = time.time()
    dt = (t1 - t0)
    tokens_processed = train_loader.B * train_loader.T * grad_accum_steps * ddp_world_size
    tokens_per_sec = tokens_processed / dt 
    dt *= 1000 # Convert to milliseconds
    if master_process:
        print(f"step {step:5d} | loss {loss_accum.item():.6f} | lr {lr:.4e} | norm {norm:.4f}, time {dt:.2f} ms, tokens/sec {tokens_per_sec:.2f}")
        with open(log_file, "a") as f:
            f.write(f"{step} train {loss_accum.item():.6f}\n")

if ddp:
    destroy_process_group() # clean up the distributed backend