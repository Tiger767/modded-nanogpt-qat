import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
import math
import json
import argparse
from dataclasses import dataclass
import tiktoken  # Make sure to install: pip install tiktoken


PARAMS = {}

def set_globals(config_path):
    print(f"Loading config from {config_path}")
    with open(config_path, 'r') as f:
        config = json.load(f)
    global ENABLE_MLP_QUANTIZATION, ENABLE_ATTN_QUANTIZATION, ENABLE_ATTN_PROJ_QUANTIZATION
    global ENABLE_MLP_ACT_QUANTIZATION, ENABLE_MLP_PRE_ACT_QUANTIZATION
    global ENABLE_EXTREME_MLP_ACT_QUANTIZATION, ENABLE_EXTREME_PRE_MLP_ACT_QUANTIZATION
    global ENABLE_ATTN_ACT_QUANTIZATION, ENABLE_ATTN_PRE_ACT_QUANTIZATION
    global ENABLE_EXTREME_ATTN_ACT_QUANTIZATION, ENABLE_EXTREME_PRE_ATTN_ACT_QUANTIZATION

    ENABLE_MLP_QUANTIZATION = config.get("ENABLE_MLP_QUANTIZATION", False)
    ENABLE_ATTN_QUANTIZATION = config.get("ENABLE_ATTN_QUANTIZATION", False)
    ENABLE_ATTN_PROJ_QUANTIZATION = config.get("ENABLE_ATTN_PROJ_QUANTIZATION", False)
    ENABLE_MLP_ACT_QUANTIZATION = config.get("ENABLE_MLP_ACT_QUANTIZATION", False)
    ENABLE_MLP_PRE_ACT_QUANTIZATION = config.get("ENABLE_MLP_PRE_ACT_QUANTIZATION", False)
    ENABLE_EXTREME_MLP_ACT_QUANTIZATION = config.get("ENABLE_EXTREME_MLP_ACT_QUANTIZATION", False)
    ENABLE_EXTREME_PRE_MLP_ACT_QUANTIZATION = config.get("ENABLE_EXTREME_PRE_MLP_ACT_QUANTIZATION", False)
    ENABLE_ATTN_ACT_QUANTIZATION = config.get("ENABLE_ATTN_ACT_QUANTIZATION", False)
    ENABLE_ATTN_PRE_ACT_QUANTIZATION = config.get("ENABLE_ATTN_PRE_ACT_QUANTIZATION", False)
    ENABLE_EXTREME_ATTN_ACT_QUANTIZATION = config.get("ENABLE_EXTREME_ATTN_ACT_QUANTIZATION", False)
    ENABLE_EXTREME_PRE_ATTN_ACT_QUANTIZATION = config.get("ENABLE_EXTREME_PRE_ATTN_ACT_QUANTIZATION", False)


# Initialize defaults to False (Normal)
set_globals("configs/normal.json")

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -----------------------------------------------------------------------------
# Simplified Quantization Functions (Inference-only)
# -----------------------------------------------------------------------------
# We replace all custom autograd.function calls with their
# standard PyTorch forward-pass equivalents for inference.

def round_grad_pt(x):
    """Replaces round_grad_pt.apply(x)"""
    return torch.round(x)

def clamp_pt(x, min_val=-1.0, max_val=1.0):
    """Replaces clamp_pt.apply(x, min_val, max_val)"""
    return torch.clamp(x, min_val, max_val)

def ternary_quantize_mean_special(x: torch.Tensor):
    min_max_value = x.abs().mean(dim=-1, keepdim=True).mean(dim=-2, keepdim=True).clamp(min=1e-5)
    quantized = clamp_pt(round_grad_pt(x / min_max_value))
    return min_max_value * quantized

def ternary_quantize(x: torch.Tensor):
    min_max_value = x.abs().mean().clamp(min=1e-5)
    quantized = clamp_pt(round_grad_pt(x / min_max_value))
    return min_max_value * quantized

def quantized_tanh(x):
    x = torch.nn.functional.tanh(x)
    return round_grad_pt(x)

def quantized_relu2(x: torch.Tensor, num_bits=8, eps=1e-5):
    x = torch.nn.functional.relu(x).square()
    xmax = x.max(-1, keepdim=True).values.clamp(min=eps)
    num_steps = (2**num_bits - 1)
    scaled_x = x / xmax * num_steps
    rounded_x = round_grad_pt(scaled_x)
    quantized_x = rounded_x / num_steps * xmax
    return quantized_x

def quantized_linear(x: torch.Tensor, num_bits=8, eps=1e-5):
    num_steps = (2**num_bits - 1) // 2
    pos = x >= 0
    xnmin = x.min(-1, keepdim=True).values.clamp(max=-eps)
    xpmax = x.max(-1, keepdim=True).values.clamp(min=eps)
    xp = round_grad_pt(x / xpmax * num_steps) / num_steps * xpmax
    xn = round_grad_pt(x / xnmin * num_steps) / num_steps * xnmin
    x = xp * pos + xn * (~pos)
    return x

# -----------------------------------------------------------------------------
# Model Modules (Copied from train_gpt.py, modified for inference)
# -----------------------------------------------------------------------------

class Quantize(nn.Module):
    def __init__(
        self,
        quantize_func,
        out_scalars=False,
        shape=None,
        device=None,
        dtype=None,
        *args,
        **kwargs
    ) -> None:
        super().__init__(*args, **kwargs)
        self.quantize_func = quantize_func
        self._post_quantized_mode = False
        self._max_value = 1.0
        self.out_scalars = None
        if out_scalars:
            self.out_scalars = nn.Parameter(
                torch.sign(
                    torch.randn((shape[0], 1, *shape[2:]), device=device, dtype=dtype)
                )
            )

    def forward(self, x):
        if self.quantize_func is None:
            return x
        # Note: We are NOT in _post_quantized_mode.
        # The training script applies quantization on the fly.
        if self.out_scalars is not None:
            return self.out_scalars * self.quantize_func(x)
        return self.quantize_func(x)

class QuantizedLinear(nn.Linear):
    def __init__(
        self, *args, weight_quantize_func=None, bias_quantize_func=None, **kwargs
    ) -> None:
        super().__init__(*args, **kwargs)
        self.quantize_weight = Quantize(weight_quantize_func)
        self.quantize_bias = Quantize(bias_quantize_func)

    def forward(self, input):
        qw = self.quantize_weight(self.weight)
        qb = self.quantize_bias(self.bias)
        return F.linear(input, qw, qb)

def norm(x: Tensor):
    return F.rms_norm(x, (x.size(-1),))

class CastedLinear(nn.Linear):
    def __init__(self, in_features: int, out_features: int, use_fp8=False, x_s=1.0, w_s=1.0, grad_s=1.0):
        super().__init__(in_features, out_features, bias=False)
        self.use_fp8 = use_fp8
        self.x_s = x_s
        self.w_s = w_s
        self.grad_s = grad_s

    def reset_parameters(self) -> None:
        with torch.no_grad():
            self.weight.zero_()

    def forward(self, x: Tensor):
        # For inference, self.training is False, so it always uses F.linear
        if self.use_fp8 and self.training:
            # This FP8 path is for training only and requires custom ops
            _x = x.flatten(0, -2)
            out: Tensor = torch.ops.nanogpt.mm(_x, self.weight, x_s=self.x_s, w_s=self.w_s, grad_s=self.grad_s)[0]
            return out.reshape(*x.shape[:-1], -1)
        else:
            # General inference path
            return F.linear(x, self.weight.type_as(x))

class Yarn(nn.Module):
    def __init__(self, head_dim, max_seq_len):
        super().__init__()
        self.head_dim = head_dim
        self.max_seq_len = max_seq_len
        self.block_size = 128 # Hardcoded from train_gpt.py args
        self.reset()

    def reset(self):
        angular_freq = (1 / 1024) ** torch.linspace(0, 1, steps=self.head_dim//4, dtype=torch.float32, device=device)
        angular_freq = torch.cat([angular_freq, angular_freq.new_zeros(self.head_dim//4)])
        t = torch.arange(self.max_seq_len, dtype=torch.float32, device=device)
        theta = torch.outer(t, angular_freq)
        # Buffers are not saved in state_dict because persistent=False
        self.cos = nn.Buffer(theta.cos().to(torch.bfloat16), persistent=False)
        self.sin = nn.Buffer(theta.sin().to(torch.bfloat16), persistent=False)
        self.angular_freq = angular_freq
        self.attn_scale = 0.1

    def apply(self, old_window: int, new_window: int, alpha: int=1, beta: int=32):
        rotations = self.block_size * old_window * self.angular_freq / (2 * torch.pi)
        scaling_factor = old_window / new_window
        interpolation_weight = torch.clamp((rotations - alpha) / (beta - alpha), 0, 1)
        self.angular_freq *= scaling_factor + interpolation_weight * (1 - scaling_factor)
        t = torch.arange(self.max_seq_len, dtype=torch.float32, device=self.angular_freq.device)
        theta = torch.outer(t, self.angular_freq)
        self.cos.copy_(theta.cos())
        self.sin.copy_(theta.sin())
        self.attn_scale *= 0.2 * math.log(new_window / old_window) + 1

def rotary(x_BTHD: Tensor, cos: Tensor, sin: Tensor):
    seq_len = x_BTHD.size(1)
    assert cos.size(0) >= seq_len
    cos, sin = (
        cos[None, :seq_len, None, :],
        sin[None, :seq_len, None, :],
    )
    x1, x2 = x_BTHD.chunk(2, dim=-1)
    y1 = x1 * cos + x2 * sin
    y2 = x1 * (-sin) + x2 * cos
    return torch.cat((y1, y2), 3)

@dataclass
class AttnArgs:
    ve: torch.Tensor
    sa_lambdas: torch.Tensor
    seqlens: torch.Tensor # Unused in this inference script, but kept for signature
    bm_size: int
    cos: torch.Tensor
    sin: torch.Tensor
    attn_scale: float

class CausalSelfAttention(nn.Module):
    def __init__(self, dim: int, head_dim: int, num_heads: int):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.dim = dim
        self.hdim = num_heads * head_dim

        assert self.hdim == self.dim, "num_heads * head_dim must equal model_dim"
        std = 0.5 * (self.dim ** -0.5)
        bound = (3 ** 0.5) * std
        self.qkvo_w = nn.Parameter(torch.empty(self.hdim, self.dim*4))
        self.quantize_qkvo_w = Quantize(ternary_quantize_mean_special) if ENABLE_ATTN_QUANTIZATION else lambda x: x
        self.quantize_qkvo_w2 = Quantize(ternary_quantize) if ENABLE_ATTN_PROJ_QUANTIZATION else lambda x: x
        self.qkvo_w.label='attn'

        with torch.no_grad():
            self.qkvo_w.view(4,self.hdim, self.dim)[:3].uniform_(-bound, bound)
            self.qkvo_w.view(4,self.hdim, self.dim)[3].zero_()

        self.attn_gate = CastedLinear(12, num_heads)
        self.attn_gate.weight.label = 'attn_gate'

    def forward(self, x: Tensor, attn_args: AttnArgs):
        B, T, _ = x.shape # batch size, sequence length
        cos, sin = attn_args.cos, attn_args.sin
        ve, sa_lambdas = attn_args.ve, attn_args.sa_lambdas
        attn_scale = attn_args.attn_scale
        # bm_size (windowing) is unused in the general-purpose attention
        # The original script uses it for flash_attn_varlen_func.
        # We replace it with standard attention, which is not windowed.

        if ENABLE_ATTN_PRE_ACT_QUANTIZATION:
            if ENABLE_EXTREME_PRE_ATTN_ACT_QUANTIZATION:
                x = quantized_tanh(x)
            else:
                x = quantized_linear(x, num_bits=8)

        q, k, v = F.linear(x, self.quantize_qkvo_w(self.qkvo_w.view(4, self.hdim, self.dim)[:3]).flatten(end_dim=1).type_as(x)).view(B, T, 3 * self.num_heads, self.head_dim).chunk(3, dim=-2)
        q, k = norm(q), norm(k)
        q, k = rotary(q, cos, sin), rotary(k, cos, sin)
        
        if ve is not None:
            v = sa_lambdas[0] * v + sa_lambdas[1] * ve.view_as(v)
        else:
            v = sa_lambdas[0] * v

        if ENABLE_ATTN_ACT_QUANTIZATION:
            if ENABLE_EXTREME_ATTN_ACT_QUANTIZATION:
                q = quantized_tanh(q)
                k = quantized_tanh(k)
            else:
                q = quantized_linear(q, num_bits=8)
                k = quantized_linear(k, num_bits=8)

# (B, T, H, D) -> (B, H, T, D)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # PyTorch 2.0+ scaled dot product attention
        # y shape is (B, H, T, D)
        y = F.scaled_dot_product_attention(q, k, v, 
                                           is_causal=True, 
                                           scale=attn_scale)

        # (B, H, T, D) -> (B, T, H, D)
        y = y.transpose(1, 2) # <-- *** FIX 1: Don't flatten heads yet ***
        
        # Apply gate: (B, T, H, D) * (B, T, H, 1) -> (B, T, H, D)
        y = y * torch.sigmoid(self.attn_gate(x[..., :self.attn_gate.weight.size(-1)])).view(B, T, self.num_heads, 1)
        
        # Now re-assemble all head outputs side by side
        y = y.contiguous().view(B, T, self.num_heads * self.head_dim) 
        
        # <-- *** FIX 2: Add missing final output projection ***
        y = F.linear(y, self.quantize_qkvo_w2(self.qkvo_w.view(4, self.hdim, self.dim)[3]).type_as(y))
        return y


class MLP(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        hdim = 4 * dim
        self.c_fc = nn.Parameter(torch.empty(dim, hdim))
        self.c_proj = nn.Parameter(torch.empty(dim, hdim))
        self.quantize_c_fc = Quantize(ternary_quantize) if ENABLE_MLP_QUANTIZATION else lambda x: x
        self.quantize_c_proj = Quantize(ternary_quantize) if ENABLE_MLP_QUANTIZATION else lambda x: x
        self.c_fc.label = 'mlp'
        self.c_proj.label = 'mlp'
        self.c_fc.lr_mul = 2.

        std = 0.5 * (dim ** -0.5)
        bound = (3 ** 0.5) * std
        with torch.no_grad():
            self.c_fc.uniform_(-bound, bound)
            self.c_proj.zero_()

    def forward(self, x: Tensor):
        if ENABLE_MLP_PRE_ACT_QUANTIZATION:
            if ENABLE_EXTREME_PRE_MLP_ACT_QUANTIZATION:
                x = quantized_tanh(x)
            else:
                x = quantized_linear(x, num_bits=8)
        x = F.linear(x, self.quantize_c_fc(self.c_fc).T.type_as(x))
        if ENABLE_MLP_ACT_QUANTIZATION:
            if ENABLE_EXTREME_MLP_ACT_QUANTIZATION:
                x = quantized_tanh(x)
            else:
                x = quantized_relu2(x, num_bits=8)
        else:
            x = F.relu(x).square()
        x = F.linear(x, self.quantize_c_fc(self.c_proj).type_as(x))
        return x

class Block(nn.Module):
    def __init__(self, dim: int, head_dim: int, num_heads: int, layer_idx: int):
        super().__init__()
        self.attn = CausalSelfAttention(dim, head_dim, num_heads) if layer_idx not in [0, 7] else None
        self.mlp = MLP(dim) if layer_idx != 0 else None

    def forward(self, x: Tensor, x0: Tensor, lambdas: Tensor, attn_args: AttnArgs):
        x = lambdas[0] * x + lambdas[1] * x0
        if self.attn is not None:
            x = x + self.attn(norm(x), attn_args)
        if self.mlp is not None:
            x = x + self.mlp(norm(x))
        return x

def next_multiple_of_n(v: float | int, *, n: int):
    return next(x for x in range(n, int(v) + 1 + n, n) if x >= v)

class GPT(nn.Module):
    def __init__(self, vocab_size: int, num_layers: int, num_heads: int, head_dim: int, model_dim: int, max_seq_len: int):
        super().__init__()
        self.vocab_size = next_multiple_of_n(vocab_size, n=128)
        self.embed = nn.Embedding(self.vocab_size, model_dim)
        self.smear_gate = CastedLinear(12, 1)
        self.smear_gate.weight.label = 'smear_gate'
        self.value_embeds = nn.ModuleList([nn.Embedding(self.vocab_size, model_dim) for _ in range(3)])
        self.blocks = nn.ModuleList([Block(model_dim, head_dim, num_heads, i) for i in range(num_layers)])
        self.yarn = Yarn(head_dim, max_seq_len)
        self.lm_head = CastedLinear(model_dim, self.vocab_size, use_fp8=False) # use_fp8=False for general inference
        
        assert num_layers % 2 == 0
        
        # The checkpoint was trained with a world_size where the padding was 0
        # (e.g., world_size=1 or 2). The total number of scalars defined in 
        # train_gpt.py before padding is 62.
        # 1. skip_weights: num_layers = 12
        # 2. block lambdas: 2 * num_layers = 24
        # 3. SA lambdas: 2 * num_layers = 24
        # 4. smear_lambda: 1
        # 5. backout_lambda: 1
        # Total = 12 + 24 + 24 + 1 + 1 = 62.
        # The checkpoint file has a tensor of size 62, so we must match that.
        total_scalars = 62 

        self.scalars = nn.Parameter(torch.empty(total_scalars))

    def forward(self, input_seq: Tensor, ws_short: int, ws_long: int):
        # input_seq is (B, T)
        B, T = input_seq.shape
        assert B == 1, "Inference script only supports batch size 1"

        ve = [value_embed(input_seq) for value_embed in self.value_embeds]
        ve = [None, ve[1], ve[2]] + [None] * (len(self.blocks) - 6) + [ve[0], ve[1], ve[2]]
        assert len(ve) == len(self.blocks)

        block_size = 128 # from train_gpt.py
        short_bm = ws_short * block_size
        long_bm = ws_long * block_size
        bm_sizes = [None, short_bm, short_bm, short_bm, long_bm, short_bm, short_bm, None, short_bm, short_bm, short_bm, long_bm]
        assert len(bm_sizes) == len(self.blocks)

        x = self.embed(input_seq) # (B, T, D)

        n_layers = len(self.blocks)
        skip_weights = self.scalars[:(n_layers // 2)]
        lambdas = self.scalars[1 * n_layers: 3 * n_layers].view(-1, 2)
        sa_lambdas = self.scalars[3 * n_layers: 5 * n_layers].view(-1, 2)
        smear_lambda = self.scalars[5 * n_layers]
        backout_lambda = self.scalars[5 * n_layers + 1]

        # smear token embed forward 1 position
        smear_gate_out = smear_lambda * torch.sigmoid(self.smear_gate(x[:, 1:, :self.smear_gate.weight.size(-1)]))
        x = torch.cat([
            x[:, :1, :], # (B, 1, D)
            x[:, 1:, :] + smear_gate_out * x[:, :-1, :] # (B, T-1, D)
        ], dim=1)
        x = x0 = norm(x) # (B, T, D)

        # U-net design
        skip_connections = []
        n = len(self.blocks) // 2

        x_backout = None
        backout_layer = 8
        
        for i in range(1,len(self.blocks)):
            attn_args = AttnArgs(
                ve=ve[i], # (B, T, D) or None
                sa_lambdas=sa_lambdas[i],
                seqlens=None, # Not used
                bm_size=bm_sizes[i],
                cos=self.yarn.cos,
                sin=self.yarn.sin,
                attn_scale=self.yarn.attn_scale
            )
            if i >= n and i<11:
                gate = torch.sigmoid(skip_weights[i - n])
                x = x + gate * skip_connections.pop()
            x = self.blocks[i](x, x0, lambdas[i], attn_args)
            if i < n:
                skip_connections.append(x)
            if i == backout_layer:
                x_backout = x

        x -= backout_lambda * x_backout
        x = norm(x)
        logits = self.lm_head(x)
        logits = 30 * torch.sigmoid(logits / 7.5)
        return logits # (B, T, Vocab)

# -----------------------------------------------------------------------------
# Inference (Generation) Logic
# -----------------------------------------------------------------------------

def generate(model, enc, prompt, n_tokens, temp=0.7, top_k=50):
    """
    Generates text from a prompt.
    """
    model.eval()
    BOS_ID = 50256 # GPT-2 BOS token
    tokens = [BOS_ID] + enc.encode(prompt)
    tokens = torch.tensor(tokens, dtype=torch.long, device=device).unsqueeze(0) # (1, T)

    # Use the final window sizes from training
    ws_final = 13
    ws_short = ws_final // 2 # 6
    ws_long = ws_final       # 13
    
    print("Generating tokens...")
    with torch.no_grad():
        for i in range(n_tokens):
            if i % 10 == 0:
                print(f"  Token {i}/{n_tokens}")
                
            # Get logits for the *last* token
            # Note: We pass the *entire* sequence each time.
            # This is inefficient but simple and correct.
            # For efficiency, K/V caching would be implemented.
            logits = model(tokens, ws_short, ws_long) # (B, T, V)
            logits = logits[:, -1, :] # (B, V)
            
            # Apply temperature
            if temp > 0:
                logits /= temp
                
                # Apply Top-K
                if top_k is not None:
                    v, _ = torch.topk(logits, top_k)
                    logits[logits < v[:, [-1]]] = -float('Inf')
                
                # Sample
                probs = F.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1) # (B, 1)
            else:
                # Greedy sampling
                next_token = torch.argmax(logits, dim=-1, keepdim=True)

            # Append
            tokens = torch.cat((tokens, next_token), dim=1)
    
    # Decode and print
    out_tokens = tokens.squeeze(0).tolist()
    if out_tokens[0] == BOS_ID:
        out_tokens = out_tokens[1:] # Remove BOS
    
    return enc.decode(out_tokens)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, required=True, help="Path to .pt checkpoint file")
    parser.add_argument('--config', type=str, required=True, help="Path to the config.json used to train this checkpoint")
    parser.add_argument('--prompt', type=str, default="The meaning of life is", help="Text prompt")
    args = parser.parse_args()

    # 1. Load Configuration to set global flags for Model Definition
    set_globals(args.config)

    # --- Configuration ---
    CHECKPOINT_PATH = args.checkpoint
    PROMPT = args.prompt
    N_TOKENS_TO_GENERATE = 100
    TEMPERATURE = 0.5
    TOP_K = 50
    # ---------------------
    
    print(f"Loading model from: {CHECKPOINT_PATH}")
    print(f"Using device: {device}")
    
    # 1. Load checkpoint
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)
    model_state_dict = checkpoint['model']
    
    # 2. Re-create the model architecture
    # These parameters must match train_gpt.py
    model_args = dict(
        vocab_size=50257,
        num_layers=12,
        num_heads=6,
        head_dim=128,
        model_dim=768,
        max_seq_len=2048 # Max context for inference
    )
    model = GPT(**model_args).to(device).to(torch.bfloat16)
    
    # 3. Re-apply Yarn schedule to get correct RoPE state
    # This is critical because RoPE buffers (cos/sin) are not
    # saved in the checkpoint (persistent=False). We must
    # reconstruct them by re-running the window size schedule.
    print("Applying YaRN schedule to model buffers...")
    
    # Hardcoded schedule from train_gpt.py
    ws_schedule = (3, 7, 11)
    ws_final = 13
    num_scheduled_iterations = 2245
    num_iterations = 2245 + 40 # 2285

    def get_ws(step: int):
        if step >= num_scheduled_iterations:
            return ws_final // 2, ws_final
        x = step / num_scheduled_iterations
        ws_idx = int(len(ws_schedule) * x)
        return ws_schedule[ws_idx] // 2, ws_schedule[ws_idx]

    model.yarn.reset() # Start from scratch
    ws_long = get_ws(0)[1] # Initial ws_long = 3
    
    # Run through the schedule that the model saw during training
    for step in range(num_iterations): # step 0 to 2284
         _, new_ws_long = get_ws(step)
         if new_ws_long != ws_long:
            model.yarn.apply(ws_long, new_ws_long)
            ws_long = new_ws_long
            
    # After the loop, ws_long is 13, which is the state
    # the model was in when it was saved.

    # 4. Clean and load weights
    print("Loading model weights...")
    unwrapped_state_dict = {}
    for k, v in model_state_dict.items():
        # Remove prefixes added by torch.compile ("_orig_mod.")
        # and DDP ("module.") if they exist.
        k = k.replace("_orig_mod.", "")
        k = k.replace("module.", "")
        unwrapped_state_dict[k] = v
    
    model.load_state_dict(unwrapped_state_dict)
    
    # 5. Initialize tokenizer
    enc = tiktoken.get_encoding("gpt2")
    
    # 6. Generate text
    print("---------------------")
    print(f"Prompt: {PROMPT}")
    print("---------------------")
    output = generate(model, enc, PROMPT, N_TOKENS_TO_GENERATE, temp=TEMPERATURE, top_k=TOP_K)
    print(output)
    print("---------------------")


if __name__ == "__main__":
    main()
