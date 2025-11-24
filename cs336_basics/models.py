import math
import torch
from torch.nn import Module
from torch.optim import Optimizer, lr_scheduler
from jaxtyping import Float
from collections.abc import Callable


class Linear(Module):
    def __init__(
        self, in_features: int, out_features: int, device: torch.device | None = None, dtype: torch.dtype | None = None
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self._factory_params = {"dtype": dtype, "device": device}
        self.weight = torch.nn.Parameter(
            torch.empty(size=(self.out_features, self.in_features), **self._factory_params)
        )
        self.reset_weight()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.einsum("...i, o i -> ...o", x, self.weight.to(x.device))

    def reset_weight(self) -> None:
        sigma = (2 / (self.in_features + self.out_features)) ** 0.5
        self.weight = torch.nn.init.trunc_normal_(self.weight, mean=0, std=sigma, a=-3 * sigma, b=3 * sigma)


class Embedding(Module):
    def __init__(self, num_embeddings: int, embedding_dim: int, device=None, dtype=None):
        super().__init__()
        self._factory_params = {"device": device, "dtype": dtype}
        self.weight = torch.nn.Parameter(torch.empty(size=(num_embeddings, embedding_dim), **self._factory_params))
        self.reset_weight()

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.weight[token_ids].to(token_ids.device)

    def reset_weight(self) -> None:
        sigma = 1
        self.weight = torch.nn.init.trunc_normal_(self.weight, mean=0, std=sigma, a=-3 * sigma, b=3 * sigma)


class RmsNorm(Module):
    def __init__(self, d_model: int, eps: float = 1e-5, device=None, dtype=None):
        super().__init__()
        self._factory_params = {"device": device, "dtype": dtype}
        self.d_model = d_model
        self.eps = eps
        self.weight = torch.nn.Parameter(torch.empty(size=(self.d_model,), **self._factory_params))

    def forward(self, x: Float[torch.Tensor, " ... d_model"]) -> torch.Tensor:
        in_dtype = x.dtype
        x = x.to(torch.float32)
        res = x / torch.sqrt(torch.sum(x**2, dim=-1, keepdim=True) / self.d_model + self.eps) * self.weight.to(x.device)

        return res.to(in_dtype)

    def reset_weight(self) -> None:
        self.weight = torch.nn.Parameter(torch.ones(size=(self.d_model,), **self._factory_params))


class SwiGLUFFN(Module):
    def __init__(self, d_model: int, d_ff: int, device=None, dtype=None):
        super().__init__()
        self._factory_params = {"device": device, "dtype": dtype}
        self.w1 = Linear(d_model, d_ff, **self._factory_params)
        self.w2 = Linear(d_ff, d_model, **self._factory_params)
        self.w3 = Linear(d_model, d_ff, **self._factory_params)

    def reset_weight(self) -> None:
        for mod in self.modules():
            mod.reset_weight()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.w1.forward(x)
        x3 = self.w3.forward(x)
        x2 = x1 * torch.sigmoid(x1) * x3

        return self.w2.forward(x2)


class RoPE(Module):
    def __init__(self, theta: float, d_k: int, max_seq_len: int, device=None, dtype=None):
        super().__init__()
        self._factory_params = {"device": device, "dtype": dtype}
        self.const_theta = theta
        self.d_k = d_k
        pos_mat = []
        for i in range(max_seq_len):
            pos_mat.append(self.get_rot_mat(i).T)  # store the transpose mat, y = Rx -> y^t = x^TR^T
        self.rot_mat: torch.Tensor = torch.stack(pos_mat)  # (max_seq_len, d_k, d_k)
        self.register_buffer(name="rotation matrix", tensor=self.rot_mat, persistent=False)

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        """
        input x: [... seq_len d_k]
        output: [... seq_len d_k]
        """
        res = torch.einsum("...k, ...kj -> ...j", x, self.rot_mat.to(x.device)[token_positions])
        return res

    def get_rot_mat(self, pos: int):
        rot_i = []
        for k in range(0, int(self.d_k / 2)):
            theta = pos / (self.const_theta ** (2 * k / self.d_k))
            rot_i.append(self.mat_func(theta, **self._factory_params))

        return torch.block_diag(*rot_i)

    @staticmethod
    def mat_func(theta: float, **kwargs) -> torch.Tensor:
        """
        theta in rad
        """
        return torch.tensor([[math.cos(theta), -math.sin(theta)], [math.sin(theta), math.cos(theta)]], **kwargs)


def softmax(x: torch.Tensor, dim: int) -> torch.Tensor:
    max_x = torch.max(x, dim=dim, keepdim=True).values
    x_stable = x - max_x
    exp_x = torch.exp(x_stable)
    sum_exp_x = torch.sum(exp_x, dim=dim, keepdim=True)
    p = exp_x / sum_exp_x

    return p


def attention(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
    """
    Q: b, ..., s, d_k
    K: b, ..., s, d_k
    V: b, ..., s, d_v
    mask: bool
    -> : b, ..., d_v
    """
    sqrt_d_k = Q.shape[-1] ** 0.5
    val = torch.einsum("... n k, ... m k -> ... n m", Q, K) / sqrt_d_k

    if mask is not None:
        neg_inf = torch.finfo(val.dtype).min
        val = val.masked_fill(~mask, value=neg_inf)
    val = softmax(val, dim=-1)
    atten = torch.einsum("... n m, ... m v -> ... n v", val, V)
    return atten


class MultiHeadAttention(Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_in: int = None,
        d_v: int = None,
        d_out: int = None,
        rope: RoPE = None,
        device=None,
        dtype=None,
    ):
        super().__init__()
        self._factory_params = {"device": device, "dtype": dtype}
        self.d_in = d_model if d_in is None else d_in
        self.d_k = self.d_model = d_model
        self.d_v = d_model if d_v is None else d_v
        self.d_out = d_model if d_out is None else d_out

        self.num_heads = num_heads
        self._d_k = d_model // num_heads

        self.q_proj = Linear(in_features=self.d_in, out_features=self.d_k, **self._factory_params)
        self.k_proj = Linear(in_features=self.d_in, out_features=self.d_k, **self._factory_params)
        self.v_proj = Linear(in_features=self.d_in, out_features=self.d_v, **self._factory_params)
        self.output_proj = Linear(in_features=self.d_v, out_features=self.d_out, **self._factory_params)
        self.rope = rope

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor = None) -> torch.Tensor:
        x_shapes = x.shape[:-1]
        Q = self.q_proj.forward(x).reshape(*x_shapes, -1, self._d_k).transpose(-2, -3)
        K = self.k_proj.forward(x).reshape(*x_shapes, -1, self._d_k).transpose(-2, -3)
        V = self.v_proj.forward(x).reshape(*x_shapes, -1, self._d_k).transpose(-2, -3)

        if self.rope is not None:
            if token_positions is None:
                # Default to sequential positions [0, 1, 2, ..., seq_len-1]
                seq_len = x_shapes[-1]
                token_positions = torch.arange(seq_len, device=x.device)
                # Broadcast to match batch dimensions if needed
                for _ in range(len(x_shapes) - 1):
                    token_positions = token_positions.unsqueeze(0)
            Q = self.rope(Q, token_positions)
            K = self.rope(K, token_positions)

        mask = torch.tril(torch.ones(x_shapes[-1], x_shapes[-1], dtype=torch.bool, device=Q.device))[
            *([None] * (len(x_shapes) - 1)), ...
        ]
        atten = attention(Q, K, V, mask).transpose(-2, -3).reshape(*x_shapes, -1)

        return self.output_proj.forward(atten)


class TransformerBlock(Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int, rope: RoPE = None, device=None, dtype=None):
        super().__init__()
        self._factory_params = {"device": device, "dtype": dtype}
        self.attn = MultiHeadAttention(d_model=d_model, num_heads=num_heads, rope=rope, **self._factory_params)
        self.ffn = SwiGLUFFN(d_model=d_model, d_ff=d_ff, **self._factory_params)
        self.ln1 = RmsNorm(d_model=d_model, **self._factory_params)
        self.ln2 = RmsNorm(d_model=d_model, **self._factory_params)

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor = None):
        y1 = x + self.attn(self.ln1(x), token_positions)
        y2 = y1 + self.ffn(self.ln2(y1))

        return y2


class Transformer(Module):
    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        num_heads: int,
        d_ff: int,
        num_layers: int,
        rope_theta: float,
        context_length: int,
        device=None,
        dtype=None,
    ):
        super().__init__()
        self._factory_params = {"device": device, "dtype": dtype}
        self.token_embeddings = Embedding(num_embeddings=vocab_size, embedding_dim=d_model, **self._factory_params)
        rope = RoPE(rope_theta, d_model // num_heads, context_length)
        self.layers = torch.nn.ModuleList(
            [
                TransformerBlock(d_model=d_model, num_heads=num_heads, d_ff=d_ff, rope=rope, **self._factory_params)
                for _ in range(num_layers)
            ]
        )
        self.ln_final = RmsNorm(d_model=d_model)
        self.lm_head = Linear(in_features=d_model, out_features=vocab_size)

    def forward(self, x: torch.Tensor):
        x = self.token_embeddings(x)
        for m_block in self.layers:
            x = m_block(x)
        x = self.ln_final(x)
        x = self.lm_head(x)
        return x


def cross_entropy_loss(x: torch.Tensor, y: torch.Tensor):
    max_logits = x.max(dim=-1, keepdim=True).values
    logsumexp = torch.log(torch.sum(torch.exp(x - max_logits), dim=-1))
    correct_logits = x[torch.arange(x.size(0)), y]
    loss = -correct_logits + max_logits.unsqueeze(-1) + logsumexp

    return loss.mean()


class AdamW(Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), weight_decay=0.01, eps=1e-8):
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        defaults = dict(lr=lr, betas=betas, weight_decay=weight_decay)
        super().__init__(params, defaults)
        self.eps = eps

    @torch.no_grad
    def step(self, closure: Callable | None = None):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            lr = group["lr"]
            beta1, beta2 = group["betas"]
            wd = group["weight_decay"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                # auto grad data
                g = p.grad.data
                # get hyper-parameters
                state = self.state[p]
                if len(state) == 0:
                    state["t"] = 0  # it is the number, so assignment no wekcopy
                    state["exp_avg"] = torch.zeros_like(p)
                    state["exp_avg_sq"] = torch.zeros_like(p)
                t = state["t"] = state["t"] + 1
                m = state["exp_avg"]
                v = state["exp_avg_sq"]
                # update 1st moment estimate
                # m.mul_(beta1).add_(g, alpha=1 - beta1)
                m[:] = beta1 * m + (1 - beta1) * g
                # update 2rd moment estimate
                # v.mul_(beta2).addcmul_(g, g, value=1 - beta2)
                v[:] = beta2 * v + (1 - beta2) * g**2
                # update learning rate
                lr_t = lr * (1 - beta2**t) ** 0.5 / (1 - beta1**t)
                # !!!!! update parameters !!!!!
                # p.data.addcdiv_(m, v.sqrt().add_(self.eps), value=-lr_t)
                p[:] = p - lr_t * m / (v**0.5 + self.eps)
                # !!!!! weight decay !!!!!
                if wd > 0:
                    # p.data.add_(p, alpha=-lr * wd)
                    p[:] = p - lr * wd * p
        return loss
