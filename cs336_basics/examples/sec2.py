# %%
import torch
from einops import rearrange, einsum


def set_seed(seed):
    import torch
    import numpy as np
    import random
    import os

    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


set_seed(0)

# %%
D = torch.randn(size=(64, 10, 512))
A = torch.randn(size=(1024, 512))
Y = D @ A.T
Y = einsum(D, A, "batch sequence d_in, d_out d_in -> batch sequence d_out")
Y = einsum(D, A, "... d_in, d_out d_in -> ... d_out")

# %%
images = torch.randn(64, 128, 128, 3)
dim_by = torch.linspace(start=0.0, end=1.0, steps=10)
dim_value = rearrange(dim_by, "dim_value -> 1 dim_value 1 1 1")
images_rearr = rearrange(images, "b height width channel -> b 1 height width channel")
dimmed_images = images_rearr * dim_value
dimmed_images = einsum(
    images,
    dim_by,
    "batch height width channel, dim_value -> batch dim_value height width channel",
)

# %%
channels_last = torch.randn(64, 32, 32, 3)
# (batch, height, width, channel)
B = torch.randn(32 * 32, 32 * 32)
## Rearrange an image tensor for mixing across all pixels
channels_last_flat = channels_last.view(-1, channels_last.size(1) * channels_last.size(2), channels_last.size(3))
print(channels_last_flat.shape)
channels_first_flat = channels_last_flat.transpose(1, 2)
print(channels_first_flat.shape)
channels_first_flat_transformed = channels_first_flat @ B.T
channels_last_flat_transformed = channels_first_flat_transformed.transpose(1, 2)
print(channels_last_flat_transformed.shape)

# %%
height = width = 32
## Rearrange replaces clunky torch view + transpose
channels_first = rearrange(channels_last, "batch height width channel -> batch channel (height width)")
print(channels_first.shape)
channels_first_transformed = einsum(
    channels_first,
    B,
    "batch channel pixel_in, pixel_out pixel_in -> batch channel pixel_out",
)
channels_last_transformed = rearrange(
    channels_first_transformed, "batch channel (height width) -> batch height width channel", height=height, width=width
)
print(channels_last_transformed.shape)
# %%
import einx

height = width = 32
channels_last_transformed = einx.dot(
    "batch row_in col_in channel, (row_out col_out) (row_in col_in)-> batch row_out col_out channel",
    channels_last,
    B,
    # col_in=width,
    row_out=height,
    # col_out=width,
)
print(channels_last_transformed.shape)
# %%
d_in = 64
d_out = 512
sigma = (2 / (d_in + d_out)) ** 0.5
linear_weights = (
    torch.randn(
        size=(512, 64),
    )
    * sigma
).clip(-3 * sigma, 3 * sigma)

# if away from the mean, resample
linear_weights = torch.nn.init.trunc_normal_(linear_weights, std=sigma, a=-3 * sigma, b=3 * sigma)
# %%
a = torch.zeros(size=(10,))
print(a.shape)
print(a.unsqueeze(0).shape)
# %%
a = torch.randn(size=(2, 2))
print(torch.block_diag(*[a.clone()] * 4))

# %%
a = torch.randn(size=(64, 10, 512))
rot_mat = torch.randn(size=(20, 512, 512))
used_rot_mat = rot_mat[:10]
torch.einsum("...k, ...kj -> ...j", a, used_rot_mat).shape

# %%
a = torch.randn(size=((2, 3, 4)))
print(a.sum(dim=0).shape)
print(a.sum(dim=1).shape)
print(a.sum(dim=2).shape)

# %%
a = torch.randn(size=(64, 10, 32))
b = a.clone()
print(torch.concatenate([a, b], dim=-1).shape)

# %%
print(torch.randn(size=(64, 10, 32))[*[None] * 3, ...].shape)
# %%
vocab_size = 50257
context_length = 1024
num_layers = 48
d_model = 1600
num_heads = 25
d_ff = 6400

num_parameters = {
    "embedding": vocab_size * d_model,
    "layers": num_layers * (d_model * d_model * 4 + d_model * d_ff * 3 + d_model + d_model),
    "ln_final": d_model,
    "lm_head": d_model * vocab_size,
}
print(
    f"num_parameters: {sum(num_parameters.values())}, used memory (sigle precision): {sum(num_parameters.values()) * 4 / 1024 / 1024 / 1024} GB"
)


# %%
from cs336_basics.models import Transformer

model = Transformer(
    vocab_size=vocab_size,
    d_model=d_model,
    num_heads=num_heads,
    d_ff=d_ff,
    num_layers=num_layers,
    rope_theta=1,
    context_length=100,
)
num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Total params: {num_params}")
assert num_params == sum(num_parameters.values())


# %%
def get_peak_memory(
    batch_size,
    vocab_size,
    context_length,
    num_layers,
    d_model,
    num_heads,
    d_ff=None,
    dtype_bytes=4, # float32
):
    dff = 4 * d_model if d_ff is None else d_ff
    token_embeddings = vocab_size * d_model
    # RMSNorm->MultiHeadAttention->RMSNorm->SwiGluFeedForward
    transformer_layers = num_layers * (d_model + 4 * d_model * d_model + d_model + 3 * d_model * dff)
    ln_norm = d_model
    lm_head = d_model * vocab_size

    model_params = token_embeddings + transformer_layers + ln_norm + lm_head
    adamw_params = model_params * 2  # m + v