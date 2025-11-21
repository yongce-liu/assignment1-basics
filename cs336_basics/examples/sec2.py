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
