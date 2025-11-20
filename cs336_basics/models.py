# %%
import torch
from einops import rearrange, einsum

def set_seed(seed):
    import torch
    import numpy as np
    import random
    import os
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
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
dim_by = torch.linspace(start=.0, end=1.0, steps=10)
dim_value = rearrange(dim_by, "dim_value -> 1 dim_value 1 1 1")
images_rearr = rearrange(images, "b height width channel -> b 1 height width channel")
dimmed_images = images_rearr * dim_value
dimmed_images = einsum(images, dim_by,
"batch height width channel, dim_value -> batch dim_value height width channel")
# %%
