# Supported Operators

Shape notation: uppercase strings are symbolic dimensions (`B`, `T`, `D`), integers are constants.

---

## Tensor shape methods

### `reshape` / `view`

```python
x.reshape(*shape)
x.view(*shape)
torch.reshape(x, shape)
```

Input: `(*dims)`
Output: `(*requested)` — at most one element of `shape` may be `-1` (automatically inferred).
For fully-constant shapes, the total element count must be preserved; mismatches are reported as `TSF1004`.

### `permute`

```python
x.permute(*order)
```

Input: `(*dims)` (rank N)
Output: `(dims[order[0]], dims[order[1]], ...)` — reorders all N axes.
Negative indices supported.

### `transpose`

```python
x.transpose(dim0, dim1)
```

Input: `(*dims)`
Output: `(*dims)` with `dim0` and `dim1` swapped. Negative indices supported.

### `flatten`

```python
x.flatten(start_dim=0, end_dim=-1)
```

Input: `(*dims)`
Output: `(*dims[:start], product(dims[start:end+1]), *dims[end+1:])`

Example: `Shape("B", 8, 32, 32)` with `flatten(1)` → `[B, 8192]`

### `squeeze`

```python
x.squeeze()        # removes all size-1 dims
x.squeeze(dim)     # removes dim only if it is size 1
```

Returns `None` if the specified dim is not size 1.

### `unsqueeze`

```python
x.unsqueeze(dim)
```

Input: `(*dims)` (rank N)
Output: `(*dims)` with a new size-1 axis inserted at position `dim`.
Valid range: `[-N-1, N]`. Negative indices count from the end of the *output* tensor (`-1` appends).

### `matmul` / `bmm`

```python
x.matmul(y)
torch.matmul(x, y)
torch.bmm(x, y)
x @ y
```

Input: `(*, M, K)` and `(*, K, N)` (both rank ≥ 2)
Output: `(*, M, N)` — batch dimensions are broadcast.
Emits `TSF1003` on inner-dimension mismatch.

### `mm`

```python
x.mm(y)
torch.mm(x, y)
```

Input: `(M, K)` and `(K, N)` — both must be rank 2 (no batch dimensions).
Output: `(M, N)`.
Emits `TSF1003` if inner dimensions do not match.

### `movedim`

```python
x.movedim(source, destination)
torch.movedim(x, source, destination)
```

Input: `(*dims)` (rank N)
Output: `(*dims)` with the specified axes moved to new positions.
`source` and `destination` may be a single integer or a tuple of integers; negative indices are supported.

### `size`

```python
x.size()       # → ShapeTupleValue (all dims)
x.size(dim)    # → IntegerValue for that axis (constant dims only; symbolic → None)
```

---

## Reduction methods

```python
x.sum()                       # → scalar (rank 0)
x.sum(dim=1)                  # → removes axis 1
x.sum(dim=1, keepdim=True)    # → keeps axis 1 as size 1
x.mean(dim=(1, 2))            # → removes axes 1 and 2 (tuple dim)
torch.amax(x, dim=1)          # functional form — same semantics
```

Supported methods: `sum`, `mean`, `max`, `min`, `amax`, `amin`, `prod`, `all`, `any`, `argmax`, `argmin`, `nanmean`, `nansum`.

Available as both tensor methods (`x.sum(...)`) and `torch.*` functions (`torch.sum(x, ...)`).
`dim` may be an integer or a tuple of integers; negative indices are supported.
When no `dim` is given, the result is a rank-0 scalar tensor.

---

## Arithmetic and broadcasting

### Element-wise ops

```python
x + y,  x - y,  x * y,  x / y,  x // y,  x % y,  x ** y
```

Input: any two tensors with broadcast-compatible shapes (NumPy/PyTorch rules).
Output: broadcast result shape. Incompatible shapes emit `TSF1006`.

### Augmented assignment

```python
x += y
x -= y
x *= y   # etc.
```

Treated as `x = x <op> y`; the result shape follows broadcast rules and updates the variable in the environment.
Useful for residual connections (`x += residual`).

### `torch.einsum`

```python
torch.einsum(subscript, t1, t2, ...)
torch.einsum(subscript, [t1, t2, ...])
```

Supports explicit-mode subscripts (those containing `->`) with single-character labels.
Ellipsis and implicit mode (no `->`) are not yet supported.

```python
# Batched matmul: (B, T, D) @ (B, D, T) → (B, T, T)
out = torch.einsum("bik,bkj->bij", q, k)

# Matrix-vector product: (M, K) @ (K,) → (M,)
out = torch.einsum("ij,j->i", A, v)

# Outer product: (M,) ⊗ (N,) → (M, N)
out = torch.einsum("i,j->ij", u, v)
```

Emits `TSF1003` if contracted dimensions have mismatched sizes (constant or symbolic).

---

## Sequence operations

### `torch.cat`

```python
torch.cat([t1, t2, ...], dim=0)
```

All tensors must have the same rank and matching sizes on all axes except `dim`.
Output: the concatenated axis size is the sum of input sizes on that axis.

### `torch.stack`

```python
torch.stack([t1, t2, ...], dim=0)
```

All tensors must have identical shapes.
Output: a new axis of size `len(tensors)` is inserted at position `dim`. Result rank = input rank + 1.

### `chunk`

```python
x.chunk(n, dim=0)
```

Input: `(*dims)`
Output: a tuple of `n` tensors.

- **Constant dim, evenly divisible:** each chunk has `dims[dim] // n` on the split axis.
- **Constant dim, not evenly divisible:** first `n-1` chunks have `ceil(dims[dim] / n)`, last chunk has the remainder.
- **Symbolic dim:** each chunk has `dims[dim]//n` as an expression.

Supports tuple-unpacking: `a, b, c = x.chunk(3, dim=-1)`.

### `split`

```python
x.split(split_size, dim=0)
torch.split(x, split_size_or_sections, dim=0)
```

`split_size` may be:

- An `int` — splits into equal-ish chunks of that size (requires the axis to be a constant).
- A `list[int]` — splits into exactly those sizes. When the axis is a constant, the section sizes must sum to it.

Output: a tuple of `TensorValue` objects, one per chunk.
Supports tuple-unpacking: `q, k, v = x.split(64, dim=-1)`.

---

## Tensor expansion

### `expand` / `expand_as`

```python
x.expand(*sizes)
x.expand_as(other)
```

`expand` broadcasts singleton dimensions to the given sizes; `-1` keeps the original dimension unchanged.
`expand_as` expands to match the shape of `other`.

Input: `(*dims)`
Output: `(*sizes)` — leading dimensions may be added.

### `repeat`

```python
x.repeat(*repeats)
```

Repeats the tensor along each dimension (copies data, unlike `expand`).

Input: `(*dims)` (rank N)
Output: each dimension `i` is multiplied by `repeats[i]`. If `len(repeats) > N`, leading dimensions of `1` are prepended first.

---

## Shape-preserving tensor methods

The following methods preserve the input shape exactly (dtype/device casts, memory layout, gradient management, in-place fill):

```python
x.contiguous()
x.float()   x.half()   x.double()
x.int()     x.long()   x.short()   x.byte()   x.bool()
x.to(dtype_or_device)
x.detach()
x.clone()
x.cpu()     x.cuda()
x.type(dtype)
x.masked_fill(mask, value)
x.fill_(value)   x.zero_()   x.normal_()   x.uniform_()
x.requires_grad_(...)
```

---

## Functional passthrough

The following `torch.*` and `torch.nn.functional.*` calls return the same shape as their first argument:

**Activations:** `relu`, `relu_`, `leaky_relu`, `gelu`, `silu`, `sigmoid`, `tanh`, `elu`, `selu`, `mish`, `hardswish`

**Normalisation:** `layer_norm`, `batch_norm`, `group_norm`, `instance_norm`, `normalize`

**Attention / masking:** `softmax`, `log_softmax`, `triu`, `tril`

**Regularisation:** `dropout`, `dropout2d`, `dropout3d`

**Element-wise predicates / unary:** `flip`, `isfinite`, `isinf`, `isnan`, `abs`, `neg`, `sign`

```python
y = F.relu(x)                  # [B, T, D] → [B, T, D]
y = torch.softmax(x, dim=-1)
y = F.layer_norm(x, x.shape[-1:])
y = torch.triu(x)
```

### `F.scaled_dot_product_attention`

```python
F.scaled_dot_product_attention(query, key, value, ...)
```

Output shape equals `query`'s shape.

---

## Tensor constructors

### Size-based constructors

```python
torch.zeros(B, T, D)
torch.zeros((B, T, D))      # single tuple arg
torch.ones(size=(B, T))     # keyword size=
torch.empty(B, T, D)
torch.randn(B, T, D)
torch.rand(B, T, D)
torch.full((B, T), fill_value)
```

Size arguments can be integer constants (→ `ConstantDim`) or variable names from the environment (→ `UnknownDim`).

### `*_like` constructors

```python
torch.zeros_like(x)
torch.ones_like(x)
torch.empty_like(x)
torch.randn_like(x)
torch.rand_like(x)
torch.full_like(x, fill_value)
```

Output shape equals `x`'s shape.

### `torch.arange`

```python
torch.arange(end)
torch.arange(start, end)
torch.arange(start, end, step)
```

Output: rank-1 tensor. When all arguments are integer constants, the exact length is computed.
Otherwise the dimension is unknown.

### `F.one_hot`

```python
F.one_hot(tensor, num_classes=N)
```

Input: `(*dims)` — integer index tensor of any rank.
Output: `(*dims, N)` — the `num_classes` size is appended as a new trailing axis.

---

## Spatial interpolation

### `F.interpolate`

```python
F.interpolate(input, size=None, scale_factor=None, mode="nearest", ...)
```

Batch and channel dimensions (first two) are always preserved. Only spatial dimensions (all dims beyond the first two) are resized.

Input: `(N, C, *spatial)` — rank ≥ 3.
Output: `(N, C, *new_spatial)`

- **`size` as a tuple:** `(H_out, W_out)` — each spatial dim is replaced by the given constant.
- **`size` as a variable** (e.g. `labels.shape[-2:]`): evaluated at analysis time when possible.
- **`scale_factor` as a float or tuple:** each constant spatial dim is multiplied. Symbolic dims with integer scale factors produce expressions (e.g. `2*H`); non-integer factors produce `?`.

When neither `size` nor `scale_factor` can be resolved, no hover is emitted (silent pass).

```python
y = F.interpolate(x, size=(64, 64), mode="bilinear")   # [B, C, H, W] → [B, C, 64, 64]
y = F.interpolate(x, scale_factor=2.0)                  # [B, C, 16, 16] → [B, C, 32, 32]
y = F.interpolate(x, size=labels.shape[-2:])             # size from another tensor's shape
```

---

## Advanced indexing and selection

### `x.diagonal` / `torch.diagonal`

```python
x.diagonal(offset=0, dim1=0, dim2=1)
torch.diagonal(x, offset=0, dim1=0, dim2=1)
```

Removes `dim1` and `dim2` from the shape and appends the diagonal length.
When both dimensions are constants, diagonal length = `max(0, min(d1, d2) - |offset|)`.
When both dimensions are the same symbolic dim and `offset=0`, the diagonal length equals that dim.
Otherwise, the diagonal length is `?`.

Input: `(*dims)` — rank ≥ 2.
Output: `(*remaining, diag_len)`

```python
y = x.diagonal(dim1=-2, dim2=-1)   # [B, 64, 64] → [B, 64]
```

### `x.index_select` / `torch.index_select`

```python
x.index_select(dim, index)
torch.index_select(x, dim, index)
```

Replaces `dims[dim]` with the number of elements in `index` (its first dimension).
When `index` is a known-shape 1-D tensor, the exact count is tracked; otherwise `?`.

Input: `(*dims)`, index: `(K,)`
Output: `(*dims)` with `dims[dim]` replaced by `K`.

```python
idx: Annotated[torch.Tensor, Shape(10)]
y = x.index_select(1, idx)   # [B, 64, H] → [B, 10, H]
```

### `torch.topk`

```python
torch.topk(input, k, dim=-1, largest=True, sorted=True)
x.topk(k, dim=-1)
```

Both the `values` and `indices` output tensors have the same shape: the selected dimension becomes `k`.
Accessing `.values` or `.indices` on the result is handled — the shape is preserved.

Input: `(*dims)`
Output: `(*dims[:dim], k, *dims[dim+1:])`

```python
top = torch.topk(x, k=10, dim=-1).values   # [B, 256] → [B, 10]
y = top.mean(dim=-1)                         # [B, 10] → [B]
```

### `torch.bincount`

```python
torch.bincount(input, weights=None, minlength=0)
```

Returns a 1-D tensor whose length depends on the maximum value in `input` — not statically determinable.
Shape is reported as `[?]`.

---

## Indexing and slicing

```python
x[0]          # integer index — removes that dimension
x[1:5]        # slice with constant bounds — size tracked (5-1 = 4)
x[1:]         # open-ended slice — original dimension preserved
x[None]       # newaxis — inserts a size-1 dimension
x[...]        # ellipsis — passes through all remaining dimensions
x[0, :, None] # combinations of the above
```

---

## Tensor attributes

```python
x.shape        # → ShapeTupleValue of all dims
x.ndim         # → IntegerValue(rank)
x.shape[i]     # → the i-th Dim (supports negative indices)
```

---

## `nn` modules

### Shape-preserving modules

The following module types pass the input shape through unchanged:

`BatchNorm1d`, `BatchNorm2d`, `BatchNorm3d`, `LayerNorm`, `Dropout`, `Dropout2d`, `Dropout3d`,
`ReLU`, `LeakyReLU`, `GELU`, `SiLU`, `Sigmoid`, `Tanh`, `ELU`, `SELU`, `PReLU`, `Mish`,
`Hardswish`, `Hardsigmoid`, `Identity`, `Softmax`

```python
class Net(nn.Module):
    def __init__(self):
        self.bn   = nn.BatchNorm2d(64)
        self.act  = nn.GELU()
        self.drop = nn.Dropout(0.1)

    def forward(self, x: Annotated[torch.Tensor, Shape("B", 64, "H", "W")]):
        y = self.bn(x)    # [B, 64, H, W]
        z = self.act(y)   # [B, 64, H, W]
        w = self.drop(z)  # [B, 64, H, W]
```

Module aliases are fully supported — `act = self.act; y = act(x)` works identically.

### `nn.Embedding`

```python
nn.Embedding(num_embeddings, embedding_dim)
```

Input: `(*indices)` — any rank.
Output: `(*indices, embedding_dim)` — the embedding dimension is appended as a new trailing axis.

```python
class Net(nn.Module):
    def __init__(self):
        self.emb = nn.Embedding(10000, 512)

    def forward(self, x: Annotated[torch.Tensor, Shape("B", "T")]):
        y = self.emb(x)   # [B, T, 512]
```

### `nn.Linear`

```python
nn.Linear(in_features, out_features)
```

Input: `(..., in_features)`
Output: `(..., out_features)` — all leading dimensions are preserved.

If the last input dimension is a `ConstantDim`, it is validated against `in_features`; a mismatch emits `TSF1007`. A symbolic last dim skips the check and still propagates `out_features`.

### `nn.Conv2d`

```python
nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1)
```

Input: `(N, C_in, H, W)` — must be rank 4.
Output: `(N, C_out, H_out, W_out)`

Formula: `H_out = floor((H + 2·padding − dilation·(kernel−1) − 1) / stride + 1)`

If the channel dimension is a `ConstantDim`, it is validated against `in_channels`; a mismatch emits `TSF1007`. A symbolic channel dim skips the check and still propagates the output shape.

### `nn.MaxPool2d` / `nn.AvgPool2d`

```python
nn.MaxPool2d(kernel_size, stride=None, padding=0, dilation=1)
nn.AvgPool2d(kernel_size, stride=None, padding=0)
```

Input: `(N, C, H, W)` — must be rank 4.
Output: `(N, C, H_out, W_out)` — `N` and `C` are preserved.

Formula: `H_out = floor((H + 2·padding − dilation·(kernel−1) − 1) / stride + 1)`

When `stride` is omitted, PyTorch defaults it to `kernel_size` (both layers).
`nn.AvgPool2d` has no `dilation` parameter; it is implicitly `(1, 1)`.

```python
class Net(nn.Module):
    def __init__(self):
        self.pool = nn.MaxPool2d(2)           # stride defaults to 2

    def forward(self, x: Annotated[torch.Tensor, Shape("B", "C", 32, 32)]):
        y = self.pool(x)   # [B, C, 16, 16]
```

### `nn.Sequential`

```python
nn.Sequential(layer1, layer2, ...)
```

Shape is propagated through each sub-module in order. Supported sub-module types are the same as those recognized directly (Linear, Conv2d, MaxPool2d, AvgPool2d, Embedding, passthrough activations/norms).

```python
class Net(nn.Module):
    def __init__(self):
        self.net = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 16),
        )

    def forward(self, x: Annotated[torch.Tensor, Shape("B", 128)]):
        y = self.net(x)   # [B, 16]
```

### `nn.MultiheadAttention`

```python
nn.MultiheadAttention(embed_dim, num_heads, dropout=0.0, ..., batch_first=False)
```

When called with `(query, key, value)`, returns a tuple `(output, attn_weights)`:

- `output`: same shape as `query`.
- `attn_weights`: shape is not tracked statically (returned as `[?, ?, ?]`).

Tuple unpacking is supported:

```python
class Net(nn.Module):
    def __init__(self):
        self.attn = nn.MultiheadAttention(64, 8, batch_first=True)
        self.proj = nn.Linear(64, 32)

    def forward(self, x: Annotated[torch.Tensor, Shape("B", "T", 64)]):
        out, _ = self.attn(x, x, x)   # out: [B, T, 64]
        y = self.proj(out)             # [B, T, 32]
```

---

## Return shape validation

When a function's return type is annotated with `Shape(...)`, the inferred shape of the return expression is compared against the declared shape. A mismatch raises `TSF1009`.

Only definite mismatches are reported (rank difference, or a constant-vs-constant dimension pair that differs). Symbolic dimensions are never flagged.

```python
def fn(x: Annotated[torch.Tensor, Shape("B", 128)]) -> Annotated[torch.Tensor, Shape("B", 64)]:
    return x   # TSF1009: Return shape [B, 128] does not match declared [B, 64]
```

---

## Near-term roadmap

Operator additions are driven by gaps in real PyTorch model code. Next likely additions:

- `torch.einsum` with ellipsis (`...`) and implicit mode
- `nn.MultiheadAttention` attention weight shape tracking
- `nn.Sequential` with `OrderedDict` argument form

Every new operator requires tests before it ships. See [Development](development.md#adding-a-new-operator).
