# Supported Operators

Shape notation: uppercase strings are symbolic dimensions (`B`, `T`, `D`), integers are constants.

## Tensor methods

### `reshape` / `view`

```python
x.reshape(*shape)
x.view(*shape)
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

### `size`

```python
x.size()       # → ShapeTupleValue (all dims)
x.size(dim)    # → IntegerValue for that axis (constant dims only; symbolic → None)
```

### `matmul`

```python
x.matmul(y)
torch.matmul(x, y)
torch.bmm(x, y)
```

Input: `(*, M, K)` and `(*, K, N)` (both rank ≥ 2)
Output: `(*, M, N)` — batch dimensions are broadcast.

---

## Module-level functions

### `torch.reshape`

```python
torch.reshape(x, shape)
```

Same semantics as `x.reshape(shape)`.

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

---

## Broadcasting operators

```python
x + y,  x - y,  x * y,  x / y,  x // y,  x % y,  x ** y
```

Input: any two tensors with broadcast-compatible shapes (NumPy/PyTorch rules).
Output: broadcast result shape. Incompatible shapes emit `TSF1006`.

---

## Indexing and slicing

```python
x[0]          # integer index — removes that dimension
x[1:5]        # slice — keeps that dimension (size not tracked)
x[None]       # newaxis — inserts a size-1 dimension
x[...]        # ellipsis — passes through all remaining dimensions
x[0, :, None] # combinations of the above
```

---

## `nn.Linear`

```python
nn.Linear(in_features, out_features)
```

Input: `(..., in_features)`
Output: `(..., out_features)` — all leading dimensions are preserved.

The last input dimension must be a `ConstantDim` equal to `in_features`. A symbolic last dim does not match. Shape mismatch emits `TSF1007`.

---

## `nn.Conv2d`

```python
nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1)
```

Input: `(N, C_in, H, W)` — must be rank 4.
Output: `(N, C_out, H_out, W_out)`

Formula: `H_out = floor((H + 2·padding − dilation·(kernel−1) − 1) / stride + 1)`

Channel dimension must equal `in_channels` (constant match). Shape mismatch emits `TSF1007`.

---

## Tensor attributes

```python
x.shape        # → ShapeTupleValue of all dims
x.ndim         # → IntegerValue(rank)
x.shape[i]     # → the i-th Dim (supports negative indices)
```

---

## Near-term roadmap

Operator additions are driven by gaps in real PyTorch model code. Next likely additions:

- Reduction ops: `sum`, `mean`, `amax`, `amin` (with `dim=` and `keepdim=`)
- `movedim`, `mm`
- Pooling layers: `nn.MaxPool2d`, `nn.AvgPool2d`
- `nn.Embedding`, `nn.LayerNorm`, `nn.BatchNorm2d`
- Limited `einsum` support
- Broader slice / index coverage

Every new operator requires tests before it ships. See [Development](development.md#adding-a-new-operator).
