# Multi-Head MLP-Mixer for PDE Solving

A novel neural operator architecture combining **MLP-Mixer** with **multi-resolution pyramid decomposition** for solving Partial Differential Equations (PDEs). Designed and implemented during a research internship at the **Weierstrass Institute for Applied Analysis and Stochastics (WIAS), Berlin**.

---

## Motivation

Standard neural PDE solvers struggle on high-frequency solutions — they learn low-frequency components well but fail to capture fine-scale structure. This project tackles that limitation with a **pyramid approach**: the solution is decomposed across multiple resolutions, and a custom MLP-Mixer-based operator learns the residuals at each level.

---

## Architecture

### `Tile_Operator` — Core Neural Operator
The central building block. Takes a coarse input field and produces a refined output at a target resolution using:

- **Patch tokenization:** the 2D input field is divided into non-overlapping patches, each flattened into a token
- **Parallel MLP streams:** `num_parallel` independent embedding streams process tokens simultaneously, allowing the model to capture multiple solution components in a single forward pass
- **`MixerLayer`:** alternates between token-mixing (cross-patch interaction) and channel-mixing (within-patch feature transformation), following the MLP-Mixer paradigm
- **Attention-style projection heads:** K projection heads compute weighted combinations of token and channel features via learned logit/value weights — a lightweight attention mechanism without the quadratic cost of self-attention

### Multi-Resolution Pyramid
The full model is a `Tile_Operator` pyramid trained **stage-wise**:

```
Input (coarse) → Level 1 → Level 2 → ... → Level 7 → Full resolution output
```

Each level receives the upsampled output of the previous level and predicts the residual at its own resolution. Multi-resolution stacks are constructed via `avg_pool2d` downsampling.

### Custom Loss: H1 Norm
Training uses a **finite-difference H1 loss** instead of standard MSE:

```
H1(u) ≈ ∫ (u² + λ·|∇u|²) dx
```

This penalizes not just pointwise error but also gradient mismatch — critical for PDE solutions where smoothness is a physical requirement.

---

## Key Design Choices

| Choice | Rationale |
|---|---|
| Parallel MLP streams | Captures multiple frequency components simultaneously |
| Stage-wise pyramid training | Prevents gradient vanishing in deep multi-scale architectures |
| H1 loss instead of L2 | Enforces derivative accuracy, not just value accuracy |
| Learnable `gamma` scaling | Controls contribution of token vs channel features adaptively |

---

## Results

- **67% lower L2 error** compared to baseline on stochastic PDE benchmarks
- **100x data reduction** (10,000 → 100 training samples) via inductive bias design
- **7x model compression** (7M → 1M parameters) with a simultaneous **31% H1 error improvement**

*Part of a first-author manuscript in preparation (supervisors: M. Eigel, J. Schütte, WIAS Berlin).*

---

## Usage

```python
model = Tile_Operator(
    patch=4,
    H=64, W=64,
    embed_dim=32,
    token_hidden=64,
    channel_hidden=64,
    num_parallel=4,
    proj_heads=4,
    num_inputs=1
)
```

Training is done stage-wise via `train_pyramid_stagewise()` — see notebook for full training loop.

---

## Requirements

```
pip install -r requirements.txt
```

---

## Related

- [Earthquake-Diffusion-Model](../Earthquake-Diffusion-Model) — another project using similar operator-inspired design principles
- MLP-Mixer (Tolstikhin et al., 2021)
- Fourier Neural Operator (Li et al., 2020)
