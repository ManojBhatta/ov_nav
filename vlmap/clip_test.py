"""
CLIP patch-level cosine similarity visualiser
Corrected version of clip_test.ipynb

Fixes applied
─────────────
1. Cosine similarity was inverted (1/sim instead of sim) — removed.
2. query_feat.device called on a numpy array after .cpu().numpy() → AttributeError — removed.
3. Plot title was hardcoded as "a red chair" while the query was "curtains" — made dynamic.
4. depth_t device was hardcoded to 'cuda' — now uses the `device` variable.
5. Added a 4th panel: patch similarity map bicubic-interpolated to 224×224 and
   alpha-blended over the RGB image so spatial correspondence is clear.
"""

import os
import math

import numpy as np
import cv2
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt
from  matplotlib.colorizer  import _ScalarMappable

# import matplotlib.colormaps as cm
import clip

# ── change these paths to point at your data ──────────────────────────────────
DATA_DIR   = "/tmp/vlmap_recording/test7"
DEPTH_FILE = os.path.join(DATA_DIR, "depth", "test7_000039.npy")
RGB_FILE   = os.path.join(DATA_DIR, "rgb",   "test7_000039.png")
QUERY      = "rug"
OVERLAY_ALPHA = 0.55   # opacity of the heatmap layer in the 4th panel
# ──────────────────────────────────────────────────────────────────────────────


# ---------------------------------------------------------------------------
# Model setup
# ---------------------------------------------------------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

clip_version  = "ViT-B/16"
clip_feat_dim = {"ViT-B/32": 512, "ViT-B/16": 512, "ViT-L/14": 768}[clip_version]
grid_dim      = {"ViT-B/32": 7,   "ViT-B/16": 14,  "ViT-L/14": 16 }[clip_version]

print("Loading CLIP model …")
clip_model, preprocess = clip.load(clip_version)
clip_model.to(device).eval()


# ---------------------------------------------------------------------------
# Patch-embedding extractor
# ---------------------------------------------------------------------------
def get_patch_embeddings(model, image_tensor):
    """
    Hook the final transformer block to capture per-patch tokens,
    project them into CLIP's shared embedding space, and L2-normalise.

    Returns
    -------
    patches_normalized : torch.Tensor, shape [num_patches, feat_dim]
        One unit-length embedding per spatial patch (all on CPU).
    """
    patch_tokens_cache = []
    dev = next(model.parameters()).device

    def _capture(module, inp, output):
        # output: [seq_len, batch, hidden_dim]  (seq_len = 1+num_patches)
        # output[0] = CLS token  →  skip it
        patch_tokens_cache.append(output[1:, :, :].detach().cpu())

    hook = model.visual.transformer.resblocks[-1].register_forward_hook(_capture)

    with torch.no_grad():
        model.encode_image(image_tensor)   # triggers the hook

    hook.remove()

    # [num_patches, 1, hidden_dim]  →  squeeze batch dim  →  [num_patches, hidden_dim]
    patches = torch.cat([p.squeeze(1) for p in patch_tokens_cache], dim=0)
    patches = model.visual.ln_post(patches.to(dev))

    num_patches = patches.shape[0]
    g = int(num_patches ** 0.5)
    print(f"Patch grid: {g}×{g} = {num_patches} patches,  hidden dim: {patches.shape[1]}")

    # Project into shared 512-dim (or 768-dim) CLIP space
    patches_projected = patches.float() @ model.visual.proj.float()   # [N, feat_dim]
    print(f"Projected shape: {patches_projected.shape}")

    # L2-normalise so dot-product == cosine similarity
    patches_normalized = patches_projected / patches_projected.norm(dim=-1, keepdim=True)
    return patches_normalized.cpu()   # [num_patches, feat_dim]


# ---------------------------------------------------------------------------
# Cosine-similarity grid
# ---------------------------------------------------------------------------
def create_similarity_grid(patch_feats_np, query_text):
    """
    Compute cosine similarity between each patch and the text query.

    patch_feats_np : np.ndarray  [num_patches, feat_dim]  (already L2-normed)
    query_text     : str

    Returns
    -------
    similarity_grid : np.ndarray  [grid_dim, grid_dim]  values in [-1, 1]
    """
    query_tokens = clip.tokenize([query_text]).to(device)

    with torch.no_grad():
        query_feat = clip_model.encode_text(query_tokens)        # [1, feat_dim]
        query_feat = query_feat / query_feat.norm(dim=-1, keepdim=True)  # unit vec

    # ── BUG FIX: dot product of two unit vectors IS cosine similarity.
    #    The original code did  1/(sim + eps)  which INVERTS the score.
    query_np   = query_feat.float().cpu().numpy()                # [1, feat_dim]
    similarity = (patch_feats_np @ query_np.T).squeeze(-1)      # [num_patches]

    similarity_grid = similarity.reshape(grid_dim, grid_dim)     # [g, g]
    print(f"Similarity range: [{similarity_grid.min():.4f}, {similarity_grid.max():.4f}]")
    return similarity_grid


# ---------------------------------------------------------------------------
# Depth loader  (plain .npy)
# ---------------------------------------------------------------------------
def load_depth(path):
    return np.load(path).astype(np.float32)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    # ── load data ──────────────────────────────────────────────────────────
    rgb_pil = Image.open(RGB_FILE).convert("RGB")
    depth   = load_depth(DEPTH_FILE)

    # ── patch embeddings ────────────────────────────────────────────────────
    # ── BUG FIX: use `device` variable, not hardcoded 'cuda'
    depth_t = torch.from_numpy(depth).reshape(1, 1, *depth.shape).to(device)
    depth_t = F.interpolate(depth_t, size=(grid_dim, grid_dim), mode="nearest")

    image_tensor = preprocess(rgb_pil).unsqueeze(0).to(device)  # [1, 3, 224, 224]
    patch_feats  = get_patch_embeddings(clip_model, image_tensor)  # [N, feat_dim]

    # ── cosine similarity grid ──────────────────────────────────────────────
    sim_grid = create_similarity_grid(patch_feats.detach().cpu().numpy(), QUERY)  # [g, g]

    # ── interpolate patch grid → 224×224 for overlay ────────────────────────
    sim_t    = torch.from_numpy(sim_grid).unsqueeze(0).unsqueeze(0)   # [1,1,g,g]
    sim_224  = F.interpolate(sim_t, size=(224, 224), mode="bicubic",
                             align_corners=False).squeeze().numpy()    # [224,224]

    # Scale to [0,1] for colormap
    vmin, vmax = sim_224.min(), sim_224.max()
    sim_norm = (sim_224 - vmin) / (vmax - vmin + 1e-8)

    # Build RGBA heatmap and blend over the resized RGB
    colormap   = plt.get_cmap("jet_r")
    heatmap_rgba = colormap(sim_norm)                     # [224,224,4]

    rgb_224 = np.array(rgb_pil.resize((224, 224))) / 255.0  # [224,224,3]
    overlay = (1 - OVERLAY_ALPHA) * rgb_224 + OVERLAY_ALPHA * heatmap_rgba[:, :, :3]
    overlay = np.clip(overlay, 0, 1)

    # ── plot ─────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 4, figsize=(18, 4.5))
    fig.suptitle(f"CLIP patch cosine similarity — query: \"{QUERY}\"", fontsize=13)

    # Panel 1 – RGB
    axes[0].imshow(rgb_pil)
    axes[0].set_title("RGB Image")
    axes[0].axis("off")

    # Panel 2 – Depth
    im2 = axes[1].imshow(depth, cmap="plasma")
    axes[1].set_title("Depth Image")
    axes[1].axis("off")
    fig.colorbar(im2, ax=axes[1], fraction=0.046, pad=0.04, label="m")

    # Panel 3 – Raw patch similarity grid (g × g)
    im3 = axes[2].imshow(sim_grid, cmap="jet_r",
                         vmin=sim_grid.min(), vmax=sim_grid.max())
    axes[2].set_title(f"Similarity grid ({grid_dim}×{grid_dim})")
    axes[2].axis("off")
    fig.colorbar(im3, ax=axes[2], fraction=0.046, pad=0.04, label="cosine sim")

    # Panel 4 – Bicubic-interpolated similarity overlaid on RGB at 224×224
    axes[3].imshow(overlay)
    # add a semi-transparent colourbar scaled to actual similarity values
    sm = _ScalarMappable(cmap="jet_r",
                           norm=plt.Normalize(vmin=vmin, vmax=vmax))
    sm.set_array([])
    fig.colorbar(sm, ax=axes[3], fraction=0.046, pad=0.04, label="cosine sim")
    axes[3].set_title("Interpolated similarity\noverlay (224×224)")
    axes[3].axis("off")

    plt.tight_layout()
    out_path = "clip_similarity_output.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved figure to {out_path}")
    plt.show()


if __name__ == "__main__":
    main()