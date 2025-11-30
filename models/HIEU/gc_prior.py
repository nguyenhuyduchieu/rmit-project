import numpy as np
import pandas as pd

def rolling_corr_adj(X: np.ndarray, win: int = 256, eps: float = 1e-8):
    """Build symmetric adjacency from rolling correlation magnitude over last win samples.
    X: [T, N] series per coin. Returns A_prior [N, N] in [0,1]."""
    T, N = X.shape
    win = min(win, T)
    Xw = X[-win:]
    C = np.corrcoef(Xw.T)
    C = np.nan_to_num(C)
    A = np.abs(C)
    np.fill_diagonal(A, 0.0)
    # normalize to [0,1]
    A = (A - A.min()) / (A.max() - A.min() + eps)
    return A

def gc_features_from_adj(A: np.ndarray):
    deg_in = A.sum(axis=1)
    deg_out = A.sum(axis=0)
    asym = deg_in - deg_out
    return np.stack([deg_in, deg_out, asym], axis=-1)

def build_gc_priors(X: np.ndarray, Lg: int = 1500):
    """Placeholder for BigVAR/BC/TY pipeline: return prior adjacency and simple GC-features.
    For now, use rolling correlation over last Lg as a proxy prior.
    X: [T, N]"""
    A = rolling_corr_adj(X, win=min(Lg, len(X)))
    feats = gc_features_from_adj(A)
    return A, feats
