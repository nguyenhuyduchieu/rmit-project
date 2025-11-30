import numpy as np

def compute_pnl_sharpe(preds: np.ndarray, trues: np.ndarray):
    # preds, trues: [N, H, 1] prices (normalized). Use horizon-1 step return.
    # Strategy: long if median forecast > last observed; short if <.
    N, H, _ = preds.shape
    # approximate next-step returns from trues
    # r_t+1 = (y_{t+1} - y_t) / |y_t| (avoid div by 0)
    y0 = trues[:, 0, 0]
    y1 = trues[:, 1, 0]
    ret = (y1 - y0) / (np.abs(y0) + 1e-8)
    # signal from median quantile vs point
    median_pred = preds[:, 0, 0]
    signal = np.sign(median_pred - y0)
    pnl = signal * ret
    mean = pnl.mean()
    std = pnl.std() + 1e-12
    sharpe = mean / std
    return {
        'pnl_mean': float(mean),
        'pnl_std': float(std),
        'sharpe': float(sharpe)
    }
