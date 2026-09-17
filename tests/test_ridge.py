import torch

from kvtransfer.ridge import MomentAccumulator, r2_score


def _planted(n=4000, p=12, q=6, noise=0.01, shift=50.0, seed=0):
    g = torch.Generator().manual_seed(seed)
    W = torch.randn(p, q, generator=g)
    b = torch.randn(q, generator=g)
    X = torch.randn(n, p, generator=g) + shift  # large mean: exercises the cancellation guard
    Y = X @ W + b + noise * torch.randn(n, q, generator=g)
    return X, Y, W, b


def test_streaming_ridge_recovers_planted_affine_map():
    X, Y, W, b = _planted()
    acc = MomentAccumulator(X.shape[1], Y.shape[1])
    for i in range(0, X.shape[0], 500):
        acc.update(X[i:i + 500], Y[i:i + 500])
    W_hat, b_hat, r2 = acc.solve(lam=1e-6)
    assert torch.allclose(W_hat, W, atol=2e-2)
    assert torch.allclose(b_hat, b, atol=1.0)  # bias absorbs shift*W error; check predictions instead
    Y_hat = X @ W_hat + b_hat
    assert r2_score(Y, Y_hat) > 0.999
    assert abs(r2 - r2_score(Y, Y_hat)) < 1e-3


def test_block_solve_matches_full_solve_on_subblock():
    X, Y, W, b = _planted(p=16, q=8)
    acc = MomentAccumulator(16, 8)
    acc.update(X, Y)
    rows = torch.tensor([0, 1, 2, 3, 8, 9])
    cols = torch.tensor([1, 5])
    W_blk, b_blk, r2_blk = acc.solve(0.01, rows=rows, cols=cols)
    sub = MomentAccumulator(6, 2)
    sub.update(X[:, rows], Y[:, cols])
    W_sub, b_sub, r2_sub = sub.solve(0.01)
    assert torch.allclose(W_blk, W_sub, atol=1e-4)
    assert torch.allclose(b_blk, b_sub, atol=1e-3)
    assert abs(r2_blk - r2_sub) < 1e-6


def test_state_dict_round_trip():
    X, Y, _, _ = _planted(n=300)
    acc = MomentAccumulator(12, 6)
    acc.update(X[:100], Y[:100])
    acc.update(X[100:], Y[100:])
    acc2 = MomentAccumulator.from_state_dict(acc.state_dict())
    W1, b1, r1 = acc.solve(0.01)
    W2, b2, r2 = acc2.solve(0.01)
    assert torch.equal(W1, W2) and torch.equal(b1, b2) and r1 == r2


def test_r2_of_pure_noise_is_near_zero():
    g = torch.Generator().manual_seed(1)
    X = torch.randn(5000, 8, generator=g)
    Y = torch.randn(5000, 4, generator=g)
    acc = MomentAccumulator(8, 4)
    acc.update(X, Y)
    _, _, r2 = acc.solve(0.0)
    assert -0.01 < r2 < 0.02
