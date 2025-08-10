import torch
import pytest

from ml.utils.graph_smooth import knn_graph, graph_laplacian_penalty


def test_knn_graph_shapes_single_batch():
    theta = torch.randn(10, 3)
    idx, w = knn_graph(theta, k=4, sigma=0.3)
    assert idx.shape == (10, 4)
    assert w.shape == (10, 4)
    assert idx.dtype == torch.long
    assert (w >= 0).all() and (w <= 1).all()


def test_knn_graph_shapes_batched():
    theta = torch.randn(2, 6, 2)
    idx, w = knn_graph(theta, k=3, sigma=0.5)
    assert idx.shape == (2, 6, 3)
    assert w.shape == (2, 6, 3)


def test_knn_graph_k_limits():
    theta = torch.randn(5, 2)
    idx, _ = knn_graph(theta, k=10, sigma=0.25)
    # cannot exceed N-1
    assert idx.shape[-1] == 4


def test_knn_graph_trivial():
    theta = torch.randn(1, 2)
    idx, w = knn_graph(theta, k=4, sigma=0.25)
    assert idx.numel() == 0
    assert w.numel() == 0


def test_graph_laplacian_penalty_scalar():
    theta = torch.tensor([[0.0], [1.0], [2.0]])
    idx, w = knn_graph(theta, k=1, sigma=1.0)

    # y linearly increasing with theta
    y = torch.tensor([0.0, 1.0, 2.0])
    pen = graph_laplacian_penalty(y, idx, w)
    assert pen > 0

    # constant y -> zero penalty
    y_const = torch.zeros(3)
    pen_const = graph_laplacian_penalty(y_const, idx, w)
    assert torch.isclose(pen_const, torch.tensor(0.0))


def test_graph_laplacian_penalty_vector():
    theta = torch.randn(2, 5, 3)
    idx, w = knn_graph(theta, k=2, sigma=0.5)
    y = torch.randn(2, 5, 7)
    pen = graph_laplacian_penalty(y, idx, w)
    assert pen.dim() == 0
    assert pen >= 0


def test_invalid_sigma():
    theta = torch.randn(4, 2)
    with pytest.raises(ValueError):
        _ = knn_graph(theta, k=2, sigma=0.0)
