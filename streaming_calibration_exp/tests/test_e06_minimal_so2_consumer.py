from __future__ import annotations

import math

import torch

from src.models.components.streaming_spint import MinimalSO2EquivariantConsumer


def _consumer() -> MinimalSO2EquivariantConsumer:
    torch.manual_seed(42)
    return MinimalSO2EquivariantConsumer(
        window_size=5,
        hidden_dim=16,
        side_mean=[0.4, -0.3, 1.2, 2.0],
        side_std=[1.5, 0.8, 0.7, 1.1],
        behavior_mean=[0.2, -0.4],
        behavior_std=[2.0, 0.6],
        behavior_scaling_factor=5.0,
    )


def _normalized_t4(consumer: MinimalSO2EquivariantConsumer) -> torch.Tensor:
    raw = torch.tensor(
        [
            [
                [1.0, 0.5, math.sqrt(1.25), 2.2],
                [-0.4, 1.2, math.sqrt(1.6), 1.7],
                [0.7, -1.1, math.sqrt(1.7), 2.8],
            ],
            [
                [0.3, -0.8, math.sqrt(0.73), 1.4],
                [1.4, 0.2, math.sqrt(2.0), 2.5],
                [-0.9, -0.6, math.sqrt(1.17), 2.1],
            ],
        ],
        dtype=torch.float32,
    )
    return (raw - consumer.side_mean) / consumer.side_std


def _rotate_normalized_t4(
    consumer: MinimalSO2EquivariantConsumer,
    side: torch.Tensor,
    angle: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    rotation = torch.tensor(
        [
            [math.cos(angle), -math.sin(angle)],
            [math.sin(angle), math.cos(angle)],
        ],
        dtype=side.dtype,
    )
    raw = side * consumer.side_std + consumer.side_mean
    rotated = raw.clone()
    rotated[..., :2] = raw[..., :2] @ rotation.T
    return (rotated - consumer.side_mean) / consumer.side_std, rotation


def test_minimal_so2_consumer_is_physically_equivariant() -> None:
    consumer = _consumer()
    neural = torch.randn(2, 5, 3)
    identity = torch.randn(2, 3, 5)
    side = _normalized_t4(consumer)
    rotated_side, rotation = _rotate_normalized_t4(consumer, side, angle=0.73)

    base = consumer.physical_output(consumer(neural, identity, side))
    rotated = consumer.physical_output(consumer(neural, identity, rotated_side))

    assert torch.allclose(rotated, base @ rotation.T, atol=2.0e-6, rtol=2.0e-6)
    _, invariant = consumer.carrier_geometry(side)
    _, rotated_invariant = consumer.carrier_geometry(rotated_side)
    assert torch.allclose(rotated_invariant, invariant, atol=2.0e-6, rtol=2.0e-6)


def test_minimal_so2_consumer_is_neuron_permutation_invariant() -> None:
    consumer = _consumer()
    neural = torch.randn(2, 5, 3)
    identity = torch.randn(2, 3, 5)
    side = _normalized_t4(consumer)
    permutation = torch.tensor([2, 0, 1])

    base = consumer(neural, identity, side)
    permuted = consumer(
        neural.index_select(2, permutation),
        identity.index_select(1, permutation),
        side.index_select(1, permutation),
    )

    assert torch.allclose(permuted, base, atol=1.0e-6, rtol=1.0e-6)


def test_minimal_so2_consumer_backpropagates_only_through_scalar_network() -> None:
    consumer = _consumer()
    neural = torch.randn(2, 5, 3)
    identity = torch.randn(2, 3, 5)
    loss = consumer(neural, identity, _normalized_t4(consumer)).square().mean()

    loss.backward()

    assert all(parameter.grad is not None for parameter in consumer.scalar_net.parameters())
