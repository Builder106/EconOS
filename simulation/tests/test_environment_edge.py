"""Edge case tests for MarketEnv simulation and economic logic functions."""
import numpy as np
import pytest

from simulation.environment import MarketEnv
from simulation.logic import (
    calculate_cpi,
    calculate_gini,
    calculate_lorenz_curve,
    calculate_real_gdp,
)


def test_zero_income_tax_payment():
    """Verify zero labor results in zero gross income and zero tax paid."""
    env = MarketEnv(num_consumers=3, num_producers=1, tax_rate=0.25)
    env.reset()
    initial_treasury = env.treasury

    # Consumers choose 0 labor, 0 consumption intent
    actions = {
        "consumer_0": np.array([0.0, 0.0], dtype=np.float32),
        "consumer_1": np.array([0.0, 0.0], dtype=np.float32),
        "consumer_2": np.array([0.0, 0.0], dtype=np.float32),
        "producer_0": np.array([0.5, 0.5], dtype=np.float32),
    }

    _obs, rewards, _terminations, _truncations, _infos = env.step(actions)
    assert env.treasury == initial_treasury == 0.0
    for i in range(3):
        assert rewards[f"consumer_{i}"] == pytest.approx(0.0)


def test_zero_producer_economy():
    """Verify simulation functions properly with zero producers."""
    env = MarketEnv(num_consumers=4, num_producers=0)
    obs, _info = env.reset()
    assert len(env.agents) == 4
    assert all("consumer" in a for a in env.agents)

    actions = {a: env.action_space(a).sample() for a in env.agents}
    obs, rewards, _terminations, _truncations, _infos = env.step(actions)
    assert len(obs) == 4
    assert len(rewards) == 4


def test_empty_actions_dict():
    """Verify empty actions dict sets agents to empty list and returns empty dicts."""
    env = MarketEnv(num_consumers=2, num_producers=1)
    env.reset()
    obs, rewards, terminations, truncations, infos = env.step({})
    assert env.agents == []
    assert obs == {}
    assert rewards == {}
    assert terminations == {}
    assert truncations == {}
    assert infos == {}


def test_100_cycle_truncation():
    """Verify 100-cycle truncation triggers after max_cycles steps."""
    env = MarketEnv(num_consumers=2, num_producers=1, max_cycles=100)
    env.reset()

    actions = {a: np.array([0.5, 0.5], dtype=np.float32) for a in env.agents}
    for step_num in range(1, 101):
        _obs, _rewards, _terminations, truncations, _infos = env.step(actions)
        if step_num < 100:
            assert not any(truncations.values())
        else:
            assert all(truncations.values())
            assert env.num_cycles == 100


def test_redistribute_treasury_empty_or_zero():
    """Verify treasury redistribution edge cases when empty or no consumers."""
    # Treasury is zero
    env = MarketEnv(num_consumers=2, num_producers=1)
    env.reset()
    assert env.redistribute_treasury() == (0.0, 0.0)

    # No consumers
    env_no_c = MarketEnv(num_consumers=0, num_producers=2)
    env_no_c.reset()
    env_no_c.treasury = 100.0
    assert env_no_c.redistribute_treasury() == (0.0, 0.0)


def test_gini_negative_and_zero_wealth():
    """Verify Gini calculation handles negative wealth, zero sum, and empty inputs."""
    assert calculate_gini([]) == 0
    assert calculate_gini([0, 0, 0]) == 0
    assert calculate_gini([-10, 10]) == 0

    # Negative wealth values with non-zero sum
    g = calculate_gini([-50, 100, 200])
    assert isinstance(g, (float, np.floating))


def test_lorenz_curve_empty_and_zero():
    """Verify Lorenz curve calculation handles empty input and all-zero wealth."""
    empty_result = calculate_lorenz_curve([])
    assert empty_result == {"quantiles": [0.0, 1.0], "cumulative_wealth": [0.0, 1.0]}

    zero_result = calculate_lorenz_curve([0, 0, 0])
    assert zero_result == {"quantiles": [0.0, 1.0], "cumulative_wealth": [0.0, 1.0]}


def test_cpi_and_real_gdp_zero_and_invalid_base():
    """Verify CPI and Real GDP calculations with zero or negative price/base."""
    assert calculate_cpi(10.0, base_price=0.0) == 100.0
    assert calculate_cpi(10.0, base_price=-5.0) == 100.0

    # Zero CPI -> Real GDP returns 0.0
    assert calculate_cpi(0.0, base_price=10.0) == 0.0
    assert calculate_real_gdp(100.0, current_price=0.0, base_price=10.0) == 0.0
