import os
import sys
import numpy as np
import numpy.testing as npt

# Ensure repository root is on sys.path so tests can import local modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from sim_utils import (
    run_monte_carlo,
    run_custom_scenario,
    summarize_ghost_frac,
    prob_ghost_share,
)


def test_run_monte_carlo_deterministic_shape_and_repeatability():
    # small-size run to keep tests fast
    n_towns = 50
    n_sims = 10
    start_year = 2023
    end_year = 2025

    out1 = run_monte_carlo(
        n_towns=n_towns,
        n_sims=n_sims,
        start_year=start_year,
        end_year=end_year,
        initial_mean=0.7,
        initial_std=0.05,
        mean_decline_slow=-0.005,
        mean_decline_base=-0.02,
        mean_decline_rapid=-0.03,
        p_slow=0.3,
        p_base=0.5,
        p_rapid=0.2,
        sigma_viability=0.02,
        ghost_threshold=0.3,
        seed=123,
        track_towns=False,
    )

    out2 = run_monte_carlo(
        n_towns=n_towns,
        n_sims=n_sims,
        start_year=start_year,
        end_year=end_year,
        initial_mean=0.7,
        initial_std=0.05,
        mean_decline_slow=-0.005,
        mean_decline_base=-0.02,
        mean_decline_rapid=-0.03,
        p_slow=0.3,
        p_base=0.5,
        p_rapid=0.2,
        sigma_viability=0.02,
        ghost_threshold=0.3,
        seed=123,
        track_towns=False,
    )

    ghost_frac_all_1, years_1, scenarios_1, *_ = out1
    ghost_frac_all_2, years_2, scenarios_2, *_ = out2

    # shapes
    assert ghost_frac_all_1.shape == (n_sims, len(np.arange(start_year, end_year + 1)))
    assert years_1.tolist() == list(np.arange(start_year, end_year + 1))
    assert len(scenarios_1) == n_sims

    # repeatability with same seed
    npt.assert_array_almost_equal(ghost_frac_all_1, ghost_frac_all_2)
    npt.assert_array_equal(years_1, years_2)
    npt.assert_array_equal(scenarios_1, scenarios_2)


def test_run_custom_scenario_and_summaries():
    # create a trivial deterministic custom scenario
    n_towns = 20
    n_sims = 5
    start_year = 2023
    end_year = 2024

    ghost_frac_all, years = run_custom_scenario(
        n_towns=n_towns,
        n_sims=n_sims,
        start_year=start_year,
        end_year=end_year,
        initial_mean=0.5,
        initial_std=0.0,  # deterministic initial viability
        decline_rate=-0.1,
        sigma_viability=0.0,
        ghost_threshold=0.4,
        seed=1,
    )

    # shape expectations
    assert ghost_frac_all.shape == (n_sims, len(years))

    # Because initial mean 0.5, no volatility, large decline, by end year most towns likely below threshold
    summary = summarize_ghost_frac(ghost_frac_all, years, years[-1])
    assert summary["Year"] == years[-1]
    assert "Mean (%)" in summary

    probs = prob_ghost_share(ghost_frac_all, years, years[-1], thresholds=(0.1, 0.5))
    assert probs["Year"] == years[-1]
    assert "P(>=10%)" in probs or any(k.startswith("P(>=" ) for k in probs.keys())


def test_run_sensitivity_analysis_basic_and_extremes():
    from sim_utils import run_sensitivity_analysis

    base_params = {
        'n_towns': 50,
        'start_year': 2023,
        'end_year': 2025,
        'initial_mean': 0.7,
        'initial_std': 0.05,
        'mean_decline_slow': -0.005,
        'mean_decline_base': -0.02,
        'mean_decline_rapid': -0.03,
        'p_slow': 0.3,
        'p_base': 0.5,
        'p_rapid': 0.2,
        'sigma_viability': 0.02,
        'ghost_threshold': 0.3,
    }

    # Basic check: initial_mean range
    param_range = np.linspace(0.6, 0.8, 5)
    df = run_sensitivity_analysis(base_params, 'initial_mean', param_range, n_sims_sensitivity=20)

    assert len(df) == len(param_range)
    expected_cols = {'param_value', 'final_mean', 'final_std', 'final_p10', 'final_p90'}
    assert expected_cols.issubset(set(df.columns))

    # No NaNs and percentages within sensible bounds [0, 100]
    assert not df[['final_mean', 'final_std', 'final_p10', 'final_p90']].isna().any().any()
    assert (df[['final_mean', 'final_p10', 'final_p90']] >= 0).all().all()
    assert (df[['final_mean', 'final_p10', 'final_p90']] <= 100).all().all()

    # Extremes for p_slow (edge cases): 0.0 and 0.8
    param_range2 = [0.0, 0.8]
    df2 = run_sensitivity_analysis(base_params, 'p_slow', param_range2, n_sims_sensitivity=20)
    assert len(df2) == 2
    # param_value column should reflect param_range2
    assert np.allclose(df2['param_value'].values.astype(float), np.array(param_range2, dtype=float))
    # final_std should be non-negative
    assert (df2['final_std'] >= 0).all()
