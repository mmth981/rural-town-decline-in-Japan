import streamlit as st
import numpy as np
import pandas as pd

@st.cache_data
def run_monte_carlo(n_towns, n_sims, start_year, end_year, initial_mean, initial_std,
                    mean_decline_slow, mean_decline_base, mean_decline_rapid,
                    p_slow, p_base, p_rapid, sigma_viability, ghost_threshold, seed=42,
                    track_towns=False, n_sample_towns=20):
    rng = np.random.default_rng(seed)
    years = np.arange(start_year, end_year + 1)
    n_years = len(years)

    initial_viability = np.clip(rng.normal(loc=initial_mean, scale=initial_std, size=n_towns), 0, 1)

    ghost_frac_all = np.zeros((n_sims, n_years))
    scenario_tracker = []

    town_trajectories = None
    town_scenario_labels = None
    if track_towns:
        sample_indices = rng.choice(n_towns, size=min(n_sample_towns, n_towns), replace=False)
        town_trajectories = {
            'slow': np.zeros((len(sample_indices), n_years)),
            'base': np.zeros((len(sample_indices), n_years)),
            'rapid': np.zeros((len(sample_indices), n_years))
        }
        scenario_tracked = {'slow': False, 'base': False, 'rapid': False}
        town_scenario_labels = []

    for s in range(n_sims):
        scenario = rng.choice(["slow", "base", "rapid"], p=[p_slow, p_base, p_rapid])
        scenario_tracker.append(scenario)

        if scenario == "slow":
            mu_decline = mean_decline_slow
        elif scenario == "base":
            mu_decline = mean_decline_base
        else:
            mu_decline = mean_decline_rapid

        v = initial_viability.copy()
        frac_list = []

        should_track = track_towns and not scenario_tracked.get(scenario, True)

        for t, year in enumerate(years):
            if year == start_year:
                ghost_frac = np.mean(v < ghost_threshold)
                frac_list.append(ghost_frac)
                if should_track:
                    town_trajectories[scenario][:, t] = v[sample_indices]
                continue

            noise = rng.normal(loc=0.0, scale=sigma_viability, size=n_towns)
            v = v + mu_decline + noise
            v = np.clip(v, 0.0, 1.0)

            ghost_frac = np.mean(v < ghost_threshold)
            frac_list.append(ghost_frac)

            if should_track:
                town_trajectories[scenario][:, t] = v[sample_indices]

        if should_track:
            scenario_tracked[scenario] = True

        ghost_frac_all[s, :] = np.array(frac_list)

    return ghost_frac_all, years, np.array(scenario_tracker), town_trajectories, initial_viability


@st.cache_data
def run_sensitivity_analysis(base_params, param_name, param_range, n_sims_sensitivity=500):
    results = []
    for val in param_range:
        params = base_params.copy()
        params[param_name] = val

        if param_name == 'p_slow':
            remaining = 1.0 - val
            params['p_base'] = remaining * 0.6
            params['p_rapid'] = remaining * 0.4
        elif param_name == 'p_base':
            remaining = 1.0 - val
            params['p_slow'] = remaining * 0.6
            params['p_rapid'] = remaining * 0.4

        ghost_frac_all, years, _, _, _ = run_monte_carlo(
            n_towns=params['n_towns'],
            n_sims=n_sims_sensitivity,
            start_year=params['start_year'],
            end_year=params['end_year'],
            initial_mean=params['initial_mean'],
            initial_std=params['initial_std'],
            mean_decline_slow=params['mean_decline_slow'],
            mean_decline_base=params['mean_decline_base'],
            mean_decline_rapid=params['mean_decline_rapid'],
            p_slow=params['p_slow'],
            p_base=params['p_base'],
            p_rapid=params['p_rapid'],
            sigma_viability=params['sigma_viability'],
            ghost_threshold=params['ghost_threshold'],
            seed=42 + int(val * 1000),
            track_towns=False
        )

        final_mean = np.mean(ghost_frac_all[:, -1]) * 100
        final_std = np.std(ghost_frac_all[:, -1]) * 100
        results.append({
            'param_value': val,
            'final_mean': final_mean,
            'final_std': final_std,
            'final_p10': np.percentile(ghost_frac_all[:, -1], 10) * 100,
            'final_p90': np.percentile(ghost_frac_all[:, -1], 90) * 100
        })

    return pd.DataFrame(results)


@st.cache_data
def run_custom_scenario(n_towns, n_sims, start_year, end_year, initial_mean, initial_std,
                        decline_rate, sigma_viability, ghost_threshold,
                        intervention_year=None, intervention_effect=0.0, seed=42):
    rng = np.random.default_rng(seed)
    years = np.arange(start_year, end_year + 1)
    n_years = len(years)

    initial_viability = np.clip(rng.normal(loc=initial_mean, scale=initial_std, size=n_towns), 0, 1)
    ghost_frac_all = np.zeros((n_sims, n_years))

    for s in range(n_sims):
        v = initial_viability.copy()
        frac_list = []

        for t, year in enumerate(years):
            if year == start_year:
                ghost_frac = np.mean(v < ghost_threshold)
                frac_list.append(ghost_frac)
                continue

            current_decline = decline_rate
            if intervention_year and year >= intervention_year:
                current_decline = decline_rate + intervention_effect

            noise = rng.normal(loc=0.0, scale=sigma_viability, size=n_towns)
            v = v + current_decline + noise
            v = np.clip(v, 0.0, 1.0)

            ghost_frac = np.mean(v < ghost_threshold)
            frac_list.append(ghost_frac)

        ghost_frac_all[s, :] = np.array(frac_list)

    return ghost_frac_all, years


def summarize_ghost_frac(ghost_frac_all, years, target_year):
    idx = np.where(years == target_year)[0][0]
    vals = ghost_frac_all[:, idx]
    return {
        "Year": target_year,
        "Mean (%)": round(float(np.mean(vals)) * 100, 1),
        "Median (%)": round(float(np.median(vals)) * 100, 1),
        "10th Percentile (%)": round(float(np.percentile(vals, 10)) * 100, 1),
        "90th Percentile (%)": round(float(np.percentile(vals, 90)) * 100, 1),
    }


def prob_ghost_share(ghost_frac_all, years, target_year, thresholds=(0.3, 0.5, 0.7)):
    idx = np.where(years == target_year)[0][0]
    vals = ghost_frac_all[:, idx]
    probs = {"Year": target_year}
    for thr in thresholds:
        probs[f"P(>={int(thr*100)}%)"] = round(float(np.mean(vals >= thr)) * 100, 1)
    return probs
