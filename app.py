import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

from sim_utils import (
    run_monte_carlo,
    run_sensitivity_analysis,
    run_custom_scenario,
    summarize_ghost_frac,
    prob_ghost_share,
)

st.set_page_config(
    page_title="Rural Town Viability Forecast",
    page_icon="🏘️",
    layout="wide"
)

st.title("Rural Town Viability Monte Carlo Simulation")
st.markdown("Forecast the fraction of rural towns becoming 'ghost towns' over time using Monte Carlo simulation.")

with st.sidebar:
    st.header("Simulation Parameters")
    
    st.subheader("Basic Settings")
    n_towns = st.slider("Number of Towns", min_value=100, max_value=1000, value=500, step=50)
    n_sims = st.slider("Number of Simulations", min_value=500, max_value=5000, value=2000, step=100)
    
    st.subheader("Time Range")
    col1, col2 = st.columns(2)
    with col1:
        start_year = st.number_input("Start Year", min_value=2020, max_value=2040, value=2023)
    with col2:
        end_year = st.number_input("End Year", min_value=2030, max_value=2100, value=2050)
    
    valid_time_range = end_year >= start_year
    if not valid_time_range:
        st.error("End Year must be greater than or equal to Start Year.")
    
    st.subheader("Initial Viability")
    initial_mean = st.slider("Mean Initial Viability", min_value=0.4, max_value=0.9, value=0.7, step=0.05)
    initial_std = st.slider("Std Dev of Initial Viability", min_value=0.05, max_value=0.2, value=0.1, step=0.01)
    
    st.subheader("Decline Rates (Annual)")
    mean_decline_slow = st.slider("Slow Decline Rate", min_value=-0.005, max_value=-0.02, value=-0.01, step=0.005, format="%.3f")
    mean_decline_base = st.slider("Base Decline Rate", min_value=-0.01, max_value=-0.03, value=-0.02, step=0.005, format="%.3f")
    mean_decline_rapid = st.slider("Rapid Decline Rate", min_value=-0.02, max_value=-0.05, value=-0.03, step=0.005, format="%.3f")
    
    st.subheader("Scenario Probabilities")
    p_slow = st.slider("P(Slow Decline)", min_value=0.0, max_value=1.0, value=0.3, step=0.05)
    p_base = st.slider("P(Base Trend)", min_value=0.0, max_value=1.0, value=0.5, step=0.05)
    p_rapid = 1.0 - p_slow - p_base
    valid_probabilities = p_rapid >= 0
    if not valid_probabilities:
        st.error("Probabilities exceed 1.0! Adjust P(Slow) or P(Base).")
        p_rapid = 0.0
    else:
        st.info(f"P(Rapid Collapse) = {p_rapid:.2f}")
    
    st.subheader("Other Parameters")
    sigma_viability = st.slider("Yearly Volatility (Std Dev)", min_value=0.01, max_value=0.05, value=0.02, step=0.005, format="%.3f")
    ghost_threshold = st.slider("Ghost Town Threshold", min_value=0.1, max_value=0.5, value=0.3, step=0.05)
    
    run_simulation = st.button("Run Simulation", type="primary", use_container_width=True)


if 'ghost_frac_all' not in st.session_state:
    st.session_state.ghost_frac_all = None
    st.session_state.years = None
    st.session_state.scenarios = None
    st.session_state.town_trajectories = None
    st.session_state.initial_viability = None
    st.session_state.saved_runs = []

can_run_simulation = valid_probabilities and valid_time_range

if run_simulation or st.session_state.ghost_frac_all is None:
    if can_run_simulation:
        with st.spinner("Running Monte Carlo simulation..."):
            ghost_frac_all, years, scenarios, town_trajectories, initial_viability = run_monte_carlo(
                n_towns, n_sims, start_year, end_year, initial_mean, initial_std,
                mean_decline_slow, mean_decline_base, mean_decline_rapid,
                p_slow, p_base, p_rapid, sigma_viability, ghost_threshold,
                track_towns=True, n_sample_towns=20
            )
            st.session_state.ghost_frac_all = ghost_frac_all
            st.session_state.years = years
            st.session_state.scenarios = scenarios
            st.session_state.town_trajectories = town_trajectories
            st.session_state.initial_viability = initial_viability
        st.success(f"Simulation complete! {n_sims:,} runs across {len(years)} years.")

if st.session_state.ghost_frac_all is not None:
    ghost_frac_all = st.session_state.ghost_frac_all
    years = st.session_state.years
    scenarios = st.session_state.scenarios
    town_trajectories = st.session_state.town_trajectories
    initial_viability = st.session_state.initial_viability
    
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "Main Forecast", 
        "Sensitivity Analysis", 
        "Town Trajectories",
        "Custom Scenarios",
        "Run Comparison"
    ])
    
    with tab1:
        st.header("Time Series Forecast")
        
        mean_path = ghost_frac_all.mean(axis=0) * 100
        median_path = np.median(ghost_frac_all, axis=0) * 100
        p10_path = np.percentile(ghost_frac_all, 10, axis=0) * 100
        p25_path = np.percentile(ghost_frac_all, 25, axis=0) * 100
        p75_path = np.percentile(ghost_frac_all, 75, axis=0) * 100
        p90_path = np.percentile(ghost_frac_all, 90, axis=0) * 100
        
        fig_ts = go.Figure()
        
        fig_ts.add_trace(go.Scatter(
            x=list(years) + list(years)[::-1],
            y=list(p10_path) + list(p90_path)[::-1],
            fill='toself',
            fillcolor='rgba(99, 110, 250, 0.2)',
            line=dict(color='rgba(255,255,255,0)'),
            name='10-90% Band',
            hoverinfo='skip'
        ))
        
        fig_ts.add_trace(go.Scatter(
            x=list(years) + list(years)[::-1],
            y=list(p25_path) + list(p75_path)[::-1],
            fill='toself',
            fillcolor='rgba(99, 110, 250, 0.4)',
            line=dict(color='rgba(255,255,255,0)'),
            name='25-75% Band',
            hoverinfo='skip'
        ))
        
        fig_ts.add_trace(go.Scatter(
            x=years, y=mean_path,
            mode='lines',
            name='Mean',
            line=dict(color='#636EFA', width=3)
        ))
        
        fig_ts.add_trace(go.Scatter(
            x=years, y=median_path,
            mode='lines',
            name='Median',
            line=dict(color='#EF553B', width=2, dash='dash')
        ))
        
        fig_ts.update_layout(
            title="Ghost Town Percentage Over Time",
            xaxis_title="Year",
            yaxis_title="Ghost Towns (%)",
            hovermode="x unified",
            legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
            height=500
        )
        
        st.plotly_chart(fig_ts, use_container_width=True)
        
        st.header("Summary Statistics")
        
        key_years = [y for y in [2030, 2040, 2050] if start_year <= y <= end_year]
        if not key_years:
            key_years = [years[len(years)//3], years[2*len(years)//3], years[-1]]
        
        summary_data = [summarize_ghost_frac(ghost_frac_all, years, y) for y in key_years]
        summary_df = pd.DataFrame(summary_data)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Ghost Town Percentages by Year")
            st.dataframe(summary_df, use_container_width=True, hide_index=True)
        
        with col2:
            st.subheader("Probability of Reaching Thresholds")
            prob_data = [prob_ghost_share(ghost_frac_all, years, y) for y in key_years]
            prob_df = pd.DataFrame(prob_data)
            st.dataframe(prob_df, use_container_width=True, hide_index=True)
        
        st.header("Distribution Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Final Year Distribution")
            final_vals = ghost_frac_all[:, -1] * 100
            
            fig_hist = px.histogram(
                x=final_vals,
                nbins=50,
                labels={'x': 'Ghost Town Percentage', 'y': 'Frequency'},
                title=f"Distribution of Ghost Town % in {years[-1]}"
            )
            fig_hist.update_layout(showlegend=False)
            st.plotly_chart(fig_hist, use_container_width=True)
        
        with col2:
            st.subheader("Scenario Breakdown")
            scenario_counts = pd.Series(scenarios).value_counts()
            fig_pie = px.pie(
                values=scenario_counts.values,
                names=scenario_counts.index,
                title="Scenario Distribution in Simulations",
                color_discrete_map={'slow': '#00CC96', 'base': '#636EFA', 'rapid': '#EF553B'}
            )
            st.plotly_chart(fig_pie, use_container_width=True)
        
        st.header("Scenario Comparison")
        
        slow_mask = scenarios == "slow"
        base_mask = scenarios == "base"
        rapid_mask = scenarios == "rapid"
        
        fig_scenario = go.Figure()
        
        if slow_mask.sum() > 0:
            slow_mean = ghost_frac_all[slow_mask].mean(axis=0) * 100
            fig_scenario.add_trace(go.Scatter(
                x=years, y=slow_mean,
                mode='lines',
                name='Slow Decline',
                line=dict(color='#00CC96', width=2)
            ))
        
        if base_mask.sum() > 0:
            base_mean = ghost_frac_all[base_mask].mean(axis=0) * 100
            fig_scenario.add_trace(go.Scatter(
                x=years, y=base_mean,
                mode='lines',
                name='Base Trend',
                line=dict(color='#636EFA', width=2)
            ))
        
        if rapid_mask.sum() > 0:
            rapid_mean = ghost_frac_all[rapid_mask].mean(axis=0) * 100
            fig_scenario.add_trace(go.Scatter(
                x=years, y=rapid_mean,
                mode='lines',
                name='Rapid Collapse',
                line=dict(color='#EF553B', width=2)
            ))
        
        fig_scenario.update_layout(
            title="Mean Ghost Town % by Scenario",
            xaxis_title="Year",
            yaxis_title="Ghost Towns (%)",
            hovermode="x unified",
            height=400
        )
        
        st.plotly_chart(fig_scenario, use_container_width=True)
        
        st.header("Export Data")
        
        col1, col2 = st.columns(2)
        
        with col1:
            summary_export = pd.DataFrame({
                'Year': years,
                'Mean (%)': mean_path,
                'Median (%)': median_path,
                '10th Percentile (%)': p10_path,
                '25th Percentile (%)': p25_path,
                '75th Percentile (%)': p75_path,
                '90th Percentile (%)': p90_path,
            })
            csv_summary = summary_export.to_csv(index=False)
            st.download_button(
                label="Download Summary Statistics (CSV)",
                data=csv_summary,
                file_name="ghost_town_summary.csv",
                mime="text/csv"
            )
        
        with col2:
            full_export = pd.DataFrame(ghost_frac_all * 100, columns=years)
            full_export.insert(0, 'Scenario', scenarios)
            csv_full = full_export.to_csv(index=False)
            st.download_button(
                label="Download Full Simulation Data (CSV)",
                data=csv_full,
                file_name="ghost_town_full_simulation.csv",
                mime="text/csv"
            )
    
    with tab2:
        st.header("Sensitivity Analysis")
        st.markdown("Explore how changing individual parameters affects the final ghost town percentage.")
        
        base_params = {
            'n_towns': n_towns,
            'start_year': start_year,
            'end_year': end_year,
            'initial_mean': initial_mean,
            'initial_std': initial_std,
            'mean_decline_slow': mean_decline_slow,
            'mean_decline_base': mean_decline_base,
            'mean_decline_rapid': mean_decline_rapid,
            'p_slow': p_slow,
            'p_base': p_base,
            'p_rapid': p_rapid,
            'sigma_viability': sigma_viability,
            'ghost_threshold': ghost_threshold
        }
        
        col1, col2 = st.columns(2)
        
        with col1:
            param_to_analyze = st.selectbox(
                "Parameter to Analyze",
                options=[
                    'initial_mean',
                    'initial_std', 
                    'mean_decline_base',
                    'sigma_viability',
                    'ghost_threshold',
                    'p_slow'
                ],
                format_func=lambda x: {
                    'initial_mean': 'Initial Viability Mean',
                    'initial_std': 'Initial Viability Std Dev',
                    'mean_decline_base': 'Base Decline Rate',
                    'sigma_viability': 'Yearly Volatility',
                    'ghost_threshold': 'Ghost Town Threshold',
                    'p_slow': 'Probability of Slow Decline'
                }.get(x, x)
            )
        
        with col2:
            n_points = st.slider("Number of Analysis Points", min_value=5, max_value=20, value=10)
        
        param_ranges = {
            'initial_mean': np.linspace(0.4, 0.9, n_points),
            'initial_std': np.linspace(0.05, 0.2, n_points),
            'mean_decline_base': np.linspace(-0.035, -0.01, n_points),
            'sigma_viability': np.linspace(0.01, 0.05, n_points),
            'ghost_threshold': np.linspace(0.1, 0.5, n_points),
            'p_slow': np.linspace(0.0, 0.8, n_points)
        }
        
        run_sensitivity = st.button("Run Sensitivity Analysis", type="primary")
        
        if run_sensitivity:
            with st.spinner(f"Running sensitivity analysis for {param_to_analyze}..."):
                sensitivity_df = run_sensitivity_analysis(
                    base_params, 
                    param_to_analyze, 
                    param_ranges[param_to_analyze],
                    n_sims_sensitivity=500
                )
                st.session_state.sensitivity_df = sensitivity_df
                st.session_state.sensitivity_param = param_to_analyze
        
        if 'sensitivity_df' in st.session_state and st.session_state.sensitivity_df is not None:
            sensitivity_df = st.session_state.sensitivity_df
            param_analyzed = st.session_state.sensitivity_param
            
            param_labels = {
                'initial_mean': 'Initial Viability Mean',
                'initial_std': 'Initial Viability Std Dev',
                'mean_decline_base': 'Base Decline Rate',
                'sigma_viability': 'Yearly Volatility',
                'ghost_threshold': 'Ghost Town Threshold',
                'p_slow': 'Probability of Slow Decline'
            }
            
            fig_sensitivity = go.Figure()
            
            fig_sensitivity.add_trace(go.Scatter(
                x=list(sensitivity_df['param_value']) + list(sensitivity_df['param_value'])[::-1],
                y=list(sensitivity_df['final_p10']) + list(sensitivity_df['final_p90'])[::-1],
                fill='toself',
                fillcolor='rgba(99, 110, 250, 0.2)',
                line=dict(color='rgba(255,255,255,0)'),
                name='10-90% Range',
                hoverinfo='skip'
            ))
            
            fig_sensitivity.add_trace(go.Scatter(
                x=sensitivity_df['param_value'],
                y=sensitivity_df['final_mean'],
                mode='lines+markers',
                name='Mean Final Ghost %',
                line=dict(color='#636EFA', width=3)
            ))
            
            fig_sensitivity.update_layout(
                title=f"Sensitivity: {param_labels.get(param_analyzed, param_analyzed)} vs Final Ghost Town %",
                xaxis_title=param_labels.get(param_analyzed, param_analyzed),
                yaxis_title=f"Ghost Towns in {end_year} (%)",
                height=500
            )
            
            st.plotly_chart(fig_sensitivity, use_container_width=True)
            
            st.subheader("Sensitivity Analysis Heatmap")
            
            sensitivity_df['range'] = sensitivity_df['final_p90'] - sensitivity_df['final_p10']
            
            fig_heatmap = px.imshow(
                sensitivity_df[['final_mean', 'final_std', 'range']].T.values,
                x=[f"{v:.3f}" for v in sensitivity_df['param_value']],
                y=['Mean (%)', 'Std Dev (%)', 'Range (P90-P10)'],
                color_continuous_scale='RdYlBu_r',
                title=f"Parameter Impact Heatmap: {param_labels.get(param_analyzed, param_analyzed)}",
                aspect='auto'
            )
            st.plotly_chart(fig_heatmap, use_container_width=True)
            
            st.subheader("Sensitivity Data Table")
            display_df = sensitivity_df.copy()
            display_df.columns = ['Parameter Value', 'Mean (%)', 'Std Dev (%)', 'P10 (%)', 'P90 (%)', 'Range']
            st.dataframe(display_df.round(2), use_container_width=True, hide_index=True)
    
    with tab3:
        st.header("Individual Town Trajectories")
        st.markdown("Track the viability of individual sample towns over time across different scenarios.")
        
        if town_trajectories is not None and isinstance(town_trajectories, dict):
            available_scenarios = [s for s in ['slow', 'base', 'rapid'] if np.any(town_trajectories.get(s, np.zeros(1)) != 0)]
            
            if available_scenarios:
                selected_scenario = st.selectbox(
                    "Select Scenario to View",
                    options=available_scenarios,
                    format_func=lambda x: {'slow': 'Slow Decline', 'base': 'Base Trend', 'rapid': 'Rapid Collapse'}.get(x, x)
                )
                
                scenario_data = town_trajectories[selected_scenario]
                n_sample = scenario_data.shape[0]
                
                col1, col2 = st.columns(2)
                with col1:
                    show_all = st.checkbox("Show all sample towns", value=True)
                with col2:
                    if not show_all:
                        selected_towns = st.multiselect(
                            "Select towns to display",
                            options=list(range(n_sample)),
                            default=list(range(min(5, n_sample))),
                            format_func=lambda x: f"Town {x+1}"
                        )
                    else:
                        selected_towns = list(range(n_sample))
                
                fig_towns = go.Figure()
                
                fig_towns.add_hline(
                    y=ghost_threshold,
                    line_dash="dash",
                    line_color="red",
                    annotation_text=f"Ghost Threshold ({ghost_threshold})",
                    annotation_position="bottom right"
                )
                
                colors = px.colors.qualitative.Set3
                for i, town_idx in enumerate(selected_towns):
                    color = colors[i % len(colors)]
                    fig_towns.add_trace(go.Scatter(
                        x=years,
                        y=scenario_data[town_idx, :],
                        mode='lines',
                        name=f'Town {town_idx + 1}',
                        line=dict(color=color, width=1.5),
                        opacity=0.8
                    ))
                
                scenario_labels = {'slow': 'Slow Decline', 'base': 'Base Trend', 'rapid': 'Rapid Collapse'}
                fig_towns.update_layout(
                    title=f"Individual Town Viability Over Time ({scenario_labels.get(selected_scenario, selected_scenario)})",
                    xaxis_title="Year",
                    yaxis_title="Viability Score",
                    yaxis=dict(range=[0, 1]),
                    height=500,
                    hovermode="x unified"
                )
                
                st.plotly_chart(fig_towns, use_container_width=True)
                
                st.subheader("Town Status Summary")
                
                final_viabilities = scenario_data[:, -1]
                initial_viabilities = scenario_data[:, 0]
                
                town_summary = pd.DataFrame({
                    'Town': [f"Town {i+1}" for i in range(n_sample)],
                    'Scenario': [scenario_labels.get(selected_scenario, selected_scenario)] * n_sample,
                    'Initial Viability': initial_viabilities.round(3),
                    'Final Viability': final_viabilities.round(3),
                    'Change': (final_viabilities - initial_viabilities).round(3),
                    'Status': ['Ghost Town' if v < ghost_threshold else 'Viable' for v in final_viabilities]
                })
                
                col1, col2, col3 = st.columns(3)
                ghost_count = sum(final_viabilities < ghost_threshold)
                with col1:
                    st.metric("Ghost Towns", f"{ghost_count}/{n_sample}")
                with col2:
                    st.metric("Average Final Viability", f"{np.mean(final_viabilities):.3f}")
                with col3:
                    st.metric("Average Decline", f"{np.mean(final_viabilities - initial_viabilities):.3f}")
                
                st.dataframe(town_summary, use_container_width=True, hide_index=True)
                
                st.subheader("Scenario Comparison: Sample Town Trajectories")
                
                fig_compare = go.Figure()
                
                fig_compare.add_hline(
                    y=ghost_threshold,
                    line_dash="dash",
                    line_color="red",
                    annotation_text=f"Ghost Threshold",
                    annotation_position="bottom right"
                )
                
                scenario_colors = {'slow': '#00CC96', 'base': '#636EFA', 'rapid': '#EF553B'}
                
                for scenario in available_scenarios:
                    s_data = town_trajectories[scenario]
                    mean_trajectory = s_data.mean(axis=0)
                    fig_compare.add_trace(go.Scatter(
                        x=years,
                        y=mean_trajectory,
                        mode='lines',
                        name=f'{scenario_labels.get(scenario, scenario)} (Avg)',
                        line=dict(color=scenario_colors.get(scenario, '#888'), width=3)
                    ))
                
                fig_compare.update_layout(
                    title="Average Sample Town Trajectory by Scenario",
                    xaxis_title="Year",
                    yaxis_title="Average Viability Score",
                    yaxis=dict(range=[0, 1]),
                    height=400,
                    hovermode="x unified"
                )
                
                st.plotly_chart(fig_compare, use_container_width=True)
                
                st.subheader("Initial vs Final Viability Distribution")
                
                fig_scatter = go.Figure()
                
                fig_scatter.add_trace(go.Scatter(
                    x=initial_viabilities,
                    y=final_viabilities,
                    mode='markers',
                    marker=dict(
                        size=12,
                        color=final_viabilities,
                        colorscale='RdYlGn',
                        showscale=True,
                        colorbar=dict(title="Final Viability")
                    ),
                    text=[f"Town {i+1}" for i in range(n_sample)],
                    hovertemplate="<b>%{text}</b><br>Initial: %{x:.3f}<br>Final: %{y:.3f}<extra></extra>"
                ))
                
                fig_scatter.add_hline(y=ghost_threshold, line_dash="dash", line_color="red")
                fig_scatter.add_vline(x=ghost_threshold, line_dash="dash", line_color="red")
                
                fig_scatter.add_trace(go.Scatter(
                    x=[0, 1], y=[0, 1],
                    mode='lines',
                    line=dict(dash='dot', color='gray'),
                    name='No Change Line',
                    showlegend=True
                ))
                
                fig_scatter.update_layout(
                    title="Initial vs Final Town Viability",
                    xaxis_title="Initial Viability",
                    yaxis_title="Final Viability",
                    xaxis=dict(range=[0, 1]),
                    yaxis=dict(range=[0, 1]),
                    height=500
                )
                
                st.plotly_chart(fig_scatter, use_container_width=True)
            else:
                st.info("No scenario data available. Run a new simulation to generate this data.")
        else:
            st.info("Town trajectory data not available. Run a new simulation to generate this data.")
    
    with tab4:
        st.header("Custom Scenario Builder")
        st.markdown("Test custom policy interventions and their effects on town viability.")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Base Scenario")
            custom_decline = st.slider(
                "Annual Decline Rate", 
                min_value=-0.05, max_value=0.0, value=-0.02, step=0.005,
                format="%.3f",
                key="custom_decline"
            )
        
        with col2:
            st.subheader("Policy Intervention")
            enable_intervention = st.checkbox("Enable Policy Intervention")
            
            if enable_intervention:
                intervention_year = st.slider(
                    "Intervention Start Year",
                    min_value=int(start_year) + 1,
                    max_value=int(end_year) - 1,
                    value=int((start_year + end_year) / 2),
                    key="intervention_year"
                )
                intervention_effect = st.slider(
                    "Intervention Effect (reduction in decline)",
                    min_value=0.0, max_value=0.03, value=0.01, step=0.005,
                    format="%.3f",
                    help="Positive value reduces the decline rate",
                    key="intervention_effect"
                )
            else:
                intervention_year = None
                intervention_effect = 0.0
        
        run_custom = st.button("Run Custom Scenario", type="primary", key="run_custom")
        
        if run_custom:
            with st.spinner("Running custom scenario simulations..."):
                baseline_results, baseline_years = run_custom_scenario(
                    n_towns, n_sims, start_year, end_year, initial_mean, initial_std,
                    custom_decline, sigma_viability, ghost_threshold,
                    intervention_year=None, intervention_effect=0.0, seed=42
                )
                
                if enable_intervention:
                    intervention_results, _ = run_custom_scenario(
                        n_towns, n_sims, start_year, end_year, initial_mean, initial_std,
                        custom_decline, sigma_viability, ghost_threshold,
                        intervention_year=intervention_year, 
                        intervention_effect=intervention_effect, 
                        seed=43
                    )
                else:
                    intervention_results = None
                
                st.session_state.custom_baseline = baseline_results
                st.session_state.custom_intervention = intervention_results
                st.session_state.custom_years = baseline_years
                st.session_state.custom_intervention_year = intervention_year
        
        if 'custom_baseline' in st.session_state and st.session_state.custom_baseline is not None:
            baseline = st.session_state.custom_baseline
            intervention = st.session_state.custom_intervention
            custom_years = st.session_state.custom_years
            int_year = st.session_state.custom_intervention_year
            
            fig_custom = go.Figure()
            
            baseline_mean = baseline.mean(axis=0) * 100
            baseline_p10 = np.percentile(baseline, 10, axis=0) * 100
            baseline_p90 = np.percentile(baseline, 90, axis=0) * 100
            
            fig_custom.add_trace(go.Scatter(
                x=list(custom_years) + list(custom_years)[::-1],
                y=list(baseline_p10) + list(baseline_p90)[::-1],
                fill='toself',
                fillcolor='rgba(239, 85, 59, 0.2)',
                line=dict(color='rgba(255,255,255,0)'),
                name='Baseline Range',
                hoverinfo='skip'
            ))
            
            fig_custom.add_trace(go.Scatter(
                x=custom_years, y=baseline_mean,
                mode='lines',
                name='Baseline (No Intervention)',
                line=dict(color='#EF553B', width=3)
            ))
            
            if intervention is not None:
                int_mean = intervention.mean(axis=0) * 100
                int_p10 = np.percentile(intervention, 10, axis=0) * 100
                int_p90 = np.percentile(intervention, 90, axis=0) * 100
                
                fig_custom.add_trace(go.Scatter(
                    x=list(custom_years) + list(custom_years)[::-1],
                    y=list(int_p10) + list(int_p90)[::-1],
                    fill='toself',
                    fillcolor='rgba(0, 204, 150, 0.2)',
                    line=dict(color='rgba(255,255,255,0)'),
                    name='Intervention Range',
                    hoverinfo='skip'
                ))
                
                fig_custom.add_trace(go.Scatter(
                    x=custom_years, y=int_mean,
                    mode='lines',
                    name='With Intervention',
                    line=dict(color='#00CC96', width=3)
                ))
                
                if int_year:
                    fig_custom.add_vline(
                        x=int_year, 
                        line_dash="dash", 
                        line_color="gray",
                        annotation_text=f"Intervention ({int_year})"
                    )
            
            fig_custom.update_layout(
                title="Custom Scenario: Policy Impact on Ghost Towns",
                xaxis_title="Year",
                yaxis_title="Ghost Towns (%)",
                height=500,
                hovermode="x unified"
            )
            
            st.plotly_chart(fig_custom, use_container_width=True)
            
            if intervention is not None:
                st.subheader("Intervention Impact Summary")
                
                final_baseline = baseline[:, -1] * 100
                final_intervention = intervention[:, -1] * 100
                reduction = final_baseline.mean() - final_intervention.mean()
                
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Baseline Final %", f"{final_baseline.mean():.1f}%")
                with col2:
                    st.metric("With Intervention", f"{final_intervention.mean():.1f}%")
                with col3:
                    st.metric("Reduction", f"{reduction:.1f}%", delta=f"-{reduction:.1f}%")
                with col4:
                    effectiveness = (reduction / final_baseline.mean()) * 100 if final_baseline.mean() > 0 else 0
                    st.metric("Effectiveness", f"{effectiveness:.1f}%")
    
    with tab5:
        st.header("Multi-Run Comparison")
        st.markdown("Save and compare multiple simulation runs with different parameters.")
        
        run_name = st.text_input("Run Name", value=f"Run {len(st.session_state.saved_runs) + 1}")
        
        if st.button("Save Current Run", key="save_run"):
            run_data = {
                'name': run_name,
                'ghost_frac_all': ghost_frac_all.copy(),
                'years': years.copy(),
                'params': {
                    'n_towns': n_towns,
                    'n_sims': n_sims,
                    'initial_mean': initial_mean,
                    'mean_decline_base': mean_decline_base,
                    'ghost_threshold': ghost_threshold
                }
            }
            st.session_state.saved_runs.append(run_data)
            st.success(f"Saved '{run_name}' to comparison list!")
        
        if len(st.session_state.saved_runs) > 0:
            st.subheader("Saved Runs")
            
            runs_info = []
            for i, run in enumerate(st.session_state.saved_runs):
                final_mean = np.mean(run['ghost_frac_all'][:, -1]) * 100
                runs_info.append({
                    'Index': i,
                    'Name': run['name'],
                    'Towns': run['params']['n_towns'],
                    'Simulations': run['params']['n_sims'],
                    'Final Ghost %': f"{final_mean:.1f}%"
                })
            
            st.dataframe(pd.DataFrame(runs_info), use_container_width=True, hide_index=True)
            
            if len(st.session_state.saved_runs) >= 2:
                st.subheader("Comparison Chart")
                
                selected_runs = st.multiselect(
                    "Select runs to compare",
                    options=list(range(len(st.session_state.saved_runs))),
                    default=list(range(min(3, len(st.session_state.saved_runs)))),
                    format_func=lambda x: st.session_state.saved_runs[x]['name']
                )
                
                if len(selected_runs) >= 1:
                    fig_compare = go.Figure()
                    
                    colors = px.colors.qualitative.Set1
                    
                    for i, run_idx in enumerate(selected_runs):
                        run = st.session_state.saved_runs[run_idx]
                        mean_path = run['ghost_frac_all'].mean(axis=0) * 100
                        
                        color = colors[i % len(colors)]
                        
                        fig_compare.add_trace(go.Scatter(
                            x=run['years'],
                            y=mean_path,
                            mode='lines',
                            name=run['name'],
                            line=dict(color=color, width=2)
                        ))
                    
                    fig_compare.update_layout(
                        title="Comparison of Simulation Runs",
                        xaxis_title="Year",
                        yaxis_title="Mean Ghost Towns (%)",
                        height=500,
                        hovermode="x unified"
                    )
                    
                    st.plotly_chart(fig_compare, use_container_width=True)
                    
                    st.subheader("Final Year Statistics Comparison")
                    
                    comparison_data = []
                    for run_idx in selected_runs:
                        run = st.session_state.saved_runs[run_idx]
                        final_vals = run['ghost_frac_all'][:, -1] * 100
                        comparison_data.append({
                            'Run': run['name'],
                            'Mean (%)': round(np.mean(final_vals), 1),
                            'Median (%)': round(np.median(final_vals), 1),
                            'Std Dev (%)': round(np.std(final_vals), 1),
                            'P10 (%)': round(np.percentile(final_vals, 10), 1),
                            'P90 (%)': round(np.percentile(final_vals, 90), 1)
                        })
                    
                    st.dataframe(pd.DataFrame(comparison_data), use_container_width=True, hide_index=True)
            
            if st.button("Clear All Saved Runs", key="clear_runs"):
                st.session_state.saved_runs = []
                st.rerun()
        else:
            st.info("No saved runs yet. Run a simulation and click 'Save Current Run' to start comparing.")

st.markdown("---")
st.markdown("""
**About this simulation:** This Monte Carlo model forecasts the fraction of rural towns that may become 
'ghost towns' (viability below threshold) over time. It considers three macro scenarios with different 
decline rates and incorporates yearly random volatility to capture unexpected shocks.
""")
