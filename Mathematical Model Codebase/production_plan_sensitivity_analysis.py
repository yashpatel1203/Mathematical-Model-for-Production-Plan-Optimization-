import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from production_plan_extended import solve_production_scheduling_extended, generate_synthetic_batch_data


def run_sensitivity_analysis_and_export_excel():
    """
    Automates the process of running sensitivity analysis for the production scheduling model
    and exports results to an Excel file with separate sheets for each scenario.
    """
    print("--- Starting Sensitivity Analysis ---")

    # Define BASE PARAMETERS for the scenarios, aligning with your provided main block setup
    BASE_H = 12  # From your generate_synthetic_batch_data call
    BASE_NMS = 2
    BASE_MS_NAMES = ['S2', 'S3']
    BASE_WORKSHOPS = {'W1': [0, 1]}

    # Base parameters for generating synthetic batch data and penalties
    BASE_NUM_BATCHES = 10
    BASE_AVG_PROD_TIME = 3
    BASE_PROD_TIME_VARIANCE = 1
    BASE_TYPE_R_PROPORTION = 0.5
    BASE_DEADLINE_FLEXIBILITY = 2
    BASE_AVG_PENALTY_MULTIPLIER = 150
    BASE_PENALTY_VARIANCE_PERCENT = 0.1

    # Base cost per hour for operating each manufacturing system (S2, S3)
    BASE_COST_PER_HOUR_ON_MACHINE = {0: 100, 1: 120}

    # List to store results from all scenarios
    all_scenario_raw_results = []

    # Helper to generate base batch data and penalties for scenarios that don't vary them
    def get_base_data_and_penalties():
        return generate_synthetic_batch_data(
            num_batches=BASE_NUM_BATCHES,
            planning_horizon_H=BASE_H,
            avg_prod_time=BASE_AVG_PROD_TIME,
            prod_time_variance=BASE_PROD_TIME_VARIANCE,
            type_R_proportion=BASE_TYPE_R_PROPORTION,
            deadline_flexibility=BASE_DEADLINE_FLEXIBILITY,
            avg_penalty_multiplier=BASE_AVG_PENALTY_MULTIPLIER,
            penalty_variance_percent=BASE_PENALTY_VARIANCE_PERCENT
        )

    # --- Scenario 1: Sensitivity to Relative Machine Costs ---
    print("\n--- Running Scenario 1: Sensitivity to Relative Machine Costs ---")
    s2_costs_to_test = [80, 100, 120, 150, 180]  # Vary S2 cost, keep S3 constant
    for cost_s2 in s2_costs_to_test:
        current_costs = {0: cost_s2, 1: BASE_COST_PER_HOUR_ON_MACHINE[1]}  # S2 varies, S3 fixed
        batches_data, penalties = get_base_data_and_penalties()  # Use base generated data

        results = solve_production_scheduling_extended(
            BASE_H, BASE_NMS, BASE_MS_NAMES, BASE_WORKSHOPS,
            batches_data, current_costs, penalties
        )
        all_scenario_raw_results.append({
            'Scenario': 'Relative Machine Costs (S2)',
            'Parameter_Varied': 'Cost S2 ($/h)',
            'Parameter_Value': cost_s2,
            'Total_Cost': results['total_cost'],
            'Rejected_Batches': results['rejected_batches_count'],
            'Status': results['status']
        })
        print(f"  S2 Cost: ${cost_s2}/h -> Total Cost: ${results['total_cost']:.2f}, Rejected: {results['rejected_batches_count']} ({results['status']})")


    # --- Scenario 2: Sensitivity to Overall Rejection Penalty Scale ---
    print("\n--- Running Scenario 2: Sensitivity to Overall Rejection Penalty Scale ---")
    penalty_multipliers = [0.1, 0.5, 1.0, 2.0, 5.0]  # Test multiplying all penalties
    for multiplier in penalty_multipliers:
        # Generate new penalties based on multiplier, keeping other data params constant
        batches_data, penalties = generate_synthetic_batch_data(
            num_batches=BASE_NUM_BATCHES,
            planning_horizon_H=BASE_H,
            avg_prod_time=BASE_AVG_PROD_TIME,
            prod_time_variance=BASE_PROD_TIME_VARIANCE,
            type_R_proportion=BASE_TYPE_R_PROPORTION,
            deadline_flexibility=BASE_DEADLINE_FLEXIBILITY,
            avg_penalty_multiplier=BASE_AVG_PENALTY_MULTIPLIER * multiplier,
            penalty_variance_percent=BASE_PENALTY_VARIANCE_PERCENT
        )

        results = solve_production_scheduling_extended(
            BASE_H, BASE_NMS, BASE_MS_NAMES, BASE_WORKSHOPS,
            batches_data, BASE_COST_PER_HOUR_ON_MACHINE, penalties
        )
        all_scenario_raw_results.append({
            'Scenario': 'Overall Rejection Penalty Scale',
            'Parameter_Varied': 'Penalty Multiplier',
            'Parameter_Value': multiplier,
            'Total_Cost': results['total_cost'],
            'Rejected_Batches': results['rejected_batches_count'],
            'Status': results['status']
        })
        print(f"  Penalty Multiplier: {multiplier:.1f}x -> Total Cost: ${results['total_cost']:.2f}, Rejected: {results['rejected_batches_count']} ({results['status']})")

    # --- Scenario 3: Sensitivity to Planning Horizon (H) ---
    print("\n--- Running Scenario 3: Sensitivity to Planning Horizon (H) ---")
    horizons_to_test = [8, 10, 12, 14, 16]
    for h_val in horizons_to_test:
        batches_data, penalties = generate_synthetic_batch_data(  # Regenerate data to match new H for EM/LM
            num_batches=BASE_NUM_BATCHES,
            planning_horizon_H=h_val,
            avg_prod_time=BASE_AVG_PROD_TIME,
            prod_time_variance=BASE_PROD_TIME_VARIANCE,
            type_R_proportion=BASE_TYPE_R_PROPORTION,
            deadline_flexibility=BASE_DEADLINE_FLEXIBILITY,
            avg_penalty_multiplier=BASE_AVG_PENALTY_MULTIPLIER,
            penalty_variance_percent=BASE_PENALTY_VARIANCE_PERCENT
        )
        results = solve_production_scheduling_extended(
            h_val, BASE_NMS, BASE_MS_NAMES, BASE_WORKSHOPS,
            batches_data, BASE_COST_PER_HOUR_ON_MACHINE, penalties
        )
        all_scenario_raw_results.append({
            'Scenario': 'Planning Horizon (H)',
            'Parameter_Varied': 'Horizon (H)',
            'Parameter_Value': h_val,
            'Total_Cost': results['total_cost'],
            'Rejected_Batches': results['rejected_batches_count'],
            'Status': results['status']
        })
        print(f"  Horizon (H): {h_val} -> Total Cost: ${results['total_cost']:.2f}, Rejected: {results['rejected_batches_count']} ({results['status']})")

    # --- Scenario 4: Sensitivity to Increased Batch Production Times ---
    print("\n--- Running Scenario 4: Sensitivity to Increased Batch Production Times ---")
    prod_time_multipliers = [0.7, 0.9, 1.0, 1.2, 1.5]
    for pt_mult in prod_time_multipliers:
        batches_data, penalties = generate_synthetic_batch_data(
            num_batches=BASE_NUM_BATCHES,
            planning_horizon_H=BASE_H,
            avg_prod_time=int(BASE_AVG_PROD_TIME * pt_mult),
            prod_time_variance=BASE_PROD_TIME_VARIANCE,
            type_R_proportion=BASE_TYPE_R_PROPORTION,
            deadline_flexibility=BASE_DEADLINE_FLEXIBILITY,
            avg_penalty_multiplier=BASE_AVG_PENALTY_MULTIPLIER,
            penalty_variance_percent=BASE_PENALTY_VARIANCE_PERCENT
        )

        results = solve_production_scheduling_extended(
            BASE_H, BASE_NMS, BASE_MS_NAMES, BASE_WORKSHOPS,
            batches_data, BASE_COST_PER_HOUR_ON_MACHINE, penalties
        )
        all_scenario_raw_results.append({
            'Scenario': 'Batch Production Times',
            'Parameter_Varied': 'Prod Time Multiplier',
            'Parameter_Value': pt_mult,
            'Total_Cost': results['total_cost'],
            'Rejected_Batches': results['rejected_batches_count'],
            'Status': results['status']
        })
        print(f"  Prod Time Multiplier: {pt_mult:.1f}x -> Total Cost: ${results['total_cost']:.2f}, Rejected: {results['rejected_batches_count']} ({results['status']})")

    # --- Scenario 5: Sensitivity to Batch Type Proportion (High 'R' demand) ---
    print("\n--- Running Scenario 5: Sensitivity to Batch Type Proportion (High 'R' demand) ---")
    r_proportions = [0.1, 0.3, 0.5, 0.7, 0.9]
    for prop_r in r_proportions:
        batches_data, penalties = generate_synthetic_batch_data(
            num_batches=BASE_NUM_BATCHES,
            planning_horizon_H=BASE_H,
            avg_prod_time=BASE_AVG_PROD_TIME,
            prod_time_variance=BASE_PROD_TIME_VARIANCE,
            type_R_proportion=prop_r,
            deadline_flexibility=BASE_DEADLINE_FLEXIBILITY,
            avg_penalty_multiplier=BASE_AVG_PENALTY_MULTIPLIER,
            penalty_variance_percent=BASE_PENALTY_VARIANCE_PERCENT
        )
        results = solve_production_scheduling_extended(
            BASE_H, BASE_NMS, BASE_MS_NAMES, BASE_WORKSHOPS,
            batches_data, BASE_COST_PER_HOUR_ON_MACHINE, penalties
        )
        all_scenario_raw_results.append({
            'Scenario': 'Batch Type Proportion (R)',
            'Parameter_Varied': 'Proportion R',
            'Parameter_Value': prop_r,
            'Total_Cost': results['total_cost'],
            'Rejected_Batches': results['rejected_batches_count'],
            'Status': results['status']
        })
        print(f"  Proportion of R types: {prop_r:.1f} -> Total Cost: ${results['total_cost']:.2f}, Rejected: {results['rejected_batches_count']} ({results['status']})")

    print("\n--- Sensitivity Analysis Complete ---")

    # Create a single DataFrame from all collected raw results
    df_all_results = pd.DataFrame(all_scenario_raw_results)

    # Export results to an Excel file with separate sheets
    output_excel_filename = "sensitivity_analysis_results.xlsx"
    with pd.ExcelWriter(output_excel_filename, engine='xlsxwriter') as writer:
        # Group by 'Scenario' and write each group to a separate sheet
        for scenario_name, group_df in df_all_results.groupby('Scenario'):
            # Drop the 'Scenario' column before writing to sheet for cleaner data
            group_df.drop(columns=['Scenario']).to_excel(writer, sheet_name=scenario_name[:30], index=False) # Limit sheet name length

    print(f"\nSensitivity analysis results exported to '{output_excel_filename}' with separate sheets for each scenario.")

    # --- Re-generate Plots (Optional, as you already have them but good for verification) ---
    # These plots use the df_all_results structure.
    # Function to create and save a plot (kept from your original script)
    def create_and_save_plot(df, scenario_name_filter, param_col, value_col, y_label, title, filename, color):
        plt.figure(figsize=(10, 6))
        scenario_df = df[df['Scenario'] == scenario_name_filter]
        plt.plot(scenario_df[param_col], scenario_df[value_col], marker='o', linestyle='-', color=color)
        plt.xlabel(param_col)
        plt.ylabel(y_label)
        plt.title(title)
        plt.grid(True, linestyle='--', alpha=0.7)
        if 'Cost' in y_label:  # Format y-axis as currency for cost plots
            plt.gca().yaxis.set_major_formatter(mticker.FormatStrFormatter('$%.0f'))
        plt.tight_layout()
        plt.savefig(filename, bbox_inches='tight')
        plt.close()

    # Plot 1: Total Cost vs. S2 Cost
    create_and_save_plot(df_all_results, 'Relative Machine Costs (S2)', 'Parameter_Value', 'Total_Cost',
                         'Total Cost ($)', 'Sensitivity of Total Cost to Machine S2 Operating Cost',
                         "sensitivity_s2_cost_vs_total_cost.png", 'blue')

    # Plot 2: Rejected Batches vs. Penalty Multiplier
    create_and_save_plot(df_all_results, 'Overall Rejection Penalty Scale', 'Parameter_Value', 'Rejected_Batches',
                         'Number of Rejected Batches',
                         'Sensitivity of Rejected Batches to Overall Rejection Penalty Scale',
                         "sensitivity_penalty_vs_rejected_batches.png", 'red')

    # Plot 3: Total Cost vs. Planning Horizon
    create_and_save_plot(df_all_results, 'Planning Horizon (H)', 'Parameter_Value', 'Total_Cost',
                         'Total Cost ($)', 'Sensitivity of Total Cost to Planning Horizon',
                         "sensitivity_horizon_vs_total_cost.png", 'green')

    # Plot 4: Rejected Batches vs. Production Time Multiplier
    create_and_save_plot(df_all_results, 'Batch Production Times', 'Parameter_Value', 'Rejected_Batches',
                         'Number of Rejected Batches', 'Sensitivity of Rejected Batches to Increased Production Times',
                         "sensitivity_prod_time_vs_rejected_batches.png", 'purple')

    # Plot 5: Rejected Batches vs. R-Type Proportion
    create_and_save_plot(df_all_results, 'Batch Type Proportion (R)', 'Parameter_Value', 'Rejected_Batches',
                         'Number of Rejected Batches', 'Sensitivity of Rejected Batches to R-Type Batch Proportion',
                         "sensitivity_r_type_prop_vs_rejected_batches.png", 'orange')


if __name__ == '__main__':
    run_sensitivity_analysis_and_export_excel()