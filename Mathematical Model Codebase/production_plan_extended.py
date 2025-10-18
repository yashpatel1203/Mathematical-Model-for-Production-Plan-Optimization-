import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from ortools.sat.python import cp_model


def generate_synthetic_batch_data(
        num_batches: int,
        planning_horizon_H: int,
        avg_prod_time: int,
        prod_time_variance: int,
        type_R_proportion: float,
        deadline_flexibility: int,  # Avg difference between LM and EM
        avg_penalty_multiplier: float = 200,  # Base penalty value to multiply by prod time
        penalty_variance_percent: float = 0.2  # +/- 20% variance
) -> tuple[dict, dict]:
    """
    Generates synthetic batch data and corresponding rejection penalties for the production planning model.

    Args:
        num_batches (int): The total number of batches to generate.
        planning_horizon_H (int): The total planning horizon (H).
        avg_prod_time (int): Average production time for a batch.
        prod_time_variance (int): Max variance from avg_prod_time for production time.
        type_R_proportion (float): Proportion of 'R' type batches (0.0 to 1.0).
        deadline_flexibility (int): Average window size for LM - EM.
        avg_penalty_multiplier (float): A base value to multiply by production time for penalty calculation.
                                        Adjust this to set the general scale of penalties.
        penalty_variance_percent (float): Percentage variance for penalties (e.g., 0.2 for +/- 20%).

    Returns:
        tuple[dict, dict]: A tuple containing two dictionaries:
                           1. BATCHES_DATA_RAW (batch characteristics).
                           2. REJECTION_PENALTY_PER_BATCH (penalties, 0-indexed for direct use).
    """
    generated_batches_data = {}
    generated_penalties = {}
    original_batch_ids_list = []  # To keep track of original 1-indexed IDs

    for i in range(1, num_batches + 1):  # Start batch IDs from 1
        original_batch_ids_list.append(i)  # Add original 1-indexed ID

        # Randomly determine batch type
        batch_type = 'R' if random.random() < type_R_proportion else 'B'

        # Generate production time
        prod_time = max(1, avg_prod_time + random.randint(-prod_time_variance, prod_time_variance))

        # Generate EM and LM
        em_lower_bound = 1
        em_upper_bound = planning_horizon_H - prod_time
        em = random.randint(em_lower_bound, em_upper_bound)

        min_lm = em + max(0, prod_time - 1)
        lm = min(planning_horizon_H, em + random.randint(0, deadline_flexibility))

        if lm < min_lm:
            lm = min_lm
        if lm < em:
            lm = em

        generated_batches_data[i] = {
            'EM': em,
            'LM': lm,
            'Type': batch_type,
            'ProductionTime': prod_time
        }

        # Generate penalty for this batch
        # A simple approach: penalty proportional to production time and a base multiplier
        base_penalty = prod_time * avg_penalty_multiplier
        variance_amount = base_penalty * penalty_variance_percent
        penalty = int(base_penalty + random.uniform(-variance_amount, variance_amount))
        generated_penalties[i] = max(100, penalty)  # Ensure a minimum penalty

    # Now, convert the penalties to be 0-indexed using original_batch_ids_list for consistency
    # with how your solve_production_scheduling function processes BATCHES_DATA
    batch_id_to_idx = {batch_id: idx for idx, batch_id in enumerate(sorted(generated_batches_data.keys()))}

    indexed_penalties = {}
    for original_id, penalty_value in generated_penalties.items():
        indexed_penalties[batch_id_to_idx[original_id]] = penalty_value

    return generated_batches_data, indexed_penalties

def solve_production_scheduling_extended(H, NMS, MS_NAMES, WORKSHOPS, BATCHES_DATA_INPUT, COST_PER_HOUR_ON_MACHINE, REJECTION_PENALTY_PER_BATCH):
    """
    Implements the production plan optimization model from the article
    'Mathematical Model for Production Plan Optimization-A Case Study of Discrete Event Systems'
    using Google OR-Tools CP-SAT solver, with an extended cost-based objective.
    """
    # Create a sorted list of original batch IDs for consistent indexing (0-indexed for Python)
    original_batch_ids = sorted(BATCHES_DATA_INPUT.keys())
    # Create a mapping from original 1-indexed ID to 0-indexed ID used in code
    batch_id_to_idx = {batch_id: idx for idx, batch_id in enumerate(original_batch_ids)}
    results = {}
    # Convert raw batch data to 0-indexed EM/LM for use with 0-indexed time horizon
    BATCHES_DATA = {}
    for original_id, data in BATCHES_DATA_INPUT.items():
        idx = batch_id_to_idx[original_id]
        BATCHES_DATA[idx] = {
            # Convert EM/LM from 1-indexed (e.g., hour 1) to 0-indexed (e.g., time slot 0)
            'EM': data['EM'] - 1,
            'LM': data['LM'] - 1,
            'Type': data['Type'],
            'ProductionTime': data['ProductionTime']
        }
    # Total number of batches (after indexing conversion)
    NP = len(BATCHES_DATA)

    # --- 2. Model Initialization ---
    model = cp_model.CpModel()

    # --- 3. Decision Variables ---

    # start_time[i_idx, l_idx]: Start time of batch i_idx on system l_idx (integer variable)
    # The domain is [EM_i, LM_i] (0-indexed).
    start_time = {}
    # end_time[i_idx, l_idx]: End time of batch i_idx on system l_idx (integer variable)
    # This is start_time + ProductionTime. Domain is derived from start_time and ProductionTime.
    end_time = {}
    # is_assigned[i_idx, l_idx]: Boolean variable, true if batch i_idx is assigned to system l_idx
    is_assigned = {}
    # interval[i_idx, l_idx]: Optional Interval variable for batch i_idx on system l_idx.
    # An optional interval is only active if its 'is_present' literal (is_assigned here) is true.
    interval = {}
    # is_rejected[i_idx]: Boolean variable, true if batch i_idx is not operated (rejected)
    is_rejected = {}

    for i_idx in range(NP):
        # A batch is either rejected or assigned to a machine.
        is_rejected[i_idx] = model.NewBoolVar(f'is_rejected_batch_{original_batch_ids[i_idx]}')

        # Define bounds for start and end times for each possible assignment
        # EM and LM are interpreted as 0-indexed earliest and latest *start* times.
        min_start_i = BATCHES_DATA[i_idx]['EM']
        max_start_i = BATCHES_DATA[i_idx]['LM']
        prod_time_i = BATCHES_DATA[i_idx]['ProductionTime']

        for l_idx in range(NMS):
            # start_time: Can range from its earliest allowed start to its latest allowed start
            # or the latest possible start that allows it to finish within the horizon H.
            start_time[i_idx, l_idx] = model.NewIntVar(
                min_start_i,
                max_start_i,
                f'start_batch_{original_batch_ids[i_idx]}_on_ms_{MS_NAMES[l_idx]}'
            )
            # end_time: Will be start_time + prod_time. Its max value can be H (exclusive end).
            end_time[i_idx, l_idx] = model.NewIntVar(
                prod_time_i, # Minimum end time (if starts at 0)
                H,           # Maximum end time (exclusive, so max time slot index is H-1)
                f'end_batch_{original_batch_ids[i_idx]}_on_ms_{MS_NAMES[l_idx]}'
            )

            is_assigned[i_idx, l_idx] = model.NewBoolVar(
                f'is_assigned_batch_{original_batch_ids[i_idx]}_to_ms_{MS_NAMES[l_idx]}'
            )

            # Create an optional interval variable.
            # This implicitly enforces `end_time - start_time == prod_time` if `is_assigned` is true.
            interval[i_idx, l_idx] = model.NewOptionalIntervalVar(
                start_time[i_idx, l_idx],
                prod_time_i,
                end_time[i_idx, l_idx],
                is_assigned[i_idx, l_idx],
                f'interval_batch_{original_batch_ids[i_idx]}_on_ms_{MS_NAMES[l_idx]}'
            )

    # --- 4. Objective Function (Extended: Minimize Total Cost) ---
    total_production_cost = []
    for i_idx in range(NP):
        for l_idx in range(NMS):
            # The production cost for a batch is its production time multiplied by the
            # cost per hour of the machine it's assigned to.
            # is_assigned[i_idx, l_idx] acts as a 0/1 multiplier here.
            cost_if_assigned = BATCHES_DATA[i_idx]['ProductionTime'] * COST_PER_HOUR_ON_MACHINE[l_idx]
            total_production_cost.append(
                cost_if_assigned * is_assigned[i_idx, l_idx]
            )

    total_rejection_penalty = []
    for i_idx in range(NP):
        # Add the penalty for this batch if it is rejected
        total_rejection_penalty.append(
            REJECTION_PENALTY_PER_BATCH[i_idx] * is_rejected[i_idx]
        )

    # Set the objective function to minimize the sum of total production costs and total rejection penalties
    model.Minimize(sum(total_production_cost) + sum(total_rejection_penalty))

    # --- 5. Constraints ---

    # Constraint 1: Each batch is either assigned to exactly one machine or it is rejected.
    # This combines aspects of Equation (5) and (6) and the definition of F_i.
    for i_idx in range(NP):
        model.Add(sum(is_assigned[i_idx, l_idx] for l_idx in range(NMS)) + is_rejected[i_idx] == 1)

    # Constraint 2: No Overlap on Manufacturing Systems (Equation 3).
    # A manufacturing system operates at most one batch in a time unit t.
    for l_idx in range(NMS):
        # Collect all intervals for this specific machine.
        # AddNoOverlap ensures that no two intervals on the same resource overlap in time.
        intervals_on_machine = [interval[i_idx, l_idx] for i_idx in range(NP)]
        model.AddNoOverlap(intervals_on_machine)

    # Constraint 3: Earliest and Latest Manufacturing Dates (Equations 7 & 8).
    # These are already incorporated into the domain definitions of `start_time` variables.
    # Additionally, ensure that if a batch is assigned, its end time is within the planning horizon.
    for i_idx in range(NP):
        for l_idx in range(NMS):
            # Ensure the activity finishes within the horizon H (exclusive upper bound for time range).
            # This is critical as `LM` is for start time, not end time of the batch.
            model.Add(end_time[i_idx, l_idx] <= H).OnlyEnforceIf(is_assigned[i_idx, l_idx])

    # Constraint 4: Batches of different types cannot be operated in the same workshop at a given unit of time.
    # (Adaptation of Equations 10-12).
    # This ensures that for any given workshop, at any given time, all machines in that workshop
    # are either idle or processing batches of the *same* type.
    for workshop_name, machine_indices in WORKSHOPS.items():
        for t in range(H): # Iterate through each time unit in the horizon
            # Boolean variables to indicate if Type 'A' (Red) or Type 'B' (Blue) batches are active
            # in this workshop at this specific time 't'.
            type_R_active_overall = model.NewBoolVar(f'type_R_overall_W{workshop_name}_T{t}')
            type_B_active_overall = model.NewBoolVar(f'type_B_overall_W{workshop_name}_T{t}')

            # List of literals (boolean variables) that would make type_R_active_overall true.
            literals_type_R_at_t = []
            # List of literals that would make type_B_active_overall true.
            literals_type_B_at_t = []

            for l_idx in machine_indices: # For each machine belonging to the current workshop
                for i_idx in range(NP): # For each batch
                    batch_type = BATCHES_DATA[i_idx]['Type']

                    # Create boolean variables to check if the batch is active on this machine at time 't'.
                    # This effectively acts as the X_i_l_t variable from the paper if it were defined for specific times.
                    is_batch_active_at_t = model.NewBoolVar(f'active_batch_{original_batch_ids[i_idx]}_ms_{MS_NAMES[l_idx]}_at_t_{t}')

                    # Define auxiliary boolean conditions for clarity and proper reification.
                    # start_cond: True if the batch's start time is <= current time 't'.
                    start_cond = model.NewBoolVar(f'start_cond_{original_batch_ids[i_idx]}_{MS_NAMES[l_idx]}_t_{t}')
                    model.Add(start_time[i_idx, l_idx] <= t).OnlyEnforceIf(start_cond)
                    model.Add(start_time[i_idx, l_idx] > t).OnlyEnforceIf(start_cond.Not())

                    # end_cond: True if the batch's end time is > current time 't'.
                    end_cond = model.NewBoolVar(f'end_cond_{original_batch_ids[i_idx]}_{MS_NAMES[l_idx]}_t_{t}')
                    model.Add(end_time[i_idx, l_idx] > t).OnlyEnforceIf(end_cond)
                    model.Add(end_time[i_idx, l_idx] <= t).OnlyEnforceIf(end_cond.Not())

                    # is_batch_active_at_t is true if AND ONLY IF:
                    # 1. The batch is assigned to this machine (is_assigned[i_idx, l_idx] is true)
                    # 2. The batch has started by or before 't' (start_cond is true)
                    # 3. The batch has not yet finished by 't' (end_cond is true)
                    model.AddBoolAnd([is_assigned[i_idx, l_idx], start_cond, end_cond]).OnlyEnforceIf(is_batch_active_at_t)
                    model.AddBoolOr([is_assigned[i_idx, l_idx].Not(), start_cond.Not(), end_cond.Not()]).OnlyEnforceIf(is_batch_active_at_t.Not())

                    # Add this boolean literal to the appropriate type list for this time 't'.
                    if batch_type == 'R': # 'R' for Red (Type 'A' in article's equations, but 'R' in data)
                        literals_type_R_at_t.append(is_batch_active_at_t)
                    elif batch_type == 'B': # 'B' for Blue (Type 'B' in article's equations, and 'B' in data)
                        literals_type_B_at_t.append(is_batch_active_at_t)

            # If any literal in literals_type_R_at_t is true, then type_R_active_overall must be true.
            # If all literals are false, then type_R_active_overall must be false.
            if literals_type_R_at_t: # Only add if there are actual literals to check
                model.AddBoolOr(literals_type_R_at_t).OnlyEnforceIf(type_R_active_overall)
                model.AddBoolAnd([literal.Not() for literal in literals_type_R_at_t]).OnlyEnforceIf(type_R_active_overall.Not())
            else: # If no 'R' type batches can ever be active here, then it's always false
                model.Add(type_R_active_overall == False)

            # Same logic for Type 'B' batches.
            if literals_type_B_at_t:
                model.AddBoolOr(literals_type_B_at_t).OnlyEnforceIf(type_B_active_overall)
                model.AddBoolAnd([literal.Not() for literal in literals_type_B_at_t]).OnlyEnforceIf(type_B_active_overall.Not())
            else:
                model.Add(type_B_active_overall == False)

            # The core workshop type constraint: A workshop cannot process both 'R' and 'B' types simultaneously.
            model.Add(type_R_active_overall + type_B_active_overall <= 1)

    # --- 6. Solve the Model ---
    solver = cp_model.CpSolver()
    # You can enable logging to see the solver's progress
    # solver.parameters.log_search_progress = True
    status = solver.Solve(model)

    # --- 7. Print Results ---
    schedule_data = [] # For CSV
    gantt_plot_data = {ms_idx: [] for ms_idx in range(NMS)} # For Matplotlib Gantt
    if status == cp_model.OPTIMAL or status == cp_model.FEASIBLE:
        print(f"Solution Status: {solver.StatusName(status)}")
        # Print the new objective value
        print(f"Objective value (total cost): {solver.ObjectiveValue():.2f}")
        results['status'] = solver.StatusName(status)
        results['total_cost'] = solver.ObjectiveValue()
        results['rejected_batches_count'] = sum(
            solver.BooleanValue(is_rejected[i_idx]) for i_idx in range(NP)
        )
        # Initialize schedule grid for visualization
        schedule_grid = {ms_name: ['-' for _ in range(H)] for ms_name in MS_NAMES}

        assigned_batches_info = []
        rejected_batches_info = []

        for i_idx in range(NP):
            original_id = original_batch_ids[i_idx]
            batch_data = BATCHES_DATA[i_idx]

            if solver.BooleanValue(is_rejected[i_idx]):
                rejected_batches_info.append(
                    f"Batch {original_id} (Type: {batch_data['Type']}, Prod Time: {batch_data['ProductionTime']}h, Penalty: ${REJECTION_PENALTY_PER_BATCH[i_idx]})"
                )
                schedule_data.append({
                    'Batch ID': original_id,
                    'Type': batch_data['Type'],
                    'Production Time (h)': batch_data['ProductionTime'],
                    'Assigned Machine': 'N/A',
                    'Start Time': 'N/A',
                    'End Time': 'N/A',
                    'Status': 'Rejected',
                    'Penalty Cost': f"$ {REJECTION_PENALTY_PER_BATCH[i_idx]:.2f}",
                    'Production Cost': f"$ {0:.2f}",
                })
            else:
                for l_idx in range(NMS):
                    if solver.BooleanValue(is_assigned[i_idx, l_idx]):
                        start_t = solver.Value(start_time[i_idx, l_idx])
                        end_t = solver.Value(end_time[i_idx, l_idx]) # Exclusive end
                        production_cost = BATCHES_DATA[i_idx]['ProductionTime'] * COST_PER_HOUR_ON_MACHINE[l_idx]
                        schedule_data.append({
                            'Batch ID': original_id,
                            'Type': batch_data['Type'],
                            'Production Time (h)': batch_data['ProductionTime'],
                            'Assigned Machine': MS_NAMES[l_idx],
                            'Start Time': start_t + 1,
                            'End Time': end_t + 1, # Inclusive end for display
                            'Status': 'Assigned',
                            'Penalty Cost': f"$ {0:.2f}",
                            'Production Cost': f"$ {production_cost:.2f}"
                        })
                        # Add data for Gantt plot: (start, duration, batch_label, type, machine_idx)
                        gantt_plot_data[l_idx].append((start_t, batch_data['ProductionTime'], f'B{original_id}', batch_data['Type'], l_idx))

                        assigned_batches_info.append(
                            f"Batch {original_id} (Type: {batch_data['Type']}, Prod Time: {batch_data['ProductionTime']}h, Cost: ${production_cost:.2f}) "
                            f"on {MS_NAMES[l_idx]} from time {start_t + 1} to {end_t + 1}"
                        )
                        # Fill the schedule grid for assigned batches
                        for t in range(start_t, end_t):
                            if t < H: # Safety check to prevent index out of bounds if end_t == H
                                schedule_grid[MS_NAMES[l_idx]][t] = f"{original_id}{batch_data['Type'][0]}" # e.g., "2B"
        results['detailed_schedule'] = schedule_data
        print("\n--- Assigned Batches ---")
        if assigned_batches_info:
            for info in assigned_batches_info:
                print(info)
        else:
            print("No batches were assigned.")

        print("\n--- Rejected Batches ---")
        if rejected_batches_info:
            for info in rejected_batches_info:
                print(info)
        else:
            print("No batches were rejected!")

        print("\n--- Production Schedule ---")
        # Header row for time slots
        header = "MS   | " + " | ".join(f"{t+1:2d}" for t in range(H))
        print(header)
        print("-" * len(header)) # Separator line

        # Rows for each manufacturing system's schedule
        for ms_name in MS_NAMES:
            row_content = " | ".join(f"{item:2s}" for item in schedule_grid[ms_name])
            print(f"{ms_name:4s} | {row_content}")

        # --- Generate CSV Output using Pandas ---
        df_schedule = pd.DataFrame(schedule_data)
        df_schedule.to_csv("production_plan_extended.csv", index=False)

        # --- Generate Matplotlib Gantt Chart Image ---
        fig, ax = plt.subplots(figsize=(12, 6))

        # Color mapping for batch types
        type_colors = {'R': 'lightcoral', 'B': 'skyblue'}
        color_patches = []
        for _type, color in type_colors.items():
            color_patches.append(mpatches.Patch(color=color, label=f'Type {_type}'))

        # Plot each assigned batch as a horizontal bar
        for ms_idx, batches in gantt_plot_data.items():
            for start, duration, label, batch_type, _ in batches:
                ax.barh(MS_NAMES[ms_idx], duration, left=start, height=0.6,
                        color=type_colors.get(batch_type, 'gray'),
                        edgecolor='black', linewidth=0.8,
                        label=f'{label} (Type {batch_type})')
                # Add batch label in the middle of the bar
                ax.text(start + duration / 2, MS_NAMES[ms_idx], label,
                        ha='center', va='center', color='black', fontsize=9)

        # Configure plot aesthetics
        ax.set_title('Production Schedule Gantt Chart (0-indexed time)', fontsize=14)
        ax.set_xlabel('Time Units (h)', fontsize=12)
        ax.set_ylabel('Manufacturing System', fontsize=12)
        ax.set_xlim(0, H)
        ax.set_xticks(np.arange(0, H + 1, 1))  # Show ticks for each hour
        ax.set_yticks(range(NMS))
        ax.set_yticklabels(MS_NAMES)
        ax.invert_yaxis()  # To show MS_NAMES from top to bottom

        ax.grid(True, linestyle='--', alpha=0.6)
        ax.legend(handles=color_patches, title="Batch Types", bbox_to_anchor=(1.05, 1), loc='upper left')

        plt.tight_layout()

        # Save plot to a base64 string
        plt.savefig("production_plan_extended.png", format='png', bbox_inches='tight')
        plt.close(fig)  # Close the figure to free up memory
    else:
        results['status'] = solver.StatusName(status)
        results['total_cost'] = float('inf') # Indicate infeasibility with a very high cost
        results['rejected_batches_count'] = NP # Assume all are implicitly rejected
        results['detailed_schedule'] = [] # No schedule possible
        print(f"No solution found. Status: {solver.StatusName(status)}")
    return results

if __name__ == '__main__':
    # --- 1. Data Definition ---
    H = 8 # H: Total number of time units (e.g., hours) in the planning horizon.
    NMS = 2 # NMS: Total number of Manufacturing Systems (machines).
    MS_NAMES = ['S2', 'S3'] # Names of machines for easier output interpretation
    WORKSHOPS = {
        'W1': [0, 1] # WORKSHOPS: Defines which machines belong to which workshop.
    }

    # --- New Data Parameters for Extension and synthetic data ---
    # BATCHES_DATA_INPUT: Characteristics of each production batch as provided in Table 1.
    # 'EM': Earliest Manufacturing Date.
    # 'LM': Latest Manufacturing Date.
    # 'Type': Batch type ('R' for Red, 'B' for Blue).
    # 'ProductionTime': Duration in hours required to process the batch.
    BATCHES_DATA_INPUT, REJECTION_PENALTY_PER_BATCH = generate_synthetic_batch_data(
        num_batches=10,
        planning_horizon_H=12,
        avg_prod_time=3,
        prod_time_variance=1,
        type_R_proportion=0.5,
        deadline_flexibility=2,
        avg_penalty_multiplier=150,
        penalty_variance_percent=0.1
    )
    # Cost per hour for operating each manufacturing system (S2, S3)
    COST_PER_HOUR_ON_MACHINE = {
        0: 100,  # Cost per hour for Machine S2 (index 0)
        1: 120   # Cost per hour for Machine S3 (index 1)
    }
    solve_production_scheduling_extended(H, NMS, MS_NAMES, WORKSHOPS, BATCHES_DATA_INPUT, COST_PER_HOUR_ON_MACHINE, REJECTION_PENALTY_PER_BATCH)