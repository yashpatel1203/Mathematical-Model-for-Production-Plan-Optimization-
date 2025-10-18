import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from ortools.sat.python import cp_model

def solve_production_scheduling(H, NMS, MS_NAMES, WORKSHOPS, BATCHES_DATA_INPUT):
    """
    Implements the production plan optimization model from the article
    'Mathematical Model for Production Plan Optimization-A Case Study of Discrete Event Systems'
    using Google OR-Tools CP-SAT solver.
    """
    # Create a sorted list of original batch IDs for consistent indexing (0-indexed for Python)
    original_batch_ids = sorted(BATCHES_DATA_INPUT.keys())
    # Create a mapping from original 1-indexed ID to 0-indexed ID used in code
    batch_id_to_idx = {batch_id: idx for idx, batch_id in enumerate(original_batch_ids)}

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

    # --- 4. Objective Function ---
    # Equation (1) from the article, interpreted as minimizing the count of rejected batches.
    # "It aims to minimize the non-operated (rejected) batches."
    model.Minimize(sum(is_rejected[i_idx] for i_idx in range(NP)))

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
        print(f"Objective value (total rejected batches): {int(solver.ObjectiveValue())}")

        # Initialize schedule grid for visualization
        schedule_grid = {ms_name: ['-' for _ in range(H)] for ms_name in MS_NAMES}

        assigned_batches_info = []
        rejected_batches_info = []

        for i_idx in range(NP):
            original_id = original_batch_ids[i_idx]
            batch_data = BATCHES_DATA[i_idx]

            if solver.BooleanValue(is_rejected[i_idx]):
                rejected_batches_info.append(
                    f"Batch {original_id} (Type: {batch_data['Type']}, Prod Time: {batch_data['ProductionTime']}h)"
                )
                schedule_data.append({
                    'Batch ID': original_id,
                    'Type': batch_data['Type'],
                    'Production Time (h)': batch_data['ProductionTime'],
                    'Assigned Machine': 'N/A',
                    'Start Time (0-indexed)': 'N/A',
                    'End Time (0-indexed)': 'N/A',
                    'Status': 'Rejected'
                })
            else:
                for l_idx in range(NMS):
                    if solver.BooleanValue(is_assigned[i_idx, l_idx]):
                        start_t = solver.Value(start_time[i_idx, l_idx])
                        end_t = solver.Value(end_time[i_idx, l_idx]) # Exclusive end

                        schedule_data.append({
                            'Batch ID': original_id,
                            'Type': batch_data['Type'],
                            'Production Time (h)': batch_data['ProductionTime'],
                            'Assigned Machine': MS_NAMES[l_idx],
                            'Start Time (0-indexed)': start_t + 1,
                            'End Time (0-indexed)': end_t + 1, # Inclusive end for display
                            'Status': 'Assigned'
                        })
                        # Add data for Gantt plot: (start, duration, batch_label, type, machine_idx)
                        gantt_plot_data[l_idx].append((start_t, batch_data['ProductionTime'], f'B{original_id}', batch_data['Type'], l_idx))

                        assigned_batches_info.append(
                            f"Batch {original_id} (Type: {batch_data['Type']}, Prod Time: {batch_data['ProductionTime']}h) "
                            f"on {MS_NAMES[l_idx]} from time {start_t + 1} to {end_t + 1}"
                        )
                        # Fill the schedule grid for assigned batches
                        for t in range(start_t, end_t):
                            if t < H: # Safety check to prevent index out of bounds if end_t == H
                                schedule_grid[MS_NAMES[l_idx]][t] = f"{original_id}{batch_data['Type'][0]}" # e.g., "2B"

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
        df_schedule.to_csv("production_plan_base.csv", index=False)

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
        plt.savefig("production_plan_base.png", format='png', bbox_inches='tight')
        plt.close(fig)  # Close the figure to free up memory
    else:
        print(f"No solution found. Status: {solver.StatusName(status)}")

if __name__ == '__main__':
    # --- 1. Data Definition ---
    H = 8 # H: Total number of time units (e.g., hours) in the planning horizon.
    NMS = 2 # NMS: Total number of Manufacturing Systems (machines).
    MS_NAMES = ['S2', 'S3'] # Names of machines for easier output interpretation
    WORKSHOPS = {
        'W1': [0, 1] # WORKSHOPS: Defines which machines belong to which workshop.
    }
    # BATCHES_DATA_INPUT: Characteristics of each production batch as provided in Table 1.
    # 'EM': Earliest Manufacturing Date.
    # 'LM': Latest Manufacturing Date.
    # 'Type': Batch type ('R' for Red, 'B' for Blue).
    # 'ProductionTime': Duration in hours required to process the batch.
    BATCHES_DATA_INPUT = {
        1: {'EM': 6, 'LM': 8, 'Type': 'R', 'ProductionTime': 2},
        2: {'EM': 3, 'LM': 4, 'Type': 'B', 'ProductionTime': 2},
        3: {'EM': 2, 'LM': 3, 'Type': 'B', 'ProductionTime': 2},
        4: {'EM': 5, 'LM': 5, 'Type': 'R', 'ProductionTime': 4},
        5: {'EM': 1, 'LM': 2, 'Type': 'B', 'ProductionTime': 1},
        6: {'EM': 5, 'LM': 7, 'Type': 'B', 'ProductionTime': 3},
        7: {'EM': 2, 'LM': 2, 'Type': 'B', 'ProductionTime': 1},
    }
    solve_production_scheduling(H, NMS, MS_NAMES, WORKSHOPS, BATCHES_DATA_INPUT)