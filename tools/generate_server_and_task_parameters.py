# generate_server_and_task_parameters.py
# Generates:
# 1) homogeneous_server_info.xlsx  -> 3 sheets: low/high/med
# 2) heterogeneous_server_info.xlsx -> 3 sheets: low/high/med
# 3) task_parameters.xlsx

import os
import pandas as pd
import random
import numpy as np
from scipy.stats import truncnorm

from config.configuration import parameters
from config.paths import DATA_DIR, ensure_dirs


NUM_EDGE_SERVERS = parameters.NUM_EDGE_SERVERS
NUM_CLOUD_SERVERS = parameters.NUM_CLOUD_SERVERS


def generate_processing_frequencies(number_of_server: int, server_type: str):
    """Generate processing frequencies for edge or cloud servers."""
    if server_type.lower() == "edge":
        return [round(random.uniform(*parameters.EDGE_PROCESSING_FREQ_RANGE), 2) for _ in range(number_of_server)]
    elif server_type.lower() == "cloud":
        return [round(random.uniform(*parameters.CLOUD_PROCESSING_FREQ_RANGE), 2) for _ in range(number_of_server)]
    else:
        raise ValueError("server_type must be 'edge' or 'cloud'")


def generate_server_info_per_state(scenario_type: str, filename: str):
    """
    Create one Excel file with exactly 3 sheets (low/high/med) for the given scenario_type.
    Each sheet includes both Edge servers and Cloud servers.

    Sheet columns:
      Server_ID, Server_Type, Processing_Frequency, Failure_Rate
    """
    failure_rates = parameters.compute_failure_rates()
    state_types = ["low", "high", "med"]  # order like your sample

    for idx, state_type in enumerate(state_types):
        server_counter = 1
        server_info = []
        columns = ["Server_ID", "Server_Type", "Processing_Frequency", "Failure_Rate"]

        # -------- Edge servers --------
        edge_freqs = generate_processing_frequencies(NUM_EDGE_SERVERS, "edge")
        for i in range(NUM_EDGE_SERVERS):
            interval = failure_rates["edge"][scenario_type][state_type]
            failure_rate = round(random.uniform(*interval), 6)

            server_info.append([
                server_counter,
                "Edge",
                edge_freqs[i],
                failure_rate
            ])
            server_counter += 1

        # -------- Cloud servers --------
        cloud_freqs = generate_processing_frequencies(NUM_CLOUD_SERVERS, "cloud")
        for i in range(NUM_CLOUD_SERVERS):
            interval = failure_rates["cloud"][scenario_type][state_type]
            failure_rate = round(random.uniform(*interval), 6)

            server_info.append([
                server_counter,
                "Cloud",
                cloud_freqs[i],
                failure_rate
            ])
            server_counter += 1

        df = pd.DataFrame(server_info, columns=columns)

        sheet_name = f"{scenario_type.capitalize()}_state_{state_type}"
        mode = "w" if idx == 0 else "a"
        with pd.ExcelWriter(filename, engine="openpyxl", mode=mode) as writer:
            df.to_excel(writer, sheet_name=sheet_name, index=False)


def generate_task_params(filename: str = "task_parameters.xlsx"):
    """
    Generate task parameters Excel:
      Task_ID, Task_Size, Computation_Demand
    """
    task_info = []
    NUM_TASKS = parameters.taskno
    TASK_SIZE_RANGE = parameters.TASK_SIZE_RANGE

    a, b = parameters.Low_demand, parameters.High_demand
    mu = (a + b) / 2
    sigma = (b - a) / 6
    lower, upper = (a - mu) / sigma, (b - mu) / sigma

    for task_id in range(1, NUM_TASKS + 1):
        # Task_Size (integer)
        task_size = np.random.randint(TASK_SIZE_RANGE[0], TASK_SIZE_RANGE[1] + 1)

        # Computation_Demand (float)
        computation_demand = truncnorm.rvs(lower, upper, loc=mu, scale=sigma)

        task_info.append([task_id, task_size, float(computation_demand)])

    task_df = pd.DataFrame(task_info, columns=["Task_ID", "Task_Size", "Computation_Demand"])
    task_df.to_excel(filename, index=False)


def main():
    """Write all Excel parameter files into data/... (not project root)."""
    ensure_dirs()

    homogeneous_path = os.path.join(DATA_DIR, "homogeneous_server_info.xlsx")
    heterogeneous_path = os.path.join(DATA_DIR, "heterogeneous_server_info.xlsx")
    tasks_path = os.path.join(DATA_DIR, "task_parameters.xlsx")

    generate_server_info_per_state("homogeneous", homogeneous_path)
    generate_server_info_per_state("heterogeneous", heterogeneous_path)
    generate_task_params(tasks_path)
    print("Parameters defined in Excel files!")


if __name__ == "__main__":
    main()
