# DRL-Based Reliable Offloading Simulator

This repository provides a **generic and modular simulator** for reliability-aware
task offloading in **distributed Edge/Cloud computing systems**.
The simulator supports pluggable Deep Reinforcement Learning (DRL) agents
(e.g., **DQN**, **PPO**, **DDPG**) and is not tied to any specific application domain
such as vehicular or RSU-based systems.

All input Excel files are stored in `data/`, and all experiment outputs are written to `results`.

---

## Project structure

- `Project_main.py`  
  Main entry point for running the simulator.

- `config/`  
  Experiment configuration and centralized paths:
  - `configuration.py` – scenario type, failure state, agent selection, hyperparameters  
  - `params.py` – unified parameter object  
  - `paths.py` – single source of truth for project paths (root, `data`, `results`)

- `core/`  
  Simulation core (environment, state representation, episode loop, tasks, servers).

- `agents/`  
  DRL agents (DQN / PPO / DDPG), implemented as interchangeable modules.

- `tools/`  
  Utility scripts (e.g., generation of input Excel parameter files).

- `io_utils/`  
  Result logging and post-processing utilities.

- `data/`  
  Input Excel files (server and task parameters).

- `results/`  
  Output Excel files generated per scenario, state, and model.

---

## Execution workflow

### 1) Pre-process (generate input Excel files)

Run this step **only if** the input Excel files are missing or need to be regenerated:

```bash
python pre_process.py
```

This script generates all required Excel files into the `data/` directory.

---

### 2) Run the simulation

```bash
python Project_main.py
```

Simulation results are automatically written to:

```
results/<scenario>_<state>_results/<model>_<scenario>_<state>.xlsx
```

---

### 3) Post-process results (optional)

```bash
python post_process.py
```

This step augments the result workbooks with additional analysis sheets and generates
a global aggregated file (e.g., `Final_Result_All.xlsx`) inside the `results` directory.

---

## Switching DRL agents and experiment setup

The learning algorithm and reliability scenario are controlled via
`config/configuration.py`, which serves as the main experiment configuration file.

Key parameters include:

- `model_summary = "dqn" | "ppo" | "ddpg"`  
  Selects the DRL algorithm used for decision making. The corresponding agent
  implementation is instantiated from the `agents/` directory.

- `SCENARIO_TYPE = "homogeneous" | "heterogeneous"`  
  Specifies how failure probabilities are distributed across computing nodes.
  In the homogeneous case, nodes share similar failure ranges, while in the
  heterogeneous case, failure characteristics vary across nodes.

- `FAILURE_STATE = "low" | "med" | "high"`  
  Defines the base reliability level of the system. This parameter directly
  affects the failure-rate values loaded from the input Excel files.

By modifying these parameters, different experimental scenarios can be executed
without any changes to the simulation core or agent implementations.

---

## Agent interface contract

The simulation core depends only on a **minimal agent interface**, which ensures
that learning algorithms can be replaced without modifying the environment logic.

- **Discrete-action agents (DQN / PPO):**  
  ```
  select_action(state) -> int
  ```

- **Continuous scoring agents (DDPG):**  
  ```
  policy(state) -> score_vector
  ```

The final action selection (e.g., `argmax` over scores) is handled inside the
simulation core.

---

## Adding a new DRL agent

New DRL algorithms can be integrated in a low-risk and incremental manner:

1. Create a new agent implementation inside the `agents/` directory
   (e.g., `agents/a2c_agent.py`).

2. Implement the required action-selection interface expected by the simulation core
   (either `select_action` or `policy`, depending on the action space).

3. Register the new agent in the model construction logic
   (e.g., within `build_model()` in `Project_main.py`).

4. Set the corresponding value of `model_summary` in `config/configuration.py`.

This design allows new learning methods to be added without altering the
simulation environment or episode loop.

---

## Notes

- All scripts should be executed from the **project root**.
- Root-level launcher scripts (`pre_process.py`, `post_process.py`) are provided to
  avoid Python import issues when running utility modules.
- Input data and results are strictly separated into `data` and `results` to ensure
  reproducibility and clean experiment management.
