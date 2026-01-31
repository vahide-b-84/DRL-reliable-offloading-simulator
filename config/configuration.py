#configuration.py
from scipy.stats import norm

class parameters:
    # ======================================================================
    # experiment setting
    # ======================================================================
    SCENARIO_TYPE = "heterogeneous"  # Options: "homogeneous", "heterogeneous"
    FAILURE_STATE = "high"  # Options: "low", "med", "high"
    model_summary = "ddpg" # Options: "dqn", "ppo","ddpg"  
    total_episodes = 5  # 100

    # ======================================================================
    # Infrastructure: servers
    # ======================================================================
    NUM_EDGE_SERVERS = 6  # 7,8
    NUM_CLOUD_SERVERS = 2  # 3,2
    serverNo = NUM_EDGE_SERVERS + NUM_CLOUD_SERVERS  # 10
    # ======================================================================
    # Workload: tasks
    # ======================================================================
    TASK_ARRIVAL_RATE = 0.5 # Task arrival time, 0.1, 0.2
    TASK_SIZE_RANGE = (10, 100)  # heter
    Low_demand, High_demand = 1, 100 # MIPS (Normal(mean=50, std=16) implied)
    taskno = 200
    # ======================================================================
    # Network model
    # ======================================================================
    # RSU-to-cloud backhaul bandwidth (Mb/s).
    rsu_to_cloud_bandwidth = 8  # Mb/s

    # ======================================================================
    # Reliability model parameters (Edge vs Cloud)
    # ======================================================================
    # Edge reliability: base failure probability for each state (low/med/high).
    # These are interpreted as failure-probability percentiles used later to derive failure rates.
    INITIAL_FAILURE_PROB_LOW_EDGE = 0.0001
    INITIAL_FAILURE_PROB_HIGH_EDGE = 0.79
    INITIAL_FAILURE_PROB_MED_EDGE = 0.55

    # Interval width around the base failure probability for:
    # - homogeneous   : smaller spread across nodes
    # - heterogeneous : larger spread across nodes
    HOMOGENEOUS_INTERVAL_EDGE = 0.10
    HETEROGENEOUS_INTERVAL_EDGE = 0.20
    # Edge processing capacity range (MIPS).
    EDGE_PROCESSING_FREQ_RANGE = (10, 15)  # MIPS
    # Cloud reliability: base failure probabilities (typically much lower than edge).
    INITIAL_FAILURE_PROB_LOW_CLOUD = 1e-6
    INITIAL_FAILURE_PROB_HIGH_CLOUD = 7.9e-6
    INITIAL_FAILURE_PROB_MED_CLOUD = 5.5e-6
    # Interval widths for cloud (small values due to very low failure probabilities).
    HOMOGENEOUS_INTERVAL_CLOUD = 1e-6
    HETEROGENEOUS_INTERVAL_CLOUD = 2e-6
    # Cloud processing capacity range (MIPS).
    CLOUD_PROCESSING_FREQ_RANGE = (30, 60)  # MIPS
    # Mapping from state IDs to labels used in FAILURE_STATE.
    STATES = {
        "S1": "low",
        "S2": "high",
        "S3": "med",
    }

    # ======================================================================
    # RL hyperparameters
    # ======================================================================
    # ----------- DDPG ----------------
    std_dev_ddpg = 0.25
    critic_lr_ddpg = 0.001   #0.002  
    actor_lr_ddpg = 0.0003 #  0.0005  # 0.0005
    gamma_ddpg = 0.85 # 0.9
    tau_ddpg = 0.005 #0.01
    buffer_capacity_ddpg = 100000 #50000
    batch_size_ddpg = 256 #64
    activation_function_ddpg ="softmax" 
    # ------------- DQN --------------
    hidden_layers_dqn = [128, 64]
    af_dqn = "relu"
    lr_dqn = 5e-4
    gamma_dqn = 0.90
    tau_dqn = 0.005
    buffer_capacity_dqn = 200_000
    batch_size_dqn = 256
    epsilon_start_dqn = 1.0
    epsilon_end_dqn = 0.01
    epsilon_decay_dqn = 300
    # ----------- PPO --------------
    hidden_layers_ppo = [64, 32]
    af_ppo = "tanh"
    actor_lr_ppo = 1e-4
    critic_lr_ppo = 5e-4
    gamma_ppo = 0.90
    clip_eps_ppo = 0.2
    k_epochs_ppo = 2
    batch_size_ppo = 64
    entropy_coef_ppo = 0.01
    reward_scale_ppo = 1
    gae_lambda_ppo = 0.95
    value_loss_coef_ppo = 0.5
    max_grad_norm_ppo = 0.5


    # ==========================================================
    # Reliability helper methods
    # ======================================================================

    @staticmethod
    def compute_failure_probabilities():
        """
        Build failure-probability intervals for edge and cloud under:
        - homogeneous scenario type
        - heterogeneous scenario type

        Output format:
            {
              'edge':  {'homogeneous': {'low': (p0, p1), ...}, 'heterogeneous': {...}},
              'cloud': {'homogeneous': {...},               'heterogeneous': {...}}
            }
        """
        return {
            "edge": {
                "homogeneous": {
                    "low": (
                        parameters.INITIAL_FAILURE_PROB_LOW_EDGE,
                        parameters.INITIAL_FAILURE_PROB_LOW_EDGE + parameters.HOMOGENEOUS_INTERVAL_EDGE,
                    ),
                    "high": (
                        parameters.INITIAL_FAILURE_PROB_HIGH_EDGE,
                        parameters.INITIAL_FAILURE_PROB_HIGH_EDGE + parameters.HOMOGENEOUS_INTERVAL_EDGE,
                    ),
                    "med": (
                        parameters.INITIAL_FAILURE_PROB_MED_EDGE,
                        parameters.INITIAL_FAILURE_PROB_MED_EDGE + parameters.HOMOGENEOUS_INTERVAL_EDGE,
                    ),
                },
                "heterogeneous": {
                    "low": (
                        parameters.INITIAL_FAILURE_PROB_LOW_EDGE,
                        parameters.INITIAL_FAILURE_PROB_LOW_EDGE + parameters.HETEROGENEOUS_INTERVAL_EDGE,
                    ),
                    "high": (
                        parameters.INITIAL_FAILURE_PROB_HIGH_EDGE,
                        parameters.INITIAL_FAILURE_PROB_HIGH_EDGE + parameters.HETEROGENEOUS_INTERVAL_EDGE,
                    ),
                    "med": (
                        parameters.INITIAL_FAILURE_PROB_MED_EDGE,
                        parameters.INITIAL_FAILURE_PROB_MED_EDGE + parameters.HETEROGENEOUS_INTERVAL_EDGE,
                    ),
                },
            },
            "cloud": {
                "homogeneous": {
                    "low": (
                        parameters.INITIAL_FAILURE_PROB_LOW_CLOUD,
                        parameters.INITIAL_FAILURE_PROB_LOW_CLOUD + parameters.HOMOGENEOUS_INTERVAL_CLOUD,
                    ),
                    "high": (
                        parameters.INITIAL_FAILURE_PROB_HIGH_CLOUD,
                        parameters.INITIAL_FAILURE_PROB_HIGH_CLOUD + parameters.HOMOGENEOUS_INTERVAL_CLOUD,
                    ),
                    "med": (
                        parameters.INITIAL_FAILURE_PROB_MED_CLOUD,
                        parameters.INITIAL_FAILURE_PROB_MED_CLOUD + parameters.HOMOGENEOUS_INTERVAL_CLOUD,
                    ),
                },
                "heterogeneous": {
                    "low": (
                        parameters.INITIAL_FAILURE_PROB_LOW_CLOUD,
                        parameters.INITIAL_FAILURE_PROB_LOW_CLOUD + parameters.HETEROGENEOUS_INTERVAL_CLOUD,
                    ),
                    "high": (
                        parameters.INITIAL_FAILURE_PROB_HIGH_CLOUD,
                        parameters.INITIAL_FAILURE_PROB_HIGH_CLOUD + parameters.HETEROGENEOUS_INTERVAL_CLOUD,
                    ),
                    "med": (
                        parameters.INITIAL_FAILURE_PROB_MED_CLOUD,
                        parameters.INITIAL_FAILURE_PROB_MED_CLOUD + parameters.HETEROGENEOUS_INTERVAL_CLOUD,
                    ),
                },
            },
        }

    @staticmethod
    def compute_failure_rates():
        """
        Convert failure-probability intervals into failure-rate intervals.

        Approach:
        - Use a Normal distribution over task demand (Low_demand..High_demand) to map probability intervals
          to percentile values via `norm.ppf`.
        - Convert percentile values to rates via inverse (1 / percentile_value).

        Output format mirrors compute_failure_probabilities().
        """
        failure_probs = parameters.compute_failure_probabilities()

        # Normal distribution parameters derived from demand bounds:
        # mean at the midpoint, std so that +/-3 std spans [Low_demand, High_demand]
        mean = (parameters.Low_demand + parameters.High_demand) / 2
        std = (parameters.High_demand - parameters.Low_demand) / 6

        def get_failure_rate_interval(prob_interval):
            # Translate probability interval into percentile values of the demand distribution.
            lower_percentile_value = norm.ppf(1 - prob_interval[0], loc=mean, scale=std)
            upper_percentile_value = norm.ppf(1 - prob_interval[1], loc=mean, scale=std)
            # Convert percentile values into rate bounds (project-specific interpretation).
            return (1 / lower_percentile_value, 1 / upper_percentile_value)

        def compute_all(rtype):
            return {
                "homogeneous": {
                    k: get_failure_rate_interval(v)
                    for k, v in failure_probs[rtype]["homogeneous"].items()
                },
                "heterogeneous": {
                    k: get_failure_rate_interval(v)
                    for k, v in failure_probs[rtype]["heterogeneous"].items()
                },
            }

        return {"edge": compute_all("edge"), "cloud": compute_all("cloud")}

    @staticmethod
    def compute_Alpha():
        """
        Compute alpha parameters used by your simulator (typically for time-varying failure or hazard models).

        For each failure-rate interval (r0, r1), alpha is computed as:
            alpha_0 = (r1 - r0) / taskno
            alpha_1 = r1

        Output format mirrors compute_failure_rates().
        """
        failure_rates = parameters.compute_failure_rates()

        def calc_alpha(rate_interval):
            return ((rate_interval[1] - rate_interval[0]) / parameters.taskno, rate_interval[1])

        def compute_all(rtype):
            return {
                "homogeneous": {k: calc_alpha(v) for k, v in failure_rates[rtype]["homogeneous"].items()},
                "heterogeneous": {k: calc_alpha(v) for k, v in failure_rates[rtype]["heterogeneous"].items()},
            }

        return {"edge": compute_all("edge"), "cloud": compute_all("cloud")}

        """
        Compute alpha parameters used by your simulator (typically for time-varying failure or hazard models).

        For each failure-rate interval (r0, r1), alpha is computed as:
            alpha_0 = (r1 - r0) / taskno
            alpha_1 = r1

        Output format mirrors compute_failure_rates().
        """
        failure_rates = parameters.compute_failure_rates()

        def calc_alpha(rate_interval):
            return ((rate_interval[1] - rate_interval[0]) / parameters.taskno, rate_interval[1])

        def compute_all(rtype):
            return {
                "homogeneous": {k: calc_alpha(v) for k, v in failure_rates[rtype]["homogeneous"].items()},
                "heterogeneous": {k: calc_alpha(v) for k, v in failure_rates[rtype]["heterogeneous"].items()},
            }

        return {"edge": compute_all("edge"), "cloud": compute_all("cloud")}