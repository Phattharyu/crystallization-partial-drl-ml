import gymnasium as gym
import pcgym  # or your environment's exact import name

print("Libraries imported successfully!")
try:
    # Test a basic step to ensure memory allocation functions normally
    import numpy as np
    from pcgym import make_env

    # =====================================================================
    # UPDATED TIMELINE PARAMETERS
    # =====================================================================
    T = 30.0       # Total time horizon = 30 min
    nsteps = 30    # N = 30 discrete time steps (resulting in dt = 1.0 min)

    # =====================================================================
    # TARGET SETPOINTS (SP) FROM TABLE 2
    # =====================================================================
    SP = {
        'CV': [1.00 for _ in range(nsteps)],
        'Ln': [15.00 for _ in range(nsteps)]
    }

    # =====================================================================
    # UPDATED ACTION SPACE (Incremental control bounds)
    # =====================================================================
    action_space = {
        'low': np.array([-1.0]),   # Minimum delta T step allowed per minute
        'high': np.array([1.0])    # Maximum delta T step allowed per minute
    }

    # =====================================================================
    # OBSERVATION SPACE (State Bounds from Table 2 + Setpoints)
    # =====================================================================
    # Order: [mu0, mu1, mu2, mu3, C, CV, Ln, CV_SP, Ln_SP]
    observation_space = {
        'low' : np.array([0.0, 0.0, 0.0, 0.0, 0.00, 0.00, 0.00, 0.00, 0.00]),
        'high' : np.array([1.0e20, 1.0e20, 1.0e20, 1.0e20, 0.50, 2.00, 20.00, 2.00, 20.00])  
    }

    # =====================================================================
    # ENVIRONMENT PARAMETERS SETUP
    # =====================================================================
    env_params = {
        'N': nsteps, 
        'tsim': T, 
        'SP': SP, 
        'o_space': observation_space, 
        'a_space': action_space,
        
        # Initial conditions (x0) including the initial setpoints
        'x0': np.array([1.50e3, 2.30e4, 1.80e6, 2.50e8, 0.16, 1.00, 15.00, 1.00, 15.00]),
        
        'r_scale': {
            'CV': 1e1,
            'Ln': 1e0
        },
        
        'model': 'crystallization', 
        
        'normalise_a': True, 
        'normalise_o': True, 
        'noise': True, 
        'integration_method': 'jax', 
        'noise_percentage': 0.01, 
    }
    # env_params['custom_reward'] = oracle_reward
    # Initialize Environment
    env = make_env(env_params)
    obs, info = env.reset()
    print("Environment initialized! Initial observation:", obs)
    env.close()
except Exception as e:
    print("An error occurred:", e)