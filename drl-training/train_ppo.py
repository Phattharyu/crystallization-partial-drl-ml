import os
import gymnasium as gym
import numpy as np
import xgboost as xgb
from pcgym import make_env
from stable_baselines3 import PPO
from custom_env_wrapper import CrystalObservationWrapper

def main():
    print("Initializing Crystallization Loop...")

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
    
    # Establish local file paths securely
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    x_scaler = os.path.join(SCRIPT_DIR, "../ml-training/X_scaler.pkl")
    mu3_scaler = os.path.join(SCRIPT_DIR, "../ml-training/y_scaler.pkl")
    xgb_model = os.path.join(SCRIPT_DIR, "../ml-training/xgb_model.ubj")

    # Apply your wrapper adapter
    env = CrystalObservationWrapper(
        env=env,
        x_scaler_path=x_scaler,
        mu3_path=mu3_scaler,
        xgb_model_path=xgb_model
    )
    
    print("Wrapper injected successfully. Commencing PPO Training...")
    
    # Setup PPO Model
    model = PPO("MlpPolicy", env, verbose=1, tensorboard_log="./ppo_tensorboard/")
    
    try:
        # Train your model 
        model.learn(total_timesteps=30000, progress_bar=True)
        model.save("ppo_crystallization_expert")
        print("Training complete! Model saved.")
    finally:
        # Safely shut down environment to prevent system memory leaks
        env.close()

if __name__ == "__main__":
    main()