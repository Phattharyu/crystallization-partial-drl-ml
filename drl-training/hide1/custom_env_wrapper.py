import os
import gymnasium as gym
import joblib
import numpy as np
import xgboost as xgb


class CrystalObservationWrapper(gym.Wrapper):

    def __init__(self, env, x_scaler_path, mu3_path, xgb_model_path):
        super().__init__(env)

        # 1. Load your separate SL scalers using joblib
        self.X_scaler = joblib.load(x_scaler_path)
        self.mu3_scaler = joblib.load(mu3_path)

        # # 2. Convert the model path to an absolute path to prevent folder mismatches
        # if not os.isabs(xgb_model_path):
        #     # If a relative path like "../ml-training/..." is given, anchor it to this script's directory
        #     base_dir = os.path.dirname(os.path.abspath(__file__))
        #     xgb_model_path = os.path.abspath(
        #         os.path.join(base_dir, xgb_model_path)
        #     )

        # 3. Load native XGBoost model using the Booster API
        # self.xgb_model = joblib.load(xgb_model_path)
        self.xgb_model = xgb.XGBRegressor()
        self.xgb_model.load_model(xgb_model_path)

    def step(self, action):
        # Step the actual pc-gym environment using the agent's action
        obs, reward, terminated, truncated, info = self.env.step(action)

        # Intercept and modify the observation using our prediction pipeline
        modified_obs = self._predict_and_inject_mu3(obs)

        return modified_obs, reward, terminated, truncated, info

    def reset(self, **kwargs):
        # Intercept and fix the initial observation when resetting the environment
        obs, info = self.env.reset(**kwargs)
        modified_obs = self._predict_and_inject_mu3(obs)
        return modified_obs, info

    def _predict_and_inject_mu3(self, partial_obs):
        # 1. Extract features required by XGBoost from the observation array
        raw_features = self._extract_features_from_obs(partial_obs)

        # Reshape to a 2D array [1, num_features] for scikit-learn
        raw_features_2d = np.array(raw_features).reshape(1, -1)

        # 2. Scale features using your input-specific joblib scaler
        X_scaled = self.X_scaler.transform(raw_features_2d)

        # 3. FIX: Convert numpy array to DMatrix because we used xgb.Booster()
        dmatrix_inputs = xgb.DMatrix(X_scaled)

        # 4. Predict the normalized mu3
        pred_normalized_mu3 = self.xgb_model.predict(dmatrix_inputs).reshape(
            -1, 1
        )

        # 5. Inverse-transform ONLY mu3 back to its real-world physical scale
        actual_mu3_value = self.mu3_scaler.inverse_transform(
            pred_normalized_mu3
        )[0][0]

        # 6. Inject the real-world mu3 value back into the array slot for the environment
        full_obs = self._inject_into_obs_array(partial_obs, actual_mu3_value)

        return full_obs

    def _extract_features_from_obs(self, obs):
        # Your initial observation printed: [-1. -1. -1. -1. -0.36 0. 0.5 0. 0.5]
        # Assuming the partial observation contains everything EXCEPT mu3, return it as-is
        return obs

    def _inject_into_obs_array(self, obs, mu3_val):
        # Create a mutable copy of the observation array
        modified_obs = np.array(obs, dtype=np.float32)

        # Identify which slot represents mu3.
        # Assuming mu3 is cut out or placed at the very last index (-1), update it:
        modified_obs[-1] = mu3_val

        return modified_obs