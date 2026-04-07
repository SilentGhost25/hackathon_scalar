"""
Training Script for the Stock Trading PPO Agent
=================================================

Trains a PPO agent on actual stock data fetched via yfinance.
Incorporates:
- Custom Gymnasium StockTradingEnv
- VecNormalize for state/reward scaling
- pandas-ta for technical indicators
- Custom CNN feature extractor (LSTM/1D-Conv)
- Walk-forward evaluation on separate test set.
"""

import os
import sys
import torch as th
import torch.nn as nn
import numpy as np
import pandas as pd
import yfinance as yf
import pandas_ta as ta

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

# Ensure project root is on path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from env.stock_env import StockTradingEnv

# ============================================================
# 1. Custom CNN Features Extractor
# ============================================================
class StockCNNFeaturesExtractor(BaseFeaturesExtractor):
    """
    Extends SB3 to use a 1D-Conv network as requested.
    Extracts features from the sequential/market data before passing to MLPs.
    """
    def __init__(self, observation_space, features_dim: int = 128):
        super(StockCNNFeaturesExtractor, self).__init__(observation_space, features_dim)
        
        n_input_channels = 1
        
        self.cnn = nn.Sequential(
            nn.Conv1d(n_input_channels, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Flatten(),
        )
        
        # Compute shape dynamically
        with th.no_grad():
            sample = th.as_tensor(observation_space.sample()[None, None, ...]).float()
            n_flatten = self.cnn(sample).shape[1]
            
        self.linear = nn.Sequential(
            nn.Linear(n_flatten, features_dim),
            nn.ReLU()
        )

    def forward(self, observations: th.Tensor) -> th.Tensor:
        # Add a channel dimension: (batch_size, seq_len) -> (batch_size, 1, seq_len)
        obs = observations.unsqueeze(1)
        return self.linear(self.cnn(obs))

# ============================================================
# 2. Data Fetching and Preparation
# ============================================================
def load_and_prepare_data(ticker="AAPL"):
    print(f"Downloading historical data for {ticker}...")
    df = yf.download(ticker, start="2020-01-01", end="2024-12-31")
    
    # Flatten MultiIndex columns if single ticker is downloaded
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    
    # Calculate indicators using pandas-ta
    print("Calculating technical indicators...")
    # These will add new columns to df
    df.ta.macd(append=True)
    df.ta.rsi(append=True)
    df.ta.bbands(append=True)
    df.ta.obv(append=True)
    
    # Drop rows with NaN values resulting from indicator lookback periods
    df.dropna(inplace=True)
    
    # Walk-forward Evaluation Split: Train 2020-2023, Test 2024
    df_train = df.loc["2020-01-01":"2023-12-31"].copy()
    df_test = df.loc["2024-01-01":"2024-12-31"].copy()
    
    # Ensure all column names are string
    df_train.columns = df_train.columns.astype(str)
    df_test.columns = df_test.columns.astype(str)
    
    print(f"Data ready. Train points: {len(df_train)}. Test points: {len(df_test)}")
    return df_train, df_test

# ============================================================
# 3. Main Training Pipeline
# ============================================================
def main():
    print("=" * 60)
    print("  Stock Trading PPO Agent - Training & Eval")
    print("=" * 60)
    
    # 1. Get Data
    df_train, df_test = load_and_prepare_data("AAPL")
    
    # 2. Setup Train Environment
    def make_train_env():
        return StockTradingEnv(df=df_train)
    
    env = DummyVecEnv([make_train_env])
    
    # CRITICAL: Normalize state & rewards!
    env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.0)
    
    # 3. Setup Architecture
    # Using the 1D-Conv -> 2 FC layers pattern
    policy_kwargs = dict(
        features_extractor_class=StockCNNFeaturesExtractor,
        features_extractor_kwargs=dict(features_dim=128),
        net_arch=dict(pi=[128, 64], vf=[128, 64])
    )
    
    model = PPO("MlpPolicy", env, verbose=1,
                learning_rate=3e-4, 
                n_steps=2048,
                batch_size=64, 
                n_epochs=10,
                policy_kwargs=policy_kwargs)
                
    # 4. Train Model
    print("Training model...")
    # total_timesteps=100_000 is good for hackathon demo
    model.learn(total_timesteps=100_000)
    
    # Save Model
    os.makedirs("saved_models", exist_ok=True)
    model.save("saved_models/ppo_trading_agent")
    env.save("saved_models/vec_normalize.pkl")
    print("\nModel saved to 'saved_models/ppo_trading_agent.zip'")
    
    # 5. Walk-forward Evaluation (2024)
    print("\n" + "=" * 60)
    print("  Walk-forward Evaluation Set (2024 Data)")
    print("=" * 60)
    
    def make_test_env():
        return StockTradingEnv(df=df_test)
        
    test_env = DummyVecEnv([make_test_env])
    
    # Load env scaling stats, but turn OFF updating them during test phase
    test_env = VecNormalize.load("saved_models/vec_normalize.pkl", test_env)
    test_env.training = False
    test_env.norm_reward = False
    
    obs = test_env.reset()
    done = False
    
    while not done:
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, done, info_list = test_env.step(action)
        
        info = info_list[0]
        if done:
            print(f"Evaluation complete. Final Test Portfolio Value: ${info['portfolio_value']:,.2f}")
            break

if __name__ == "__main__":
    main()
