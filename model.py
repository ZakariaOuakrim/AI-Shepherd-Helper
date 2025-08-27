import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
import gym
from gym import spaces
import matplotlib.pyplot as plt
from collections import deque
import random
from datetime import datetime, timedelta
import json
import os

class GrazingEnvironment(gym.Env):
    """
    Custom environment for grazing route optimization with constraints
    """
    def __init__(self, data_folder='grazing_data'):
        super(GrazingEnvironment, self).__init__()

        # Load the generated data
        self.load_data(data_folder)

        # Environment parameters
        self.n_zones = 10
        self.max_days = 90  # 3 months simulation
        self.herd_size = 150

        # State space: [current_zone, day_of_year, weather_features, vegetation_features, herd_status, constraint_flags]
        # State size: 1 + 1 + 3 + 2 + 3 + 10 (zone accessibility) = 20
        self.observation_space = spaces.Box(
            low=0, high=1, shape=(20,), dtype=np.float32
        )

        # Action space: move to zone 0-9
        self.action_space = spaces.Discrete(self.n_zones)

        # Track zone usage for consecutive day limits
        self.zone_usage_history = {}
        self.last_zone_exit = {}

        # Reset environment
        self.reset()

    def load_data(self, data_folder):
        """Load all the generated grazing data including constraints"""
        try:
            # Load constraints first
            with open(f'{data_folder}/grazing_constraints.json', 'r') as f:
                self.constraints = json.load(f)
            print("✓ Grazing constraints loaded")

            # Load vegetation data
            self.vegetation_df = pd.read_csv(f'{data_folder}/vegetation_data.csv')
            self.vegetation_df['date'] = pd.to_datetime(self.vegetation_df['date'])

            # Load weather data
            self.weather_df = pd.read_csv(f'{data_folder}/weather_data.csv')
            self.weather_df['date'] = pd.to_datetime(self.weather_df['date'])

            # Load topographical data
            self.topo_df = pd.read_csv(f'{data_folder}/topographical_data.csv')

            # Load water sources
            self.water_df = pd.read_csv(f'{data_folder}/water_sources.csv')

            # Load livestock data
            self.livestock_df = pd.read_csv(f'{data_folder}/livestock_tracking.csv')

            print("✓ All data loaded successfully!")

        except Exception as e:
            print(f"⚠️ Error loading data: {e}")
            print("📋 Generating sample data...")
            self._generate_sample_data()

    def _generate_sample_data(self):
        """Generate sample data if files are not available"""
        # Create basic constraints if file missing
        self.constraints = {
            "zone_restrictions": {
                "protected_areas": [8, 9],
                "seasonal_closures": {
                    "breeding_season": {
                        "zones": [3, 7],
                        "months": [4, 5, 6],
                        "reason": "Wildlife breeding protection"
                    }
                },
                "weather_based": {
                    "flood_prone": {
                        "zones": [4, 6],
                        "trigger": "rainfall > 15mm"
                    }
                }
            },
            "carrying_capacity_limits": {
                "max_sheep_per_hectare": 8,
                "max_consecutive_days": 7,
                "recovery_period_days": 14
            },
            "water_access_requirements": {
                "max_distance_from_water_km": 2.0,
                "zones_without_water": [8, 9]
            }
        }

        # Sample vegetation data
        dates = pd.date_range('2024-01-01', '2024-12-31', freq='W')
        veg_data = []
        for date in dates:
            for zone in range(1, 11):
                month = date.month
                if month in [3, 4, 5]:  # Spring
                    ndvi = 0.6 + np.random.normal(0, 0.1)
                elif month in [9, 10, 11]:  # Autumn
                    ndvi = 0.5 + np.random.normal(0, 0.1)
                else:
                    ndvi = 0.3 + np.random.normal(0, 0.1)

                # Apply constraints
                accessible = zone not in self.constraints["zone_restrictions"]["protected_areas"]
                if not accessible:
                    ndvi = 0.0

                veg_data.append({
                    'date': date,
                    'zone_id': zone,
                    'ndvi': max(0, min(1, ndvi)),
                    'biomass_kg_per_hectare': max(0, ndvi * 1000),
                    'carrying_capacity_sheep_per_hectare': max(0, ndvi * 10),
                    'accessible': accessible
                })

        self.vegetation_df = pd.DataFrame(veg_data)

        # Sample weather data
        weather_data = []
        for date in dates:
            weather_data.append({
                'date': date,
                'temperature': 20 + np.random.normal(0, 8),
                'humidity': 60 + np.random.normal(0, 15),
                'rainfall': max(0, np.random.exponential(3))
            })
        self.weather_df = pd.DataFrame(weather_data)

    def is_zone_accessible(self, zone, day_of_year):
        """Check if a zone is accessible based on all constraints"""
        current_date = datetime(2024, 1, 1) + timedelta(days=day_of_year)
        month = current_date.month

        # Check protected areas
        if (zone + 1) in self.constraints["zone_restrictions"]["protected_areas"]:
            return False, "Protected area", -100

        # Check seasonal closures
        seasonal_closures = self.constraints["zone_restrictions"]["seasonal_closures"]
        for closure_name, closure_data in seasonal_closures.items():
            if ((zone + 1) in closure_data["zones"] and
                month in closure_data["months"]):
                return False, closure_data["reason"], -80

        # Check consecutive day limits
        max_days = self.constraints["carrying_capacity_limits"]["max_consecutive_days"]
        if zone in self.zone_usage_history:
            if self.zone_usage_history[zone] >= max_days:
                # Check if recovery period has passed
                recovery_days = self.constraints["carrying_capacity_limits"]["recovery_period_days"]
                if (zone in self.last_zone_exit and
                    day_of_year - self.last_zone_exit[zone] < recovery_days):
                    return False, "Recovery period needed", -60

        # Check weather-based restrictions
        zone_quality = self.get_zone_quality(zone, day_of_year)
        weather_restrictions = self.constraints["zone_restrictions"]["weather_based"]

        if "flood_prone" in weather_restrictions:
            if ((zone + 1) in weather_restrictions["flood_prone"]["zones"] and
                zone_quality["rainfall"] > 15):
                return False, "Flood risk", -70

        if "extreme_temperature" in weather_restrictions:
            if ((zone + 1) in weather_restrictions["extreme_temperature"]["zones"] and
                zone_quality["temperature"] < 5):
                return False, "Extreme cold", -50

        # Check water access
        if (zone + 1) in self.constraints["water_access_requirements"]["zones_without_water"]:
            return False, "No water access", -90

        return True, "Accessible", 0

    def get_zone_accessibility_vector(self, day_of_year):
        """Get accessibility for all zones as a vector"""
        accessibility = np.zeros(self.n_zones, dtype=np.float32)
        for zone in range(self.n_zones):
            accessible, _, _ = self.is_zone_accessible(zone, day_of_year)
            accessibility[zone] = 1.0 if accessible else 0.0
        return accessibility

    def get_valid_actions(self, day_of_year):
        """Get list of valid actions (accessible zones)"""
        valid_actions = []
        for zone in range(self.n_zones):
            accessible, _, _ = self.is_zone_accessible(zone, day_of_year)
            if accessible:
                valid_actions.append(zone)
        return valid_actions if valid_actions else [0]  # Fallback to zone 0 if none available

    def get_zone_quality(self, zone, day_of_year):
        """Get current quality metrics for a zone"""
        # Get vegetation data for this zone and time
        current_date = datetime(2024, 1, 1) + timedelta(days=day_of_year)

        # Find closest vegetation data
        veg_data = self.vegetation_df[
            (self.vegetation_df['zone_id'] == zone + 1) &  # zones are 1-indexed in data
            (abs((self.vegetation_df['date'] - current_date).dt.days) <= 7)
        ]

        if len(veg_data) > 0:
            ndvi = veg_data.iloc[0]['ndvi']
            carrying_capacity = veg_data.iloc[0]['carrying_capacity_sheep_per_hectare']
            accessible = veg_data.iloc[0].get('accessible', True)
        else:
            # Default values if no data found
            ndvi = 0.4
            carrying_capacity = 4.0
            accessible = True

        # Get weather data
        weather_data = self.weather_df[
            abs((self.weather_df['date'] - current_date).dt.days) <= 3
        ]

        if len(weather_data) > 0:
            temperature = weather_data.iloc[0]['temperature']
            rainfall = weather_data.iloc[0]['rainfall']
        else:
            temperature = 20
            rainfall = 0

        return {
            'ndvi': ndvi,
            'carrying_capacity': carrying_capacity,
            'temperature': temperature,
            'rainfall': rainfall,
            'accessible': accessible
        }

    def calculate_reward(self, zone, action, day_of_year):
        """Calculate reward for taking an action with constraint penalties"""
        # Check if action violates constraints
        accessible, reason, penalty = self.is_zone_accessible(action, day_of_year)

        # Heavy penalty for constraint violations
        if not accessible:
            self.last_reward_breakdown = {
                'constraint_violation': penalty,
                'vegetation': 0,
                'overgrazing_penalty': 0,
                'weather': 0,
                'movement_penalty': 0,
                'season_bonus': 0,
                'total': penalty
            }
            return penalty

        zone_quality = self.get_zone_quality(action, day_of_year)

        # Base reward from vegetation quality (only if accessible)
        vegetation_reward = zone_quality['ndvi'] * 100 if zone_quality['accessible'] else 0

        # Carrying capacity reward (avoid overgrazing)
        if zone_quality['carrying_capacity'] > 0:
            capacity_utilization = self.herd_size / (zone_quality['carrying_capacity'] * 10)
            if capacity_utilization > 1.0:
                overgrazing_penalty = -50 * (capacity_utilization - 1.0)
            else:
                overgrazing_penalty = 0
        else:
            overgrazing_penalty = -30  # No capacity available

        # Weather comfort reward
        temp_comfort = 1 - abs(zone_quality['temperature'] - 22) / 30
        weather_reward = temp_comfort * 20

        # Movement penalty (encourage staying in good zones, but not too long)
        if action != zone:
            movement_penalty = -5
        else:
            # Small penalty for staying too long in same zone
            days_in_zone = self.zone_usage_history.get(action, 0)
            max_days = self.constraints["carrying_capacity_limits"]["max_consecutive_days"]
            if days_in_zone > max_days * 0.7:  # Start penalizing at 70% of limit
                movement_penalty = -10 * (days_in_zone / max_days)
            else:
                movement_penalty = 0

        # Time-based adjustments
        season_bonus = 0
        month = ((day_of_year // 30) % 12) + 1
        if month in [3, 4, 5, 9, 10, 11]:  # Good grazing months
            season_bonus = 10

        # Bonus for using diverse zones (exploration)
        diversity_bonus = len(set(self.zone_usage_history.keys())) * 2

        # Water access bonus (implicit - zones without water already penalized)
        water_bonus = 5 if (action + 1) not in self.constraints["water_access_requirements"]["zones_without_water"] else 0

        total_reward = (vegetation_reward + overgrazing_penalty + weather_reward +
                       movement_penalty + season_bonus + diversity_bonus + water_bonus)

        # Track reward components for analysis
        self.last_reward_breakdown = {
            'constraint_violation': 0,
            'vegetation': vegetation_reward,
            'overgrazing_penalty': overgrazing_penalty,
            'weather': weather_reward,
            'movement_penalty': movement_penalty,
            'season_bonus': season_bonus,
            'diversity_bonus': diversity_bonus,
            'water_bonus': water_bonus,
            'total': total_reward
        }

        return total_reward

    def get_state(self):
        """Get current state representation including constraint information"""
        zone_quality = self.get_zone_quality(self.current_zone, self.current_day)
        accessibility_vector = self.get_zone_accessibility_vector(self.current_day)

        state = np.array([
            self.current_zone / 9.0,  # Normalized current zone
            self.current_day / 365.0,  # Normalized day of year
            min(max(zone_quality['temperature'] / 40.0, -1.0), 1.0),  # Normalized temperature (fixed bounds)
            min(zone_quality['rainfall'] / 20.0, 1.0),  # Normalized rainfall
            min(zone_quality['ndvi'], 1.0),  # NDVI (already 0-1)
            min(zone_quality['carrying_capacity'] / 20.0, 1.0),  # Normalized carrying capacity
            self.herd_health / 100.0,  # Herd health (0-100)
            min(self.days_in_current_zone / 30.0, 1.0),  # Days in zone (normalized to month)
            min(max(self.cumulative_reward / 1000.0, -1.0), 1.0),  # Normalized cumulative reward
            min(self.grazing_pressure / 100.0, 1.0),  # Grazing pressure
            *accessibility_vector  # 10 values for zone accessibility
        ], dtype=np.float32)

        return state

    def update_zone_usage(self, old_zone, new_zone, day):
        """Update zone usage tracking"""
        if old_zone == new_zone:
            # Staying in same zone
            self.zone_usage_history[new_zone] = self.zone_usage_history.get(new_zone, 0) + 1
        else:
            # Moving to new zone
            if old_zone is not None:
                self.last_zone_exit[old_zone] = day
            self.zone_usage_history[new_zone] = 1

    def reset(self):
        """Reset environment to initial state"""
        self.current_zone = 0
        self.current_day = 0
        self.herd_health = 100.0
        self.days_in_current_zone = 0
        self.cumulative_reward = 0
        self.grazing_pressure = 0
        self.episode_rewards = []
        self.zone_history = []

        # Reset constraint tracking
        self.zone_usage_history = {}
        self.last_zone_exit = {}

        return self.get_state()

    def step(self, action):
        """Take action and return new state, reward, done, info"""
        # Record action
        self.zone_history.append((self.current_day, self.current_zone, action))

        # Calculate reward (includes constraint checking)
        reward = self.calculate_reward(self.current_zone, action, self.current_day)

        # Update zone usage tracking
        old_zone = self.current_zone
        self.update_zone_usage(old_zone, action, self.current_day)

        # Move to new zone
        self.current_zone = action

        # Update counters
        if old_zone == action:
            self.days_in_current_zone += 1
        else:
            self.days_in_current_zone = 1

        self.current_day += 1
        self.cumulative_reward += reward
        self.episode_rewards.append(reward)

        # Update herd health based on zone quality and constraints
        zone_quality = self.get_zone_quality(self.current_zone, self.current_day)
        accessible, _, _ = self.is_zone_accessible(self.current_zone, self.current_day)

        if accessible:
            health_change = (zone_quality['ndvi'] - 0.3) * 2
        else:
            health_change = -5  # Health penalty for being in restricted zone

        self.herd_health = np.clip(self.herd_health + health_change, 0, 100)

        # Update grazing pressure
        self.grazing_pressure = min(100, self.days_in_current_zone * 5)

        # Check if episode is done
        done = self.current_day >= self.max_days or self.herd_health <= 0

        # Info for debugging
        accessible, reason, _ = self.is_zone_accessible(self.current_zone, self.current_day)
        info = {
            'herd_health': self.herd_health,
            'days_in_zone': self.days_in_current_zone,
            'zone_quality': zone_quality,
            'reward_breakdown': getattr(self, 'last_reward_breakdown', {}),
            'cumulative_reward': self.cumulative_reward,
            'zone_accessible': accessible,
            'restriction_reason': reason,
            'valid_actions': self.get_valid_actions(self.current_day),
            'constraint_violations': sum(1 for r in self.episode_rewards if r < -50)
        }

        return self.get_state(), reward, done, info


class PPONetwork(nn.Module):
    """
    Neural network for PPO agent with constraint awareness
    """
    def __init__(self, state_dim, action_dim, hidden_dim=64):  # Reduced size
        super(PPONetwork, self).__init__()

        # Shared feature extraction layers (more stable architecture)
        self.shared_layers = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim//2),
            nn.ReLU()
        )

        # Actor head (policy)
        self.actor = nn.Sequential(
            nn.Linear(hidden_dim//2, hidden_dim//4),
            nn.ReLU(),
            nn.Linear(hidden_dim//4, action_dim)
        )

        # Critic head (value function)
        self.critic = nn.Sequential(
            nn.Linear(hidden_dim//2, hidden_dim//4),
            nn.ReLU(),
            nn.Linear(hidden_dim//4, 1)
        )

        # Initialize weights properly
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight, gain=0.5)  # Smaller gain for stability
            torch.nn.init.constant_(m.bias, 0)

    def forward(self, state):
        features = self.shared_layers(state)
        return features

    def get_action_probs(self, state, valid_actions_mask=None):
        features = self.forward(state)
        logits = self.actor(features)

        # Apply action masking for invalid actions
        if valid_actions_mask is not None:
            # Safer masking - use large negative values instead of log
            mask = torch.where(valid_actions_mask == float('-inf'), -1e8, 0.0)
            logits = logits + mask

        return F.softmax(logits, dim=-1)

    def get_value(self, state):
        features = self.forward(state)
        return self.critic(features)

    def get_action_and_value(self, state, valid_actions_mask=None):
        features = self.forward(state)
        logits = self.actor(features)

        # Apply action masking
        if valid_actions_mask is not None:
            mask = torch.where(valid_actions_mask == float('-inf'), -1e8, 0.0)
            logits = logits + mask

        probs = F.softmax(logits, dim=-1)
        value = self.critic(features)
        return probs, value


class PPOAgent:
    """
    PPO Agent for constrained grazing route optimization
    """
    def __init__(self, state_dim, action_dim, lr=1e-4, gamma=0.95, eps_clip=0.15, k_epochs=3):  # More conservative params
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.k_epochs = k_epochs
        self.action_dim = action_dim

        # Networks
        self.policy = PPONetwork(state_dim, action_dim)
        self.old_policy = PPONetwork(state_dim, action_dim)
        self.old_policy.load_state_dict(self.policy.state_dict())

        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr, eps=1e-5)  # More stable optimizer

        # Experience buffer
        self.memory = {
            'states': [],
            'actions': [],
            'rewards': [],
            'is_terminals': [],
            'log_probs': [],
            'valid_actions_masks': []
        }

        # Training metrics
        self.training_rewards = []
        self.training_losses = []
        self.constraint_violations = []

    def create_valid_actions_mask(self, valid_actions):
        """Create mask for valid actions"""
        mask = torch.full((self.action_dim,), float('-inf'))
        for action in valid_actions:
            mask[action] = 0.0
        return mask

    def select_action(self, state, valid_actions=None, training=True):
        """Select action using current policy with constraint masking"""
        state = torch.FloatTensor(state).unsqueeze(0)

        # Create action mask if constraints provided
        valid_actions_mask = None
        if valid_actions is not None and len(valid_actions) > 0:
            valid_actions_mask = self.create_valid_actions_mask(valid_actions).unsqueeze(0)
        elif valid_actions is not None and len(valid_actions) == 0:
            # Emergency: if no valid actions, allow all (shouldn't happen with proper constraints)
            valid_actions = list(range(self.action_dim))
            valid_actions_mask = self.create_valid_actions_mask(valid_actions).unsqueeze(0)

        with torch.no_grad():
            probs = self.old_policy.get_action_probs(state, valid_actions_mask)

            # Check for NaN values
            if torch.isnan(probs).any():
                print("Warning: NaN detected in action probabilities, using uniform distribution")
                probs = torch.ones_like(probs) / probs.size(-1)
                if valid_actions_mask is not None:
                    probs = probs * torch.exp(valid_actions_mask)
                    probs = probs / probs.sum(dim=-1, keepdim=True)

            # Add small epsilon to avoid zero probabilities
            probs = probs + 1e-8
            probs = probs / probs.sum(dim=-1, keepdim=True)

            dist = Categorical(probs)
            action = dist.sample()

            if training:
                log_prob = dist.log_prob(action)
                self.memory['log_probs'].append(log_prob)
                self.memory['valid_actions_masks'].append(valid_actions_mask.squeeze() if valid_actions_mask is not None else None)

        return action.item()

    def store_experience(self, state, action, reward, is_terminal):
        """Store experience in memory"""
        self.memory['states'].append(state)
        self.memory['actions'].append(action)
        self.memory['rewards'].append(reward)
        self.memory['is_terminals'].append(is_terminal)

    def update(self):
        """Update policy using PPO with constraint awareness"""
        if len(self.memory['states']) == 0:
            return 0

        # Convert lists to tensors
        states = torch.FloatTensor(np.array(self.memory['states']))
        actions = torch.LongTensor(self.memory['actions'])

        # Check for valid log_probs
        if len(self.memory['log_probs']) == 0:
            return 0

        old_log_probs = torch.stack(self.memory['log_probs']).detach()

        # Handle valid action masks
        valid_masks = []
        for mask in self.memory['valid_actions_masks']:
            if mask is not None:
                valid_masks.append(mask)
            else:
                valid_masks.append(torch.zeros(self.action_dim))
        valid_actions_masks = torch.stack(valid_masks) if valid_masks else None

        # Calculate discounted rewards
        rewards = []
        discounted_reward = 0
        for reward, is_terminal in zip(reversed(self.memory['rewards']), reversed(self.memory['is_terminals'])):
            if is_terminal:
                discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward)
            rewards.insert(0, discounted_reward)

        rewards = torch.FloatTensor(rewards)
        # Normalize rewards more carefully
        if rewards.std() > 1e-8:
            rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-8)
        else:
            rewards = rewards - rewards.mean()

        # Count constraint violations
        violation_count = sum(1 for r in self.memory['rewards'] if r < -50)
        self.constraint_violations.append(violation_count)

        # PPO update
        total_loss = 0
        for epoch in range(self.k_epochs):
            try:
                # Get current policy outputs with constraint masking
                probs, state_values = self.policy.get_action_and_value(states, valid_actions_masks)

                # Check for NaN values in probs
                if torch.isnan(probs).any():
                    print(f"Warning: NaN detected in probabilities at epoch {epoch}")
                    break

                # Add small epsilon to avoid log(0)
                probs = probs + 1e-8
                probs = probs / probs.sum(dim=-1, keepdim=True)

                dist = Categorical(probs)
                new_log_probs = dist.log_prob(actions)

                # Check for NaN in log_probs
                if torch.isnan(new_log_probs).any():
                    print(f"Warning: NaN detected in new log probs at epoch {epoch}")
                    break

                # Calculate ratio
                ratio = torch.exp(new_log_probs - old_log_probs)

                # Clip ratio to prevent extreme values
                ratio = torch.clamp(ratio, 0.1, 10.0)

                # Calculate advantages
                advantages = rewards - state_values.squeeze()

                # Actor loss (PPO clip)
                surr1 = ratio * advantages
                surr2 = torch.clamp(ratio, 1 - self.eps_clip, 1 + self.eps_clip) * advantages
                actor_loss = -torch.min(surr1, surr2).mean()

                # Critic loss
                critic_loss = F.mse_loss(state_values.squeeze(), rewards)

                # Entropy bonus for exploration (but constrained to valid actions)
                entropy = dist.entropy().mean()

                # Total loss
                loss = actor_loss + 0.5 * critic_loss - 0.001 * entropy  # Smaller entropy bonus

                # Check for NaN in loss
                if torch.isnan(loss):
                    print(f"Warning: NaN detected in loss at epoch {epoch}")
                    break

                # Backward pass
                self.optimizer.zero_grad()
                loss.backward()

                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 0.5)

                self.optimizer.step()

                total_loss += loss.item()

            except Exception as e:
                print(f"Error during update at epoch {epoch}: {e}")
                break

        # Update old policy
        self.old_policy.load_state_dict(self.policy.state_dict())

        # Clear memory
        self.clear_memory()

        avg_loss = total_loss / self.k_epochs if self.k_epochs > 0 else 0
        self.training_losses.append(avg_loss)
        return avg_loss

    def clear_memory(self):
        """Clear experience buffer"""
        for key in self.memory:
            self.memory[key] = []

    def save_model(self, filepath):
        """Save trained model"""
        torch.save({
            'policy_state_dict': self.policy.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'training_rewards': self.training_rewards,
            'training_losses': self.training_losses,
            'constraint_violations': self.constraint_violations
        }, filepath)
        print(f"Model saved to {filepath}")

    def load_model(self, filepath):
        """Load trained model"""
        checkpoint = torch.load(filepath)
        self.policy.load_state_dict(checkpoint['policy_state_dict'])
        self.old_policy.load_state_dict(checkpoint['policy_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.training_rewards = checkpoint.get('training_rewards', [])
        self.training_losses = checkpoint.get('training_losses', [])
        self.constraint_violations = checkpoint.get('constraint_violations', [])
        print(f"Model loaded from {filepath}")


def train_ppo_agent(episodes=1000, update_frequency=20, data_folder='grazing_data'):
    """
    Train PPO agent for constrained grazing route optimization
    """
    print("🚀 Starting PPO training with constraint integration...")

    # Create environment and agent
    env = GrazingEnvironment(data_folder)
    agent = PPOAgent(
        state_dim=env.observation_space.shape[0],
        action_dim=env.action_space.n,
        lr=1e-4  # More conservative learning rate
    )
    print("Environment and agent created.")

    # Training metrics
    episode_rewards = []
    running_reward = 0
    total_violations = 0

    for episode in range(episodes):
        state = env.reset()
        episode_reward = 0
        episode_violations = 0
        steps = 0

        while True:
            # Get valid actions for current state
            valid_actions = env.get_valid_actions(env.current_day)

            # Select action (constrained)
            action = agent.select_action(state, valid_actions)

            # Take step
            next_state, reward, done, info = env.step(action)

            # Store experience
            agent.store_experience(state, action, reward, done)

            state = next_state
            episode_reward += reward
            steps += 1

            # Track violations
            if reward < -50:  # Constraint violation threshold
                episode_violations += 1

            if done or steps > 200:  # Add step limit
                break

        # Update running reward
        running_reward = 0.05 * episode_reward + (1 - 0.05) * running_reward
        episode_rewards.append(episode_reward)
        agent.training_rewards.append(episode_reward)
        total_violations += episode_violations

        # Update agent
        if episode % update_frequency == 0 and episode > 0:
            loss = agent.update()
            avg_violations = total_violations / update_frequency
            print(f"Episode {episode:4d} | Avg Reward: {running_reward:7.2f} | Loss: {loss:.4f} | "
                  f"Herd Health: {info.get('herd_health', 0):.1f} | Violations/ep: {avg_violations:.2f}")
            total_violations = 0

        # Save model periodically
        if episode % 200 == 0 and episode > 0:
            agent.save_model(f'ppo_constrained_model_episode_{episode}.pth')

    print("✅ Training completed!")

    # Save final model
    agent.save_model('ppo_constrained_model_final.pth')

    # Plot training progress
    plt.figure(figsize=(15, 10))

    plt.subplot(2, 3, 1)
    plt.plot(episode_rewards)
    plt.title('Episode Rewards')
    plt.xlabel('Episode')
    plt.ylabel('Reward')

    plt.subplot(2, 3, 2)
    if agent.training_losses:
        plt.plot(agent.training_losses)
        plt.title('Training Loss')
        plt.xlabel('Update')
        plt.ylabel('Loss')

    plt.subplot(2, 3, 3)
    # Plot constraint violations over time
    if agent.constraint_violations:
        plt.plot(agent.constraint_violations)
        plt.title('Constraint Violations per Update')
        plt.xlabel('Update')
        plt.ylabel('Violations')

    plt.subplot(2, 3, 4)
    # Moving average of rewards
    window_size = min(100, len(episode_rewards)//10)
    if len(episode_rewards) >= window_size:
        moving_avg = pd.Series(episode_rewards).rolling(window=window_size).mean()
        plt.plot(moving_avg)
        plt.title(f'Moving Average Rewards (window={window_size})')
        plt.xlabel('Episode')
        plt.ylabel('Average Reward')

    plt.subplot(2, 3, 5)
    # Reward distribution
    plt.hist(episode_rewards, bins=30, alpha=0.7)
    plt.title('Reward Distribution')
    plt.xlabel('Reward')
    plt.ylabel('Frequency')

    plt.subplot(2, 3, 6)
    # Cumulative violations
    if agent.constraint_violations:
        cumulative_violations = np.cumsum(agent.constraint_violations)
        plt.plot(cumulative_violations)
        plt.title('Cumulative Constraint Violations')
        plt.xlabel('Update')
        plt.ylabel('Total Violations')

    plt.tight_layout()
    plt.savefig('ppo_constrained_training_progress.png', dpi=150, bbox_inches='tight')
    plt.show()

    return agent, env


def evaluate_agent(agent, env, episodes=10):
    """
    Evaluate trained agent with constraint analysis
    """
    print("Evaluating trained agent with constraint analysis...")

    evaluation_rewards = []
    route_analyses = []
    constraint_compliance = []

    for episode in range(episodes):
        state = env.reset()
        episode_reward = 0
        route = []
        violations = 0
        valid_action_count = 0
        total_actions = 0
        steps = 0

        while True:
            valid_actions = env.get_valid_actions(env.current_day)
            action = agent.select_action(state, valid_actions, training=False)
            next_state, reward, done, info = env.step(action)

            # Track constraint compliance
            if reward < -50:
                violations += 1
            if action in valid_actions:
                valid_action_count += 1
            total_actions += 1

            route.append({
                'day': env.current_day - 1,
                'zone': action,
                'reward': reward,
                'herd_health': info['herd_health'],
                'ndvi': info['zone_quality']['ndvi'],
                'accessible': info['zone_accessible'],
                'valid_actions': len(valid_actions),
                'constraint_violation': reward < -50
            })

            state = next_state
            episode_reward += reward
            steps += 1

            if done or steps > 200:
                break

        compliance_rate = valid_action_count / total_actions if total_actions > 0 else 0
        constraint_compliance.append(compliance_rate)

        evaluation_rewards.append(episode_reward)
        route_analyses.append({
            'episode': episode,
            'total_reward': episode_reward,
            'route': route,
            'final_health': info['herd_health'],
            'zone_diversity': len(set([r['zone'] for r in route])),
            'violations': violations,
            'compliance_rate': compliance_rate
        })

    avg_reward = np.mean(evaluation_rewards)
    std_reward = np.std(evaluation_rewards)
    avg_compliance = np.mean(constraint_compliance)

    print(f"Evaluation Results:")
    print(f"   Average Reward: {avg_reward:.2f} ± {std_reward:.2f}")
    print(f"   Best Episode: {max(evaluation_rewards):.2f}")
    print(f"   Worst Episode: {min(evaluation_rewards):.2f}")
    print(f"   Average Constraint Compliance: {avg_compliance:.2%}")
    print(f"   Episodes with Violations: {sum(1 for r in route_analyses if r['violations'] > 0)}/{episodes}")

    # Analyze best route
    compliant_routes = [r for r in route_analyses if r['compliance_rate'] > 0.95]
    if compliant_routes:
        best_episode = max(compliant_routes, key=lambda x: x['total_reward'])
        print(f"\nBest Compliant Route Analysis (Episode {best_episode['episode']}):")
    else:
        best_episode = route_analyses[np.argmax(evaluation_rewards)]
        print(f"\nBest Route Analysis (Episode {best_episode['episode']}) - Note: May have violations:")

    print(f"   Total Reward: {best_episode['total_reward']:.2f}")
    print(f"   Final Herd Health: {best_episode['final_health']:.1f}")
    print(f"   Zone Diversity: {best_episode['zone_diversity']}/10 zones used")
    print(f"   Violations: {best_episode['violations']}")
    print(f"   Compliance Rate: {best_episode['compliance_rate']:.2%}")

    # Visualization
    route = best_episode['route']
    days = [r['day'] for r in route]
    zones = [r['zone'] for r in route]
    health = [r['herd_health'] for r in route]
    ndvi = [r['ndvi'] for r in route]
    violations = [r['constraint_violation'] for r in route]

    plt.figure(figsize=(15, 10))

    plt.subplot(2, 3, 1)
    plt.plot(days, zones, 'o-', markersize=3)
    violation_days = [days[i] for i, v in enumerate(violations) if v]
    violation_zones = [zones[i] for i, v in enumerate(violations) if v]
    if violation_days:
        plt.scatter(violation_days, violation_zones, c='red', s=50, alpha=0.7, label='Violations')
        plt.legend()
    plt.title('Zone Selection Over Time')
    plt.xlabel('Day')
    plt.ylabel('Zone')
    plt.grid(True)

    plt.subplot(2, 3, 2)
    plt.plot(days, health, 'g-')
    plt.title('Herd Health Over Time')
    plt.xlabel('Day')
    plt.ylabel('Health')
    plt.grid(True)

    plt.subplot(2, 3, 3)
    plt.plot(days, ndvi, 'b-')
    plt.title('Vegetation Quality (NDVI) Over Time')
    plt.xlabel('Day')
    plt.ylabel('NDVI')
    plt.grid(True)

    plt.subplot(2, 3, 4)
    rewards = [r['reward'] for r in route]
    colors = ['red' if r < -50 else 'blue' for r in rewards]
    plt.bar(days, rewards, color=colors, alpha=0.7)
    plt.title('Daily Rewards (Red = Violations)')
    plt.xlabel('Day')
    plt.ylabel('Reward')
    plt.grid(True)

    plt.subplot(2, 3, 5)
    zone_counts = {}
    for zone in zones:
        zone_counts[zone] = zone_counts.get(zone, 0) + 1
    plt.bar(zone_counts.keys(), zone_counts.values())
    plt.title('Zone Usage Frequency')
    plt.xlabel('Zone')
    plt.ylabel('Days Used')
    plt.grid(True)

    plt.subplot(2, 3, 6)
    valid_actions_count = [r['valid_actions'] for r in route]
    plt.plot(days, valid_actions_count, 'purple', marker='.')
    plt.title('Available Valid Actions Over Time')
    plt.xlabel('Day')
    plt.ylabel('Number of Valid Actions')
    plt.grid(True)

    plt.tight_layout()
    plt.savefig('constrained_agent_evaluation_results.png', dpi=150, bbox_inches='tight')
    plt.show()

    return route_analyses


def get_recommendations(agent, env, start_day=0, days_ahead=14):
    """
    Get constrained grazing recommendations for the next period
    """
    print(f"Generating constrained grazing recommendations for next {days_ahead} days...")

    # Reset environment to specific start day
    env.reset()
    env.current_day = start_day

    recommendations = []
    state = env.get_state()

    for day in range(days_ahead):
        # Get valid actions (constraint-compliant)
        valid_actions = env.get_valid_actions(env.current_day)

        # Get best valid action
        if valid_actions:
            best_action = agent.select_action(state, valid_actions, training=False)
            confidence = 1.0 / len(valid_actions)  # Simple confidence measure
        else:
            best_action = 0
            confidence = 0.0

        # Get zone quality info for all zones
        zone_qualities = []
        zone_constraints = []
        for zone in range(10):
            quality = env.get_zone_quality(zone, env.current_day)
            accessible, reason, penalty = env.is_zone_accessible(zone, env.current_day)
            zone_qualities.append(quality)
            zone_constraints.append({
                'accessible': accessible,
                'reason': reason,
                'penalty': penalty
            })

        recommendations.append({
            'day': env.current_day,
            'recommended_zone': best_action,
            'confidence': confidence,
            'valid_actions': valid_actions,
            'zone_qualities': zone_qualities,
            'zone_constraints': zone_constraints,
            'date': (datetime(2024, 1, 1) + timedelta(days=env.current_day)).strftime('%Y-%m-%d'),
            'constraint_compliant': best_action in valid_actions if valid_actions else False
        })

        # Step forward
        next_state, reward, done, info = env.step(best_action)
        state = next_state

        if done:
            break

    return recommendations


# Main execution
if __name__ == "__main__":
    print("AI Shepherd: PPO-based Constrained Grazing Route Optimization System")
    print("=" * 70)

    try:
        # Train the agent with constraints
        agent, env = train_ppo_agent(episodes=1000, data_folder='grazing_data')  # Reduced episodes for testing

        # Evaluate the agent
        evaluation_results = evaluate_agent(agent, env, episodes=5)

        # Get recommendations
        recommendations = get_recommendations(agent, env, start_day=60, days_ahead=7)

        # Print sample recommendations
        print("\nSample Recommendations (Next 7 days):")
        print("Day | Date       | Zone | Confidence | Valid Actions | Compliant")
        print("-" * 65)
        for i, rec in enumerate(recommendations[:7]):
            compliant = "✓" if rec['constraint_compliant'] else "✗"
            print(f"{rec['day']:2d}  | {rec['date']} | {rec['recommended_zone']:2d}   | {rec['confidence']:.3f}      | "
                  f"{len(rec['valid_actions']):2d}            | {compliant}")

        print(f"\nSystem ready for deployment with constraint awareness!")
        print("Use agent.select_action(state, valid_actions) for constraint-compliant decisions")
        print("Check generated plots for comprehensive training and evaluation analysis")

    except Exception as e:
        print(f"Error during execution: {e}")
        import traceback
        traceback.print_exc()