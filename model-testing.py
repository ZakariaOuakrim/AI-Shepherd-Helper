import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns
from datetime import datetime, timedelta
import json
import torch
from collections import defaultdict

class GrazingZoneMapper:
    """
    Visualize and analyze grazing zones with constraints
    """
    def __init__(self, data_folder='grazing_data'):
        self.data_folder = data_folder
        self.load_data()
        self.setup_zone_grid()
    
    def load_data(self):
        """Load all grazing data"""
        try:
            # Load constraints
            with open(f'{self.data_folder}/grazing_constraints.json', 'r') as f:
                self.constraints = json.load(f)
            
            # Load other data
            self.vegetation_df = pd.read_csv(f'{self.data_folder}/vegetation_data.csv')
            self.vegetation_df['date'] = pd.to_datetime(self.vegetation_df['date'])
            
            self.weather_df = pd.read_csv(f'{self.data_folder}/weather_data.csv')
            self.weather_df['date'] = pd.to_datetime(self.weather_df['date'])
            
            self.topo_df = pd.read_csv(f'{self.data_folder}/topographical_data.csv')
            self.water_df = pd.read_csv(f'{self.data_folder}/water_sources.csv')
            
            print("Data loaded successfully for visualization")
            
        except Exception as e:
            print(f"Error loading data: {e}")
            self.create_sample_data()
    
    def create_sample_data(self):
        """Create sample data for visualization"""
        self.constraints = {
            "zone_restrictions": {
                "protected_areas": [8, 9],
                "seasonal_closures": {
                    "breeding_season": {"zones": [3, 7], "months": [4, 5, 6]}
                }
            },
            "water_access_requirements": {"zones_without_water": [8, 9]}
        }
    
    def setup_zone_grid(self):
        """Create a 2D grid representation of zones"""
        # Arrange zones in a 3x4 grid for better visualization
        self.zone_positions = {
            1: (0, 0), 2: (0, 1), 3: (0, 2), 4: (0, 3),
            5: (1, 0), 6: (1, 1), 7: (1, 2), 8: (1, 3),
            9: (2, 0), 10: (2, 1)
        }
        
        # Create reverse mapping
        self.position_to_zone = {v: k for k, v in self.zone_positions.items()}
    
    def get_zone_characteristics(self, zone_id, date=None):
        """Get characteristics for a specific zone"""
        if date is None:
            date = datetime(2024, 6, 15)  # Mid-year default
        
        # Get vegetation data
        veg_data = self.vegetation_df[
            (self.vegetation_df['zone_id'] == zone_id) &
            (abs((self.vegetation_df['date'] - date).dt.days) <= 7)
        ]
        
        if len(veg_data) > 0:
            ndvi = veg_data.iloc[0]['ndvi']
            accessible = veg_data.iloc[0].get('accessible', True)
        else:
            ndvi = np.random.uniform(0.2, 0.8)
            accessible = zone_id not in self.constraints["zone_restrictions"]["protected_areas"]
        
        # Zone characteristics
        characteristics = {
            'ndvi': ndvi,
            'accessible': accessible,
            'protected': zone_id in self.constraints["zone_restrictions"]["protected_areas"],
            'no_water': zone_id in self.constraints["water_access_requirements"].get("zones_without_water", []),
            'seasonal_closure': any(zone_id in closure["zones"] 
                                  for closure in self.constraints["zone_restrictions"]["seasonal_closures"].values()),
            'terrain_type': 'mountainous' if zone_id in [8, 9, 10] else 'hills' if zone_id in [3, 4, 7] else 'valley'
        }
        
        return characteristics
    
    def create_zone_map(self, title="Grazing Zone Map", date=None, save_path=None):
        """Create a comprehensive zone map"""
        if date is None:
            date = datetime(2024, 6, 15)
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle(f'{title} - {date.strftime("%Y-%m-%d")}', fontsize=16, fontweight='bold')
        
        # 1. Vegetation Quality Map
        self._create_vegetation_map(ax1, date)
        
        # 2. Constraints Map
        self._create_constraints_map(ax2, date)
        
        # 3. Terrain & Water Map
        self._create_terrain_map(ax3)
        
        # 4. Zone Information Panel
        self._create_info_panel(ax4, date)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
        return fig
    
    def _create_vegetation_map(self, ax, date):
        """Create vegetation quality heatmap"""
        grid = np.zeros((3, 4))
        
        for zone_id in range(1, 11):
            if zone_id in self.zone_positions:
                row, col = self.zone_positions[zone_id]
                chars = self.get_zone_characteristics(zone_id, date)
                grid[row, col] = chars['ndvi']
        
        # Fill remaining positions with NaN
        for i in range(3):
            for j in range(4):
                if (i, j) not in self.position_to_zone:
                    grid[i, j] = np.nan
        
        im = ax.imshow(grid, cmap='YlGn', vmin=0, vmax=1, aspect='equal')
        
        # Add zone labels and values
        for zone_id in range(1, 11):
            if zone_id in self.zone_positions:
                row, col = self.zone_positions[zone_id]
                chars = self.get_zone_characteristics(zone_id, date)
                ax.text(col, row, f'Z{zone_id}\n{chars["ndvi"]:.2f}', 
                       ha='center', va='center', fontweight='bold',
                       color='white' if chars['ndvi'] < 0.5 else 'black')
        
        ax.set_title('Vegetation Quality (NDVI)', fontweight='bold')
        ax.set_xticks([])
        ax.set_yticks([])
        plt.colorbar(im, ax=ax, label='NDVI Value')
    
    def _create_constraints_map(self, ax, date):
        """Create constraints visualization"""
        grid = np.zeros((3, 4))
        colors = []
        
        for zone_id in range(1, 11):
            if zone_id in self.zone_positions:
                row, col = self.zone_positions[zone_id]
                chars = self.get_zone_characteristics(zone_id, date)
                
                # Color coding: 0=accessible, 1=protected, 2=no water, 3=seasonal
                if chars['protected']:
                    grid[row, col] = 3  # Red
                elif chars['no_water']:
                    grid[row, col] = 2  # Orange
                elif chars['seasonal_closure'] and date.month in [4, 5, 6]:
                    grid[row, col] = 1  # Yellow
                else:
                    grid[row, col] = 0  # Green
        
        # Fill remaining positions
        for i in range(3):
            for j in range(4):
                if (i, j) not in self.position_to_zone:
                    grid[i, j] = -1
        
        # Custom colormap
        colors = ['green', 'yellow', 'orange', 'red', 'white']
        cmap = LinearSegmentedColormap.from_list('constraints', colors, N=5)
        
        im = ax.imshow(grid, cmap=cmap, vmin=-1, vmax=3, aspect='equal')
        
        # Add zone labels and constraint info
        constraint_labels = {0: 'OK', 1: 'SEASONAL', 2: 'NO H₂O', 3: 'PROTECTED', -1: ''}
        
        for zone_id in range(1, 11):
            if zone_id in self.zone_positions:
                row, col = self.zone_positions[zone_id]
                label_key = int(grid[row, col])
                ax.text(col, row, f'Z{zone_id}\n{constraint_labels[label_key]}', 
                       ha='center', va='center', fontweight='bold', fontsize=9)
        
        ax.set_title('Zone Constraints', fontweight='bold')
        ax.set_xticks([])
        ax.set_yticks([])
        
        # Custom legend
        legend_elements = [
            patches.Patch(color='green', label='Accessible'),
            patches.Patch(color='yellow', label='Seasonal Closure'),
            patches.Patch(color='orange', label='No Water'),
            patches.Patch(color='red', label='Protected Area')
        ]
        ax.legend(handles=legend_elements, loc='center', bbox_to_anchor=(1.15, 0.5))
    
    def _create_terrain_map(self, ax):
        """Create terrain and elevation map"""
        # Simulate elevation data
        elevations = {
            1: 1650, 2: 1680, 3: 1720, 4: 1600,
            5: 1670, 6: 1590, 7: 1750, 8: 1950,
            9: 2000, 10: 1700
        }
        
        grid = np.zeros((3, 4))
        
        for zone_id in range(1, 11):
            if zone_id in self.zone_positions:
                row, col = self.zone_positions[zone_id]
                grid[row, col] = elevations.get(zone_id, 1650)
        
        # Fill remaining positions
        for i in range(3):
            for j in range(4):
                if (i, j) not in self.position_to_zone:
                    grid[i, j] = np.nan
        
        im = ax.imshow(grid, cmap='terrain', aspect='equal')
        
        # Add zone labels and elevation
        for zone_id in range(1, 11):
            if zone_id in self.zone_positions:
                row, col = self.zone_positions[zone_id]
                elev = elevations.get(zone_id, 1650)
                ax.text(col, row, f'Z{zone_id}\n{elev}m', 
                       ha='center', va='center', fontweight='bold', color='white')
        
        # Add water sources
        water_zones = [z for z in range(1, 11) if z not in self.constraints["water_access_requirements"].get("zones_without_water", [])]
        for zone_id in water_zones:
            if zone_id in self.zone_positions:
                row, col = self.zone_positions[zone_id]
                ax.scatter(col, row, marker='o', s=100, c='blue', alpha=0.7, edgecolors='white', linewidth=2)
        
        ax.set_title('Terrain & Water Sources', fontweight='bold')
        ax.set_xticks([])
        ax.set_yticks([])
        plt.colorbar(im, ax=ax, label='Elevation (m)')
    
    def _create_info_panel(self, ax, date):
        """Create information panel"""
        ax.axis('off')
        
        info_text = f"""
ZONE INFORMATION - {date.strftime('%B %d, %Y')}

LEGEND:
🟢 Accessible zones
🟡 Seasonal restrictions  
🟠 No water access
🔴 Protected areas
💧 Water sources available

CONSTRAINTS ACTIVE:
• Protected Areas: Zones {self.constraints['zone_restrictions']['protected_areas']}
• No Water: Zones {self.constraints['water_access_requirements'].get('zones_without_water', [])}
• Seasonal (Apr-Jun): Zones {self.constraints['zone_restrictions']['seasonal_closures'].get('breeding_season', {}).get('zones', [])}

TERRAIN TYPES:
• Valleys: Zones 1, 2, 5, 6 (lower elevation)
• Hills: Zones 3, 4, 7 (moderate elevation)  
• Mountains: Zones 8, 9, 10 (high elevation)

OPTIMAL ZONES (Current Conditions):
"""
        
        # Calculate optimal zones
        optimal_zones = []
        for zone_id in range(1, 11):
            chars = self.get_zone_characteristics(zone_id, date)
            if chars['accessible'] and not chars['protected'] and not chars['no_water']:
                if chars['ndvi'] > 0.5:
                    optimal_zones.append(zone_id)
        
        info_text += f"Zones {optimal_zones} (High NDVI + Accessible)"
        
        ax.text(0.05, 0.95, info_text, transform=ax.transAxes, fontsize=10,
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))


class ModelTester:
    """
    Test and visualize AI Shepherd model performance
    """
    def __init__(self, agent, env, mapper):
        self.agent = agent
        self.env = env
        self.mapper = mapper
    
    def test_model_single_run(self, start_day=0, duration=30, visualize=True):
        """Test model for a single run and visualize the path"""
        print(f"Testing model for {duration} days starting from day {start_day}")
        
        # Reset environment
        state = self.env.reset()
        self.env.current_day = start_day
        
        # Track the journey
        journey = []
        total_reward = 0
        violations = 0
        
        for day in range(duration):
            current_date = datetime(2024, 1, 1) + timedelta(days=self.env.current_day)
            
            # Get valid actions
            valid_actions = self.env.get_valid_actions(self.env.current_day)
            
            # Get model's action
            action = self.agent.select_action(state, valid_actions, training=False)
            
            # Take step
            next_state, reward, done, info = self.env.step(action)
            
            # Record journey
            journey.append({
                'day': self.env.current_day - 1,
                'date': current_date,
                'zone': action,
                'reward': reward,
                'herd_health': info['herd_health'],
                'valid_actions': valid_actions.copy(),
                'zone_quality': info['zone_quality'].copy(),
                'accessible': info['zone_accessible'],
                'reason': info.get('restriction_reason', 'Accessible'),
                'violation': reward < -50
            })
            
            total_reward += reward
            if reward < -50:
                violations += 1
            
            state = next_state
            
            if done:
                break
        
        # Results summary
        results = {
            'journey': journey,
            'total_reward': total_reward,
            'average_reward': total_reward / len(journey),
            'violations': violations,
            'final_health': journey[-1]['herd_health'],
            'zones_used': len(set(j['zone'] for j in journey)),
            'compliance_rate': 1 - (violations / len(journey))
        }
        
        print(f"Test Results:")
        print(f"  Total Reward: {total_reward:.2f}")
        print(f"  Average Daily Reward: {results['average_reward']:.2f}")
        print(f"  Constraint Violations: {violations}/{len(journey)}")
        print(f"  Compliance Rate: {results['compliance_rate']:.1%}")
        print(f"  Final Herd Health: {results['final_health']:.1f}")
        print(f"  Zones Used: {results['zones_used']}/10")
        
        if visualize:
            self.visualize_journey(journey, results)
        
        return results
    
    def visualize_journey(self, journey, results):
        """Create comprehensive journey visualization"""
        fig = plt.figure(figsize=(20, 16))
        
        # Create main path visualization
        ax_main = plt.subplot2grid((4, 3), (0, 0), colspan=2, rowspan=2)
        self._create_path_map(ax_main, journey)
        
        # Zone usage chart
        ax_usage = plt.subplot2grid((4, 3), (0, 2))
        self._create_zone_usage_chart(ax_usage, journey)
        
        # Health and reward over time
        ax_health = plt.subplot2grid((4, 3), (1, 2))
        self._create_health_chart(ax_health, journey)
        
        # Daily rewards
        ax_rewards = plt.subplot2grid((4, 3), (2, 0))
        self._create_rewards_chart(ax_rewards, journey)
        
        # NDVI quality over time
        ax_ndvi = plt.subplot2grid((4, 3), (2, 1))
        self._create_ndvi_chart(ax_ndvi, journey)
        
        # Constraint violations timeline
        ax_violations = plt.subplot2grid((4, 3), (2, 2))
        self._create_violations_chart(ax_violations, journey)
        
        # Summary statistics
        ax_summary = plt.subplot2grid((4, 3), (3, 0), colspan=3)
        self._create_summary_panel(ax_summary, results, journey)
        
        plt.suptitle(f'AI Shepherd Journey Analysis - {len(journey)} Days', 
                     fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig('ai_shepherd_journey_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def _create_path_map(self, ax, journey):
        """Create the main path visualization on zone map"""
        # Create base zone map
        grid = np.ones((3, 4)) * 0.5  # Gray background
        
        # Color zones by average NDVI during journey
        zone_ndvi_avg = defaultdict(list)
        for j in journey:
            zone_ndvi_avg[j['zone']].append(j['zone_quality']['ndvi'])
        
        for zone_id, ndvi_values in zone_ndvi_avg.items():
            if zone_id < 10 and (zone_id + 1) in self.mapper.zone_positions:
                row, col = self.mapper.zone_positions[zone_id + 1]
                grid[row, col] = np.mean(ndvi_values)
        
        # Fill empty positions
        for i in range(3):
            for j in range(4):
                if (i, j) not in self.mapper.position_to_zone:
                    grid[i, j] = np.nan
        
        im = ax.imshow(grid, cmap='YlGn', vmin=0, vmax=1, aspect='equal', alpha=0.7)
        
        # Add zone labels
        for zone_id in range(1, 11):
            if zone_id in self.mapper.zone_positions:
                row, col = self.mapper.zone_positions[zone_id]
                ax.text(col, row, f'Z{zone_id}', ha='center', va='center', 
                       fontweight='bold', fontsize=12)
        
        # Draw the path
        path_x, path_y = [], []
        colors = []
        sizes = []
        
        for i, step in enumerate(journey):
            zone_id = step['zone'] + 1  # Convert to 1-indexed
            if zone_id in self.mapper.zone_positions:
                row, col = self.mapper.zone_positions[zone_id]
                path_x.append(col)
                path_y.append(row)
                
                # Color based on violation
                colors.append('red' if step['violation'] else 'blue')
                sizes.append(100 + i * 3)  # Growing size over time
        
        # Draw path lines
        if len(path_x) > 1:
            ax.plot(path_x, path_y, 'k-', alpha=0.6, linewidth=2, label='Path')
        
        # Draw points
        scatter = ax.scatter(path_x, path_y, c=colors, s=sizes, alpha=0.8, 
                           edgecolors='white', linewidth=1)
        
        # Mark start and end
        if path_x:
            ax.scatter(path_x[0], path_y[0], marker='s', s=200, c='green', 
                      edgecolors='white', linewidth=2, label='Start')
            ax.scatter(path_x[-1], path_y[-1], marker='*', s=300, c='gold', 
                      edgecolors='white', linewidth=2, label='End')
        
        ax.set_title('Grazing Path on Zone Map', fontweight='bold')
        ax.set_xticks([])
        ax.set_yticks([])
        ax.legend(loc='upper right', bbox_to_anchor=(1, 1))
        
        # Add colorbar
        plt.colorbar(im, ax=ax, label='Average NDVI')
    
    def _create_zone_usage_chart(self, ax, journey):
        """Create zone usage frequency chart"""
        zone_counts = defaultdict(int)
        for step in journey:
            zone_counts[step['zone'] + 1] += 1
        
        zones = list(range(1, 11))
        counts = [zone_counts[z] for z in zones]
        
        bars = ax.bar(zones, counts, color='skyblue', alpha=0.7)
        
        # Highlight most used zone
        if counts:
            max_idx = counts.index(max(counts))
            bars[max_idx].set_color('orange')
        
        ax.set_title('Zone Usage Frequency', fontweight='bold')
        ax.set_xlabel('Zone')
        ax.set_ylabel('Days Used')
        ax.set_xticks(zones)
        ax.grid(axis='y', alpha=0.3)
    
    def _create_health_chart(self, ax, journey):
        """Create herd health over time chart"""
        days = [step['day'] for step in journey]
        health = [step['herd_health'] for step in journey]
        
        ax.plot(days, health, 'g-', linewidth=2, marker='o', markersize=3)
        ax.fill_between(days, health, alpha=0.3, color='green')
        
        ax.set_title('Herd Health Over Time', fontweight='bold')
        ax.set_xlabel('Day')
        ax.set_ylabel('Health')
        ax.set_ylim(0, 100)
        ax.grid(alpha=0.3)
        
        # Add critical health line
        ax.axhline(y=20, color='red', linestyle='--', alpha=0.7, label='Critical')
        ax.legend()
    
    def _create_rewards_chart(self, ax, journey):
        """Create daily rewards chart"""
        days = [step['day'] for step in journey]
        rewards = [step['reward'] for step in journey]
        
        colors = ['red' if r < -50 else 'blue' for r in rewards]
        ax.bar(days, rewards, color=colors, alpha=0.7)
        
        ax.set_title('Daily Rewards', fontweight='bold')
        ax.set_xlabel('Day')
        ax.set_ylabel('Reward')
        ax.axhline(y=0, color='black', linestyle='-', alpha=0.5)
        ax.axhline(y=-50, color='red', linestyle='--', alpha=0.7, label='Violation Threshold')
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
    
    def _create_ndvi_chart(self, ax, journey):
        """Create NDVI quality over time"""
        days = [step['day'] for step in journey]
        ndvi = [step['zone_quality']['ndvi'] for step in journey]
        
        ax.plot(days, ndvi, 'g-', linewidth=2, marker='o', markersize=3)
        ax.fill_between(days, ndvi, alpha=0.3, color='green')
        
        ax.set_title('Vegetation Quality (NDVI)', fontweight='bold')
        ax.set_xlabel('Day')
        ax.set_ylabel('NDVI')
        ax.set_ylim(0, 1)
        ax.grid(alpha=0.3)
        
        # Add quality thresholds
        ax.axhline(y=0.6, color='green', linestyle='--', alpha=0.7, label='Good')
        ax.axhline(y=0.3, color='orange', linestyle='--', alpha=0.7, label='Poor')
        ax.legend()
    
    def _create_violations_chart(self, ax, journey):
        """Create constraint violations timeline"""
        days = [step['day'] for step in journey]
        violations = [1 if step['violation'] else 0 for step in journey]
        
        ax.fill_between(days, violations, alpha=0.5, color='red', step='pre')
        
        ax.set_title('Constraint Violations', fontweight='bold')
        ax.set_xlabel('Day')
        ax.set_ylabel('Violation')
        ax.set_ylim(0, 1.2)
        ax.set_yticks([0, 1])
        ax.set_yticklabels(['No', 'Yes'])
        ax.grid(axis='x', alpha=0.3)
    
    def _create_summary_panel(self, ax, results, journey):
        """Create summary statistics panel"""
        ax.axis('off')
        
        # Calculate additional statistics
        avg_ndvi = np.mean([step['zone_quality']['ndvi'] for step in journey])
        zone_switches = sum(1 for i in range(1, len(journey)) 
                          if journey[i]['zone'] != journey[i-1]['zone'])
        
        most_used_zone = max(set(step['zone'] + 1 for step in journey),
                           key=lambda x: sum(1 for step in journey if step['zone'] + 1 == x))
        
        summary_text = f"""
JOURNEY SUMMARY STATISTICS
══════════════════════════════════════════════════════════════════════

PERFORMANCE METRICS:
  • Total Reward: {results['total_reward']:.2f}
  • Average Daily Reward: {results['average_reward']:.2f}  
  • Final Herd Health: {results['final_health']:.1f}/100
  
CONSTRAINT COMPLIANCE:
  • Total Violations: {results['violations']}/{len(journey)}
  • Compliance Rate: {results['compliance_rate']:.1%}
  • Violation Days: {[step['day'] for step in journey if step['violation']]}

GRAZING PATTERNS:  
  • Zones Used: {results['zones_used']}/10 available zones
  • Most Used Zone: Zone {most_used_zone}
  • Zone Switches: {zone_switches} times
  • Average NDVI: {avg_ndvi:.3f}

OPERATIONAL EFFICIENCY:
  • Days per Zone: {len(journey)/results['zones_used']:.1f} average
  • Health Stability: {'STABLE' if abs(journey[0]['herd_health'] - journey[-1]['herd_health']) < 10 else 'VARIABLE'}
  • Resource Utilization: {'HIGH' if avg_ndvi > 0.5 else 'MODERATE' if avg_ndvi > 0.3 else 'LOW'}
"""
        
        ax.text(0.02, 0.98, summary_text, transform=ax.transAxes, fontsize=11,
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
    
    def compare_scenarios(self, scenarios):
        """Compare multiple test scenarios"""
        results = []
        
        for scenario in scenarios:
            print(f"\nTesting scenario: {scenario['name']}")
            result = self.test_model_single_run(
                start_day=scenario.get('start_day', 0),
                duration=scenario.get('duration', 30),
                visualize=False
            )
            result['scenario_name'] = scenario['name']
            results.append(result)
        
        # Create comparison visualization
        self._visualize_scenario_comparison(results)
        
        return results
    
    def _visualize_scenario_comparison(self, results):
        """Visualize comparison of multiple scenarios"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        scenarios = [r['scenario_name'] for r in results]
        
        # Total rewards comparison
        total_rewards = [r['total_reward'] for r in results]
        ax1.bar(scenarios, total_rewards, color='skyblue', alpha=0.7)
        ax1.set_title('Total Rewards by Scenario')
        ax1.set_ylabel('Total Reward')
        ax1.tick_params(axis='x', rotation=45)
        
        # Compliance rates
        compliance_rates = [r['compliance_rate'] * 100 for r in results]
        ax2.bar(scenarios, compliance_rates, color='green', alpha=0.7)
        ax2.set_title('Constraint Compliance Rates')
        ax2.set_ylabel('Compliance Rate (%)')
        ax2.set_ylim(0, 100)
        ax2.tick_params(axis='x', rotation=45)
        
        # Zone diversity
        zone_diversity = [r['zones_used'] for r in results]
        ax3.bar(scenarios, zone_diversity, color='orange', alpha=0.7)
        ax3.set_title('Zone Diversity (Zones Used)')
        ax3.set_ylabel('Number of Zones')
        ax3.set_ylim(0, 10)
        ax3.tick_params(axis='x', rotation=45)
        
        # Final herd health
        final_health = [r['final_health'] for r in results]
        ax4.bar(scenarios, final_health, color='red', alpha=0.7)
        ax4.set_title('Final Herd Health')
        ax4.set_ylabel('Health Level')
        ax4.set_ylim(0, 100)
        ax4.tick_params(axis='x', rotation=45)
        
        plt.suptitle('Scenario Comparison Analysis', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig('scenario_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()


def run_comprehensive_test(model_path='ppo_constrained_model_final.pth', data_folder='grazing_data'):
    """
    Run comprehensive testing suite for AI Shepherd model
    """
    print("AI Shepherd Model Testing Suite")
    print("=" * 50)
    
    # Load the environment (without agent for now)
    from ppo_constrained_complete import GrazingEnvironment, PPOAgent
    
    # Create environment and mapper
    env = GrazingEnvironment(data_folder)
    mapper = GrazingZoneMapper(data_folder)
    
    # Create and load agent
    agent = PPOAgent(
        state_dim=env.observation_space.shape[0],
        action_dim=env.action_space.n,
        lr=3e-4
    )
    
    try:
        agent.load_model(model_path)
        print(f"Model loaded successfully from {model_path}")
    except Exception as e:
        print(f"Warning: Could not load model from {model_path}: {e}")
        print("Using randomly initialized agent for testing structure")
    
    # Create tester
    tester = ModelTester(agent, env, mapper)
    
    print("\n1. Creating Zone Maps...")
    # Create zone maps for different seasons
    dates_to_map = [
        datetime(2024, 3, 15),  # Spring
        datetime(2024, 6, 15),  # Summer  
        datetime(2024, 9, 15),  # Autumn
        datetime(2024, 12, 15)  # Winter
    ]
    
    for date in dates_to_map:
        season = ["Winter", "Winter", "Winter", "Spring", "Spring", "Spring", 
                 "Summer", "Summer", "Summer", "Autumn", "Autumn", "Winter"][date.month-1]
        mapper.create_zone_map(f"Zone Map - {season}", date, 
                             f'zone_map_{season.lower()}.png')
    
    print("\n2. Running Single Model Test...")
    # Test model for spring season
    spring_results = tester.test_model_single_run(
        start_day=74,  # Mid-March
        duration=30,
        visualize=True
    )
    
    print("\n3. Running Scenario Comparisons...")
    # Define test scenarios
    scenarios = [
        {"name": "Spring_Optimal", "start_day": 74, "duration": 30},    # March 15
        {"name": "Summer_Challenging", "start_day": 165, "duration": 30}, # June 15  
        {"name": "Autumn_Recovery", "start_day": 257, "duration": 30},   # Sept 15
        {"name": "Winter_Survival", "start_day": 348, "duration": 30},   # Dec 15
        {"name": "Long_Term", "start_day": 0, "duration": 90}            # Full quarter
    ]
    
    comparison_results = tester.compare_scenarios(scenarios)
    
    # Print detailed comparison
    print("\nDetailed Scenario Results:")
    print("-" * 80)
    print(f"{'Scenario':<20} {'Reward':<10} {'Compliance':<12} {'Health':<8} {'Zones':<6}")
    print("-" * 80)
    
    for result in comparison_results:
        print(f"{result['scenario_name']:<20} {result['total_reward']:<10.1f} "
              f"{result['compliance_rate']:<12.1%} {result['final_health']:<8.1f} "
              f"{result['zones_used']:<6}")
    
    print("\n4. Constraint Analysis...")
    # Analyze constraint impact
    constraint_analysis = analyze_constraint_patterns(tester, env)
    
    print("\nTesting Complete! Generated files:")
    print("- zone_map_spring.png, zone_map_summer.png, zone_map_autumn.png, zone_map_winter.png")
    print("- ai_shepherd_journey_analysis.png")  
    print("- scenario_comparison.png")
    print("- constraint_analysis.png")
    
    return {
        'zone_maps': dates_to_map,
        'single_test': spring_results,
        'scenario_comparison': comparison_results,
        'constraint_analysis': constraint_analysis
    }


def analyze_constraint_patterns(tester, env):
    """
    Analyze how constraints affect model decisions over different time periods
    """
    print("Analyzing constraint patterns...")
    
    # Test same period with and without specific constraints
    analysis_results = {}
    
    # Store original constraints
    original_constraints = env.constraints.copy()
    
    # Test scenarios with different constraint sets
    constraint_tests = [
        {
            'name': 'All_Constraints',
            'constraints': original_constraints
        },
        {
            'name': 'No_Seasonal',
            'constraints': {
                **original_constraints,
                'zone_restrictions': {
                    **original_constraints['zone_restrictions'],
                    'seasonal_closures': {}
                }
            }
        },
        {
            'name': 'No_Water_Limits',
            'constraints': {
                **original_constraints,
                'water_access_requirements': {
                    **original_constraints['water_access_requirements'],
                    'zones_without_water': []
                }
            }
        },
        {
            'name': 'No_Protected',
            'constraints': {
                **original_constraints,
                'zone_restrictions': {
                    **original_constraints['zone_restrictions'],
                    'protected_areas': []
                }
            }
        }
    ]
    
    # Run tests with different constraint sets
    for test in constraint_tests:
        env.constraints = test['constraints']
        result = tester.test_model_single_run(
            start_day=120,  # May 1st - peak constraint period
            duration=21,    # 3 weeks
            visualize=False
        )
        analysis_results[test['name']] = result
    
    # Restore original constraints
    env.constraints = original_constraints
    
    # Create constraint analysis visualization
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    
    # Rewards comparison
    test_names = list(analysis_results.keys())
    rewards = [analysis_results[name]['total_reward'] for name in test_names]
    
    bars1 = ax1.bar(test_names, rewards, color=['red', 'orange', 'yellow', 'green'], alpha=0.7)
    ax1.set_title('Impact of Constraints on Total Reward')
    ax1.set_ylabel('Total Reward')
    ax1.tick_params(axis='x', rotation=45)
    
    # Compliance rates
    compliance = [analysis_results[name]['compliance_rate'] * 100 for name in test_names]
    bars2 = ax2.bar(test_names, compliance, color=['red', 'orange', 'yellow', 'green'], alpha=0.7)
    ax2.set_title('Constraint Compliance Rates')
    ax2.set_ylabel('Compliance Rate (%)')
    ax2.set_ylim(0, 100)
    ax2.tick_params(axis='x', rotation=45)
    
    # Zone diversity
    diversity = [analysis_results[name]['zones_used'] for name in test_names]
    bars3 = ax3.bar(test_names, diversity, color=['red', 'orange', 'yellow', 'green'], alpha=0.7)
    ax3.set_title('Zone Diversity Usage')
    ax3.set_ylabel('Zones Used')
    ax3.set_ylim(0, 10)
    ax3.tick_params(axis='x', rotation=45)
    
    # Violations
    violations = [analysis_results[name]['violations'] for name in test_names]
    bars4 = ax4.bar(test_names, violations, color=['red', 'orange', 'yellow', 'green'], alpha=0.7)
    ax4.set_title('Constraint Violations')
    ax4.set_ylabel('Number of Violations')
    ax4.tick_params(axis='x', rotation=45)
    
    plt.suptitle('Constraint Impact Analysis', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('constraint_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Print analysis summary
    print("\nConstraint Impact Summary:")
    print("-" * 50)
    baseline = analysis_results['All_Constraints']
    for name, result in analysis_results.items():
        if name != 'All_Constraints':
            reward_change = ((result['total_reward'] - baseline['total_reward']) / 
                           abs(baseline['total_reward'])) * 100
            print(f"{name}: {reward_change:+.1f}% reward change, "
                  f"{result['violations']} violations vs {baseline['violations']}")
    
    return analysis_results


# Example usage and testing functions
def quick_test():
    """Quick test function to verify everything works"""
    print("Running quick test of AI Shepherd system...")
    
    # Create mapper and show one zone map
    mapper = GrazingZoneMapper()
    mapper.create_zone_map("Quick Test Zone Map")
    
    print("Quick test complete! Zone map displayed.")
    return "Test successful"


def demo_scenario():
    """Create a demo scenario for presentation"""
    print("Creating demo scenario...")
    
    # This would use synthetic data for demo purposes
    demo_journey = [
        {'day': 0, 'zone': 1, 'reward': 45.2, 'herd_health': 98.5, 'violation': False, 
         'zone_quality': {'ndvi': 0.65}, 'accessible': True},
        {'day': 1, 'zone': 1, 'reward': 42.8, 'herd_health': 97.2, 'violation': False,
         'zone_quality': {'ndvi': 0.63}, 'accessible': True},
        {'day': 2, 'zone': 2, 'reward': 38.5, 'herd_health': 96.8, 'violation': False,
         'zone_quality': {'ndvi': 0.58}, 'accessible': True},
        # ... more demo data
    ]
    
    print("Demo scenario created. In real usage, this would show actual model decisions.")
    return demo_journey


if __name__ == "__main__":
    print("AI Shepherd Testing Suite")
    print("Choose an option:")
    print("1. Quick Test (just show zone map)")
    print("2. Full Test Suite (requires trained model)")  
    print("3. Demo Scenario (synthetic data)")
    
    choice = input("Enter choice (1, 2, or 3): ").strip()
    
    if choice == "1":
        quick_test()
    elif choice == "2":
        # Run full test suite
        results = run_comprehensive_test()
        print("Full test suite completed!")
    elif choice == "3":
        demo_scenario()
    else:
        print("Invalid choice. Running quick test...")
        quick_test()

print("""
=== AI SHEPHERD TESTING COMPLETE ===

To continue our conversation about your AI shepherd system, you could ask about:

• "How can I improve the model's performance in winter months?"
• "What modifications would help with larger herd sizes?"  
• "How do I add new types of constraints to the system?"
• "Can you explain the reward function design choices?"
• "How would I deploy this model in a real-world IoT system?"
• "What data collection improvements would help training?"
• "How do I handle dynamic weather constraints in real-time?"

Or ask about any specific aspect of the implementation, training, or results you'd like to explore further.
""")