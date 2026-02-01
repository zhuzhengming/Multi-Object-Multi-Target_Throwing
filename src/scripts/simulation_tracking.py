import sys
import os
import numpy as np
import time
import random
import yaml

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(current_dir, "../"))

from trajectory_generator import TrajectoryGenerator

class SimulationTracking:
    def __init__(self, box_position=None):
        self.box_position = box_position if box_position is not None else np.array([1.3, 0.07, -0.158])
        
        self.hedgehog_path = os.path.abspath(os.path.join(current_dir, '../hedgehog_data'))
        self.brt_path = os.path.abspath(os.path.join(current_dir, '../brt_data'))
        self.robot_path = os.path.abspath(os.path.join(current_dir, '../description/iiwa7_allegro_throwing.xml'))
        self.test_config_path = os.path.abspath(os.path.join(current_dir, '../config/test_samples_random.yaml'))
        
        self.q_min = np.array([-2.96705972839, -2.09439510239, -2.96705972839, -2.09439510239, -2.96705972839,
                          -2.09439510239, -3.05432619099])
        self.q_max = -self.q_min
        
        self.generator = TrajectoryGenerator(
            self.q_max, self.q_min,
            self.hedgehog_path, self.brt_path,
            self.robot_path, self.box_position
        )
        self.test_groups = []
        self.load_test_config()

    def load_test_config(self):
        if os.path.exists(self.test_config_path):
            with open(self.test_config_path, 'r') as f:
                try:
                    data = yaml.safe_load(f)
                    self.test_groups = data.get('test_groups', [])
                    print(f"Successfully loaded {len(self.test_groups)} test groups from YAML")
                except yaml.YAMLError as exc:
                    print(f"Error loading YAML config: {exc}")
        else:
            print(f"Test config file not found: {self.test_config_path}")

    def run_multi_throwing_sim(self, mode='greedy', use_config=False):
        if not hasattr(self, 'sim_group_count'):
            self.sim_group_count = 0

        if use_config and self.test_groups:
            # Loop through groups in config
            group = self.test_groups[self.sim_group_count % len(self.test_groups)]
            box_positions = np.array([np.array(b) for b in group['boxes']])
            group_id = group.get('group_id', self.sim_group_count + 1)
            
            print("\n" + "="*60)
            print(f">>> Running Custom Test Group (Group ID: {group_id})")
            for i, b in enumerate(box_positions):
                print(f"    Box {i+1}: x={b[0]:>7.3f}, y={b[1]:>7.3f}, z={b[2]:>7.3f}")
            print("="*60 + "\n")
            
            self.sim_group_count += 1
        else:
            # Random generation using Polar Coordinates
            def generate_random_box():
                # Continuous range: 1.2 < r < 2.5
                r = random.uniform(1.5, 2.5)
                theta = random.uniform(0, 2 * np.pi)
                x = r * np.cos(theta)
                y = r * np.sin(theta)
                z = random.uniform(-0.2, 0.2)
                return np.array([x, y, z])

            box1 = generate_random_box()
            box2 = generate_random_box()
            box3 = generate_random_box()

            # box1[2] = -0.3
            # box2[2] = 0.0
            # box3[2] = 0.3
            
            print("\n" + "="*60)
            print(f">>> Group {self.sim_group_count + 1} (Auto-generated Random Samples)")
            for i, b in enumerate([box1, box2, box3]):
                print(f"    Box {i+1}: x={b[0]:>7.3f}, y={b[1]:>7.3f}, z={b[2]:>7.3f}")
            print("="*60 + "\n")
            
            self.sim_group_count += 1
            box_positions = np.array([box1, box2, box3])
        
        if mode == 'greedy':
            self.generator.solve_multi_targets(box_positions, animate=True, full_search=True)
        elif mode == 'random':
            self.generator.solve_multi_targets(box_positions, animate=True, full_search=False, random_select=True)
        

    def run_single_throwing_sim(self, posture='posture1'):
        self.generator.solve(animate=True, posture=posture)

if __name__ == "__main__":
    sim = SimulationTracking()
    
    try:
        while True:
            
            choice = '2'
            
            if choice == '1':
                sim.run_multi_throwing_sim(mode='greedy', use_config=False)
            elif choice == '2':
                sim.run_multi_throwing_sim(mode='greedy', use_config=True)
            elif choice == '3':
                sim.run_multi_throwing_sim(mode='random', use_config=False)
            elif choice == '4':
                sim.run_multi_throwing_sim(mode='greedy', use_config=True)

            time.sleep(1)
                
    except KeyboardInterrupt:
        print("\n simulation tracking exit")
    except Exception as e:
        print(f"error: {e}")
