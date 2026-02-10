import sys
import os
import numpy as np
import time
import random
import yaml
import csv

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

    def run_multi_throwing_sim(self, mode='greedy', use_config=False, k=None, animate=True):
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
            
            print("\n" + "="*60)
            print(f">>> Group {self.sim_group_count + 1} (Auto-generated Random Samples)")
            for i, b in enumerate([box1, box2, box3]):
                print(f"    Box {i+1}: x={b[0]:>7.3f}, y={b[1]:>7.3f}, z={b[2]:>7.3f}")
            print("="*60 + "\n")
            
            self.sim_group_count += 1
            box_positions = np.array([box1, box2, box3])
        
        if mode == 'greedy':
            res = self.generator.solve_multi_targets(box_positions, animate=animate, full_search=True, k=k)
        elif mode == 'random':
            res = self.generator.solve_multi_targets(box_positions, animate=animate, full_search=False, random_select=True, k=k)
        
        if res and len(res) >= 6:
            return res[3], res[4], res[5] # search_time, best_duration, total_energy
        return None, None, None
        

    def test_k_influence(self, k_values=[1, 5, 10, 20, 40, 60, 100, 200, None], use_config=True, box_positions=None):
        print("\n" + "#"*60)
        print("Testing the influence of parameter k on performance and energy")
        print("#"*60 + "\n")
        
        results = []
        for k in k_values:
            print(f"\n--- Testing k = {k} ---")
            
            if box_positions is not None:
                # Direct solve if box_positions are provided
                res = self.generator.solve_multi_targets(box_positions, animate=False, full_search=True, k=k)
                if res and len(res) >= 6:
                    comp_time, exe_time, energy = res[3], res[4], res[5]
                else:
                    comp_time, exe_time, energy = None, None, None
            else:
                # Reset group count to use the same config for each k if use_config is True
                self.sim_group_count = 0 
                comp_time, exe_time, energy = self.run_multi_throwing_sim(mode='greedy', use_config=use_config, k=k, animate=False)
            
            if comp_time is not None:
                results.append({
                    'k': k,
                    'computation_time': comp_time,
                    'execution_time': exe_time,
                    'energy': energy
                })
                print(f"Result for k={k}: Computation Time = {comp_time:.4f}s, Execution Time = {exe_time:.4f}s, Energy = {energy:.4f}J")

        # Save results to /output
        output_dir = os.path.abspath(os.path.join(current_dir, '../output'))
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename = f"k_experiment_{timestamp}.csv"
        save_path = os.path.join(output_dir, filename)
        
        # Write to CSV
        keys = results[0].keys() if results else []
        with open(save_path, 'w', newline='') as f:
            dict_writer = csv.DictWriter(f, fieldnames=keys)
            dict_writer.writeheader()
            dict_writer.writerows(results)
            
        print(f"\nExperiment data saved to: {save_path}")

        print("\n" + "="*105)
        print(f"{'k':>5} | {'Comp Time (s)':>15} | {'Exec Time (s)':>15} | {'Total Time (s)':>15} | {'Energy (J)':>15}")
        print("-" * 105)
        for res in results:
            total_time = res['computation_time'] + res['execution_time']
            k_str = "Full" if res['k'] is None else str(res['k'])
            print(f"{k_str:>5} | {res['computation_time']:>15.4f} | {res['execution_time']:>15.4f} | {total_time:>15.4f} | {res['energy']:>15.4f}")
        print("="*105 + "\n")
        return save_path

if __name__ == "__main__":
    sim = SimulationTracking()
    
    # Example box positions for experiment
    box1 = np.array([1.25, 0.35, -0.1])
    box2 = np.array([0.4, 1.3, -0.1])
    box3 = np.array([1.5, -0.5, 0.0])
    example_boxes = np.array([box1, box2, box3])
    
    try:
        
        choice = '4' 
        
        if choice == '1':
            sim.run_multi_throwing_sim(mode='greedy', use_config=False)
        elif choice == '2':
            sim.run_multi_throwing_sim(mode='greedy', use_config=True)
        elif choice == '3':
            sim.run_multi_throwing_sim(mode='random', use_config=False)
        elif choice == '4':
            sim.test_k_influence(k_values=[1, 5, 10, 20, 50, None], box_positions=example_boxes)
        elif choice == '5':
            sim.test_k_influence(use_config=True)
        elif choice == 'q':
            sys.exit(0)
            
    except KeyboardInterrupt:
        print("\n simulation tracking exit")
    except Exception as e:
        print(f"error: {e}")
