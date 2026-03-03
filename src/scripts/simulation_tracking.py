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
        self.test_config_path = os.path.abspath(os.path.join(current_dir, '../config/test_samples.yaml'))
        
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

    def generate_random_box(self, r_range=(1.5, 2.5), z_range=(-0.2, 0.2)):
        # Random generation using Polar Coordinates
        r = random.uniform(*r_range)
        theta = random.uniform(0, 2 * np.pi)
        x = r * np.cos(theta)
        y = r * np.sin(theta)
        z = random.uniform(*z_range)
        return np.array([x, y, z])

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
            box1 = self.generate_random_box()
            box2 = self.generate_random_box()
            box3 = self.generate_random_box()
            
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
        

    def test_k_influence(self, k_values=[1, 5, 10, 20, 50, None], use_config=True, box_positions=None, save_filename=None, save_to_csv=True):
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
                    'computation_time': round(comp_time, 3),
                    'execution_time': round(exe_time, 3),
                    'energy': round(energy, 3)
                })
                print(f"Result for k={k}: Computation Time = {comp_time:.3f}s, Execution Time = {exe_time:.3f}s, Energy = {energy:.3f}J")

        if not results:
            return None

        # Save results to /output if save_to_csv is True
        if save_to_csv:
            output_dir = os.path.abspath(os.path.join(current_dir, '../output'))
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            
            if save_filename is None:
                timestamp = time.strftime("%Y%m%d_%H%M%S")
                filename = f"k_experiment_{timestamp}.csv"
            else:
                filename = save_filename
                
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
            print(f"{k_str:>5} | {res['computation_time']:>15.3f} | {res['execution_time']:>15.3f} | {total_time:>15.3f} | {res['energy']:>15.3f}")
        print("="*105 + "\n")
        return results

    def run_k_test_batch(self, num_experiments=10, k_values=[1, 5, 10, 20, 50, 80, None], save_filename=None):
        print(f"\nStarting Batch k-influence Experiment ({num_experiments} rounds)")
        all_results = []
        
        for i in range(num_experiments):
            print(f"\n{'='*25} Experiment Round {i+1}/{num_experiments} {'='*25}")
            # Use smaller radius and z-range for higher success probability as found in testing
            box1 = self.generate_random_box(r_range=(0.8, 1.2), z_range=(-0.1, 0.0))
            box2 = self.generate_random_box(r_range=(0.8, 1.2), z_range=(-0.1, 0.0))
            box3 = self.generate_random_box(r_range=(0.8, 1.2), z_range=(-0.1, 0.0))
            box4 = self.generate_random_box(r_range=(0.8, 1.2), z_range=(-0.1, 0.0))
            box_positions = np.array([box1, box2, box3, box4])
            
            round_results = self.test_k_influence(k_values=k_values, box_positions=box_positions, save_to_csv=False)
            
            if round_results:
                for res in round_results:
                    res['experiment_round'] = i + 1
                    all_results.append(res)
                
                # Add a divider row between rounds
                if i < num_experiments - 1:
                    divider = {key: "--------------" for key in all_results[-1].keys()}
                    all_results.append(divider)
            else:
                print(f"Round {i+1} failed to find valid candidates.")

        if all_results:
            output_dir = os.path.abspath(os.path.join(current_dir, '../output'))
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            filename = save_filename if save_filename else f"k_batch_experiment_{timestamp}.csv"
            save_path = os.path.join(output_dir, filename)
            
            keys = all_results[0].keys()
            with open(save_path, 'w', newline='') as f:
                dict_writer = csv.DictWriter(f, fieldnames=keys)
                dict_writer.writeheader()
                dict_writer.writerows(all_results)
            print(f"\nAll {num_experiments} experiments completed. Batch data saved to: {save_path}")
        else:
            print("\nNo experiments succeeded.")

if __name__ == "__main__":
    sim = SimulationTracking()

    # Example box positions for experiment
    box1 = np.array([1.25, 0.35, -0.1])
    box2 = np.array([0.4, 1.3, -0.1])
    box3 = np.array([1.0, -0.8, 0.0])
    example_boxes = np.array([box1, box2])

    try:
        choice = '6'

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
        elif choice == '6':
            sim.run_k_test_batch(num_experiments=10)
        elif choice == 'q':
            sys.exit(0)

    except KeyboardInterrupt:
        print("\n simulation tracking exit")
    except Exception as e:
        print(f"error: {e}")
