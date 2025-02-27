#!/usr/bin/env python3
import os
import subprocess
import argparse
import csv
import time
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def run_scene_generator(script_path, scene_type, copies, output_dir):
    """Run the scene generator script to create test scenes"""
    cmd = [
        "python", script_path,
        "--scene-type", scene_type,
        "--copies", str(copies),
        "--output-dir", output_dir
    ]
    
    print(f"Running scene generator: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        print(f"Error generating scene: {result.stderr}")
        return None
    
    # Extract scene file names from output
    scene_file = None
    constraint_file = None
    for line in result.stdout.splitlines():
        if "Scene file:" in line:
            scene_file = line.split("Scene file:")[-1].strip()
        if "Constraint file:" in line:
            constraint_file = line.split("Constraint file:")[-1].strip()
    
    return scene_file, constraint_file

def run_benchmarks(scene_files, constraint_files, executable_path, output_file, 
                  num_frames, collision_methods, constraint_methods, repetitions):
    """Run benchmark for all specified scenes"""
    
    # Create output file with header if it doesn't exist
    if not os.path.exists(output_file) or os.path.getsize(output_file) == 0:
        with open(output_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                'SceneFile', 'ConstraintFile', 'NumFrames', 'CollisionMethod',
                'ConstraintMethod', 'TotalRunTime(ms)', 'TotalFrames',
                'CollisionDetectionTime(ms)', 'CollisionResolutionTime(ms)',
                'ConstraintResolutionTime(ms)', 'CollisionChecks',
                'CollisionsDetected', 'ConstraintIterations', 'ConstraintsResolved'
            ])
    
    # Run all benchmarks in nested loops
    for rep in range(repetitions):
        for i, (scene_file, constraint_file) in enumerate(zip(scene_files, constraint_files)):
            for collision_method in collision_methods:
                for constraint_method in constraint_methods:
                    print(f"\nRunning benchmark {rep+1}/{repetitions}: "
                          f"Scene {i+1}/{len(scene_files)}, "
                          f"Collision method {collision_method}, "
                          f"Constraint method {constraint_method}")
                    
                    # Run the benchmark executable directly
                    cmd = [
                        executable_path,
                        scene_file,
                        constraint_file,
                        str(num_frames),
                        str(collision_method),
                        str(constraint_method),
                        output_file
                    ]
                    
                    print(f"Running: {' '.join(cmd)}")
                    result = subprocess.run(cmd, capture_output=True, text=True)
                    
                    if result.returncode != 0:
                        print(f"Error running benchmark: {result.stderr}")
                    
                    # Short delay between benchmarks to let system settle
                    time.sleep(1)

def analyze_results(results_file, output_dir):
    """Analyze benchmark results and generate plots"""
    if not os.path.exists(results_file):
        print(f"Error: Results file {results_file} not found")
        return
    
    # Load benchmark results
    results = pd.read_csv(results_file)
    
    # Extract scene complexity from scene file names
    results['Complexity'] = results['SceneFile'].str.extract(r'(\d+)').astype(int)
    
    # Create output directory for plots if needed
    plots_dir = os.path.join(output_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)
    
    # Average results across repetitions
    grouped = results.groupby(['Complexity', 'CollisionMethod', 'ConstraintMethod'])
    avg_results = grouped.mean().reset_index()
    
    # Plot total run time vs complexity
    plt.figure(figsize=(10, 6))
    
    # Plot for each collision method
    for collision_method in sorted(avg_results['CollisionMethod'].unique()):
        cm_data = avg_results[avg_results['CollisionMethod'] == collision_method]
        
        # For each constraint method
        for constraint_method in sorted(cm_data['ConstraintMethod'].unique()):
            data = cm_data[cm_data['ConstraintMethod'] == constraint_method]
            
            # Sort by complexity
            data = data.sort_values('Complexity')
            
            # Plot line
            label = f"Collision: {collision_method}, Constraint: {constraint_method}"
            plt.plot(data['Complexity'], data['TotalRunTime(ms)'] / data['NumFrames'], 
                    marker='o', label=label)
    
    plt.xlabel('Scene Complexity (Number of Objects)')
    plt.ylabel('Average Time per Frame (ms)')
    plt.title('Performance Scaling with Scene Complexity')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(plots_dir, 'performance_scaling.png'), dpi=300, bbox_inches='tight')
    
    # Create more detailed plots
    collision_methods_names = {
        0: "Brute Force",
        1: "Spatial Hash",
        2: "BVH"
    }
    
    constraint_methods_names = {
        0: "Sequential",
        1: "Islands",
        2: "Parallel Islands",
        3: "Propagation"
    }
    
    # Plot timing breakdown by complexity
    for (collision_method, constraint_method), group_data in avg_results.groupby(['CollisionMethod', 'ConstraintMethod']):
        plt.figure(figsize=(10, 6))
        
        group_data = group_data.sort_values('Complexity')
        
        # Stack plot showing collision detection, collision resolution, and constraint resolution
        plt.stackplot(group_data['Complexity'], 
                      group_data['CollisionDetectionTime(ms)'] / group_data['NumFrames'],
                      group_data['CollisionResolutionTime(ms)'] / group_data['NumFrames'],
                      group_data['ConstraintResolutionTime(ms)'] / group_data['NumFrames'],
                      labels=['Collision Detection', 'Collision Resolution', 'Constraint Resolution'])
        
        plt.xlabel('Scene Complexity (Number of Objects)')
        plt.ylabel('Time per Frame (ms)')
        title = f'Time Breakdown - Collision: {collision_methods_names.get(collision_method, collision_method)}, '
        title += f'Constraint: {constraint_methods_names.get(constraint_method, constraint_method)}'
        plt.title(title)
        plt.legend()
        plt.grid(True)
        
        filename = f'breakdown_c{collision_method}_s{constraint_method}.png'
        plt.savefig(os.path.join(plots_dir, filename), dpi=300, bbox_inches='tight')
        plt.close()
    
    # Generate summary report
    with open(os.path.join(output_dir, "benchmark_summary.txt"), 'w') as f:
        f.write("==== Rigid Body Simulation Benchmark Summary ====\n\n")
        
        # Overall performance ranking
        f.write("Performance Ranking (Average ms per frame):\n")
        ranking = avg_results.groupby(['CollisionMethod', 'ConstraintMethod'])['TotalRunTime(ms)'].mean() / \
                 avg_results.groupby(['CollisionMethod', 'ConstraintMethod'])['NumFrames'].mean()
        ranking = ranking.sort_values()
        
        for i, ((col_method, constr_method), value) in enumerate(ranking.items()):
            col_name = collision_methods_names.get(col_method, f"Method {col_method}")
            constr_name = constraint_methods_names.get(constr_method, f"Method {constr_method}")
            f.write(f"{i+1}. Collision: {col_name}, Constraint: {constr_name}: {value:.2f} ms\n")
        
        # Scaling behavior
        f.write("\nScaling Behavior:\n")
        for (col_method, constr_method), group in avg_results.groupby(['CollisionMethod', 'ConstraintMethod']):
            if len(group) >= 2:
                group = group.sort_values('Complexity')
                complexity_ratio = group['Complexity'].iloc[-1] / group['Complexity'].iloc[0]
                time_ratio = (group['TotalRunTime(ms)'].iloc[-1] / group['NumFrames'].iloc[-1]) / \
                             (group['TotalRunTime(ms)'].iloc[0] / group['NumFrames'].iloc[0])
                
                col_name = collision_methods_names.get(col_method, f"Method {col_method}")
                constr_name = constraint_methods_names.get(constr_method, f"Method {constr_method}")
                
                f.write(f"Collision: {col_name}, Constraint: {constr_name}:\n")
                f.write(f"  - Complexity increased by {complexity_ratio:.1f}x\n")
                f.write(f"  - Runtime increased by {time_ratio:.1f}x\n")
                
                # Calculate approximate Big-O notation
                if complexity_ratio > 1:
                    big_o = np.log(time_ratio) / np.log(complexity_ratio)
                    notation = ""
                    if big_o < 1.2:
                        notation = "O(n)"
                    elif big_o < 1.8:
                        notation = "O(n log n)"
                    elif big_o < 2.2:
                        notation = "O(n²)"
                    elif big_o < 2.8:
                        notation = "O(n²log n)"
                    else:
                        notation = f"O(n^{big_o:.1f})"
                    
                    f.write(f"  - Approximate scaling: {notation} (exponent: {big_o:.2f})\n")
                
                f.write("\n")

def main():
    parser = argparse.ArgumentParser(description='Run rigid body simulation benchmarks with scene generation')
    parser.add_argument('--generator-script', required=True, help='Path to testbench_generator.py script')
    parser.add_argument('--executable', required=True, help='Path to benchmark_runner executable')
    parser.add_argument('--output-dir', default='./benchmark_results', help='Output directory for results')
    parser.add_argument('--num-frames', type=int, default=100, help='Number of frames to simulate per benchmark')
    parser.add_argument('--repetitions', type=int, default=3, help='Number of times to repeat each benchmark')
    
    # Scene generation parameters
    parser.add_argument('--scene-type', default='stress', choices=['stress', 'two_chain', 'two_cylinder', 'cube'],
                        help='Type of scene to generate')
    parser.add_argument('--copies-range', type=int, nargs='+', default=[1, 2, 4, 8, 16], 
                        help='Number of copies to generate for each scene')
    
    # Method selection
    parser.add_argument('--collision-methods', type=int, nargs='+', default=[0, 1], 
                        help='Collision methods to test (0=Brute, 1=SpatialHash, 2=BVH)')
    parser.add_argument('--constraint-methods', type=int, nargs='+', default=[0, 1, 2, 3],
                        help='Constraint methods to test (0=Sequential, 1=Islands, 2=ParallelIslands, 3=PropagationIslands)')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Results file
    results_file = os.path.join(args.output_dir, "benchmark_results.csv")
    
    # Generate scenes and run benchmarks
    scene_files = []
    constraint_files = []
    
    for copies in args.copies_range:
        print(f"Generating scene with {copies} copies...")
        scene_file, constraint_file = run_scene_generator(
            args.generator_script,
            args.scene_type,
            copies,
            args.output_dir
        )
        
        if scene_file and constraint_file:
            scene_files.append(os.path.basename(scene_file))
            constraint_files.append(os.path.basename(constraint_file))
    
    if scene_files:
        # Run benchmarks
        run_benchmarks(
            scene_files,
            constraint_files,
            args.executable,
            results_file,
            args.num_frames,
            args.collision_methods,
            args.constraint_methods,
            args.repetitions
        )
        
        # Analyze results
        analyze_results(results_file, args.output_dir)
    else:
        print("No scenes were generated successfully.")

if __name__ == "__main__":
    main()