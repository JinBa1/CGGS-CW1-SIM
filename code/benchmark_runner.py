#!/usr/bin/env python3
import os
import subprocess
import argparse
import csv
import time
from itertools import product

def run_benchmark(executable_path, scene_file, constraint_file, num_frames, collision_method, 
                 constraint_method, output_file, data_path):
    """Run a single benchmark with specified parameters"""
    
    scene_path = os.path.join(data_path, scene_file)
    constraint_path = os.path.join(data_path, constraint_file)
    
    # Check if scene and constraint files exist
    if not os.path.exists(scene_path):
        print(f"Error: Scene file {scene_path} does not exist")
        return None
    if not os.path.exists(constraint_path):
        print(f"Error: Constraint file {constraint_path} does not exist")
        return None
    
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
        return None
    
    # Return stdout for additional parsing if needed
    return result.stdout

def create_output_header(output_file):
    """Create CSV header for the output file if it doesn't exist"""
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

def main():
    parser = argparse.ArgumentParser(description='Run rigid body simulation benchmarks')
    parser.add_argument('--executable', required=True, help='Path to benchmark_runner executable')
    parser.add_argument('--data-path', required=True, help='Path to data directory containing scene files')
    parser.add_argument('--output', default='benchmark_results.csv', help='Output CSV file for results')
    parser.add_argument('--num-frames', type=int, default=100, help='Number of frames to simulate per benchmark')
    parser.add_argument('--repetitions', type=int, default=3, help='Number of times to repeat each benchmark')
    
    # Arguments for scene selection
    parser.add_argument('--scenes', nargs='+', default=[], help='Scene files to benchmark (without .txt extension)')
    parser.add_argument('--scene-pattern', default=None, help='Pattern for scene files (e.g., "cube_*-scene.txt")')
    
    # Method selection
    parser.add_argument('--collision-methods', type=int, nargs='+', default=[0, 1], 
                        help='Collision methods to test (0=Brute, 1=SpatialHash, 2=BVH)')
    parser.add_argument('--constraint-methods', type=int, nargs='+', default=[0, 1, 2, 3],
                        help='Constraint methods to test (0=Sequential, 1=Islands, 2=ParallelIslands, 3=PropagationIslands)')
    
    args = parser.parse_args()
    
    # Create output file with header
    create_output_header(args.output)
    
    # Get scene files to test
    scene_files = []
    constraint_files = []
    
    if args.scenes:
        for scene in args.scenes:
            scene_files.append(f"{scene}-scene.txt")
            constraint_files.append(f"{scene}-constraints.txt")
    elif args.scene_pattern:
        import glob
        # Get all scene files matching the pattern
        matched_files = glob.glob(os.path.join(args.data_path, args.scene_pattern))
        for scene_path in matched_files:
            scene_file = os.path.basename(scene_path)
            scene_files.append(scene_file)
            
            # Construct corresponding constraint file name
            base_name = scene_file.replace('-scene.txt', '')
            constraint_file = f"{base_name}-constraints.txt"
            if os.path.exists(os.path.join(args.data_path, constraint_file)):
                constraint_files.append(constraint_file)
            else:
                print(f"Warning: No constraint file found for {scene_file}")
                constraint_files.append(None)
    else:
        # Default scene files if none specified
        scene_files = ["cube_6-scene.txt", "stress-scene.txt"]
        constraint_files = ["cube_6-constraints.txt", "stress-constraints.txt"]
    
    # Run all benchmarks
    for rep in range(args.repetitions):
        for i, (scene_file, constraint_file) in enumerate(zip(scene_files, constraint_files)):
            if constraint_file is None:
                continue
                
            for collision_method, constraint_method in product(args.collision_methods, args.constraint_methods):
                print(f"\nRunning benchmark {rep+1}/{args.repetitions}: "
                      f"Scene {i+1}/{len(scene_files)}, "
                      f"Collision method {collision_method}, "
                      f"Constraint method {constraint_method}")
                
                run_benchmark(
                    args.executable,
                    scene_file,
                    constraint_file,
                    args.num_frames,
                    collision_method,
                    constraint_method,
                    args.output,
                    args.data_path
                )
                
                # Short delay between benchmarks to let system settle
                time.sleep(1)

if __name__ == "__main__":
    main()