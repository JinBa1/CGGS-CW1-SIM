#!/usr/bin/env python3
import os
import argparse
import numpy as np
import math
import re

class MultiSceneGenerator:
    def __init__(self, output_dir="./"):
        self.output_dir = output_dir
        
        # Initialize scene data
        self.scene_data = {
            'stress': {
                'objects': [],
                'constraints': []
            },
            'two_chain': {
                'objects': [],
                'constraints': []
            },
            'two_cylinder': {
                'objects': [],
                'constraints': []
            },
            'cube': {
                'objects': [],
                'constraints': []
            }
        }
        
        # Load scene data
        self._load_scene_data()
    
    def _load_scene_data(self):
        """Load all the scene data from the reference files"""
        # Define scene files
        scene_files = {
            'stress': ('stress-scene.txt', 'stress-constraints.txt'),
            'two_chain': ('two_chain-scene.txt', 'two_chain-constraints.txt'),
            'two_cylinder': ('two_cylinder-scene.txt', 'two_cylinder-constraints.txt'),
            'cube': ('cube-scene.txt', 'cube-constraints.txt')
        }
        
        for scene_type, (scene_file, constraint_file) in scene_files.items():
            scene_path = "/home/jin/cggs/CGGS-CW1-Simulation/data/" + scene_file
            constraint_path = "/home/jin/cggs/CGGS-CW1-Simulation/data/" + constraint_file
            # Try to load scene from the current directory
            if os.path.exists(scene_path):
                self._parse_scene_file(scene_type, scene_path)
                if os.path.exists(constraint_path):
                    self._parse_constraint_file(scene_type, constraint_path)
            # Or look in data directory
            elif os.path.exists(os.path.join('data', scene_file)):
                self._parse_scene_file(scene_type, os.path.join('data', scene_file))
                if os.path.exists(os.path.join('data', constraint_file)):
                    self._parse_constraint_file(scene_type, os.path.join('data', constraint_file))
            else:
                print(f"Warning: Could not find scene file {scene_file}")
    
    def _parse_scene_file(self, scene_type, filename):
        """Parse a scene file into internal representation"""
        try:
            with open(filename, 'r') as f:
                lines = f.readlines()
                
                # First line is the number of objects
                num_objects = int(lines[0].strip())
                
                # Parse each object
                for i in range(1, num_objects + 1):
                    if i < len(lines):
                        line = lines[i].strip()
                        parts = re.split(r'\s+', line)
                        
                        # Handle variable whitespace in the file
                        if len(parts) >= 10:  # Ensure we have enough parts
                            mesh_type = parts[0]
                            density = float(parts[1])
                            is_fixed = int(parts[2])
                            position = (float(parts[3]), float(parts[4]), float(parts[5]))
                            orientation = (float(parts[6]), float(parts[7]), float(parts[8]), float(parts[9]))
                            
                            self.scene_data[scene_type]['objects'].append({
                                'type': mesh_type,
                                'density': density,
                                'fixed': is_fixed,
                                'position': position,
                                'orientation': orientation
                            })
            
            print(f"Loaded {len(self.scene_data[scene_type]['objects'])} objects from {filename}")
        except Exception as e:
            print(f"Error loading scene file {filename}: {e}")
    
    def _parse_constraint_file(self, scene_type, filename):
        """Parse a constraint file into internal representation"""
        try:
            with open(filename, 'r') as f:
                lines = f.readlines()
                
                # First line is the number of constraints
                num_constraints = int(lines[0].strip())
                
                # Parse each constraint
                for i in range(1, num_constraints + 1):
                    if i < len(lines):
                        line = lines[i].strip()
                        parts = re.split(r'\s+', line)
                        
                        if len(parts) >= 6:  # Ensure we have enough parts
                            mesh1 = int(parts[0])
                            vertex1 = int(parts[1])
                            mesh2 = int(parts[2])
                            vertex2 = int(parts[3])
                            lower_bound = float(parts[4])
                            upper_bound = float(parts[5])
                            
                            self.scene_data[scene_type]['constraints'].append({
                                'mesh1': mesh1,
                                'vertex1': vertex1,
                                'mesh2': mesh2,
                                'vertex2': vertex2,
                                'lower_bound': lower_bound,
                                'upper_bound': upper_bound
                            })
            
            print(f"Loaded {len(self.scene_data[scene_type]['constraints'])} constraints from {filename}")
        except Exception as e:
            print(f"Error loading constraint file {filename}: {e}")
    
    def _get_scene_bounds(self, scene_type):
        """Get the min and max bounds for a scene"""
        if not self.scene_data[scene_type]['objects']:
            return (0, 0), (0, 0)
        
        x_values = [obj['position'][0] for obj in self.scene_data[scene_type]['objects']]
        z_values = [obj['position'][2] for obj in self.scene_data[scene_type]['objects']]
        
        x_bounds = (min(x_values), max(x_values))
        z_bounds = (min(z_values), max(z_values))
        
        return x_bounds, z_bounds
    
    def generate_scene(self, scene_type, num_copies):
        """Generate a scene with multiple copies of the specified scene layout"""
        if not self.scene_data[scene_type]['objects']:
            print(f"Error: No objects found for scene type '{scene_type}'")
            return []
        
        # Get the original scene bounds
        x_bounds, z_bounds = self._get_scene_bounds(scene_type)
        
        # Calculate the size of the original scene
        scene_width = x_bounds[1] - x_bounds[0]
        scene_depth = z_bounds[1] - z_bounds[0]
        
        # Add some spacing between scenes (20% of scene size)
        spacing_x = max(scene_width * 0.2, 20.0)  # Minimum spacing of 20 units
        spacing_z = max(scene_depth * 0.2, 20.0)
        
        # Total width and depth of a single scene with spacing
        total_width = scene_width + spacing_x
        total_depth = scene_depth + spacing_z
        
        # Determine layout (try to create a roughly square arrangement)
        grid_dim = math.ceil(math.sqrt(num_copies))
        
        # Generate scene copies
        all_objects = []
        base_indices = {}  # Maps (copy, original_index) to new_index
        
        copy_count = 0
        for row in range(grid_dim):
            for col in range(grid_dim):
                if copy_count >= num_copies:
                    break
                
                # Calculate offset for this copy
                offset_x = col * total_width
                offset_z = row * total_depth
                
                # Add objects for this copy
                for i, obj in enumerate(self.scene_data[scene_type]['objects']):
                    # Apply offset to position
                    orig_pos = obj['position']
                    new_pos = (orig_pos[0] + offset_x, orig_pos[1], orig_pos[2] + offset_z)
                    
                    # Create a copy of the object with new position
                    new_obj = obj.copy()
                    new_obj['position'] = new_pos
                    
                    # Add to the scene and track the new index
                    new_index = len(all_objects)
                    all_objects.append(new_obj)
                    base_indices[(copy_count, i)] = new_index
                
                copy_count += 1
        
        return all_objects, base_indices
    
    def generate_constraints(self, scene_type, base_indices, num_copies):
        """Generate constraints based on the original constraints, mapped to the new object indices"""
        all_constraints = []
        
        if not self.scene_data[scene_type]['constraints']:
            print(f"Warning: No constraints found for scene type '{scene_type}'")
            return all_constraints
        
        # For each copy, add the constraints with remapped indices
        for copy in range(num_copies):
            for constraint in self.scene_data[scene_type]['constraints']:
                # Map the original mesh indices to the new indices in this copy
                new_mesh1 = base_indices.get((copy, constraint['mesh1']))
                new_mesh2 = base_indices.get((copy, constraint['mesh2']))
                
                # Only add if both meshes exist in this copy
                if new_mesh1 is not None and new_mesh2 is not None:
                    new_constraint = {
                        'mesh1': new_mesh1,
                        'vertex1': constraint['vertex1'],
                        'mesh2': new_mesh2,
                        'vertex2': constraint['vertex2'],
                        'lower_bound': constraint['lower_bound'],
                        'upper_bound': constraint['upper_bound']
                    }
                    all_constraints.append(new_constraint)
        
        return all_constraints
    
    def write_scene_file(self, objects, filename="generated-scene.txt"):
        """Write the scene to a file"""
        filepath = os.path.join(self.output_dir, filename)
        
        with open(filepath, 'w') as f:
            # Write number of meshes
            f.write(f"{len(objects)}\n")
            
            # Write each mesh
            for obj in objects:
                position = obj['position']
                orientation = obj['orientation']
                
                f.write(f"{obj['type']}\t{obj['density']}\t{obj['fixed']}\t"
                       f"{position[0]} {position[1]} {position[2]}\t"
                       f"{orientation[0]} {orientation[1]} {orientation[2]} {orientation[3]}\n")
        
        print(f"Scene file written to {filepath}")
        return filepath
    
    def write_constraint_file(self, constraints, filename="generated-constraints.txt"):
        """Write the constraints to a file"""
        filepath = os.path.join(self.output_dir, filename)
        
        with open(filepath, 'w') as f:
            # Write number of constraints
            f.write(f"{len(constraints)}\n")
            
            # Write each constraint
            for c in constraints:
                f.write(f"{c['mesh1']} {c['vertex1']} {c['mesh2']} {c['vertex2']} "
                       f"{c['lower_bound']} {c['upper_bound']}\n")
        
        print(f"Constraint file written to {filepath}")
        return filepath


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate test scenes for rigid body simulation")
    
    parser.add_argument("--scene-type", choices=["stress", "two_chain", "two_cylinder", "cube"], 
                        default="stress", help="Type of scene to duplicate")
    parser.add_argument("--copies", type=int, default=4, help="Number of scene copies to generate")
    parser.add_argument("--output-dir", default="/home/jin/cggs/CGGS-CW1-Simulation/data/", help="Output directory for scene files")
    parser.add_argument("--scene-file", default=None, help="Output scene filename")
    parser.add_argument("--constraint-file", default=None, help="Output constraint filename")
    
    args = parser.parse_args()
    
    # Set default filenames based on scene type if not specified
    if args.scene_file is None:
        args.scene_file = f"{args.scene_type}_{args.copies}-scene.txt"
    if args.constraint_file is None:
        args.constraint_file = f"{args.scene_type}_{args.copies}-constraints.txt"
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Initialize generator
    generator = MultiSceneGenerator(args.output_dir)
    
    # Generate scene
    objects, base_indices = generator.generate_scene(args.scene_type, args.copies)
    
    # Generate constraints
    constraints = generator.generate_constraints(args.scene_type, base_indices, args.copies)
    
    # Write files
    scene_path = generator.write_scene_file(objects, args.scene_file)
    constraint_path = generator.write_constraint_file(constraints, args.constraint_file)
    
    print(f"Generated {len(objects)} objects and {len(constraints)} constraints")
    print(f"Scene file: {scene_path}")
    print(f"Constraint file: {constraint_path}")