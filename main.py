import numpy as np
import cv2
import open3d as o3d
from pathlib import Path
import os
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm
import json
import trimesh
import shutil

def process_mesh(mesh_path):
    """
    Process a mesh file to calculate its volume.
    
    Args:
        mesh_path: Path to the mesh file
        
    Returns:
        Dictionary containing volume in ml and file path, or None if invalid
    """
    if not mesh_path.endswith('.obj'):
        return None
    
    mesh = trimesh.load(mesh_path, force='mesh', process=False, fast_load=True)
    
    if mesh.is_empty or mesh.volume == 0:
        raise ValueError("Empty mesh or zero volume.")
    
    volume_ml = mesh.volume * 1e-3  # Convert from mm^3 to ml
    return {
        'Volume(ml)': round(volume_ml, 2),
        'Path': mesh_path
    }


def process_instance(mesh_path, scale_factor, dst):
    """
    Process a single instance to create a 3D reconstruction with proper scaling.
    
    Args:
        mesh_path: Path to the mesh file
        scale_factor: Scale factor to apply to the mesh
        dst: Destination path for the processed mesh
        
    Returns:
        Volume in ml after processing
        
    Note:
        Coordinate system:
        ^
        |y
        |
        |
        |------>x
       /
      /
     /
    v z
    """
    mesh = trimesh.load(mesh_path, force='mesh', process=False, fast_load=True)
    
    if mesh.is_empty or mesh.volume == 0:
        raise ValueError("Empty mesh or zero volume.")
    
    # Scale mesh
    scale_value = 1e3 * scale_factor
    mesh.apply_scale(scale_value)
    mesh.export(dst)
    
    result = process_mesh(dst.as_posix())
    if result is None:
        raise ValueError("Failed to process mesh after scaling")
    
    return result['Volume(ml)']


def get_scale_factor(real_size_anno_path):
    """
    Calculate scale factor based on real size annotations.
    
    Args:
        real_size_anno_path: Path to the annotation JSON file
        
    Returns:
        Calculated scale factor
    """
    gt_size = {
        "knife": 18.8,      # cm
        "fork": 18.3,       # cm
        "black_plate": 26.67, # cm
        "white_plate": 22.86, # cm
    }
    
    with open(real_size_anno_path) as f:
        anno = json.load(f)
    
    # Validate that at least one annotation is not -1
    if all(value == -1 for value in anno.values()):
        raise ValueError("All annotation values are -1")
    
    # Calculate scale factor using least squares
    s1, s2 = 0, 0
    for key, value in anno.items():
        if value == -1:
            continue
        s1 += gt_size[key] / 100 * value
        s2 += value ** 2
    
    return s1 / s2


def extract_scale_from_transform(transform):
    """
    Extract scale factors from a transformation matrix.
    
    Args:
        transform: 4x4 transformation matrix
        
    Returns:
        Tuple of (scale_x, scale_y, scale_z)
    """
    # Extract the upper-left 3x3 part of the matrix
    linear_transform = transform[:3, :3]
    
    # Calculate the scale factors
    scale_x = np.linalg.norm(linear_transform[:, 0])
    scale_y = np.linalg.norm(linear_transform[:, 1])
    scale_z = np.linalg.norm(linear_transform[:, 2])
    
    return scale_x, scale_y, scale_z


def create_submission_directory():
    """Create and return the submission directory."""
    submission_dir = Path('submit_model')
    shutil.rmtree(submission_dir, ignore_errors=True)
    submission_dir.mkdir(parents=True, exist_ok=True)
    return submission_dir


def calculate_world_scales(df):
    """
    Calculate world scale factors for each combo index.
    
    Args:
        df: DataFrame containing the metadata
        
    Returns:
        Dictionary mapping combo index to world scale factor
    """
    world_scale_dict = {}
    df_grouped = df.groupby('Combo Index')
    
    for combo_index, df_combo in tqdm(df_grouped, total=len(df_grouped), desc="Calculating world scales"):
        root = Path(df_combo.iloc[0]['root'])
        real_size_anno = root / 'ref_anno.json'
        world_scale = get_scale_factor(real_size_anno)
        world_scale_dict[combo_index] = world_scale
        print(f"{'='*50} Combo {combo_index}: {world_scale}")
    
    return world_scale_dict


def process_all_instances(df, world_scale_dict, submission_dir):
    """
    Process all instances and calculate volumes.
    
    Args:
        df: DataFrame containing the metadata
        world_scale_dict: Dictionary of world scale factors
        submission_dir: Directory to save processed meshes
        
    Returns:
        List of tuples containing (uid, volume_ml)
    """
    results = []
    
    for index, row in tqdm(df.iterrows(), total=len(df), desc="Processing instances"):
        combo_index = row['Combo Index']
        instance_id = row['instance_id']
        image_path = Path(row['image'])
        root = image_path.parent
        mesh_path = root / f"part_mesh/obj_{instance_id}_color_mesh.obj"
        
        print(f"Processing: {mesh_path}")
        
        # Get scale factors
        world_scale = world_scale_dict[combo_index]
        
        # Load alignment matrix
        align_matrix_json = json.load(open(root / 'align.json'))
        align_matrix = np.array(align_matrix_json[str(instance_id)])
        local_scale = extract_scale_from_transform(align_matrix)[0]
        
        print(f"{'='*50} World scale: {world_scale}, Local scale: {local_scale}")
        
        # Process instance
        uid = row['Food Index']
        dst = submission_dir / f'{uid}.obj'
        volume_ml = process_instance(mesh_path, world_scale * local_scale, dst)
        
        print(f"{'='*50} {row['Food Name']}: {volume_ml} ml")
        
        results.append((uid, volume_ml))
    
    return results


def main():
    """Main function to process all food reconstruction data."""
    # Create submission directory
    submission_dir = create_submission_directory()
    
    # Load metadata
    df = pd.read_csv('meta_new.csv')
    
    # Calculate world scales for each combo
    world_scale_dict = calculate_world_scales(df)
    
    # Process all instances
    results = process_all_instances(df, world_scale_dict, submission_dir)
    
    # Save results
    df_results = pd.DataFrame(results, columns=['id', 'predicted'])
    df_results.to_csv('submit.csv', index=False)
    print(f"Results saved to submit.csv with {len(results)} entries")


if __name__ == "__main__":
    main()