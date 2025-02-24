#!/usr/bin/env python

print("importing libraries")
import cv2
import numpy as np
import yaml
import os
import argparse
import sys
import matplotlib.pyplot as plt
from pathlib import Path

# Import Kalibr libraries
import aslam_cv as acv
import aslam_cv_backend as acvb
import aslam_cameras_april as acv_april
import kalibr_common as kc
import sm

def parse_arguments():
    parser = argparse.ArgumentParser(description='Camera calibration using AprilGrid')
    parser.add_argument('--models', 
                       default=['pinhole-radtan'],
                       nargs='+',
                       help='Camera model (e.g., pinhole-radtan, omni-radtan)')
    parser.add_argument('--target', 
                       default='aprilgrid.yaml',
                       help='Path to target configuration YAML')
    parser.add_argument('--image-directory',
                       default='/home/venti/Dataset/IMU_CAM_Calib6/calib_picture5',
                       help='Directory containing calibration images')
    return parser.parse_args()

def create_target_config():
    """Create default AprilGrid configuration"""
    config = {
        'target_type': 'aprilgrid',
        'tagCols': 6,
        'tagRows': 6,
        'tagSize': 0.02,
        'tagSpacing': 0.3,
        'codeOffset': 0
    }
    
    # Save config if it doesn't exist
    if not os.path.exists('aprilgrid.yaml'):
        with open('aprilgrid.yaml', 'w') as f:
            yaml.dump(config, f)
            
    return config

class CalibrationVisualizer:
    def __init__(self):
        plt.ion()
        self.fig = plt.figure(figsize=(15, 5))
        self.ax1 = self.fig.add_subplot(131)
        self.ax2 = self.fig.add_subplot(132)
        self.ax3 = self.fig.add_subplot(133)
        plt.tight_layout()
        
        self.reprojection_errors = []
        self.focal_lengths = []
        
    def update_plots(self, image, corners, intrinsics, error=None):
        # Detection visualization
        self.ax1.clear()
        self.ax1.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        if corners is not None:
            self.ax1.plot(corners[:, 0], corners[:, 1], 'r+')
        self.ax1.set_title('AprilGrid Detection')
        
        # Intrinsics evolution
        if intrinsics is not None:
            self.focal_lengths.append(intrinsics[0])
            self.ax2.clear()
            self.ax2.plot(self.focal_lengths)
            self.ax2.set_title('Focal Length Evolution')
            
        # Reprojection error
        if error is not None:
            self.reprojection_errors.append(error)
            self.ax3.clear()
            self.ax3.plot(self.reprojection_errors)
            self.ax3.set_title('Reprojection Error')
            
        plt.pause(0.01)

def main():
    # Parse arguments
    args = parse_arguments()
    
    # Create default target config if needed
    target_config = create_target_config()
    
    # Initialize visualization
    visualizer = CalibrationVisualizer()
    
    # Available camera models
    camera_models = {
        'pinhole-radtan': acvb.DistortedPinhole,
        'pinhole-equi': acvb.EquidistantPinhole,
        'pinhole-fov': acvb.FovPinhole,
        'omni-none': acvb.Omni,
        'omni-radtan': acvb.DistortedOmni,
        'eucm-none': acvb.ExtendedUnified,
        'ds-none': acvb.DoubleSphere
    }
    
    print("Initializing calibration...")
    
    # Create target parameters
    target_params = kc.CalibrationTargetParameters(args.target)
    
    # Initialize cameras
    cameras = []
    for model_name in args.models:
        if model_name not in camera_models:
            raise ValueError(f"Unknown camera model: {model_name}")
            
        print(f"Initializing camera with model: {model_name}")
        camera_model = camera_models[model_name]
        
        # Create camera geometry
        camera = kc.CameraGeometry(camera_model, target_params)
        cameras.append(camera)
    
    # Process images
    image_files = list(Path(args.image_directory).glob('*.png'))
    if not image_files:
        raise ValueError(f"No PNG images found in {args.image_directory}")
        
    print(f"Found {len(image_files)} images")
    
    # Process each image
    for img_file in image_files:
        print(f"\nProcessing {img_file.name}")
        
        # Load image
        image = cv2.imread(str(img_file))
        if image is None:
            print(f"Failed to load {img_file}")
            continue
            
        # Extract corners for each camera
        for cam_idx, camera in enumerate(cameras):
            # Extract grid corners
            observations = kc.extractCornersFromImage(
                image, 
                camera.target.detector,
                noTransformation=True
            )
            
            if observations:
                corners = observations[0].getCornersImageFrame()
                # Initialize geometry if not done
                if not camera.geometry.isInitialized():
                    camera.initGeometryFromObservations([observations[0]])
                    print(f"Initialized camera {cam_idx} geometry")
                
                # Get current parameters
                proj_params = camera.geometry.projection().getParameters().flatten()
                
                # Calculate reprojection error
                error = observations[0].computeResidualsAndJacobians()[0]
                mean_error = np.mean(np.abs(error))
                
                # Update visualization
                visualizer.update_plots(image, corners, proj_params, mean_error)
                
                print(f"Camera {cam_idx} reprojection error: {mean_error:.6f}")
                print(f"Current parameters: {proj_params}")
            else:
                print(f"No corners detected for camera {cam_idx}")
    
    # Save results
    print("\nSaving calibration results...")
    base_path = Path(args.image_directory).parent
    for cam_idx, camera in enumerate(cameras):
        result_file = base_path / f"camera_{cam_idx}_calibration.yaml"
        camera.saveParameters(str(result_file))
        print(f"Camera {cam_idx} parameters saved to: {result_file}")
    
    print("\nCalibration completed!")
    plt.ioff()
    plt.show()

if __name__ == "__main__":
    main()