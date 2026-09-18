#!/usr/bin/env python3
"""
ROS Node for MLP-based Reachability-Aware Point Cloud Masking
==============================================================

This ROS node loads a trained MLP implicit reachability model and uses it to:
  1. Subscribe to a point cloud in an arbitrary sensor frame
  2. Transform the point cloud to a specified robot base frame
  3. Query the MLP model in batch to get reachability and manipulability scores
  4. Mask/annotate the point cloud based on reachability thresholds
  5. Publish the masked cloud for visualization in RViz

Configuration via ROS parameters:
  ~model_path: Path to trained PyTorch model (model.pth)
  ~config_path: Path to model config (config.json)
  ~input_cloud_topic: Input point cloud topic (e.g., '/env_cloud')
  ~output_cloud_topic: Output masked point cloud topic (e.g., '/reachable_cloud')
  ~target_frame: Base frame for transformations (e.g., 'base_link')
  ~p_reach_threshold: Reachability probability threshold [0, 1] for masking
  ~min_manip: Minimum manipulability score threshold for filtering
  ~downsample_rate: Keep every N-th point (1 = keep all, 10 = keep every 10th)
  ~use_soft_masking: If true, color points by reachability instead of hard masking
  ~device: PyTorch device ('cpu' or 'cuda')

RViz visualization:
  - Subscribe to ~output_cloud_topic with a PointCloud2 display
  - For hard masking: only reachable points are published
  - For soft masking: all points are published with intensity/rgb encoding reachability

Usage:
  rosrun piper_reachable_region reachability_mask_node.py

Or in a launch file:
  <node pkg="piper_reachable_region" type="reachability_mask_node.py" name="reachability_mask">
    <param name="model_path" value="$(find piper_reachable_region)/data/model.pth" />
    <param name="config_path" value="$(find piper_reachable_region)/data/config.json" />
    <param name="input_cloud_topic" value="/depth_camera/points" />
    <param name="output_cloud_topic" value="/reachable_cloud" />
    <param name="target_frame" value="base_link" />
    <param name="p_reach_threshold" value="0.5" />
    <param name="min_manip" value="0.0" />
    <param name="use_soft_masking" value="false" />
  </node>

Author: Teleop Team
"""

import json
import sys
import traceback
from pathlib import Path

import numpy as np
import rospy
import torch

import tf2_ros
from tf2_sensor_msgs import tf2_sensor_msgs

from sensor_msgs.msg import PointCloud2, PointField
from sensor_msgs import point_cloud2
from std_msgs.msg import Header


class ReachabilityMLP(torch.nn.Module):
    """
    Lightweight MLP for reachability prediction.
    Must match the architecture used in training.
    """
    
    def __init__(self, input_dim=3, hidden_dim=128, num_layers=4, output_manip=True):
        super(ReachabilityMLP, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.output_manip = output_manip
        
        # Shared backbone
        layers = []
        layers.append(torch.nn.Linear(input_dim, hidden_dim))
        layers.append(torch.nn.ReLU())
        for _ in range(num_layers - 1):
            layers.append(torch.nn.Linear(hidden_dim, hidden_dim))
            layers.append(torch.nn.ReLU())
        self.backbone = torch.nn.Sequential(*layers)
        
        # Reachability head
        self.reach_head = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim, hidden_dim // 2),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim // 2, 1),
            torch.nn.Sigmoid()
        )
        
        # Manipulability head
        if self.output_manip:
            self.manip_head = torch.nn.Sequential(
                torch.nn.Linear(hidden_dim, hidden_dim // 2),
                torch.nn.ReLU(),
                torch.nn.Linear(hidden_dim // 2, 1),
                torch.nn.Softplus()
            )
    
    def forward(self, xyz):
        x = self.backbone(xyz)
        p_reach = self.reach_head(x)
        m_pred = self.manip_head(x) if self.output_manip else None
        return p_reach, m_pred


class ReachabilityMaskNode:
    """
    ROS node for reachability-aware point cloud masking.
    """
    
    def __init__(self):
        """
        Initialize the node: load model, set up TF, create subscribers/publishers.
        """
        rospy.loginfo("Initializing ReachabilityMaskNode...")
        
        # Get parameters
        self.model_path = rospy.get_param('~model_path', 'model.pth')
        self.config_path = rospy.get_param('~config_path', 'config.json')
        self.input_cloud_topic = rospy.get_param('~input_cloud_topic', '/env_cloud')
        self.output_cloud_topic = rospy.get_param('~output_cloud_topic', '/reachable_cloud')
        self.target_frame = rospy.get_param('~target_frame', 'base_link')
        self.p_reach_threshold = rospy.get_param('~p_reach_threshold', 0.5)
        self.min_manip = rospy.get_param('~min_manip', 0.0)
        self.downsample_rate = rospy.get_param('~downsample_rate', 1)
        self.use_soft_masking = rospy.get_param('~use_soft_masking', False)
        device_str = rospy.get_param('~device', 'cpu')
        
        rospy.loginfo(f"Model path: {self.model_path}")
        rospy.loginfo(f"Config path: {self.config_path}")
        rospy.loginfo(f"Input topic: {self.input_cloud_topic}")
        rospy.loginfo(f"Output topic: {self.output_cloud_topic}")
        rospy.loginfo(f"Target frame: {self.target_frame}")
        rospy.loginfo(f"P_reach threshold: {self.p_reach_threshold}")
        rospy.loginfo(f"Min manipulability: {self.min_manip}")
        rospy.loginfo(f"Downsample rate: {self.downsample_rate}")
        rospy.loginfo(f"Use soft masking: {self.use_soft_masking}")
        
        # Setup device
        if device_str == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device_str)
        rospy.loginfo(f"Using device: {self.device}")
        
        # Load config
        try:
            with open(self.config_path, 'r') as f:
                self.config = json.load(f)
            rospy.loginfo("Config loaded successfully")
        except Exception as e:
            rospy.logerr(f"Failed to load config: {e}")
            raise
        
        # Extract normalization parameters
        self.xyz_min = np.array(self.config['normalization']['xyz_min'])
        self.xyz_max = np.array(self.config['normalization']['xyz_max'])
        self.manip_max = self.config['normalization']['manip_max']
        self.manip_min = self.config['normalization']['manip_min']
        
        # Load model
        try:
            model_config = self.config['model_architecture']
            self.model = ReachabilityMLP(
                input_dim=model_config['input_dim'],
                hidden_dim=model_config['hidden_dim'],
                num_layers=model_config['num_layers'],
                output_manip=model_config['output_manip'],
            ).to(self.device)
            self.model.load_state_dict(torch.load(self.model_path, map_location=self.device))
            self.model.eval()
            rospy.loginfo("Model loaded successfully")
        except Exception as e:
            rospy.logerr(f"Failed to load model: {e}")
            raise
        
        # Setup TF listener
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)
        rospy.sleep(0.5)  # Wait for TF to be populated
        
        # Setup subscribers and publishers
        self.cloud_sub = rospy.Subscriber(self.input_cloud_topic, PointCloud2, self.cloud_callback, queue_size=1)
        self.cloud_pub = rospy.Publisher(self.output_cloud_topic, PointCloud2, queue_size=1)
        
        rospy.loginfo("ReachabilityMaskNode initialized successfully")
    
    def normalize_xyz(self, xyz):
        """
        Normalize 3D positions to [-1, 1]^3 using stored bounding box.
        
        Args:
            xyz: (N, 3) array of 3D positions in base frame
            
        Returns:
            xyz_normalized: (N, 3) array normalized to [-1, 1]^3
        """
        xyz_center = (self.xyz_min + self.xyz_max) / 2
        xyz_half_size = (self.xyz_max - self.xyz_min) / 2
        xyz_norm = (xyz - xyz_center) / (xyz_half_size + 1e-8)
        return np.clip(xyz_norm, -1.0, 1.0)  # Clip to [-1, 1]
    
    def denormalize_manip(self, m_norm):
        """
        Denormalize manipulability from normalized space to original scale.
        
        Args:
            m_norm: (N,) normalized manipulability scores
            
        Returns:
            m: (N,) denormalized manipulability
        """
        # Simple linear denormalization: scale back to original range
        manip_range = self.manip_max - self.manip_min
        m = self.manip_min + m_norm * manip_range
        return m
    
    def pointcloud2_to_numpy(self, cloud_msg):
        """
        Convert ROS PointCloud2 message to NumPy array.
        
        Args:
            cloud_msg: sensor_msgs/PointCloud2
            
        Returns:
            points: (N, 3) array of xyz coordinates
            rgb: (N,) array of RGB values if available, else None
            intensity: (N,) array of intensity if available, else None
        """
        # Extract xyz
        points_list = list(point_cloud2.read_points(cloud_msg, field_names=('x', 'y', 'z'), skip_nans=True))
        if len(points_list) == 0:
            rospy.logwarn("Empty point cloud received")
            return None, None, None
        
        points = np.array(points_list, dtype=np.float32)
        
        # Try to extract rgb or intensity
        rgb = None
        intensity = None
        
        try:
            rgb_list = list(point_cloud2.read_points(cloud_msg, field_names=('rgb',), skip_nans=False))
            if rgb_list and rgb_list[0][0] is not None:
                rgb = np.array([x[0] for x in rgb_list], dtype=np.float32)
        except Exception:
            pass
        
        try:
            intensity_list = list(point_cloud2.read_points(cloud_msg, field_names=('intensity',), skip_nans=False))
            if intensity_list and intensity_list[0][0] is not None:
                intensity = np.array([x[0] for x in intensity_list], dtype=np.float32)
        except Exception:
            pass
        
        return points, rgb, intensity
    
    def numpy_to_pointcloud2(self, points, frame_id, p_reach=None, m_pred=None, rgb=None):
        """
        Convert NumPy array to ROS PointCloud2 message.
        
        Args:
            points: (N, 3) array of xyz
            frame_id: frame ID string
            p_reach: (N,) optional reachability probabilities
            m_pred: (N,) optional manipulability scores
            rgb: (N,) optional RGB values
            
        Returns:
            PointCloud2 message
        """
        if len(points) == 0:
            cloud_msg = PointCloud2()
            cloud_msg.header.frame_id = frame_id
            cloud_msg.height = 0
            cloud_msg.width = 0
            return cloud_msg
        
        # Prepare fields
        fields = [
            PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
        ]
        
        # Package data
        data_list = []
        for i in range(len(points)):
            x, y, z = points[i, 0], points[i, 1], points[i, 2]
            data = struct.pack('fff', x, y, z)
            
            # Add intensity if available
            if p_reach is not None:
                # Encode reachability as intensity [0, 1]
                intensity_val = p_reach[i]
                data += struct.pack('f', intensity_val)
            
            if m_pred is not None and 'intensity' not in [f.name for f in fields]:
                manip_val = m_pred[i]
                data += struct.pack('f', manip_val)
            
            data_list.append(data)
        
        # Create the cloud message
        if p_reach is not None:
            # Create cloud with intensity field for reachability
            header = Header()
            header.stamp = rospy.get_rostime()
            header.frame_id = frame_id

            fields = [
                PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
                PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
                PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
                PointField(name='intensity', offset=12, datatype=PointField.FLOAT32, count=1),
            ]

            cloud_msg = point_cloud2.create_cloud(
                header,
                fields,
                [(points[i, 0], points[i, 1], points[i, 2], p_reach[i]) for i in range(len(points))]
            )
        else:
            cloud_msg = point_cloud2.create_cloud_xyz32(rospy.get_rostime(), frame_id, points)

        return cloud_msg
    
    def cloud_callback(self, cloud_msg):
        """
        Callback for incoming point clouds.
        
        Process:
          1. Transform cloud to target frame
          2. Extract xyz and convert to NumPy
          3. Normalize xyz
          4. Query MLP in batch
          5. Apply masking
          6. Publish result
        """
        try:
            # Get transform
            try:
                transform = self.tf_buffer.lookup_transform(
                    self.target_frame,
                    cloud_msg.header.frame_id,
                    rospy.Time(0),
                    timeout=rospy.Duration(1.0)
                )
            except tf2_ros.TransformException as e:
                rospy.logwarn(f"TF lookup failed: {e}")
                return
            
            # Transform cloud
            try:
                cloud_transformed = tf2_sensor_msgs.do_transform_cloud(cloud_msg, transform)
            except Exception as e:
                rospy.logwarn(f"Cloud transformation failed: {e}")
                return
            
            # Extract points
            points, rgb, intensity = self.pointcloud2_to_numpy(cloud_transformed)
            if points is None:
                return
            
            # Downsample if requested
            if self.downsample_rate > 1:
                points = points[::self.downsample_rate]
                if rgb is not None:
                    rgb = rgb[::self.downsample_rate]
                if intensity is not None:
                    intensity = intensity[::self.downsample_rate]
            
            # Normalize xyz
            xyz_norm = self.normalize_xyz(points)
            
            # Batch inference
            with torch.no_grad():
                xyz_tensor = torch.from_numpy(xyz_norm).float().to(self.device)
                p_reach, m_pred = self.model(xyz_tensor)
                p_reach = p_reach.squeeze().cpu().numpy()
                if m_pred is not None:
                    m_pred = m_pred.squeeze().cpu().numpy()
                    # Denormalize manipulability
                    m_pred = self.denormalize_manip(m_pred)
            
            # Apply masking
            if self.use_soft_masking:
                # Soft masking: keep all points, color by reachability
                masked_points = points
                masked_p_reach = p_reach
                masked_m_pred = m_pred
            else:
                # Hard masking: keep only reachable points
                mask = p_reach >= self.p_reach_threshold
                if self.min_manip > 0 and m_pred is not None:
                    mask = mask & (m_pred >= self.min_manip)
                
                masked_points = points[mask]
                masked_p_reach = p_reach[mask] if self.use_soft_masking or len(masked_points) > 0 else np.array([])
                masked_m_pred = m_pred[mask] if m_pred is not None and len(masked_points) > 0 else None

            # print(f"masked_points: {masked_points.shape}")
            # print(f"raw points: {points.shape}")
            
            # Publish result
            if len(masked_points) > 0:
                output_cloud = self.numpy_to_pointcloud2(
                    masked_points,
                    cloud_transformed.header.frame_id,
                    p_reach=masked_p_reach,
                    m_pred=masked_m_pred
                )
                self.cloud_pub.publish(output_cloud)
                rospy.logdebug(f"Published {len(masked_points)} points (reachable: {np.sum(p_reach >= self.p_reach_threshold)} / {len(points)})")
            else:
                rospy.logwarn("No points passed reachability threshold; publishing empty cloud")
                empty_cloud = PointCloud2()
                empty_cloud.header = cloud_transformed.header
                self.cloud_pub.publish(empty_cloud)
        
        except Exception as e:
            rospy.logerr(f"Error in cloud callback: {e}")
            traceback.print_exc()


def main():
    """
    Main entry point for the ROS node.
    """
    rospy.init_node('reachability_mask_node')
    
    try:
        node = ReachabilityMaskNode()
        rospy.loginfo("Spinning...")
        rospy.spin()
    except KeyboardInterrupt:
        rospy.loginfo("Shutting down")
    except Exception as e:
        rospy.logerr(f"Fatal error: {e}")
        traceback.print_exc()


if __name__ == '__main__':
    import struct
    main()

