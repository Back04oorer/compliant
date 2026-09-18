#ifndef MY_CONTROLLER_PLUGIN_DM_CONTROLLER_H
#define MY_CONTROLLER_PLUGIN_DM_CONTROLLER_H

#include <controller_interface/controller.h>
#include <hardware_interface/joint_command_interface.h>
#include <pluginlib/class_list_macros.hpp>
#include <ros/ros.h>
#include <std_msgs/Float64.h>
#include <geometry_msgs/Vector3Stamped.h>
#include <dm_common/HybridJointInterface.h>
#include <mutex>
#include <map>
#include <string>
#include <vector>

namespace damiao
{


class DmController : public controller_interface::Controller<HybridJointInterface>
{
public:
  DmController() = default;
  ~DmController() = default;

  bool init(HybridJointInterface* robot_hw, ros::NodeHandle& nh) override;
  void starting(const ros::Time& time) override;
  void update(const ros::Time& time, const ros::Duration& period) override;
  void stopping(const ros::Time& time) override;

private:
  std::vector<HybridJointHandle> hybridJointHandles_;
  ros::Subscriber repulsive_force_sub_;
  ros::NodeHandle nh_;
  
  // Thread-safe storage for repulsive force values
  std::mutex repulsive_force_mutex_;
  double repulsive_force_x_;
  double repulsive_force_y_;
  
  // Structure to hold axis-specific parameters
  struct AxisParams {
    std::string motor_name;
    double kp_default;
    double kd_default;
    double repulsive_force_threshold;
    double pos_haptic_positive;
    double pos_haptic_negative;
    double kd_haptic;
    bool has_haptic;  // Whether haptic parameters are available
  };
  
  // Control parameters (loaded from ROS parameters)
  AxisParams x_params_;  // Parameters for x axis (joint1_motor)
  AxisParams y_params_;  // Parameters for y axis (joint2_motor)
  AxisParams z_params_;  // Parameters for z axis (joint0_motor)
  
  // Mapping from motor name to index in hybridJointHandles_
  std::map<std::string, size_t> motor_index_map_;
  
  void repulsiveForceCallback(const geometry_msgs::Vector3Stamped::ConstPtr& msg);
  
  // Helper function to load axis parameters
  void loadAxisParams(ros::NodeHandle& nh, const std::string& axis_name, AxisParams& params);

};

}  // namespace my_controller_plugin

#endif  // MY_CONTROLLER_PLUGIN_DM_CONTROLLER_H