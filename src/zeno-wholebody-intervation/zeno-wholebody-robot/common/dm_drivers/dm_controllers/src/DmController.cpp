#include "dm_controllers/DmController.h"
#include <ros/console.h>

namespace damiao
{

	// Helper function to load axis parameters
	void DmController::loadAxisParams(ros::NodeHandle &nh, const std::string &axis_name, AxisParams &params)
	{
		ros::NodeHandle axis_nh(nh, axis_name);
		
		axis_nh.param("motor_name", params.motor_name, std::string(""));
		axis_nh.param("kp_default", params.kp_default, 5.0);
		axis_nh.param("kd_default", params.kd_default, 0.1);
		
		// Check if haptic parameters exist
		if (axis_nh.hasParam("repulsive_force_threshold"))
		{
			params.has_haptic = true;
			axis_nh.param("repulsive_force_threshold", params.repulsive_force_threshold, 5.0);
			axis_nh.param("pos_haptic_positive", params.pos_haptic_positive, 0.15);
			axis_nh.param("pos_haptic_negative", params.pos_haptic_negative, -0.15);
			axis_nh.param("kd_haptic", params.kd_haptic, 2.0);
		}
		else
		{
			params.has_haptic = false;
			params.repulsive_force_threshold = 0.0;
			params.pos_haptic_positive = 0.0;
			params.pos_haptic_negative = 0.0;
			params.kd_haptic = 0.0;
		}
	}

	bool DmController::init(HybridJointInterface *robot_hw, ros::NodeHandle &nh)
	{
		std::cerr << "Successfully got HybridEffort joint interface" << std::endl;
		std::cerr << "[DmController::init] NodeHandle namespace: " << nh.getNamespace() << std::endl;
		std::cerr << "[DmController::init] NodeHandle resolved name: " << nh.resolveName("") << std::endl;

		// Load axis parameters from hierarchical structure
		loadAxisParams(nh, "x", x_params_);
		loadAxisParams(nh, "y", y_params_);
		loadAxisParams(nh, "z", z_params_);

		// Collect all motor names and create handles
		std::vector<std::string> joint_names;
		if (!x_params_.motor_name.empty())
			joint_names.push_back(x_params_.motor_name);
		if (!y_params_.motor_name.empty())
			joint_names.push_back(y_params_.motor_name);
		if (!z_params_.motor_name.empty())
			joint_names.push_back(z_params_.motor_name);

		// Create handles and build index map
		for (size_t i = 0; i < joint_names.size(); ++i)
		{
			hybridJointHandles_.push_back(robot_hw->getHandle(joint_names[i]));
			motor_index_map_[joint_names[i]] = i;
		}

		// Initialize repulsive force values
		repulsive_force_x_ = 0.0;
		repulsive_force_y_ = 0.0;

		// Subscribe to repulsive force vector topic
		nh_ = nh;
		repulsive_force_sub_ = nh_.subscribe("/repulsive_force_vector", 1, &DmController::repulsiveForceCallback, this);

		std::cerr << "[DmController] Parameters loaded:" << std::endl;
		std::cerr << "  x axis - motor: " << x_params_.motor_name 
		          << ", kp_default: " << x_params_.kp_default 
		          << ", kd_default: " << x_params_.kd_default;
		if (x_params_.has_haptic)
		{
			std::cerr << ", repulsive_force_threshold: " << x_params_.repulsive_force_threshold
			          << ", pos_haptic_positive: " << x_params_.pos_haptic_positive
			          << ", pos_haptic_negative: " << x_params_.pos_haptic_negative
			          << ", kd_haptic: " << x_params_.kd_haptic;
		}
		std::cerr << std::endl;
		
		std::cerr << "  y axis - motor: " << y_params_.motor_name 
		          << ", kp_default: " << y_params_.kp_default 
		          << ", kd_default: " << y_params_.kd_default;
		if (y_params_.has_haptic)
		{
			std::cerr << ", repulsive_force_threshold: " << y_params_.repulsive_force_threshold
			          << ", pos_haptic_positive: " << y_params_.pos_haptic_positive
			          << ", pos_haptic_negative: " << y_params_.pos_haptic_negative
			          << ", kd_haptic: " << y_params_.kd_haptic;
		}
		std::cerr << std::endl;
		
		std::cerr << "  z axis - motor: " << z_params_.motor_name 
		          << ", kp_default: " << z_params_.kp_default 
		          << ", kd_default: " << z_params_.kd_default;
		if (z_params_.has_haptic)
		{
			std::cerr << ", repulsive_force_threshold: " << z_params_.repulsive_force_threshold
			          << ", pos_haptic_positive: " << z_params_.pos_haptic_positive
			          << ", pos_haptic_negative: " << z_params_.pos_haptic_negative
			          << ", kd_haptic: " << z_params_.kd_haptic;
		}
		std::cerr << std::endl;

		return true;
	}

	void DmController::starting(const ros::Time &time)
	{
		// ROS_INFO("DmController started.");
		std::cerr << "DmController started." << std::endl;
	}

	void DmController::update(const ros::Time &time, const ros::Duration &period)
	{
		// Get repulsive force values (thread-safe)
		double repulsive_x, repulsive_y;
		{
			std::lock_guard<std::mutex> lock(repulsive_force_mutex_);
			repulsive_x = repulsive_force_x_;
			repulsive_y = repulsive_force_y_;
		}

		// Process x axis (joint1_motor)
		if (motor_index_map_.find(x_params_.motor_name) != motor_index_map_.end())
		{
			size_t x_idx = motor_index_map_[x_params_.motor_name];
			double kp_x = x_params_.kp_default + std::abs(repulsive_x);
			double pos_des_x = 0.0;
			double kd_x = x_params_.kd_default;
			
			if (x_params_.has_haptic)
			{
				if (repulsive_x < -x_params_.repulsive_force_threshold)
				{
					pos_des_x = x_params_.pos_haptic_positive;
					kd_x = x_params_.kd_haptic;
				}
				else if (repulsive_x > x_params_.repulsive_force_threshold)
				{
					pos_des_x = x_params_.pos_haptic_negative;
					kd_x = x_params_.kd_haptic;
				}
			}
			
			hybridJointHandles_[x_idx].setCommand(pos_des_x, 0.0, kp_x, kd_x, 0.0);
		}

		// Process y axis (joint2_motor)
		if (motor_index_map_.find(y_params_.motor_name) != motor_index_map_.end())
		{
			size_t y_idx = motor_index_map_[y_params_.motor_name];
			double kp_y = y_params_.kp_default + std::abs(repulsive_y);
			double pos_des_y = 0.0;
			double kd_y = y_params_.kd_default;
			
			if (y_params_.has_haptic)
			{
				if (repulsive_y < -y_params_.repulsive_force_threshold)
				{
					pos_des_y = y_params_.pos_haptic_negative;
					kd_y = y_params_.kd_haptic;
				}
				else if (repulsive_y > y_params_.repulsive_force_threshold)
				{
					pos_des_y = y_params_.pos_haptic_positive;
					kd_y = y_params_.kd_haptic;
				}
			}
			
			hybridJointHandles_[y_idx].setCommand(pos_des_y, 0.0, kp_y, kd_y, 0.0);
		}

		// Process z axis (joint0_motor) - no haptic feedback
		if (motor_index_map_.find(z_params_.motor_name) != motor_index_map_.end())
		{
			size_t z_idx = motor_index_map_[z_params_.motor_name];
			hybridJointHandles_[z_idx].setCommand(0.0, 0.0, z_params_.kp_default, z_params_.kd_default, 0.0);
		}
	}

	void DmController::stopping(const ros::Time &time)
	{
		// ROS_INFO("DmController stopped.");
		std::cerr << "DmController stop." << std::endl;
	}

	void DmController::repulsiveForceCallback(const geometry_msgs::Vector3Stamped::ConstPtr &msg)
	{
		std::lock_guard<std::mutex> lock(repulsive_force_mutex_);
		repulsive_force_x_ = msg->vector.x;
		repulsive_force_y_ = msg->vector.y;
	}

} // namespace damiao

// 注册插件
PLUGINLIB_EXPORT_CLASS(damiao::DmController, controller_interface::ControllerBase);