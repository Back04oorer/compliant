std_msgs/Header header

# Motor names (corresponds to indices in arrays below)
string[] names

# Current state values
float64[] position      # Current position of each motor
float64[] velocity      # Current velocity of each motor
float64[] effort        # Current effort/torque of each motor

# Command values
float64[] cmd_position  # Commanded position of each motor
float64[] cmd_velocity  # Commanded velocity of each motor
float64[] cmd_effort    # Commanded effort/torque of each motor

# Control parameters
float64[] kp            # Proportional gain for each motor
float64[] kd            # Derivative gain for each motor

