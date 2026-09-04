// Copyright 2026 EngineAI
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <yaml-cpp/yaml.h>

#include <cmath>
#include <cstring>

#include <algorithm>
#include <chrono>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include <MNN/Interpreter.hpp>
#include <MNN/Tensor.hpp>

#include <geometry_msgs/msg/twist.hpp>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/imu.hpp>
#include <sensor_msgs/msg/joint_state.hpp>
#include <std_msgs/msg/float64_multi_array.hpp>
#include <std_srvs/srv/empty.hpp>

namespace
{

constexpr std::size_t kJointCount = 24;
constexpr std::size_t kActionCount = 22;
constexpr std::size_t kObservationCount = 72;
constexpr std::size_t kHistorySteps = 15;
constexpr std::size_t kPolicyInputCount =
  kObservationCount * kHistorySteps + 3;

const std::vector<std::string> kAllJoints = {
  "J00_HIP_PITCH_L", "J01_HIP_ROLL_L", "J02_HIP_YAW_L",
  "J03_KNEE_PITCH_L", "J04_ANKLE_PITCH_L", "J05_ANKLE_ROLL_L",
  "J06_HIP_PITCH_R", "J07_HIP_ROLL_R", "J08_HIP_YAW_R",
  "J09_KNEE_PITCH_R", "J10_ANKLE_PITCH_R", "J11_ANKLE_ROLL_R",
  "J12_WAIST_YAW",
  "J13_SHOULDER_PITCH_L", "J14_SHOULDER_ROLL_L",
  "J15_SHOULDER_YAW_L", "J16_ELBOW_PITCH_L", "J17_ELBOW_YAW_L",
  "J18_SHOULDER_PITCH_R", "J19_SHOULDER_ROLL_R",
  "J20_SHOULDER_YAW_R", "J21_ELBOW_PITCH_R", "J22_ELBOW_YAW_R",
  "J23_HEAD_YAW",
};

const std::vector<double> kJointLowerLimits = {
  -3.141, -0.436, -1.57, -0.3491, -0.6807, -0.2618,
  -3.141, -2.094, -4.014, -0.3491, -0.6807, -0.2618,
  -4.014,
  -2.9671, -0.6108, -2.618, -2.1948, -2.618,
  -2.9671, -2.3562, -2.618, -2.1948, -2.618,
  -0.6109,
};

const std::vector<double> kJointUpperLimits = {
  2.443, 2.094, 4.014, 2.3911, 0.7243, 0.2618,
  2.443, 0.436, 1.57, 2.3911, 0.7243, 0.2618,
  1.57,
  2.7925, 2.3562, 2.618, 0.7374, 2.618,
  2.7925, 0.6108, 2.618, 0.7374, 2.618,
  0.6109,
};

std::vector<double> flatten_groups(const YAML::Node & groups)
{
  std::vector<double> values;
  for (const auto & group : groups) {
    for (const auto & value : group) {
      values.push_back(value.as<double>());
    }
  }
  return values;
}

Eigen::Vector3d read_vector3(const YAML::Node & node)
{
  if (!node || node.size() != 3) {
    throw std::runtime_error("Expected a three-element vector in policy config");
  }
  return Eigen::Vector3d(
    node[0].as<double>(), node[1].as<double>(), node[2].as<double>());
}

class MnnPolicy
{
public:
  explicit MnnPolicy(const std::string & model_path)
  {
    interpreter_.reset(MNN::Interpreter::createFromFile(model_path.c_str()));
    if (!interpreter_) {
      throw std::runtime_error("Failed to load MNN model: " + model_path);
    }

    MNN::ScheduleConfig schedule;
    schedule.numThread = 1;
    session_ = interpreter_->createSession(schedule);
    if (!session_) {
      throw std::runtime_error("Failed to create MNN inference session");
    }
    input_ = interpreter_->getSessionInput(session_, nullptr);
    output_ = interpreter_->getSessionOutput(session_, nullptr);
    if (!input_ || !output_) {
      throw std::runtime_error("MNN model has no default input or output");
    }
    if (
      input_->elementSize() != static_cast<int>(kPolicyInputCount) ||
      output_->elementSize() != static_cast<int>(kActionCount))
    {
      throw std::runtime_error(
              "Unexpected MNN tensor sizes: input=" +
              std::to_string(input_->elementSize()) + " output=" +
              std::to_string(output_->elementSize()));
    }
  }

  ~MnnPolicy()
  {
    if (interpreter_ && session_) {
      interpreter_->releaseSession(session_);
      interpreter_->releaseModel();
    }
  }

  std::vector<float> infer(const std::vector<float> & observation)
  {
    if (observation.size() != kPolicyInputCount) {
      throw std::runtime_error("Policy input vector has the wrong size");
    }

    // Session tensors may use an internal packed layout. Always transfer
    // through host tensors with the model's logical (CAFFE) dimension order
    // instead of treating the session buffer as a flat float array.
    MNN::Tensor host_input(input_, MNN::Tensor::CAFFE);
    std::memcpy(
      host_input.host<float>(), observation.data(),
      observation.size() * sizeof(float));
    input_->copyFromHostTensor(&host_input);
    if (interpreter_->runSession(session_) != MNN::NO_ERROR) {
      throw std::runtime_error("MNN policy inference failed");
    }
    MNN::Tensor host_output(output_, MNN::Tensor::CAFFE);
    output_->copyToHostTensor(&host_output);
    const float * output_data = host_output.host<float>();
    return std::vector<float>(output_data, output_data + kActionCount);
  }

private:
  std::shared_ptr<MNN::Interpreter> interpreter_;
  MNN::Session * session_{nullptr};
  MNN::Tensor * input_{nullptr};
  MNN::Tensor * output_{nullptr};
};

}  // namespace

class Pm01WalkingPolicyNode : public rclcpp::Node
{
public:
  Pm01WalkingPolicyNode()
  : Node("pm01_walking_policy")
  {
    const auto config_file = declare_parameter<std::string>("config_file", "");
    const auto model_path = declare_parameter<std::string>("model_path", "");
    command_timeout_ = declare_parameter<double>("command_timeout", 0.5);
    sensor_timeout_ = declare_parameter<double>("sensor_timeout", 0.25);
    max_tilt_ = declare_parameter<double>("max_tilt", 1.0);
    simulation_action_clip_ =
      declare_parameter<double>("simulation_action_clip", 0.0);
    max_target_step_ = declare_parameter<double>("max_target_step", 0.0);
    reset_settle_time_ = declare_parameter<double>("reset_settle_time", 0.0);
    control_arms_ = declare_parameter<bool>("control_arms", true);
    const auto command_topic = declare_parameter<std::string>(
      "command_topic", "/pm01_sdk_pd_controller/commands");
    const auto imu_topic = declare_parameter<std::string>(
      "imu_topic", "/pm01/imu");
    if (config_file.empty() || model_path.empty()) {
      throw std::runtime_error("config_file and model_path parameters are required");
    }

    load_config(config_file);
    policy_ = std::make_unique<MnnPolicy>(model_path);
    joint_positions_.assign(kJointCount, 0.0);
    joint_velocities_.assign(kJointCount, 0.0);
    previous_action_.assign(kActionCount, 0.0F);
    history_.assign(kObservationCount * kHistorySteps, 0.0F);
    last_targets_ = default_positions_;

    for (std::size_t index = 0; index < kAllJoints.size(); ++index) {
      joint_index_[kAllJoints[index]] = index;
    }
    for (const auto & name : active_joint_names_) {
      const auto iterator = joint_index_.find(name);
      if (iterator == joint_index_.end()) {
        throw std::runtime_error("Unknown active joint in config: " + name);
      }
      active_joint_indices_.push_back(iterator->second);
    }

    joint_subscription_ = create_subscription<sensor_msgs::msg::JointState>(
      "/joint_states", rclcpp::SensorDataQoS(),
      std::bind(
        &Pm01WalkingPolicyNode::on_joint_state, this,
        std::placeholders::_1));
    imu_subscription_ = create_subscription<sensor_msgs::msg::Imu>(
      imu_topic, rclcpp::SensorDataQoS(),
      std::bind(
        &Pm01WalkingPolicyNode::on_imu, this, std::placeholders::_1));
    velocity_subscription_ = create_subscription<geometry_msgs::msg::Twist>(
      "/cmd_vel", 10,
      std::bind(
        &Pm01WalkingPolicyNode::on_cmd_vel, this,
        std::placeholders::_1));
    command_publisher_ =
      create_publisher<std_msgs::msg::Float64MultiArray>(
      command_topic, 10);
    reset_service_ = create_service<std_srvs::srv::Empty>(
      "/pm01_walking_policy/reset",
      std::bind(
        &Pm01WalkingPolicyNode::on_reset, this,
        std::placeholders::_1, std::placeholders::_2));

    timer_ = create_wall_timer(
      std::chrono::duration<double>(control_dt_),
      std::bind(&Pm01WalkingPolicyNode::control_step, this));
    RCLCPP_INFO(
      get_logger(),
      "PM01 MNN walking policy ready: input=%zu, output=%zu, rate=%.1f Hz",
      kPolicyInputCount, kActionCount, 1.0 / control_dt_);
  }

private:
  void load_config(const std::string & path)
  {
    const YAML::Node config = YAML::LoadFile(path);
    if (
      config["num_observations"].as<std::size_t>() != kObservationCount ||
      config["num_include_obs_steps"].as<std::size_t>() != kHistorySteps)
    {
      throw std::runtime_error("Policy observation dimensions do not match the MNN model");
    }

    active_joint_names_ =
      config["active_joint_names"].as<std::vector<std::string>>();
    default_positions_ = flatten_groups(config["default_joint_q"]);
    action_scales_ = flatten_groups(config["action_scale"]);
    if (
      active_joint_names_.size() != kActionCount ||
      default_positions_.size() != kJointCount ||
      action_scales_.size() != kActionCount)
    {
      throw std::runtime_error("Policy joint arrays have invalid dimensions");
    }

    position_observation_scale_ =
      config["observation_scale_dof_pos"].as<double>();
    velocity_observation_scale_ =
      config["observation_scale_dof_vel"].as<double>();
    angular_velocity_scale_ =
      config["observation_scale_angular_vel"].as<double>();
    gravity_scale_ = config["observation_scale_quat"].as<double>();
    command_observation_scale_ = Eigen::Vector3d(
      config["observation_scale_linear_vel"].as<double>(),
      config["observation_scale_linear_vel"].as<double>(), 1.0);
    observation_clip_ = config["observation_clip"].as<double>();
    action_clip_ = config["action_clip"].as<double>();
    control_dt_ = config["control_dt"].as<double>();
    command_scale_positive_ = read_vector3(config["command_scale_pos"]);
    command_scale_negative_ = read_vector3(config["command_scale_neg"]);
    imu_install_bias_ = read_vector3(config["imu_install_bias"]);

    const double sampling_frequency =
      config["remote_command_sampling_frequency"].as<double>();
    const double cutoff_frequency =
      config["remote_command_cut_off_frequency"].as<double>();
    const bool enable_filter =
      config["enable_remote_command_lpf"].as<bool>();
    if (
      enable_filter && sampling_frequency > 0.0 &&
      cutoff_frequency > 0.0)
    {
      const double dt = 1.0 / sampling_frequency;
      const double rc = 1.0 / (2.0 * M_PI * cutoff_frequency);
      command_filter_weight_ = dt / (dt + rc);
    } else {
      command_filter_weight_ = 1.0;
    }
  }

  void on_joint_state(const sensor_msgs::msg::JointState::SharedPtr message)
  {
    std::lock_guard<std::mutex> lock(mutex_);
    if (
      awaiting_post_reset_sensors_ &&
      rclcpp::Time(message->header.stamp) <= reset_sensor_epoch_)
    {
      return;
    }
    if (
      message->position.size() != message->name.size() ||
      message->velocity.size() != message->name.size())
    {
      return;
    }
    std::vector<bool> found(kJointCount, false);
    for (std::size_t source = 0; source < message->name.size(); ++source) {
      const auto iterator = joint_index_.find(message->name[source]);
      if (iterator == joint_index_.end()) {
        continue;
      }
      const std::size_t target = iterator->second;
      joint_positions_[target] = message->position[source];
      joint_velocities_[target] = message->velocity[source];
      found[target] = true;
    }
    joints_ready_ = std::all_of(
      found.begin(), found.end(), [](bool value) {
        return value;
      });
    if (joints_ready_) {
      last_joint_state_time_ = now();
    }
  }

  void on_imu(const sensor_msgs::msg::Imu::SharedPtr message)
  {
    std::lock_guard<std::mutex> lock(mutex_);
    if (
      awaiting_post_reset_sensors_ &&
      rclcpp::Time(message->header.stamp) <= reset_sensor_epoch_)
    {
      return;
    }
    const auto & orientation = message->orientation;
    Eigen::Quaterniond quaternion(
      orientation.w, orientation.x, orientation.y, orientation.z);
    if (!std::isfinite(quaternion.norm()) || quaternion.norm() < 1.0e-9) {
      return;
    }
    imu_orientation_ = quaternion.normalized();
    imu_angular_velocity_ = Eigen::Vector3d(
      message->angular_velocity.x,
      message->angular_velocity.y,
      message->angular_velocity.z);
    imu_ready_ = imu_angular_velocity_.allFinite();
    if (imu_ready_) {
      last_imu_time_ = now();
    }
  }

  void on_cmd_vel(const geometry_msgs::msg::Twist::SharedPtr message)
  {
    std::lock_guard<std::mutex> lock(mutex_);
    const Eigen::Vector3d normalized(
      std::clamp(message->linear.x, -1.0, 1.0),
      std::clamp(message->linear.y, -1.0, 1.0),
      std::clamp(message->angular.z, -1.0, 1.0));
    for (Eigen::Index index = 0; index < 3; ++index) {
      command_target_[index] =
        normalized[index] *
        (normalized[index] >= 0.0 ?
        command_scale_positive_[index] : command_scale_negative_[index]);
    }
    last_command_time_ = now();
    command_received_ = true;
  }

  Eigen::Matrix3d install_rotation() const
  {
    return (
      Eigen::AngleAxisd(imu_install_bias_.z(), Eigen::Vector3d::UnitZ()) *
      Eigen::AngleAxisd(imu_install_bias_.y(), Eigen::Vector3d::UnitY()) *
      Eigen::AngleAxisd(imu_install_bias_.x(), Eigen::Vector3d::UnitX())
    ).toRotationMatrix();
  }

  void control_step()
  {
    std::lock_guard<std::mutex> lock(mutex_);
    const auto current_time = now();
    if (
      !joints_ready_ || !imu_ready_ ||
      (current_time - last_joint_state_time_).seconds() > sensor_timeout_ ||
      (current_time - last_imu_time_).seconds() > sensor_timeout_)
    {
      RCLCPP_WARN_THROTTLE(
        get_logger(), *get_clock(), 2000,
        "Waiting for fresh sensors (joint_states=%s, imu=%s)",
        joints_ready_ ? "ready" : "missing",
        imu_ready_ ? "ready" : "missing");
      return;
    }

    Eigen::Vector3d target = command_target_;
    const bool command_fresh =
      command_received_ &&
      (current_time - last_command_time_).seconds() <= command_timeout_;
    if (!command_fresh) {
      target.setZero();
    }
    filtered_command_ +=
      command_filter_weight_ * (target - filtered_command_);

    const Eigen::Matrix3d local_rotation =
      imu_orientation_.toRotationMatrix();
    const Eigen::Matrix3d body_rotation =
      local_rotation * install_rotation().transpose();
    const Eigen::Vector3d body_angular_velocity =
      body_rotation.transpose() * local_rotation * imu_angular_velocity_;
    const Eigen::Vector3d projected_gravity =
      -body_rotation.transpose() * Eigen::Vector3d::UnitZ();
    const double tilt = std::acos(
      std::clamp(-projected_gravity.z(), -1.0, 1.0));

    if (halted_) {
      publish_targets(default_positions_);
      return;
    }
    if (current_time < policy_enable_after_) {
      publish_targets(default_positions_);
      return;
    }
    if (awaiting_post_reset_sensors_) {
      awaiting_post_reset_sensors_ = false;
      policy_enabled_ = true;
      history_ready_ = false;
      std::fill(previous_action_.begin(), previous_action_.end(), 0.0F);
      RCLCPP_INFO(
        get_logger(),
        "Fresh post-reset sensors received; enabling walking policy");
    }
    if (!policy_enabled_) {
      publish_targets(default_positions_);
      double maximum_joint_error = 0.0;
      double maximum_joint_velocity = 0.0;
      std::size_t maximum_error_joint = 0;
      std::size_t maximum_velocity_joint = 0;
      for (const std::size_t joint : active_joint_indices_) {
        const double error =
          std::abs(joint_positions_[joint] - default_positions_[joint]);
        const double velocity = std::abs(joint_velocities_[joint]);
        if (error > maximum_joint_error) {
          maximum_joint_error = error;
          maximum_error_joint = joint;
        }
        if (velocity > maximum_joint_velocity) {
          maximum_joint_velocity = velocity;
          maximum_velocity_joint = joint;
        }
      }
      if (
        tilt < 0.25 && maximum_joint_error < 0.15 &&
        maximum_joint_velocity < 1.0)
      {
        policy_enabled_ = true;
        history_ready_ = false;
        std::fill(previous_action_.begin(), previous_action_.end(), 0.0F);
        RCLCPP_INFO(
          get_logger(),
          "Robot settled; enabling policy (tilt=%.3f, joint_error=%.3f, "
          "joint_velocity=%.3f)",
          tilt, maximum_joint_error, maximum_joint_velocity);
      } else {
        RCLCPP_INFO_THROTTLE(
          get_logger(), *get_clock(), 1000,
          "Waiting for settled reset pose (tilt=%.3f, joint_error=%.3f %s, "
          "joint_velocity=%.3f %s)",
          tilt, maximum_joint_error, kAllJoints[maximum_error_joint].c_str(),
          maximum_joint_velocity,
          kAllJoints[maximum_velocity_joint].c_str());
      }
      return;
    }
    if (tilt > max_tilt_) {
      halted_ = true;
      command_target_.setZero();
      filtered_command_.setZero();
      publish_targets(default_positions_);
      RCLCPP_ERROR(
        get_logger(),
        "Policy halted because base tilt reached %.3f rad; call "
        "/pm01/reset_stand before continuing",
        tilt);
      return;
    }

    std::vector<float> observation(kObservationCount, 0.0F);
    for (std::size_t action = 0; action < kActionCount; ++action) {
      const std::size_t joint = active_joint_indices_[action];
      observation[action] = static_cast<float>(
        (joint_positions_[joint] - default_positions_[joint]) *
        position_observation_scale_);
      observation[kActionCount + action] = static_cast<float>(
        joint_velocities_[joint] * velocity_observation_scale_);
      observation[2 * kActionCount + action] = previous_action_[action];
    }
    for (std::size_t axis = 0; axis < 3; ++axis) {
      observation[3 * kActionCount + axis] = static_cast<float>(
        body_angular_velocity[axis] * angular_velocity_scale_);
      observation[3 * kActionCount + 3 + axis] = static_cast<float>(
        projected_gravity[axis] * gravity_scale_);
    }
    for (auto & value : observation) {
      value = std::clamp(
        value, static_cast<float>(-observation_clip_),
        static_cast<float>(observation_clip_));
    }

    if (!history_ready_) {
      for (std::size_t step = 0; step < kHistorySteps; ++step) {
        std::copy(
          observation.begin(), observation.end(),
          history_.begin() + step * kObservationCount);
      }
      history_ready_ = true;
    } else {
      std::memmove(
        history_.data(), history_.data() + kObservationCount,
        (history_.size() - kObservationCount) * sizeof(float));
      std::copy(
        observation.begin(), observation.end(),
        history_.end() - kObservationCount);
    }

    std::vector<float> policy_input(kPolicyInputCount, 0.0F);
    std::copy(history_.begin(), history_.end(), policy_input.begin());
    const Eigen::Vector3d command_observation =
      filtered_command_.cwiseProduct(command_observation_scale_);
    for (std::size_t axis = 0; axis < 3; ++axis) {
      policy_input[kObservationCount * kHistorySteps + axis] =
        static_cast<float>(command_observation[axis]);
    }

    std::vector<float> action;
    try {
      action = policy_->infer(policy_input);
    } catch (const std::exception & error) {
      RCLCPP_ERROR_THROTTLE(
        get_logger(), *get_clock(), 2000, "%s", error.what());
      return;
    }

    if (!first_inference_logged_) {
      const auto action_range =
        std::minmax_element(action.begin(), action.end());
      RCLCPP_INFO(
        get_logger(),
        "First policy inference: gravity=[%.3f, %.3f, %.3f], "
        "angular_velocity=[%.3f, %.3f, %.3f], action_range=[%.3f, %.3f]",
        projected_gravity.x(), projected_gravity.y(), projected_gravity.z(),
        body_angular_velocity.x(), body_angular_velocity.y(),
        body_angular_velocity.z(), *action_range.first, *action_range.second);
      first_inference_logged_ = true;
    }

    auto targets = default_positions_;
    const double effective_action_clip =
      simulation_action_clip_ > 0.0 ?
      std::min(action_clip_, simulation_action_clip_) : action_clip_;
    for (std::size_t index = 0; index < kActionCount; ++index) {
      if (!std::isfinite(action[index])) {
        RCLCPP_ERROR(get_logger(), "MNN policy produced a non-finite action");
        return;
      }
      previous_action_[index] = std::clamp(
        action[index], static_cast<float>(-effective_action_clip),
        static_cast<float>(effective_action_clip));
      const std::size_t joint = active_joint_indices_[index];
      if (!control_arms_ && joint >= 13 && joint <= 22) {
        previous_action_[index] = 0.0F;
        continue;
      }
      targets[joint] += previous_action_[index] * action_scales_[index];
      targets[joint] = std::clamp(
        targets[joint], kJointLowerLimits[joint], kJointUpperLimits[joint]);
    }
    if (max_target_step_ > 0.0 && last_targets_.size() == targets.size()) {
      for (std::size_t joint = 0; joint < targets.size(); ++joint) {
        targets[joint] = std::clamp(
          targets[joint],
          last_targets_[joint] - max_target_step_,
          last_targets_[joint] + max_target_step_);
      }
      // If an optional actuator slew limit changes the commanded action, feed
      // the action that was actually applied back into the next observation.
      for (std::size_t index = 0; index < kActionCount; ++index) {
        const std::size_t joint = active_joint_indices_[index];
        if (std::abs(action_scales_[index]) > 1.0e-12) {
          previous_action_[index] = static_cast<float>(
            (targets[joint] - default_positions_[joint]) /
            action_scales_[index]);
        }
      }
    }
    publish_targets(targets);
  }

  void publish_targets(const std::vector<double> & targets)
  {
    last_targets_ = targets;
    std_msgs::msg::Float64MultiArray message;
    message.data = targets;
    command_publisher_->publish(message);
  }

  void on_reset(
    const std::shared_ptr<std_srvs::srv::Empty::Request> request,
    std::shared_ptr<std_srvs::srv::Empty::Response> response)
  {
    (void)request;
    (void)response;
    std::lock_guard<std::mutex> lock(mutex_);
    std::fill(previous_action_.begin(), previous_action_.end(), 0.0F);
    std::fill(history_.begin(), history_.end(), 0.0F);
    history_ready_ = false;
    command_target_.setZero();
    filtered_command_.setZero();
    command_received_ = false;
    policy_enabled_ = false;
    halted_ = false;
    first_inference_logged_ = false;
    joints_ready_ = false;
    imu_ready_ = false;
    awaiting_post_reset_sensors_ = true;
    reset_sensor_epoch_ = now();
    policy_enable_after_ =
      reset_sensor_epoch_ + rclcpp::Duration::from_seconds(
      std::max(0.0, reset_settle_time_));
    publish_targets(default_positions_);
    RCLCPP_INFO(get_logger(), "Walking policy state reset");
  }

  std::mutex mutex_;
  std::unique_ptr<MnnPolicy> policy_;
  std::unordered_map<std::string, std::size_t> joint_index_;
  std::vector<std::string> active_joint_names_;
  std::vector<std::size_t> active_joint_indices_;
  std::vector<double> default_positions_;
  std::vector<double> action_scales_;
  std::vector<double> joint_positions_;
  std::vector<double> joint_velocities_;
  std::vector<double> last_targets_;
  std::vector<float> previous_action_;
  std::vector<float> history_;

  Eigen::Quaterniond imu_orientation_{Eigen::Quaterniond::Identity()};
  Eigen::Vector3d imu_angular_velocity_{Eigen::Vector3d::Zero()};
  Eigen::Vector3d imu_install_bias_{Eigen::Vector3d::Zero()};
  Eigen::Vector3d command_target_{Eigen::Vector3d::Zero()};
  Eigen::Vector3d filtered_command_{Eigen::Vector3d::Zero()};
  Eigen::Vector3d command_scale_positive_{Eigen::Vector3d::Ones()};
  Eigen::Vector3d command_scale_negative_{Eigen::Vector3d::Ones()};
  Eigen::Vector3d command_observation_scale_{2.0, 2.0, 1.0};

  double position_observation_scale_{1.0};
  double velocity_observation_scale_{0.05};
  double angular_velocity_scale_{1.0};
  double gravity_scale_{1.0};
  double observation_clip_{100.0};
  double action_clip_{100.0};
  double control_dt_{0.01};
  double command_timeout_{0.5};
  double sensor_timeout_{0.25};
  double max_tilt_{1.0};
  double simulation_action_clip_{0.0};
  double max_target_step_{0.0};
  double reset_settle_time_{0.0};
  double command_filter_weight_{1.0};
  bool joints_ready_{false};
  bool imu_ready_{false};
  bool history_ready_{false};
  bool command_received_{false};
  bool policy_enabled_{false};
  bool halted_{false};
  bool first_inference_logged_{false};
  bool control_arms_{true};
  bool awaiting_post_reset_sensors_{false};
  rclcpp::Time last_joint_state_time_{0, 0, RCL_ROS_TIME};
  rclcpp::Time last_imu_time_{0, 0, RCL_ROS_TIME};
  rclcpp::Time last_command_time_{0, 0, RCL_ROS_TIME};
  rclcpp::Time policy_enable_after_{0, 0, RCL_ROS_TIME};
  rclcpp::Time reset_sensor_epoch_{0, 0, RCL_ROS_TIME};

  rclcpp::Subscription<sensor_msgs::msg::JointState>::SharedPtr
    joint_subscription_;
  rclcpp::Subscription<sensor_msgs::msg::Imu>::SharedPtr imu_subscription_;
  rclcpp::Subscription<geometry_msgs::msg::Twist>::SharedPtr
    velocity_subscription_;
  rclcpp::Publisher<std_msgs::msg::Float64MultiArray>::SharedPtr
    command_publisher_;
  rclcpp::Service<std_srvs::srv::Empty>::SharedPtr reset_service_;
  rclcpp::TimerBase::SharedPtr timer_;
};

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  try {
    rclcpp::spin(std::make_shared<Pm01WalkingPolicyNode>());
  } catch (const std::exception & error) {
    RCLCPP_FATAL(
      rclcpp::get_logger("pm01_walking_policy"), "%s", error.what());
    rclcpp::shutdown();
    return 1;
  }
  rclcpp::shutdown();
  return 0;
}
