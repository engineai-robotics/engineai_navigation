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

#include <tf2_ros/buffer.h>
#include <tf2_ros/transform_listener.h>

#include <Eigen/Geometry>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <deque>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include <geometry_msgs/msg/transform_stamped.hpp>
#include <livox_ros_driver2/msg/custom_msg.hpp>
#include "mid360_preprocessor/resampler.hpp"
#include <nav_msgs/msg/odometry.hpp>
#include <rclcpp/executors/multi_threaded_executor.hpp>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <sensor_msgs/msg/point_field.hpp>
#include <sensor_msgs/point_cloud2_iterator.hpp>
#include <std_msgs/msg/header.hpp>
#include <tf2_eigen/tf2_eigen.hpp>

namespace mid360_preprocessor
{
namespace
{

constexpr double kPi = 3.14159265358979323846;
constexpr auto kOutputPollPeriod = std::chrono::milliseconds(10);

Eigen::Vector3d vector3_parameter(
  rclcpp::Node & node, const std::string & name,
  const std::vector<double> & default_value)
{
  const auto value = node.declare_parameter<std::vector<double>>(name, default_value);
  if (value.size() != 3U) {
    throw std::invalid_argument(name + " must contain exactly three values");
  }
  const Eigen::Vector3d result(value[0], value[1], value[2]);
  if (!result.allFinite()) {
    throw std::invalid_argument(name + " must contain finite values");
  }
  return result;
}

bool has_float32_xyz(const sensor_msgs::msg::PointCloud2 & cloud)
{
  bool has_x = false;
  bool has_y = false;
  bool has_z = false;
  for (const auto & field : cloud.fields) {
    if (field.name == "x") {
      has_x = field.datatype == sensor_msgs::msg::PointField::FLOAT32;
    } else if (field.name == "y") {
      has_y = field.datatype == sensor_msgs::msg::PointField::FLOAT32;
    } else if (field.name == "z") {
      has_z = field.datatype == sensor_msgs::msg::PointField::FLOAT32;
    }
  }
  return has_x && has_y && has_z;
}

double stamp_seconds(const builtin_interfaces::msg::Time & stamp)
{
  return static_cast<double>(stamp.sec) + 1.0e-9 * stamp.nanosec;
}

Eigen::Isometry3d pose_from_odometry(const nav_msgs::msg::Odometry & message)
{
  Eigen::Isometry3d pose = Eigen::Isometry3d::Identity();
  pose.translation() = Eigen::Vector3d(
    message.pose.pose.position.x,
    message.pose.pose.position.y,
    message.pose.pose.position.z);
  Eigen::Quaterniond rotation(
    message.pose.pose.orientation.w,
    message.pose.pose.orientation.x,
    message.pose.pose.orientation.y,
    message.pose.pose.orientation.z);
  if (rotation.norm() > 1.0e-9) {
    pose.linear() = rotation.normalized().toRotationMatrix();
  }
  return pose;
}

Eigen::Isometry3d interpolate_pose(
  const Eigen::Isometry3d & first, const Eigen::Isometry3d & second,
  double alpha)
{
  Eigen::Isometry3d pose = Eigen::Isometry3d::Identity();
  pose.translation() =
    (1.0 - alpha) * first.translation() + alpha * second.translation();
  const Eigen::Quaterniond first_rotation(first.linear());
  const Eigen::Quaterniond second_rotation(second.linear());
  pose.linear() =
    first_rotation.slerp(alpha, second_rotation).normalized().toRotationMatrix();
  return pose;
}

template<typename Point>
sensor_msgs::msg::PointCloud2 make_xyz_cloud(
  const std_msgs::msg::Header & header,
  uint32_t height,
  uint32_t width,
  bool is_dense,
  const std::vector<Point> & points)
{
  sensor_msgs::msg::PointCloud2 cloud;
  cloud.header = header;
  sensor_msgs::PointCloud2Modifier modifier(cloud);
  modifier.setPointCloud2FieldsByString(1, "xyz");
  modifier.resize(points.size());
  cloud.height = height;
  cloud.width = width;
  cloud.row_step = cloud.point_step * cloud.width;
  cloud.is_dense = is_dense;

  uint8_t * ptr = cloud.data.data();
  const uint32_t step = cloud.point_step;
  for (const auto & point : points) {
    auto * xyz = reinterpret_cast<float *>(ptr);
    xyz[0] = static_cast<float>(point.x());
    xyz[1] = static_cast<float>(point.y());
    xyz[2] = static_cast<float>(point.z());
    ptr += step;
  }
  return cloud;
}

struct StampedPose
{
  double stamp{0.0};
  Eigen::Isometry3d pose{Eigen::Isometry3d::Identity()};
};

struct CloudFrame
{
  std_msgs::msg::Header header;
  std::vector<Eigen::Vector3d> points;
  Eigen::Isometry3d pose{Eigen::Isometry3d::Identity()};
  bool pose_valid{false};
};

rclcpp::QoS sensor_keep_last(size_t depth)
{
  return rclcpp::SensorDataQoS().keep_last(depth);
}

bool parse_xyz_points(
  const sensor_msgs::msg::PointCloud2 & message,
  std::vector<Eigen::Vector3d> & points)
{
  if (!has_float32_xyz(message)) {
    return false;
  }
  const std::size_t input_count =
    static_cast<std::size_t>(message.width) * message.height;
  points.clear();
  points.reserve(input_count);
  try {
    sensor_msgs::PointCloud2ConstIterator<float> x(message, "x");
    sensor_msgs::PointCloud2ConstIterator<float> y(message, "y");
    sensor_msgs::PointCloud2ConstIterator<float> z(message, "z");
    for (; x != x.end(); ++x, ++y, ++z) {
      const Eigen::Vector3d point(*x, *y, *z);
      if (point.allFinite()) {
        points.push_back(point);
      }
    }
  } catch (const std::runtime_error &) {
    points.clear();
    return false;
  }
  return true;
}

}  // namespace

class Mid360PreprocessorNode : public rclcpp::Node
{
public:
  Mid360PreprocessorNode()
  : Node("mid360_preprocessor"),
    tf_buffer_(get_clock()),
    tf_listener_(tf_buffer_)
  {
    input_type_ = declare_parameter<std::string>("input_type", "custom_msg");
    std::string default_input_topic;
    if (input_type_ == "custom_msg") {
      default_input_topic = "/livox/lidar";
    } else if (input_type_ == "pointcloud2") {
      default_input_topic = "/livox/lidar/pointcloud";
    } else {
      throw std::invalid_argument(
              "input_type must be either 'custom_msg' or 'pointcloud2'");
    }
    input_topic_ = declare_parameter<std::string>("input_topic", default_input_topic);
    output_topic_ = declare_parameter<std::string>(
      "output_topic", "/navigation/lidar/points");
    local_map_topic_ = declare_parameter<std::string>(
      "local_map_topic", "/navigation/lidar/local_map");
    target_frame_ = declare_parameter<std::string>(
      "target_frame", "LINK_TORSO_YAW");
    source_frame_override_ = declare_parameter<std::string>(
      "source_frame_override", "");
    use_tf_ = declare_parameter<bool>("use_tf", true);
    allow_static_fallback_ = declare_parameter<bool>(
      "allow_static_fallback", true);
    tf_timeout_seconds_ = declare_parameter<double>("tf_timeout", 0.05);

    config_.vertical_channels = declare_parameter<int>("vertical_channels", 16);
    config_.vertical_min_deg = declare_parameter<double>(
      "vertical_fov_min_deg", -45.0);
    config_.vertical_max_deg = declare_parameter<double>(
      "vertical_fov_max_deg", 0.0);
    config_.horizontal_min_deg = declare_parameter<double>(
      "horizontal_fov_min_deg", -60.0);
    config_.horizontal_max_deg = declare_parameter<double>(
      "horizontal_fov_max_deg", 60.0);
    config_.horizontal_resolution_deg = declare_parameter<double>(
      "horizontal_resolution_deg", 3.0);
    config_.min_distance = declare_parameter<double>("min_distance", 0.1);
    config_.max_distance = declare_parameter<double>("max_distance", 5.0);
    config_.neighbor_fill_radius_deg = declare_parameter<double>(
      "neighbor_fill_radius_deg", 3.0);
    config_.max_range_fill_height_above_ground = declare_parameter<double>(
      "max_range_fill_height_above_ground", 0.1);
    config_.origin = vector3_parameter(
      *this, "ray_origin", {0.105, 0.0, 0.185});
    config_.crop_x_min = declare_parameter<double>("crop_x_min", -7.0);
    config_.crop_x_max = declare_parameter<double>("crop_x_max", 7.0);
    config_.crop_y_min = declare_parameter<double>("crop_y_min", -7.0);
    config_.crop_y_max = declare_parameter<double>("crop_y_max", 7.0);
    config_.crop_z_min = declare_parameter<double>("crop_z_min", -0.8);
    config_.crop_z_max = declare_parameter<double>("crop_z_max", 0.7);
    config_.static_z_margin = declare_parameter<double>(
      "static_z_margin", 0.6);
    temporal_frames_ = declare_parameter<int>("temporal_frames", 5);
    if (temporal_frames_ < 1) {
      throw std::invalid_argument("temporal_frames must be at least 1");
    }
    odom_topic_ = declare_parameter<std::string>(
      "odom_topic", "/lio/robo/odom");
    static_map_topic_ = declare_parameter<std::string>(
      "static_map_topic", "/lio/cloud_world");
    odom_timeout_seconds_ = declare_parameter<double>("odom_timeout", 0.2);
    if (!std::isfinite(odom_timeout_seconds_) || odom_timeout_seconds_ < 0.0) {
      throw std::invalid_argument("odom_timeout must be finite and non-negative");
    }
    output_rate_ = declare_parameter<double>("output_rate", 10.0);
    local_map_rate_ = declare_parameter<double>("local_map_rate", 2.0);
    voxel_size_ = declare_parameter<double>("voxel_size", 0.05);
    local_map_publish_voxel_size_ = declare_parameter<double>(
      "local_map_publish_voxel_size", 0.05);
    map_cache_margin_deg_ = declare_parameter<double>(
      "map_cache_margin_deg", 120.0);
    map_cache_elevation_margin_deg_ = declare_parameter<double>(
      "map_cache_elevation_margin_deg", 15.0);
    if (!std::isfinite(map_cache_margin_deg_) || map_cache_margin_deg_ < 0.0 ||
      !std::isfinite(map_cache_elevation_margin_deg_) ||
      map_cache_elevation_margin_deg_ < 0.0)
    {
      throw std::invalid_argument(
              "map cache margins must be finite and non-negative");
    }
    map_cache_resolution_divisor_ = declare_parameter<int>(
      "map_cache_resolution_divisor", 3);
    if (map_cache_resolution_divisor_ < 1) {
      throw std::invalid_argument(
              "map_cache_resolution_divisor must be at least 1");
    }
    map_cache_config_ = expand_grid_config(
      config_, map_cache_margin_deg_, map_cache_elevation_margin_deg_,
      map_cache_resolution_divisor_);
    static_block_size_ = declare_parameter<double>("static_block_size", 2.0);
    static_match_voxel_size_ = declare_parameter<double>(
      "static_match_voxel_size", 0.0);
    if (!std::isfinite(static_match_voxel_size_) ||
      static_match_voxel_size_ < 0.0)
    {
      throw std::invalid_argument(
              "static_match_voxel_size must be finite and non-negative");
    }
    if (!std::isfinite(output_rate_) || output_rate_ < 0.0) {
      throw std::invalid_argument("output_rate must be finite and non-negative");
    }
    if (!std::isfinite(local_map_rate_) || local_map_rate_ < 0.0) {
      throw std::invalid_argument(
              "local_map_rate must be finite and non-negative");
    }
    if (!std::isfinite(voxel_size_) || voxel_size_ < 0.0) {
      throw std::invalid_argument("voxel_size must be finite and non-negative");
    }
    if (
      !std::isfinite(local_map_publish_voxel_size_) ||
      local_map_publish_voxel_size_ < 0.0)
    {
      throw std::invalid_argument(
              "local_map_publish_voxel_size must be finite and non-negative");
    }
    if (voxel_size_ > 0.0 &&
      !static_world_.configure(voxel_size_, static_block_size_))
    {
      throw std::invalid_argument(
              "static_block_size must be finite and at least voxel_size");
    }
    if (output_rate_ > 0.0) {
      output_min_interval_ = std::chrono::duration_cast<
        std::chrono::steady_clock::duration>(
        std::chrono::duration<double>(1.0 / output_rate_));
    }

    fallback_translation_ = vector3_parameter(
      *this, "fallback_translation", {0.12438064, 0.0, 0.21289480});
    const Eigen::Vector3d fallback_rpy_deg = vector3_parameter(
      *this, "fallback_rpy_deg", {50.0, 0.0, 90.0});
    fallback_rotation_ =
      Eigen::AngleAxisd(fallback_rpy_deg.z() * kPi / 180.0, Eigen::Vector3d::UnitZ()) *
      Eigen::AngleAxisd(fallback_rpy_deg.y() * kPi / 180.0, Eigen::Vector3d::UnitY()) *
      Eigen::AngleAxisd(fallback_rpy_deg.x() * kPi / 180.0, Eigen::Vector3d::UnitX());

    std::string config_error;
    if (!validate_config(config_, config_error)) {
      throw std::invalid_argument(config_error);
    }
    if (
      target_frame_.empty() || input_topic_.empty() || output_topic_.empty() ||
      local_map_topic_.empty() ||
      !std::isfinite(tf_timeout_seconds_) || tf_timeout_seconds_ < 0.0)
    {
      throw std::invalid_argument("topics, target_frame and tf_timeout must be valid");
    }
    if (local_map_rate_ > 0.0 &&
      (odom_topic_.empty() || static_map_topic_.empty()))
    {
      throw std::invalid_argument(
              "odom_topic and static_map_topic must be set when local_map_rate > 0");
    }

    subscription_callback_group_ = create_callback_group(
      rclcpp::CallbackGroupType::MutuallyExclusive);
    output_callback_group_ = create_callback_group(
      rclcpp::CallbackGroupType::MutuallyExclusive);
    local_map_callback_group_ = create_callback_group(
      rclcpp::CallbackGroupType::MutuallyExclusive);
    // Ingesting the world map must never wait on the local map timer,
    // otherwise the static layer stops growing while the window is cropped.
    static_map_callback_group_ = create_callback_group(
      rclcpp::CallbackGroupType::MutuallyExclusive);

    publisher_ = create_publisher<sensor_msgs::msg::PointCloud2>(
      output_topic_, sensor_keep_last(1));
    if (local_map_rate_ > 0.0) {
      local_map_publisher_ = create_publisher<sensor_msgs::msg::PointCloud2>(
        local_map_topic_, sensor_keep_last(1));
    }
    rclcpp::SubscriptionOptions subscription_options;
    subscription_options.callback_group = subscription_callback_group_;
    rclcpp::SubscriptionOptions local_map_subscription_options;
    local_map_subscription_options.callback_group = local_map_callback_group_;
    rclcpp::SubscriptionOptions static_map_subscription_options;
    static_map_subscription_options.callback_group = static_map_callback_group_;
    if (local_map_rate_ > 0.0) {
      odom_subscription_ = create_subscription<nav_msgs::msg::Odometry>(
        odom_topic_, rclcpp::QoS(20),
        std::bind(
          &Mid360PreprocessorNode::odom_callback, this,
          std::placeholders::_1),
        subscription_options);
      static_map_subscription_ =
        create_subscription<sensor_msgs::msg::PointCloud2>(
        static_map_topic_, rclcpp::QoS(10).reliable(),
        std::bind(
          &Mid360PreprocessorNode::static_map_callback, this,
          std::placeholders::_1),
        static_map_subscription_options);
    }
    if (input_type_ == "custom_msg") {
      custom_subscription_ =
        create_subscription<livox_ros_driver2::msg::CustomMsg>(
        input_topic_, sensor_keep_last(1),
        std::bind(
          &Mid360PreprocessorNode::custom_callback, this,
          std::placeholders::_1),
        subscription_options);
    } else {
      pointcloud_subscription_ =
        create_subscription<sensor_msgs::msg::PointCloud2>(
        input_topic_, sensor_keep_last(1),
        std::bind(
          &Mid360PreprocessorNode::pointcloud_callback, this,
          std::placeholders::_1),
        subscription_options);
    }

    output_timer_ = create_wall_timer(
      kOutputPollPeriod,
      std::bind(&Mid360PreprocessorNode::output_timer_callback, this),
      output_callback_group_);
    if (local_map_rate_ > 0.0) {
      const auto local_map_period =
        std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::duration<double>(1.0 / local_map_rate_));
      local_map_timer_ = create_wall_timer(
        local_map_period,
        std::bind(&Mid360PreprocessorNode::local_map_timer_callback, this),
        local_map_callback_group_);
    }

    RCLCPP_INFO(
      get_logger(),
      "Resampling %s on %s to %dx%zu PointCloud2 on %s in frame %s "
      "(local_map %s at %.1f Hz, static %s, output_rate %.1f Hz, "
      "voxel %.3f m, publish_voxel %.3f m, match_voxel %.3f m, "
      "static_block %.1f m, map_cache %.0f..%.0f x %.0f..%.0f deg "
      "at %.2f deg (%zux%d), crop x %.1f..%.1f m / "
      "y %.1f..%.1f m / z %.1f..%.1f m, odom %s)",
      input_type_.c_str(), input_topic_.c_str(), config_.vertical_channels,
      horizontal_samples(config_), output_topic_.c_str(), target_frame_.c_str(),
      local_map_rate_ > 0.0 ? local_map_topic_.c_str() : "disabled",
      local_map_rate_,
      local_map_rate_ > 0.0 ? static_map_topic_.c_str() : "disabled",
      output_rate_, voxel_size_, publish_voxel_size(), match_voxel_size(),
      static_block_size_,
      map_cache_config_.horizontal_min_deg,
      map_cache_config_.horizontal_max_deg,
      map_cache_config_.vertical_min_deg,
      map_cache_config_.vertical_max_deg,
      map_cache_config_.horizontal_resolution_deg,
      horizontal_samples(map_cache_config_),
      map_cache_config_.vertical_channels,
      config_.crop_x_min, config_.crop_x_max,
      config_.crop_y_min, config_.crop_y_max,
      config_.crop_z_min, config_.crop_z_max,
      local_map_rate_ > 0.0 ? odom_topic_.c_str() : "disabled");
  }

private:
  double publish_voxel_size() const
  {
    if (local_map_publish_voxel_size_ > 0.0) {
      return local_map_publish_voxel_size_;
    }
    return voxel_size_;
  }

  double match_voxel_size() const
  {
    if (static_match_voxel_size_ > 0.0) {
      return static_match_voxel_size_;
    }
    return publish_voxel_size();
  }

  Eigen::Isometry3d source_to_target(
    const std_msgs::msg::Header & header, bool & success)
  {
    success = false;
    const std::string source_frame =
      source_frame_override_.empty() ? header.frame_id : source_frame_override_;
    if (source_frame.empty()) {
      RCLCPP_ERROR_THROTTLE(
        get_logger(), *get_clock(), 2000,
        "Input frame is empty; set source_frame_override");
      return Eigen::Isometry3d::Identity();
    }
    if (source_frame == target_frame_) {
      success = true;
      return Eigen::Isometry3d::Identity();
    }

    if (use_tf_) {
      try {
        const auto transform = tf_buffer_.lookupTransform(
          target_frame_, source_frame, rclcpp::Time(header.stamp),
          rclcpp::Duration::from_seconds(tf_timeout_seconds_));
        success = true;
        return tf2::transformToEigen(transform);
      } catch (const tf2::TransformException & error) {
        if (!allow_static_fallback_) {
          RCLCPP_WARN_THROTTLE(
            get_logger(), *get_clock(), 2000,
            "Cannot transform %s to %s: %s", source_frame.c_str(),
            target_frame_.c_str(), error.what());
          return Eigen::Isometry3d::Identity();
        }
        RCLCPP_WARN_THROTTLE(
          get_logger(), *get_clock(), 5000,
          "Cannot transform %s to %s (%s); using configured static fallback",
          source_frame.c_str(), target_frame_.c_str(), error.what());
      }
    }

    if (allow_static_fallback_) {
      Eigen::Isometry3d fallback = Eigen::Isometry3d::Identity();
      fallback.linear() = fallback_rotation_;
      fallback.translation() = fallback_translation_;
      success = true;
      return fallback;
    }
    RCLCPP_WARN_THROTTLE(
      get_logger(), *get_clock(), 2000,
      "TF is disabled and static fallback is not allowed");
    return Eigen::Isometry3d::Identity();
  }

  void custom_callback(
    const livox_ros_driver2::msg::CustomMsg::ConstSharedPtr message)
  {
    std::lock_guard<std::mutex> lock(input_mutex_);
    latest_custom_ = message;
    latest_pointcloud_.reset();
    has_new_input_ = true;
  }

  void pointcloud_callback(
    const sensor_msgs::msg::PointCloud2::ConstSharedPtr message)
  {
    std::lock_guard<std::mutex> lock(input_mutex_);
    latest_pointcloud_ = message;
    latest_custom_.reset();
    has_new_input_ = true;
  }

  void odom_callback(const nav_msgs::msg::Odometry::ConstSharedPtr message)
  {
    std::lock_guard<std::mutex> lock(odom_mutex_);
    odom_history_.push_back(
      StampedPose{
        stamp_seconds(message->header.stamp),
        pose_from_odometry(*message)});
    while (odom_history_.size() > 200U) {
      odom_history_.pop_front();
    }
  }

  bool body_pose_at(
    const std_msgs::msg::Header & header, Eigen::Isometry3d & pose)
  {
    std::lock_guard<std::mutex> lock(odom_mutex_);
    if (odom_history_.empty()) {
      return false;
    }
    const double stamp = stamp_seconds(header.stamp);
    if (stamp <= odom_history_.front().stamp) {
      if (odom_history_.front().stamp - stamp > odom_timeout_seconds_) {
        return false;
      }
      pose = odom_history_.front().pose;
      return true;
    }
    if (stamp >= odom_history_.back().stamp) {
      if (stamp - odom_history_.back().stamp > odom_timeout_seconds_) {
        return false;
      }
      pose = odom_history_.back().pose;
      return true;
    }
    for (std::size_t index = 1; index < odom_history_.size(); ++index) {
      if (stamp > odom_history_[index].stamp) {
        continue;
      }
      const StampedPose & first = odom_history_[index - 1U];
      const StampedPose & second = odom_history_[index];
      const double span = second.stamp - first.stamp;
      const double alpha =
        span > 1.0e-9 ? (stamp - first.stamp) / span : 0.0;
      pose = interpolate_pose(first.pose, second.pose, alpha);
      return true;
    }
    return false;
  }

  bool latest_body_pose(Eigen::Isometry3d & pose)
  {
    std::lock_guard<std::mutex> lock(odom_mutex_);
    if (odom_history_.empty()) {
      return false;
    }
    pose = odom_history_.back().pose;
    return true;
  }

  bool output_rate_allows(
    const std::chrono::steady_clock::time_point & now) const
  {
    if (output_rate_ <= 0.0) {
      return true;
    }
    if (last_output_time_.time_since_epoch().count() == 0) {
      return true;
    }
    return now - last_output_time_ >= output_min_interval_;
  }

  void output_timer_callback()
  {
    const auto now = std::chrono::steady_clock::now();
    if (!output_rate_allows(now)) {
      return;
    }

    livox_ros_driver2::msg::CustomMsg::ConstSharedPtr custom;
    sensor_msgs::msg::PointCloud2::ConstSharedPtr pointcloud;
    {
      std::lock_guard<std::mutex> lock(input_mutex_);
      if (!has_new_input_) {
        return;
      }
      custom = latest_custom_;
      pointcloud = latest_pointcloud_;
      has_new_input_ = false;
    }

    last_output_time_ = now;
    if (custom) {
      process_custom(custom);
    } else if (pointcloud) {
      process_pointcloud(pointcloud);
    }
  }

  void process_custom(
    const livox_ros_driver2::msg::CustomMsg::ConstSharedPtr & message)
  {
    bool transform_ok = false;
    const Eigen::Isometry3d transform =
      source_to_target(message->header, transform_ok);
    if (!transform_ok) {
      return;
    }
    std::vector<Eigen::Vector3d> points;
    points.reserve(message->points.size());
    for (const auto & point : message->points) {
      const Eigen::Vector3d lidar_point(point.x, point.y, point.z);
      if (!lidar_point.allFinite()) {
        continue;
      }
      const Eigen::Vector3d body = transform * lidar_point;
      if (keep_body_point(body, config_)) {
        points.push_back(body);
      }
    }
    process_body_cloud(
      std::move(points), message->header, message->points.size());
  }

  void process_pointcloud(
    const sensor_msgs::msg::PointCloud2::ConstSharedPtr & message)
  {
    if (!has_float32_xyz(*message)) {
      RCLCPP_ERROR_THROTTLE(
        get_logger(), *get_clock(), 2000,
        "PointCloud2 input must contain FLOAT32 x, y and z fields");
      return;
    }
    bool transform_ok = false;
    const Eigen::Isometry3d transform =
      source_to_target(message->header, transform_ok);
    if (!transform_ok) {
      return;
    }

    std::vector<Eigen::Vector3d> points;
    const std::size_t input_count =
      static_cast<std::size_t>(message->width) * message->height;
    points.reserve(input_count);
    try {
      sensor_msgs::PointCloud2ConstIterator<float> x(*message, "x");
      sensor_msgs::PointCloud2ConstIterator<float> y(*message, "y");
      sensor_msgs::PointCloud2ConstIterator<float> z(*message, "z");
      for (; x != x.end(); ++x, ++y, ++z) {
        const Eigen::Vector3d lidar_point(*x, *y, *z);
        if (!lidar_point.allFinite()) {
          continue;
        }
        const Eigen::Vector3d body = transform * lidar_point;
        if (keep_body_point(body, config_)) {
          points.push_back(body);
        }
      }
    } catch (const std::runtime_error & error) {
      RCLCPP_ERROR_THROTTLE(
        get_logger(), *get_clock(), 2000,
        "Cannot parse PointCloud2 input: %s", error.what());
      return;
    }
    process_body_cloud(std::move(points), message->header, input_count);
  }

  void process_body_cloud(
    std::vector<Eigen::Vector3d> body_points,
    const std_msgs::msg::Header & input_header,
    std::size_t input_count)
  {
    const auto start = std::chrono::steady_clock::now();
    if (voxel_size_ > 0.0) {
      body_points = voxel_downsample(
        body_points, voxel_size_, config_.origin);
    }

    CloudFrame frame;
    frame.header = input_header;
    frame.points = std::move(body_points);
    frame.pose_valid = body_pose_at(input_header, frame.pose);
    const std::size_t cropped_count = frame.points.size();

    std::vector<double> ranges = resample_ranges(frame.points, config_);
    std::vector<double> map_ranges;
    Eigen::Isometry3d map_pose = Eigen::Isometry3d::Identity();
    bool map_pose_valid = false;
    {
      std::lock_guard<std::mutex> lock(map_ranges_mutex_);
      map_ranges = map_ranges_;
      map_pose = map_ranges_pose_;
      map_pose_valid = map_ranges_pose_valid_;
    }
    if (!map_ranges.empty()) {
      // The cache spans a wider FOV than the policy grid, so it always has
      // to be resampled down; with an unknown pose difference that is just
      // a crop back to the training window.
      const Eigen::Isometry3d previous_to_current =
        (map_pose_valid && frame.pose_valid) ?
        Eigen::Isometry3d(frame.pose.inverse() * map_pose) :
        Eigen::Isometry3d::Identity();
      map_ranges = reproject_ranges(
        map_ranges, previous_to_current, map_cache_config_, config_);
      ranges = merge_ranges(ranges, map_ranges);
    }
    ranges = fill_neighbor_ranges(ranges, config_);
    ranges = fill_max_range_above_ground(ranges, config_);
    std_msgs::msg::Header output_header = input_header;
    output_header.frame_id = target_frame_;
    const auto endpoints = ranges_to_endpoints(ranges, config_);
    publisher_->publish(
      make_xyz_cloud(
        output_header,
        static_cast<uint32_t>(config_.vertical_channels),
        static_cast<uint32_t>(horizontal_samples(config_)),
        true, endpoints));

    {
      std::lock_guard<std::mutex> lock(scan_mutex_);
      latest_scan_ = std::move(frame);
      scan_ready_ = true;
    }

    const double elapsed_ms =
      std::chrono::duration<double, std::milli>(
      std::chrono::steady_clock::now() - start).count();
    RCLCPP_INFO_THROTTLE(
      get_logger(), *get_clock(), 2000,
      "points pipeline %.1f ms (input=%zu crop=%zu map=%zu)",
      elapsed_ms, input_count, cropped_count, local_map_size_.load());
  }

  void static_map_callback(
    const sensor_msgs::msg::PointCloud2::ConstSharedPtr message)
  {
    if (!message) {
      return;
    }
    std::vector<Eigen::Vector3d> world_points;
    if (!parse_xyz_points(*message, world_points)) {
      RCLCPP_ERROR_THROTTLE(
        get_logger(), *get_clock(), 2000,
        "Static map %s must contain FLOAT32 x, y and z fields",
        static_map_topic_.c_str());
      return;
    }
    if (!(voxel_size_ > 0.0)) {
      RCLCPP_WARN_THROTTLE(
        get_logger(), *get_clock(), 5000,
        "voxel_size is 0; static layer cannot grow");
      return;
    }

    std::size_t inserted = 0U;
    {
      std::lock_guard<std::mutex> lock(static_map_mutex_);
      inserted = static_world_.insert(world_points);
    }
    static_map_messages_.fetch_add(1U);
    static_map_inserted_.fetch_add(inserted);
  }

  // Collects the accumulated world map inside the yaw-aligned ground window.
  // Returns the layer in the current body frame and fills the world-anchored
  // occupancy used to tell map points apart from fresh obstacles.
  std::vector<Eigen::Vector3d> crop_static_layer(
    const Eigen::Isometry3d & world_from_body,
    VoxelSet & occupied_world)
  {
    const Eigen::Isometry3d world_from_yaw =
      yaw_aligned_world_from_body(world_from_body);
    const Eigen::Isometry3d yaw_from_world = world_from_yaw.inverse();
    const Eigen::Isometry3d body_from_world = world_from_body.inverse();
    const double robot_world_z = world_from_body.translation().z();
    Eigen::Vector3d aabb_min;
    Eigen::Vector3d aabb_max;
    static_crop_aabb_in_world(
      world_from_yaw, config_, voxel_size_, aabb_min, aabb_max);

    std::vector<Eigen::Vector3d> candidates;
    {
      std::lock_guard<std::mutex> lock(static_map_mutex_);
      candidates = static_world_.collect_in_aabb(aabb_min, aabb_max);
    }
    if (candidates.empty()) {
      return {};
    }

    std::vector<Eigen::Vector3d> kept_world;
    kept_world.reserve(candidates.size());
    for (const auto & world : candidates) {
      const Eigen::Vector3d yaw = yaw_from_world * world;
      if (keep_static_map_point(yaw, world, robot_world_z, config_)) {
        kept_world.push_back(world);
      }
    }

    occupied_world = voxel_set_from_points(kept_world, match_voxel_size());

    std::vector<Eigen::Vector3d> static_publish;
    static_publish.reserve(kept_world.size());
    for (const auto & world : kept_world) {
      static_publish.push_back(body_from_world * world);
    }
    const double publish_voxel = publish_voxel_size();
    if (publish_voxel > 0.0) {
      static_publish = voxel_downsample(
        static_publish, publish_voxel, config_.origin);
    }
    return static_publish;
  }

  void local_map_timer_callback()
  {
    CloudFrame frame;
    {
      std::lock_guard<std::mutex> lock(scan_mutex_);
      if (!scan_ready_) {
        return;
      }
      frame = latest_scan_;
    }

    VoxelSet occupied_world;
    std::vector<Eigen::Vector3d> static_publish;
    std::size_t static_world_size = 0U;
    std::size_t static_block_count = 0U;
    {
      std::lock_guard<std::mutex> lock(static_map_mutex_);
      static_world_size = static_world_.size();
      static_block_count = static_world_.block_count();
    }
    Eigen::Isometry3d map_pose = frame.pose;
    bool stale_pose = false;
    if (!frame.pose_valid) {
      stale_pose = latest_body_pose(map_pose);
      if (stale_pose) {
        RCLCPP_WARN_THROTTLE(
          get_logger(), *get_clock(), 2000,
          "No %s at the scan stamp; aligning %s with the newest odometry",
          odom_topic_.c_str(), static_map_topic_.c_str());
      }
    }
    if (frame.pose_valid || stale_pose) {
      static_publish = crop_static_layer(map_pose, occupied_world);
    } else if (static_world_size > 0U) {
      RCLCPP_WARN_THROTTLE(
        get_logger(), *get_clock(), 2000,
        "No %s to align %s; publishing obstacle layer only",
        odom_topic_.c_str(), static_map_topic_.c_str());
    }

    std::vector<Eigen::Vector3d> obstacles = frame.points;
    if (!occupied_world.empty()) {
      obstacles = points_outside_world_voxels(
        frame.points, map_pose, occupied_world, match_voxel_size());
    }
    const double publish_voxel = publish_voxel_size();
    if (publish_voxel > 0.0) {
      obstacles = voxel_downsample(
        obstacles, publish_voxel, config_.origin);
    }

    std::vector<Eigen::Vector3d> map_points;
    map_points.reserve(static_publish.size() + obstacles.size());
    map_points.insert(
      map_points.end(), static_publish.begin(), static_publish.end());
    map_points.insert(map_points.end(), obstacles.begin(), obstacles.end());
    local_map_size_.store(map_points.size());
    const auto ranges = resample_ranges(map_points, map_cache_config_);
    {
      std::lock_guard<std::mutex> lock(map_ranges_mutex_);
      map_ranges_ = ranges;
      map_ranges_pose_ = map_pose;
      map_ranges_pose_valid_ = frame.pose_valid || stale_pose;
    }
    RCLCPP_INFO_THROTTLE(
      get_logger(), *get_clock(), 2000,
      "local map static_world=%zu blocks=%zu (%zu msgs, %zu voxels added) "
      "publish=%zu obstacles=%zu",
      static_world_size, static_block_count,
      static_map_messages_.load(), static_map_inserted_.load(),
      static_publish.size(), obstacles.size());

    if (
      !local_map_publisher_ ||
      local_map_publisher_->get_subscription_count() == 0)
    {
      return;
    }
    std_msgs::msg::Header header = frame.header;
    header.frame_id = target_frame_;
    local_map_publisher_->publish(
      make_xyz_cloud(
        header, 1U,
        static_cast<uint32_t>(map_points.size()), true, map_points));
  }

  std::string input_type_;
  std::string input_topic_;
  std::string output_topic_;
  std::string local_map_topic_;
  std::string target_frame_;
  std::string source_frame_override_;
  bool use_tf_{true};
  bool allow_static_fallback_{true};
  double tf_timeout_seconds_{0.05};
  int temporal_frames_{5};
  std::string odom_topic_;
  std::string static_map_topic_;
  double odom_timeout_seconds_{0.2};
  double output_rate_{10.0};
  double local_map_rate_{2.0};
  double voxel_size_{0.05};
  double local_map_publish_voxel_size_{0.05};
  double static_block_size_{2.0};
  double static_match_voxel_size_{0.0};
  double map_cache_margin_deg_{120.0};
  double map_cache_elevation_margin_deg_{15.0};
  int map_cache_resolution_divisor_{3};
  GridConfig map_cache_config_;
  GridConfig config_;
  Eigen::Vector3d fallback_translation_{Eigen::Vector3d::Zero()};
  Eigen::Matrix3d fallback_rotation_{Eigen::Matrix3d::Identity()};
  std::deque<StampedPose> odom_history_;
  std::chrono::steady_clock::duration output_min_interval_{};
  std::chrono::steady_clock::time_point last_output_time_{};

  std::mutex input_mutex_;
  std::mutex odom_mutex_;
  std::mutex scan_mutex_;
  std::mutex static_map_mutex_;
  std::mutex map_ranges_mutex_;
  livox_ros_driver2::msg::CustomMsg::ConstSharedPtr latest_custom_;
  sensor_msgs::msg::PointCloud2::ConstSharedPtr latest_pointcloud_;
  bool has_new_input_{false};
  CloudFrame latest_scan_;
  bool scan_ready_{false};
  StaticWorldMap static_world_;
  std::atomic<std::size_t> static_map_messages_{0};
  std::atomic<std::size_t> static_map_inserted_{0};
  std::vector<double> map_ranges_;
  Eigen::Isometry3d map_ranges_pose_{Eigen::Isometry3d::Identity()};
  bool map_ranges_pose_valid_{false};
  std::atomic<std::size_t> local_map_size_{0};

  tf2_ros::Buffer tf_buffer_;
  tf2_ros::TransformListener tf_listener_;
  rclcpp::CallbackGroup::SharedPtr subscription_callback_group_;
  rclcpp::CallbackGroup::SharedPtr output_callback_group_;
  rclcpp::CallbackGroup::SharedPtr local_map_callback_group_;
  rclcpp::CallbackGroup::SharedPtr static_map_callback_group_;
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr publisher_;
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr
    local_map_publisher_;
  rclcpp::Subscription<livox_ros_driver2::msg::CustomMsg>::SharedPtr
    custom_subscription_;
  rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr
    pointcloud_subscription_;
  rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr odom_subscription_;
  rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr
    static_map_subscription_;
  rclcpp::TimerBase::SharedPtr output_timer_;
  rclcpp::TimerBase::SharedPtr local_map_timer_;
};

}  // namespace mid360_preprocessor

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  try {
    auto node =
      std::make_shared<mid360_preprocessor::Mid360PreprocessorNode>();
    rclcpp::executors::MultiThreadedExecutor executor(
      rclcpp::ExecutorOptions(), 3U);
    executor.add_node(node);
    executor.spin();
  } catch (const std::exception & error) {
    RCLCPP_FATAL(
      rclcpp::get_logger("mid360_preprocessor"), "%s", error.what());
    rclcpp::shutdown();
    return 1;
  }
  rclcpp::shutdown();
  return 0;
}
