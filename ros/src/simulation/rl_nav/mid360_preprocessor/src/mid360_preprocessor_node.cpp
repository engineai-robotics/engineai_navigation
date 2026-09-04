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
#include <chrono>
#include <cmath>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <geometry_msgs/msg/transform_stamped.hpp>
#include <livox_ros_driver2/msg/custom_msg.hpp>
#include "mid360_preprocessor/resampler.hpp"
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
    config_.origin = vector3_parameter(
      *this, "ray_origin", {0.105, 0.0, 0.185});

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
      !std::isfinite(tf_timeout_seconds_) || tf_timeout_seconds_ < 0.0)
    {
      throw std::invalid_argument("topics, target_frame and tf_timeout must be valid");
    }

    publisher_ = create_publisher<sensor_msgs::msg::PointCloud2>(
      output_topic_, rclcpp::SensorDataQoS());
    if (input_type_ == "custom_msg") {
      custom_subscription_ =
        create_subscription<livox_ros_driver2::msg::CustomMsg>(
        input_topic_, rclcpp::SensorDataQoS(),
        std::bind(
          &Mid360PreprocessorNode::custom_callback, this,
          std::placeholders::_1));
    } else {
      pointcloud_subscription_ =
        create_subscription<sensor_msgs::msg::PointCloud2>(
        input_topic_, rclcpp::SensorDataQoS(),
        std::bind(
          &Mid360PreprocessorNode::pointcloud_callback, this,
          std::placeholders::_1));
    }

    RCLCPP_INFO(
      get_logger(),
      "Resampling %s on %s to %dx%zu PointCloud2 on %s in frame %s",
      input_type_.c_str(), input_topic_.c_str(), config_.vertical_channels,
      horizontal_samples(config_), output_topic_.c_str(), target_frame_.c_str());
  }

private:
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
    bool transform_ok = false;
    const Eigen::Isometry3d transform =
      source_to_target(message->header, transform_ok);
    if (!transform_ok) {
      return;
    }
    std::vector<Eigen::Vector3d> points;
    points.reserve(message->points.size());
    for (const auto & point : message->points) {
      points.emplace_back(point.x, point.y, point.z);
    }
    process(transform_points(points, transform), message->header);
  }

  void pointcloud_callback(
    const sensor_msgs::msg::PointCloud2::ConstSharedPtr message)
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
    points.reserve(
      static_cast<std::size_t>(message->width) * message->height);
    try {
      sensor_msgs::PointCloud2ConstIterator<float> x(*message, "x");
      sensor_msgs::PointCloud2ConstIterator<float> y(*message, "y");
      sensor_msgs::PointCloud2ConstIterator<float> z(*message, "z");
      for (; x != x.end(); ++x, ++y, ++z) {
        points.emplace_back(*x, *y, *z);
      }
    } catch (const std::runtime_error & error) {
      RCLCPP_ERROR_THROTTLE(
        get_logger(), *get_clock(), 2000,
        "Cannot parse PointCloud2 input: %s", error.what());
      return;
    }
    process(transform_points(points, transform), message->header);
  }

  void process(
    const std::vector<Eigen::Vector3d> & points,
    const std_msgs::msg::Header & input_header)
  {
    const auto ranges = resample_ranges(points, config_);
    const auto endpoints = ranges_to_endpoints(ranges, config_);

    sensor_msgs::msg::PointCloud2 output;
    output.header = input_header;
    output.header.frame_id = target_frame_;
    sensor_msgs::PointCloud2Modifier modifier(output);
    modifier.setPointCloud2FieldsByString(1, "xyz");
    modifier.resize(endpoints.size());
    output.height = static_cast<uint32_t>(config_.vertical_channels);
    output.width = static_cast<uint32_t>(horizontal_samples(config_));
    output.row_step = output.point_step * output.width;
    output.is_dense = false;

    sensor_msgs::PointCloud2Iterator<float> x(output, "x");
    sensor_msgs::PointCloud2Iterator<float> y(output, "y");
    sensor_msgs::PointCloud2Iterator<float> z(output, "z");
    for (const auto & endpoint : endpoints) {
      *x = endpoint.x();
      *y = endpoint.y();
      *z = endpoint.z();
      ++x;
      ++y;
      ++z;
    }
    publisher_->publish(std::move(output));
  }

  std::string input_type_;
  std::string input_topic_;
  std::string output_topic_;
  std::string target_frame_;
  std::string source_frame_override_;
  bool use_tf_{true};
  bool allow_static_fallback_{true};
  double tf_timeout_seconds_{0.05};
  GridConfig config_;
  Eigen::Vector3d fallback_translation_{Eigen::Vector3d::Zero()};
  Eigen::Matrix3d fallback_rotation_{Eigen::Matrix3d::Identity()};

  tf2_ros::Buffer tf_buffer_;
  tf2_ros::TransformListener tf_listener_;
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr publisher_;
  rclcpp::Subscription<livox_ros_driver2::msg::CustomMsg>::SharedPtr
    custom_subscription_;
  rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr
    pointcloud_subscription_;
};

}  // namespace mid360_preprocessor

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  try {
    rclcpp::spin(
      std::make_shared<mid360_preprocessor::Mid360PreprocessorNode>());
  } catch (const std::exception & error) {
    RCLCPP_FATAL(
      rclcpp::get_logger("mid360_preprocessor"), "%s", error.what());
    rclcpp::shutdown();
    return 1;
  }
  rclcpp::shutdown();
  return 0;
}
