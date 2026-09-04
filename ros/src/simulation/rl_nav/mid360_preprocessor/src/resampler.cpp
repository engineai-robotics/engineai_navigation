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

#include "mid360_preprocessor/resampler.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>

namespace mid360_preprocessor
{
namespace
{

constexpr double kPi = 3.14159265358979323846;
constexpr double kAngleToleranceDeg = 1.0e-9;

double degrees(double radians)
{
  return radians * 180.0 / kPi;
}

double radians(double degrees_value)
{
  return degrees_value * kPi / 180.0;
}

}  // namespace

bool validate_config(const GridConfig & config, std::string & error)
{
  if (config.vertical_channels < 2) {
    error = "vertical_channels must be at least 2";
    return false;
  }
  if (
    !std::isfinite(config.vertical_min_deg) ||
    !std::isfinite(config.vertical_max_deg) ||
    config.vertical_min_deg >= config.vertical_max_deg)
  {
    error = "vertical FOV must be finite and increasing";
    return false;
  }
  if (
    !std::isfinite(config.horizontal_min_deg) ||
    !std::isfinite(config.horizontal_max_deg) ||
    config.horizontal_min_deg >= config.horizontal_max_deg)
  {
    error = "horizontal FOV must be finite and increasing";
    return false;
  }
  if (
    !std::isfinite(config.horizontal_resolution_deg) ||
    config.horizontal_resolution_deg <= 0.0)
  {
    error = "horizontal_resolution_deg must be finite and positive";
    return false;
  }
  if (
    !std::isfinite(config.min_distance) ||
    !std::isfinite(config.max_distance) ||
    config.min_distance < 0.0 ||
    config.min_distance >= config.max_distance)
  {
    error = "distance limits must be finite and satisfy 0 <= min < max";
    return false;
  }
  if (!config.origin.allFinite()) {
    error = "ray origin must be finite";
    return false;
  }
  error.clear();
  return true;
}

std::size_t horizontal_samples(const GridConfig & config)
{
  const double span =
    config.horizontal_max_deg - config.horizontal_min_deg;
  return static_cast<std::size_t>(
    std::ceil(span / config.horizontal_resolution_deg)) + 1U;
}

std::vector<Eigen::Vector3d> transform_points(
  const std::vector<Eigen::Vector3d> & points,
  const Eigen::Isometry3d & transform)
{
  std::vector<Eigen::Vector3d> transformed;
  transformed.reserve(points.size());
  for (const auto & point : points) {
    if (point.allFinite()) {
      transformed.emplace_back(transform * point);
    }
  }
  return transformed;
}

std::vector<double> resample_ranges(
  const std::vector<Eigen::Vector3d> & points_in_target,
  const GridConfig & config)
{
  const std::size_t width = horizontal_samples(config);
  const std::size_t height =
    static_cast<std::size_t>(config.vertical_channels);
  std::vector<double> ranges(
    width * height, std::numeric_limits<double>::quiet_NaN());
  const double vertical_step =
    (config.vertical_max_deg - config.vertical_min_deg) /
    static_cast<double>(config.vertical_channels - 1);

  for (const auto & point : points_in_target) {
    if (!point.allFinite()) {
      continue;
    }
    const Eigen::Vector3d relative = point - config.origin;
    const double range = relative.norm();
    if (
      !std::isfinite(range) ||
      range < config.min_distance ||
      range > config.max_distance)
    {
      continue;
    }

    const double azimuth = degrees(std::atan2(relative.y(), relative.x()));
    const double elevation = degrees(
      std::atan2(
        relative.z(), std::hypot(relative.x(), relative.y())));
    if (
      azimuth < config.horizontal_min_deg - kAngleToleranceDeg ||
      azimuth > config.horizontal_max_deg + kAngleToleranceDeg ||
      elevation < config.vertical_min_deg - kAngleToleranceDeg ||
      elevation > config.vertical_max_deg + kAngleToleranceDeg)
    {
      continue;
    }

    const auto horizontal_index = static_cast<int64_t>(std::llround(
        (azimuth - config.horizontal_min_deg) /
        config.horizontal_resolution_deg));
    const auto vertical_index = static_cast<int64_t>(std::llround(
        (elevation - config.vertical_min_deg) / vertical_step));
    if (
      horizontal_index < 0 ||
      horizontal_index >= static_cast<int64_t>(width) ||
      vertical_index < 0 ||
      vertical_index >= static_cast<int64_t>(height))
    {
      continue;
    }

    const double horizontal_center =
      config.horizontal_min_deg +
      static_cast<double>(horizontal_index) *
      config.horizontal_resolution_deg;
    const double vertical_center =
      config.vertical_min_deg +
      static_cast<double>(vertical_index) * vertical_step;
    if (
      horizontal_center > config.horizontal_max_deg + 1.0e-9 ||
      std::abs(azimuth - horizontal_center) >
      0.5 * config.horizontal_resolution_deg + 1.0e-9 ||
      std::abs(elevation - vertical_center) >
      0.5 * vertical_step + 1.0e-9)
    {
      continue;
    }

    const std::size_t index =
      static_cast<std::size_t>(vertical_index) * width +
      static_cast<std::size_t>(horizontal_index);
    if (!std::isfinite(ranges[index]) || range < ranges[index]) {
      ranges[index] = range;
    }
  }
  return ranges;
}

std::vector<Eigen::Vector3f> ranges_to_endpoints(
  const std::vector<double> & ranges,
  const GridConfig & config)
{
  const std::size_t width = horizontal_samples(config);
  const std::size_t height =
    static_cast<std::size_t>(config.vertical_channels);
  const float nan = std::numeric_limits<float>::quiet_NaN();
  std::vector<Eigen::Vector3f> endpoints(
    width * height, Eigen::Vector3f(nan, nan, nan));
  if (ranges.size() != endpoints.size()) {
    return endpoints;
  }

  const double vertical_step =
    (config.vertical_max_deg - config.vertical_min_deg) /
    static_cast<double>(config.vertical_channels - 1);
  for (std::size_t vertical_index = 0; vertical_index < height; ++vertical_index) {
    const double elevation = radians(
      config.vertical_min_deg +
      static_cast<double>(vertical_index) * vertical_step);
    const double cos_elevation = std::cos(elevation);
    for (std::size_t horizontal_index = 0; horizontal_index < width;
      ++horizontal_index)
    {
      const std::size_t index = vertical_index * width + horizontal_index;
      if (!std::isfinite(ranges[index])) {
        continue;
      }
      const double azimuth = radians(
        config.horizontal_min_deg +
        static_cast<double>(horizontal_index) *
        config.horizontal_resolution_deg);
      const Eigen::Vector3d direction(
        cos_elevation * std::cos(azimuth),
        cos_elevation * std::sin(azimuth),
        std::sin(elevation));
      endpoints[index] =
        (config.origin + ranges[index] * direction).cast<float>();
    }
  }
  return endpoints;
}

}  // namespace mid360_preprocessor
