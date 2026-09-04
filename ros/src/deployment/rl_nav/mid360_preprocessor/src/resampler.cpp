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
#include <functional>
#include <limits>
#include <unordered_map>
#include <utility>

namespace mid360_preprocessor
{
namespace
{

constexpr double kPi = 3.14159265358979323846;

double degrees(double radians)
{
  return radians * 180.0 / kPi;
}

double radians(double degrees_value)
{
  return degrees_value * kPi / 180.0;
}

bool in_policy_range(double range, const GridConfig & config)
{
  return std::isfinite(range) &&
         range >= config.min_distance &&
         range <= config.max_distance;
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
  if (
    !std::isfinite(config.neighbor_fill_radius_deg) ||
    config.neighbor_fill_radius_deg < 0.0)
  {
    error = "neighbor_fill_radius_deg must be finite and non-negative";
    return false;
  }
  if (!std::isfinite(config.max_range_fill_height_above_ground)) {
    error = "max_range_fill_height_above_ground must be finite";
    return false;
  }
  if (!config.origin.allFinite()) {
    error = "ray origin must be finite";
    return false;
  }
  if (
    !std::isfinite(config.crop_x_min) ||
    !std::isfinite(config.crop_x_max) ||
    config.crop_x_min >= config.crop_x_max)
  {
    error = "crop_x limits must be finite and increasing";
    return false;
  }
  if (
    !std::isfinite(config.crop_y_min) ||
    !std::isfinite(config.crop_y_max) ||
    config.crop_y_min >= config.crop_y_max)
  {
    error = "crop_y limits must be finite and increasing";
    return false;
  }
  if (
    !std::isfinite(config.crop_z_min) ||
    !std::isfinite(config.crop_z_max) ||
    config.crop_z_min >= config.crop_z_max)
  {
    error = "crop_z limits must be finite and increasing";
    return false;
  }
  if (
    !std::isfinite(config.static_z_margin) ||
    config.static_z_margin < 0.0)
  {
    error = "static_z_margin must be finite and non-negative";
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

bool keep_body_point(
  const Eigen::Vector3d & point_in_target,
  const GridConfig & config)
{
  if (!point_in_target.allFinite()) {
    return false;
  }
  if (
    point_in_target.x() < config.crop_x_min ||
    point_in_target.x() > config.crop_x_max ||
    point_in_target.y() < config.crop_y_min ||
    point_in_target.y() > config.crop_y_max ||
    point_in_target.z() < config.crop_z_min ||
    point_in_target.z() > config.crop_z_max)
  {
    return false;
  }
  const double range = (point_in_target - config.origin).norm();
  return std::isfinite(range) && range >= config.min_distance;
}

bool keep_static_map_point(
  const Eigen::Vector3d & point_yaw,
  const Eigen::Vector3d & point_world,
  double robot_world_z,
  const GridConfig & config)
{
  if (
    !point_yaw.allFinite() || !point_world.allFinite() ||
    !std::isfinite(robot_world_z))
  {
    return false;
  }
  if (
    point_yaw.x() < config.crop_x_min ||
    point_yaw.x() > config.crop_x_max ||
    point_yaw.y() < config.crop_y_min ||
    point_yaw.y() > config.crop_y_max)
  {
    return false;
  }
  const double height = point_world.z() - robot_world_z;
  if (
    height < config.crop_z_min - config.static_z_margin ||
    height > config.crop_z_max)
  {
    return false;
  }
  const double range = (point_yaw - config.origin).norm();
  return std::isfinite(range) && range >= config.min_distance;
}

Eigen::Isometry3d yaw_aligned_world_from_body(
  const Eigen::Isometry3d & world_from_body)
{
  const Eigen::Vector3d x_axis = world_from_body.linear().col(0);
  const double yaw = std::atan2(x_axis.y(), x_axis.x());
  Eigen::Isometry3d world_from_yaw = Eigen::Isometry3d::Identity();
  world_from_yaw.linear() =
    Eigen::AngleAxisd(yaw, Eigen::Vector3d::UnitZ()).toRotationMatrix();
  world_from_yaw.translation() = world_from_body.translation();
  return world_from_yaw;
}

void static_crop_aabb_in_world(
  const Eigen::Isometry3d & world_from_yaw,
  const GridConfig & config,
  double margin,
  Eigen::Vector3d & aabb_min,
  Eigen::Vector3d & aabb_max)
{
  const double pad = std::max(0.0, margin);
  const double robot_z = world_from_yaw.translation().z();
  bool first = true;
  for (int ix = 0; ix < 2; ++ix) {
    for (int iy = 0; iy < 2; ++iy) {
      const Eigen::Vector3d corner(
        ix ? config.crop_x_max : config.crop_x_min,
        iy ? config.crop_y_max : config.crop_y_min,
        0.0);
      const Eigen::Vector3d world = world_from_yaw * corner;
      if (first) {
        aabb_min = world;
        aabb_max = world;
        first = false;
      } else {
        aabb_min = aabb_min.cwiseMin(world);
        aabb_max = aabb_max.cwiseMax(world);
      }
    }
  }
  const Eigen::Vector3d xy_pad(pad, pad, 0.0);
  aabb_min -= xy_pad;
  aabb_max += xy_pad;
  aabb_min.z() =
    robot_z + config.crop_z_min - config.static_z_margin - pad;
  aabb_max.z() = robot_z + config.crop_z_max + pad;
}

std::vector<Eigen::Vector3d> filter_body_points(
  const std::vector<Eigen::Vector3d> & points_in_target,
  const GridConfig & config)
{
  std::vector<Eigen::Vector3d> filtered;
  filtered.reserve(points_in_target.size());
  for (const auto & point : points_in_target) {
    if (keep_body_point(point, config)) {
      filtered.push_back(point);
    }
  }
  return filtered;
}

std::vector<Eigen::Vector3d> transform_filter_body_points(
  const std::vector<Eigen::Vector3d> & points,
  const Eigen::Isometry3d & transform,
  const GridConfig & config)
{
  std::vector<Eigen::Vector3d> filtered;
  filtered.reserve(points.size());
  for (const auto & point : points) {
    if (!point.allFinite()) {
      continue;
    }
    const Eigen::Vector3d body = transform * point;
    if (keep_body_point(body, config)) {
      filtered.push_back(body);
    }
  }
  return filtered;
}

std::size_t VoxelKeyHash::operator()(const VoxelKey & key) const
{
  std::size_t hash = std::hash<int64_t>{}(key.x);
  hash ^= std::hash<int64_t>{}(key.y) + 0x9e3779b9 + (hash << 6) +
  (hash >> 2);
  hash ^= std::hash<int64_t>{}(key.z) + 0x9e3779b9 + (hash << 6) +
  (hash >> 2);
  return hash;
}

VoxelKey voxel_key(const Eigen::Vector3d & point, double voxel_size)
{
  const double inverse_size = 1.0 / voxel_size;
  return VoxelKey{
    static_cast<int64_t>(std::floor(point.x() * inverse_size)),
    static_cast<int64_t>(std::floor(point.y() * inverse_size)),
    static_cast<int64_t>(std::floor(point.z() * inverse_size))};
}

bool point_in_aabb(
  const Eigen::Vector3d & point,
  const Eigen::Vector3d & aabb_min,
  const Eigen::Vector3d & aabb_max)
{
  return point.x() >= aabb_min.x() && point.x() <= aabb_max.x() &&
         point.y() >= aabb_min.y() && point.y() <= aabb_max.y() &&
         point.z() >= aabb_min.z() && point.z() <= aabb_max.z();
}

void insert_voxel(
  std::unordered_map<VoxelKey, Eigen::Vector3d, VoxelKeyHash> & voxels,
  const Eigen::Vector3d & point,
  double voxel_size,
  const Eigen::Vector3d & origin)
{
  if (!point.allFinite() || !(voxel_size > 0.0)) {
    return;
  }
  const auto inserted = voxels.emplace(voxel_key(point, voxel_size), point);
  if (!inserted.second) {
    const double current =
      (inserted.first->second - origin).squaredNorm();
    const double candidate = (point - origin).squaredNorm();
    if (candidate < current) {
      inserted.first->second = point;
    }
  }
}

std::unordered_map<VoxelKey, Eigen::Vector3d, VoxelKeyHash> voxels_from_points(
  const std::vector<Eigen::Vector3d> & points,
  double voxel_size,
  const Eigen::Vector3d & origin)
{
  std::unordered_map<VoxelKey, Eigen::Vector3d, VoxelKeyHash> voxels;
  if (!(voxel_size > 0.0)) {
    return voxels;
  }
  voxels.reserve(points.size());
  for (const auto & point : points) {
    insert_voxel(voxels, point, voxel_size, origin);
  }
  return voxels;
}

std::vector<Eigen::Vector3d> voxel_downsample(
  const std::vector<Eigen::Vector3d> & points,
  double voxel_size,
  const Eigen::Vector3d & origin)
{
  if (!(voxel_size > 0.0) || points.size() < 2U) {
    return points;
  }

  const auto voxels = voxels_from_points(points, voxel_size, origin);
  std::vector<Eigen::Vector3d> downsampled;
  downsampled.reserve(voxels.size());
  for (const auto & voxel : voxels) {
    downsampled.push_back(voxel.second);
  }
  return downsampled;
}

std::vector<Eigen::Vector3d> points_outside_voxels(
  const std::vector<Eigen::Vector3d> & points,
  const std::unordered_map<VoxelKey, Eigen::Vector3d, VoxelKeyHash> & occupied,
  double voxel_size)
{
  if (!(voxel_size > 0.0) || occupied.empty()) {
    return points;
  }

  std::vector<Eigen::Vector3d> outliers;
  outliers.reserve(points.size());
  for (const auto & point : points) {
    if (!point.allFinite()) {
      continue;
    }
    if (occupied.find(voxel_key(point, voxel_size)) == occupied.end()) {
      outliers.push_back(point);
    }
  }
  return outliers;
}

VoxelSet voxel_set_from_points(
  const std::vector<Eigen::Vector3d> & points,
  double voxel_size)
{
  VoxelSet voxels;
  if (!(voxel_size > 0.0)) {
    return voxels;
  }
  voxels.reserve(points.size());
  for (const auto & point : points) {
    if (!point.allFinite()) {
      continue;
    }
    voxels.insert(voxel_key(point, voxel_size));
  }
  return voxels;
}

std::vector<Eigen::Vector3d> points_outside_world_voxels(
  const std::vector<Eigen::Vector3d> & points_in_body,
  const Eigen::Isometry3d & world_from_body,
  const VoxelSet & occupied_world,
  double voxel_size)
{
  if (!(voxel_size > 0.0) || occupied_world.empty()) {
    return points_in_body;
  }

  std::vector<Eigen::Vector3d> outliers;
  outliers.reserve(points_in_body.size());
  for (const auto & point : points_in_body) {
    if (!point.allFinite()) {
      continue;
    }
    const Eigen::Vector3d world = world_from_body * point;
    if (occupied_world.find(voxel_key(world, voxel_size)) ==
      occupied_world.end())
    {
      outliers.push_back(point);
    }
  }
  return outliers;
}

bool StaticWorldMap::configure(double voxel_size, double block_size)
{
  if (
    !std::isfinite(voxel_size) || !(voxel_size > 0.0) ||
    !std::isfinite(block_size) || block_size < voxel_size)
  {
    return false;
  }
  if (voxel_size != voxel_size_ || block_size != block_size_) {
    clear();
  }
  voxel_size_ = voxel_size;
  block_size_ = block_size;
  return true;
}

void StaticWorldMap::clear()
{
  blocks_.clear();
  size_ = 0U;
}

std::size_t StaticWorldMap::insert(
  const std::vector<Eigen::Vector3d> & world_points)
{
  if (!(voxel_size_ > 0.0) || !(block_size_ > 0.0)) {
    return 0U;
  }
  std::size_t inserted = 0U;
  for (const auto & point : world_points) {
    if (!point.allFinite()) {
      continue;
    }
    auto & block = blocks_[voxel_key(point, block_size_)];
    if (block.emplace(voxel_key(point, voxel_size_), point).second) {
      ++inserted;
    }
  }
  size_ += inserted;
  return inserted;
}

std::vector<Eigen::Vector3d> StaticWorldMap::collect_in_aabb(
  const Eigen::Vector3d & aabb_min,
  const Eigen::Vector3d & aabb_max) const
{
  std::vector<Eigen::Vector3d> points;
  if (
    blocks_.empty() || !(block_size_ > 0.0) ||
    !aabb_min.allFinite() || !aabb_max.allFinite())
  {
    return points;
  }

  const VoxelKey first = voxel_key(aabb_min, block_size_);
  const VoxelKey last = voxel_key(aabb_max, block_size_);
  if (last.x < first.x || last.y < first.y || last.z < first.z) {
    return points;
  }
  const auto span = static_cast<double>(last.x - first.x + 1) *
    static_cast<double>(last.y - first.y + 1) *
    static_cast<double>(last.z - first.z + 1);

  points.reserve(4096U);
  const auto collect_block = [&](const Block & block) {
      for (const auto & voxel : block) {
        if (point_in_aabb(voxel.second, aabb_min, aabb_max)) {
          points.push_back(voxel.second);
        }
      }
    };

  // Scanning every block is cheaper once the window spans more blocks than
  // the map holds, which happens for degenerate or very large windows.
  if (span > static_cast<double>(blocks_.size())) {
    for (const auto & block : blocks_) {
      collect_block(block.second);
    }
    return points;
  }

  for (int64_t x = first.x; x <= last.x; ++x) {
    for (int64_t y = first.y; y <= last.y; ++y) {
      for (int64_t z = first.z; z <= last.z; ++z) {
        const auto block = blocks_.find(VoxelKey{x, y, z});
        if (block != blocks_.end()) {
          collect_block(block->second);
        }
      }
    }
  }
  return points;
}

std::size_t StaticWorldMap::size() const
{
  return size_;
}

std::size_t StaticWorldMap::block_count() const
{
  return blocks_.size();
}

bool StaticWorldMap::empty() const
{
  return size_ == 0U;
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
    if (!std::isfinite(range) || range < config.min_distance) {
      continue;
    }

    const double azimuth = degrees(std::atan2(relative.y(), relative.x()));
    const double elevation = degrees(
      std::atan2(
        relative.z(), std::hypot(relative.x(), relative.y())));
    // No explicit FOV guard here: the index range and the half-step centre
    // check below own that job exactly. Clamping to the nominal FOV first
    // would leave the edge bins only half as wide as the interior ones, so
    // the bottom row and the +-60 deg columns would catch half the returns.
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
    if (range > config.max_distance) {
      if (!std::isfinite(ranges[index])) {
        ranges[index] = std::numeric_limits<double>::infinity();
      }
      continue;
    }
    if (!in_policy_range(ranges[index], config) || range < ranges[index]) {
      ranges[index] = range;
    }
  }
  return ranges;
}

std::vector<double> fill_neighbor_ranges(
  const std::vector<double> & ranges,
  const GridConfig & config)
{
  const std::size_t width = horizontal_samples(config);
  const std::size_t height =
    static_cast<std::size_t>(config.vertical_channels);
  if (ranges.size() != width * height) {
    return ranges;
  }
  if (config.neighbor_fill_radius_deg <= 0.0) {
    return ranges;
  }

  const double vertical_step =
    (config.vertical_max_deg - config.vertical_min_deg) /
    static_cast<double>(config.vertical_channels - 1);
  const int k_horizontal = static_cast<int>(std::ceil(
      config.neighbor_fill_radius_deg / config.horizontal_resolution_deg -
      1.0e-12));
  const int k_vertical = static_cast<int>(std::ceil(
      config.neighbor_fill_radius_deg / vertical_step - 1.0e-12));
  if (k_horizontal <= 0 && k_vertical <= 0) {
    return ranges;
  }

  const double radius_sq =
    config.neighbor_fill_radius_deg * config.neighbor_fill_radius_deg;
  std::vector<double> filled = ranges;
  for (int vertical_index = 0;
    vertical_index < static_cast<int>(height); ++vertical_index)
  {
    for (int horizontal_index = 0;
      horizontal_index < static_cast<int>(width); ++horizontal_index)
    {
      const std::size_t index =
        static_cast<std::size_t>(vertical_index) * width +
        static_cast<std::size_t>(horizontal_index);
      if (in_policy_range(ranges[index], config) ||
        std::isinf(ranges[index]))
      {
        continue;
      }

      double best_angle_sq = std::numeric_limits<double>::infinity();
      double best_range = std::numeric_limits<double>::quiet_NaN();
      for (int delta_v = -k_vertical; delta_v <= k_vertical; ++delta_v) {
        for (int delta_h = -k_horizontal; delta_h <= k_horizontal; ++delta_h) {
          if (delta_v == 0 && delta_h == 0) {
            continue;
          }
          const int neighbor_v = vertical_index + delta_v;
          const int neighbor_h = horizontal_index + delta_h;
          if (
            neighbor_v < 0 || neighbor_v >= static_cast<int>(height) ||
            neighbor_h < 0 || neighbor_h >= static_cast<int>(width))
          {
            continue;
          }
          const double angle_sq =
            static_cast<double>(delta_v) * vertical_step *
            static_cast<double>(delta_v) * vertical_step +
            static_cast<double>(delta_h) * config.horizontal_resolution_deg *
            static_cast<double>(delta_h) * config.horizontal_resolution_deg;
          if (angle_sq > radius_sq + 1.0e-9) {
            continue;
          }
          const double candidate = ranges[
            static_cast<std::size_t>(neighbor_v) * width +
            static_cast<std::size_t>(neighbor_h)];
          if (!in_policy_range(candidate, config)) {
            continue;
          }
          if (
            angle_sq + 1.0e-9 < best_angle_sq ||
            (std::abs(angle_sq - best_angle_sq) <= 1.0e-9 &&
            candidate < best_range))
          {
            best_angle_sq = angle_sq;
            best_range = candidate;
          }
        }
      }
      if (in_policy_range(best_range, config)) {
        filled[index] = best_range;
      }
    }
  }
  return filled;
}

std::vector<double> fill_max_range_above_ground(
  const std::vector<double> & ranges,
  const GridConfig & config)
{
  const std::size_t width = horizontal_samples(config);
  const std::size_t height =
    static_cast<std::size_t>(config.vertical_channels);
  if (
    ranges.size() != width * height ||
    config.max_range_fill_height_above_ground < 0.0)
  {
    return ranges;
  }

  const double vertical_step =
    (config.vertical_max_deg - config.vertical_min_deg) /
    static_cast<double>(config.vertical_channels - 1);
  const double min_z =
    config.crop_z_min + config.max_range_fill_height_above_ground;
  std::vector<double> filled = ranges;
  for (std::size_t vertical_index = 0; vertical_index < height;
    ++vertical_index)
  {
    const double elevation = radians(
      config.vertical_min_deg +
      static_cast<double>(vertical_index) * vertical_step);
    const double endpoint_z =
      config.origin.z() + config.max_distance * std::sin(elevation);
    if (endpoint_z + 1.0e-9 < min_z) {
      continue;
    }
    for (std::size_t horizontal_index = 0; horizontal_index < width;
      ++horizontal_index)
    {
      const std::size_t index = vertical_index * width + horizontal_index;
      if (!in_policy_range(filled[index], config)) {
        filled[index] = config.max_distance;
      }
    }
  }
  return filled;
}

std::vector<double> merge_ranges(
  const std::vector<double> & first,
  const std::vector<double> & second)
{
  if (first.size() != second.size()) {
    return first;
  }
  std::vector<double> merged = first;
  for (std::size_t index = 0; index < merged.size(); ++index) {
    const double first_range = merged[index];
    const double second_range = second[index];
    const bool first_ok = std::isfinite(first_range);
    const bool second_ok = std::isfinite(second_range);
    if (first_ok && second_ok) {
      if (second_range < first_range) {
        merged[index] = second_range;
      }
    } else if (!first_ok && second_ok) {
      merged[index] = second_range;
    } else if (!first_ok && !second_ok) {
      if (std::isinf(first_range) || std::isinf(second_range)) {
        merged[index] = std::numeric_limits<double>::infinity();
      }
    }
  }
  return merged;
}

std::vector<Eigen::Vector3f> ranges_to_endpoints(
  const std::vector<double> & ranges,
  const GridConfig & config)
{
  const std::size_t width = horizontal_samples(config);
  const std::size_t height =
    static_cast<std::size_t>(config.vertical_channels);
  std::vector<Eigen::Vector3f> endpoints(
    width * height, Eigen::Vector3f::Zero());
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
      if (!in_policy_range(ranges[index], config)) {
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

GridConfig expand_grid_config(
  const GridConfig & config,
  double azimuth_margin_deg,
  double elevation_margin_deg,
  int resolution_divisor)
{
  GridConfig expanded = config;
  if (config.vertical_channels < 2 ||
    !(config.horizontal_resolution_deg > 0.0))
  {
    return expanded;
  }

  // Grow by whole steps so the extra bins stay centred on the training
  // angles; cropping back to the training grid is then exact. Azimuth stops
  // at a full circle and elevation at the poles, because atan2 never reports
  // anything beyond that and the extra bins could never be filled.
  const auto whole_steps = [](double margin, double step, double headroom) {
      if (!std::isfinite(margin) || margin <= 0.0 || !(step > 0.0)) {
        return 0;
      }
      const auto wanted = static_cast<int>(std::ceil(margin / step));
      const auto available = static_cast<int>(std::floor(headroom / step));
      return std::max(0, std::min(wanted, available));
    };

  const int columns = whole_steps(
    azimuth_margin_deg, config.horizontal_resolution_deg,
    std::min(
      config.horizontal_min_deg + 180.0, 180.0 - config.horizontal_max_deg));
  expanded.horizontal_min_deg -=
    static_cast<double>(columns) * config.horizontal_resolution_deg;
  expanded.horizontal_max_deg +=
    static_cast<double>(columns) * config.horizontal_resolution_deg;

  const double vertical_step =
    (config.vertical_max_deg - config.vertical_min_deg) /
    static_cast<double>(config.vertical_channels - 1);
  const int rows = whole_steps(
    elevation_margin_deg, vertical_step,
    std::min(
      config.vertical_min_deg + 90.0, 90.0 - config.vertical_max_deg));
  expanded.vertical_min_deg -= static_cast<double>(rows) * vertical_step;
  expanded.vertical_max_deg += static_cast<double>(rows) * vertical_step;

  // Subdividing the step keeps every training centre on the lattice while
  // adding samples in between. Without it a large rotation stretches the
  // spacing between reprojected samples past one training step and whole
  // columns come out empty.
  const int divisor = std::max(1, resolution_divisor);
  expanded.horizontal_resolution_deg =
    config.horizontal_resolution_deg / static_cast<double>(divisor);
  expanded.vertical_channels =
    (config.vertical_channels - 1 + 2 * rows) * divisor + 1;
  return expanded;
}

std::vector<double> reproject_ranges(
  const std::vector<double> & ranges,
  const Eigen::Isometry3d & previous_to_current,
  const GridConfig & source_config,
  const GridConfig & target_config)
{
  const std::size_t width = horizontal_samples(source_config);
  const std::size_t height =
    static_cast<std::size_t>(source_config.vertical_channels);
  if (ranges.size() != width * height || source_config.vertical_channels < 2) {
    return ranges;
  }

  const double vertical_step =
    (source_config.vertical_max_deg - source_config.vertical_min_deg) /
    static_cast<double>(source_config.vertical_channels - 1);
  std::vector<Eigen::Vector3d> points;
  points.reserve(ranges.size());
  for (std::size_t vertical_index = 0; vertical_index < height;
    ++vertical_index)
  {
    const double elevation = radians(
      source_config.vertical_min_deg +
      static_cast<double>(vertical_index) * vertical_step);
    const double cos_elevation = std::cos(elevation);
    for (std::size_t horizontal_index = 0; horizontal_index < width;
      ++horizontal_index)
    {
      const std::size_t index = vertical_index * width + horizontal_index;
      if (!in_policy_range(ranges[index], source_config)) {
        continue;
      }
      const double azimuth = radians(
        source_config.horizontal_min_deg +
        static_cast<double>(horizontal_index) *
        source_config.horizontal_resolution_deg);
      const Eigen::Vector3d direction(
        cos_elevation * std::cos(azimuth),
        cos_elevation * std::sin(azimuth),
        std::sin(elevation));
      points.push_back(
        previous_to_current *
        (source_config.origin + ranges[index] * direction));
    }
  }
  return resample_ranges(points, target_config);
}

std::vector<double> reproject_ranges(
  const std::vector<double> & ranges,
  const Eigen::Isometry3d & previous_to_current,
  const GridConfig & config)
{
  return reproject_ranges(ranges, previous_to_current, config, config);
}

}  // namespace mid360_preprocessor
