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

#ifndef MID360_PREPROCESSOR__RESAMPLER_HPP_
#define MID360_PREPROCESSOR__RESAMPLER_HPP_

#include <Eigen/Core>
#include <Eigen/Geometry>

#include <cstddef>
#include <cstdint>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace mid360_preprocessor
{

struct GridConfig
{
  int vertical_channels{16};
  double vertical_min_deg{-45.0};
  double vertical_max_deg{0.0};
  double horizontal_min_deg{-60.0};
  double horizontal_max_deg{60.0};
  double horizontal_resolution_deg{3.0};
  double min_distance{0.1};
  double max_distance{5.0};
  double neighbor_fill_radius_deg{3.0};
  double max_range_fill_height_above_ground{0.1};
  Eigen::Vector3d origin{0.105, 0.0, 0.185};
  double crop_x_min{-7.0};
  double crop_x_max{7.0};
  double crop_y_min{-7.0};
  double crop_y_max{7.0};
  double crop_z_min{-0.8};
  double crop_z_max{0.7};
  double static_z_margin{0.6};
};

bool validate_config(const GridConfig & config, std::string & error);

std::size_t horizontal_samples(const GridConfig & config);

std::vector<Eigen::Vector3d> transform_points(
  const std::vector<Eigen::Vector3d> & points,
  const Eigen::Isometry3d & transform);

bool keep_body_point(
  const Eigen::Vector3d & point_in_target,
  const GridConfig & config);

bool keep_static_map_point(
  const Eigen::Vector3d & point_yaw,
  const Eigen::Vector3d & point_world,
  double robot_world_z,
  const GridConfig & config);

Eigen::Isometry3d yaw_aligned_world_from_body(
  const Eigen::Isometry3d & world_from_body);

void static_crop_aabb_in_world(
  const Eigen::Isometry3d & world_from_yaw,
  const GridConfig & config,
  double margin,
  Eigen::Vector3d & aabb_min,
  Eigen::Vector3d & aabb_max);

std::vector<Eigen::Vector3d> filter_body_points(
  const std::vector<Eigen::Vector3d> & points_in_target,
  const GridConfig & config);

std::vector<Eigen::Vector3d> transform_filter_body_points(
  const std::vector<Eigen::Vector3d> & points,
  const Eigen::Isometry3d & transform,
  const GridConfig & config);

struct VoxelKey
{
  int64_t x{0};
  int64_t y{0};
  int64_t z{0};

  bool operator==(const VoxelKey & other) const
  {
    return x == other.x && y == other.y && z == other.z;
  }
};

struct VoxelKeyHash
{
  std::size_t operator()(const VoxelKey & key) const;
};

VoxelKey voxel_key(const Eigen::Vector3d & point, double voxel_size);

bool point_in_aabb(
  const Eigen::Vector3d & point,
  const Eigen::Vector3d & aabb_min,
  const Eigen::Vector3d & aabb_max);

void insert_voxel(
  std::unordered_map<VoxelKey, Eigen::Vector3d, VoxelKeyHash> & voxels,
  const Eigen::Vector3d & point,
  double voxel_size,
  const Eigen::Vector3d & origin);

std::unordered_map<VoxelKey, Eigen::Vector3d, VoxelKeyHash> voxels_from_points(
  const std::vector<Eigen::Vector3d> & points,
  double voxel_size,
  const Eigen::Vector3d & origin);

std::vector<Eigen::Vector3d> voxel_downsample(
  const std::vector<Eigen::Vector3d> & points,
  double voxel_size,
  const Eigen::Vector3d & origin);

std::vector<Eigen::Vector3d> points_outside_voxels(
  const std::vector<Eigen::Vector3d> & points,
  const std::unordered_map<VoxelKey, Eigen::Vector3d, VoxelKeyHash> & occupied,
  double voxel_size);

using VoxelSet = std::unordered_set<VoxelKey, VoxelKeyHash>;

VoxelSet voxel_set_from_points(
  const std::vector<Eigen::Vector3d> & points,
  double voxel_size);

std::vector<Eigen::Vector3d> points_outside_world_voxels(
  const std::vector<Eigen::Vector3d> & points_in_body,
  const Eigen::Isometry3d & world_from_body,
  const VoxelSet & occupied_world,
  double voxel_size);

// Insert-only world map of the Super-LIO cloud. Voxels are grouped into
// coarse blocks so that cropping a local window costs the window size
// instead of the whole map.
class StaticWorldMap
{
public:
  bool configure(double voxel_size, double block_size);

  void clear();

  std::size_t insert(const std::vector<Eigen::Vector3d> & world_points);

  std::vector<Eigen::Vector3d> collect_in_aabb(
    const Eigen::Vector3d & aabb_min,
    const Eigen::Vector3d & aabb_max) const;

  std::size_t size() const;

  std::size_t block_count() const;

  bool empty() const;

private:
  using Block = std::unordered_map<VoxelKey, Eigen::Vector3d, VoxelKeyHash>;

  std::unordered_map<VoxelKey, Block, VoxelKeyHash> blocks_;
  std::size_t size_{0};
  double voxel_size_{0.0};
  double block_size_{0.0};
};

std::vector<double> resample_ranges(
  const std::vector<Eigen::Vector3d> & points_in_target,
  const GridConfig & config);

std::vector<double> fill_neighbor_ranges(
  const std::vector<double> & ranges,
  const GridConfig & config);

std::vector<double> fill_max_range_above_ground(
  const std::vector<double> & ranges,
  const GridConfig & config);

std::vector<double> merge_ranges(
  const std::vector<double> & first,
  const std::vector<double> & second);

std::vector<Eigen::Vector3f> ranges_to_endpoints(
  const std::vector<double> & ranges,
  const GridConfig & config);

// Grows the FOV by whole grid steps and optionally subdivides each step, so
// that a cached range image keeps context outside the training window and
// enough angular density to survive being re-binned after a rotation.
// Training bin centres stay on the returned lattice, so cropping back is
// exact. resolution_divisor of 1 keeps the training resolution.
GridConfig expand_grid_config(
  const GridConfig & config,
  double azimuth_margin_deg,
  double elevation_margin_deg,
  int resolution_divisor = 1);

std::vector<double> reproject_ranges(
  const std::vector<double> & ranges,
  const Eigen::Isometry3d & previous_to_current,
  const GridConfig & source_config,
  const GridConfig & target_config);

std::vector<double> reproject_ranges(
  const std::vector<double> & ranges,
  const Eigen::Isometry3d & previous_to_current,
  const GridConfig & config);

}  // namespace mid360_preprocessor

#endif  // MID360_PREPROCESSOR__RESAMPLER_HPP_
