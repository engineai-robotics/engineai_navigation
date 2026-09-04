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
#include <string>
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
  Eigen::Vector3d origin{0.105, 0.0, 0.185};
};

bool validate_config(const GridConfig & config, std::string & error);

std::size_t horizontal_samples(const GridConfig & config);

std::vector<Eigen::Vector3d> transform_points(
  const std::vector<Eigen::Vector3d> & points,
  const Eigen::Isometry3d & transform);

std::vector<double> resample_ranges(
  const std::vector<Eigen::Vector3d> & points_in_target,
  const GridConfig & config);

std::vector<Eigen::Vector3f> ranges_to_endpoints(
  const std::vector<double> & ranges,
  const GridConfig & config);

}  // namespace mid360_preprocessor

#endif  // MID360_PREPROCESSOR__RESAMPLER_HPP_
