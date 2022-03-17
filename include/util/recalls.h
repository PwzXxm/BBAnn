#pragma once

#include <algorithm>
#include <cassert>
#include <fstream>
#include <numeric>
#include <set>
#include <string>
#include <tuple>
#include <vector>

namespace bbann {
namespace util {

std::tuple<std::vector<int32_t>, std::vector<float>, std::vector<int32_t>>
read_gt_file(const std::string gt_file) {
  int32_t nq, total_res;

  std::ifstream reader(gt_file, std::ios::binary | std::ios::in);
  reader.read((char *)&nq, sizeof(int32_t));
  reader.read((char *)&total_res, sizeof(int32_t));

  std::vector<int32_t> ids(total_res);
  std::vector<float> dists(total_res);
  std::vector<int32_t> lims(nq + 1, 0);

  for (size_t i = 1; i < nq + 1; ++i) {
    reader.read((char *)&lims[i], sizeof(int32_t));
    lims[i] += lims[i - 1];
  }

  assert(lims.back() == total_res);

  for (size_t i = 0; i < total_res; ++i) {
    reader.read((char *)&ids[i], sizeof(int32_t));
  }

  for (size_t i = 0; i < total_res; ++i) {
    reader.read((char *)&dists[i], sizeof(float));
  }

  reader.close();

  return std::make_tuple(ids, dists, lims);
}

/*
// check benchmark/plotting/eval_range_search.py
std::tuple<std::vector<float>, std::vector<float>> range_pr_multiple_thresholds(
    std::vector<int32_t> &lims_ref,
    std::vector<int32_t> &ids_ref,
    std::vector<uint64_t> &lims_new,
    std::vector<uint32_t> &dists_new,
    std::vector<uint32_t> &ids_new,
    std::vector<float> &threshold) {
  size_t nt = threshold.size();
  size_t nq = lims_ref.size()-1;

  // sort ref and new
  for (int i = 0; i < )

  std::vector<float> precisions(nt);
  std::vector<float> recalls(nt);

  for (size_t i = 0; i < nt; ++i) {
    float p, r;
    std::tie(p, r) = counts_to_pr();
    precisions[i] = p;
    recalls[i] = r;
  }

  return std::make_tuple(precisions, recalls);
}

float compute_ap(const std::tuple<std::vector<int32_t>, std::vector<float>,
                                          std::vector<int32_t>> &gt,
                 const std::tuple<std::vector<uint32_t>, std::vector<uint32_t>,
                                          std::vector<uint64_t>> &res
                 ) {
  std::vector<uint32_t> res_ids, res_dists;
  std::vector<uint64_t> res_lims;
  std::vector<int32_t> gt_ids, gt_lims;
  std::vector<float> gt_dists;

  std::tie(res_ids, res_dists, res_lims) = res;
  std::tie(gt_ids, gt_dists, gt_lims) = gt;

  if (res_dists.size() == 0) return 0;

  float max_dist = *std::max_element(res_dists.begin(), res_dists.end());
  const float beg = -0.001, nt = 100;
  std::vector<float> thresholds(nt);
  float interval = (max_dist - beg) / nt;
  for (size_t i = 0; i < nt; ++i) {
    thresholds[i] = beg + i * interval;
  }

  std::vector<float> precisions, recalls;
  std::tie(precisions, recalls) = range_pr_multiple_thresholds(gt_lims, gt_ids,
res_lims, res_dists, res_ids, thresholds);

  float accu = 0;
  size_t n = precisions.size();
  for (size_t i = 0; i < n-1; ++i) {
    float x0 = recalls[i], x1 = recalls[i+1];
    float y0 = recalls[i], y1 = recalls[i+1];
    accu += (x1 - x0) * (y1 + y0) / 2;
  }
  return accu;
}
*/

float compute_range_search_recall_only(
    std::tuple<std::vector<int32_t>, std::vector<float>, std::vector<int32_t>>
        &gt,
    std::tuple<std::vector<uint32_t>, std::vector<uint32_t>,
               std::vector<uint64_t>> &res) {
  std::vector<uint32_t> res_ids, res_dists;
  std::vector<uint64_t> res_lims;
  std::vector<int32_t> gt_ids, gt_lims;
  std::vector<float> gt_dists;

  std::tie(res_ids, res_dists, res_lims) = res;
  std::tie(gt_ids, gt_dists, gt_lims) = gt;

  size_t nq = gt_lims.size() - 1;
  size_t nres = res_ids.size();
  size_t ngt = gt_ids.size();

  size_t ninter = 0;
  for (size_t i = 1; i < nq; ++i) {
    std::set<int32_t> inter;
    std::set<int32_t> res_ids_set(res_ids.begin() + res_lims[i - 1],
                                  res_ids.begin() + res_lims[i]);
    std::set<int32_t> gt_ids_set(gt_ids.begin() + gt_lims[i - 1],
                                 gt_ids.begin() + gt_lims[i]);
    std::set_intersection(res_ids_set.begin(), res_ids_set.end(),
                          gt_ids_set.begin(), gt_ids_set.end(),
                          std::inserter(inter, inter.begin()));
    ninter += inter.size();
  }

  float recall = ninter * 1.0f / ngt;
  float precision = ninter * 1.0f / nres;

  return (1 + precision) * recall / 2;
}

template<typename DISTT, typename IDT>
float range_search_recall(
    std::tuple<std::vector<uint32_t>, std::vector<uint32_t>,
               std::vector<uint64_t>> &res,
    std::string groundTruthFilePath) {
  auto gt = read_gt_file(groundTruthFilePath);
  return compute_range_search_recall_only(gt, res);
}

} // namespace util
} // namespace bbann
