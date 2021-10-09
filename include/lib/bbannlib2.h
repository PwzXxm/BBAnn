#pragma once
#include "ann_interface.h"

#include <iostream>
#include <memory>
#include <stdint.h>
#include <string>

namespace bbann{

struct BBAnnParameters {
  std::string dataFilePath;
  std::string indexPrefixPath;
  std::string queryPath;
  std::string groundTruthFilePath;
  bbann::MetricType metric;
  int K = 20; // top k.
  int hnswM = 32;
  int hnswefC = 500;
  int K1 = 20;
  int blockSize = 1;
  int nProbe = 2;
};

template <typename dataT, typename paraT>
struct BBAnnIndex2 : public BuildIndexFactory<BBAnnIndex<dataT, paraT>, paraT>,
                    public AnnIndexInterface<dataT, paraT> {

  using parameterType = paraT;
  using dataType = dataT;
  using distanceT = typename TypeWrapper<dataT>::distanceT;

public:
  BBAnnIndex2(MetricType metric) : metric_(metric) {
    std::cout << "BBAnnIndex constructor" << std::endl;
  }
  MetricType metric_;

  bool LoadIndex(std::string &indexPathPrefix);

  void BatchSearchCpp(const dataT *pquery, uint64_t dim, uint64_t numQuery,
                      uint64_t knn, const paraT para, uint32_t *answer_ids,
                      distanceT *answer_dists) override;

  std::shared_ptr<hnswlib::HierarchicalNSW<float>> index_hnsw_;
  std::string indexPrefix_;

  static void BuildIndexImpl(const paraT para);
};

}// namespace bbann