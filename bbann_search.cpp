#include "lib/bbannlib2.h"
#include "util/constants.h"
#include "util/recalls.h"
#include "util/utils_inline.h"

#include <chrono>
#include <iostream>

template <typename DATAT, typename DISTT>
void search(int topk, float radius, const bbann::BBAnnParameters para) {
  bbann::BBAnnIndex2<DATAT, DISTT> index(para.metric);
  std::string index_path = para.indexPrefixPath;
  index.LoadIndex(index_path, para);

  DATAT *pquery = nullptr;
  uint32_t nq, dim;
  bbann::util::read_bin_file<DATAT>(para.queryPath, pquery, nq, dim);

  if (topk == -1) {
    auto start = std::chrono::high_resolution_clock::now();
    auto rst = index.RangeSearchCpp(pquery, dim, nq, radius, para);
    double dur = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now() - start).count();
    float acc = bbann::util::range_search_recall(rst, para.groundTruthFilePath);
    std::cout << dur << ", " << nq / dur << ", " << acc << std::endl;
  } else {
    // knn
  }
}

int main(int argc, char **argv) {
  TimeRecorder rc("main");
  if (argc != 16) {
    std::cout << "Usage: << " << argv[0] << " data_type(float or uint8 or int8)"
              << " index path"
              << " query data file"
              << " answer file"
              << " ground truth file"
              << " nprobe"
              << " hnsw ef"
              << " topk (-1 indicates range search)"
              << " K1"
              << " metric type"
              << " page per block"
              << " vector_use_sq (0 or 1)"
              << " use_hnsw_sq (0 or 1)"
              << " radius"
              << " radius factor"
              << " range search probe count" << std::endl;
    return 1;
  }

  bbann::BBAnnParameters para;

  // std::string answer_file(argv[4]);
  std::string index_path(argv[2]);
  if ('/' != *index_path.rbegin())
    index_path += '/';
  para.indexPrefixPath = index_path;
  para.queryPath = std::string(argv[3]);
  std::string answerFilePath(argv[4]);
  para.groundTruthFilePath = std::string(argv[5]);
  para.nProbe = std::stoi(argv[6]);
  para.efSearch = std::stoi(argv[7]);
  int topk = std::stoi(argv[8]); // -1 indicates range search
  para.K1 = std::stoi(argv[9]);
  para.metric = bbann::util::get_metric_type_by_name(std::string(argv[10]));
  para.blockSize = std::stoul(argv[11]) * PAGESIZE;
  para.vector_use_sq = std::stoi(argv[12]) == 0 ? false : true;
  para.use_hnsw_sq = std::stoi(argv[13]) == 0 ? false : true;
  float radius = std::stof(argv[14]);
  para.radiusFactor = std::stof(argv[15]);
  para.rangeSearchProbeCount = std::stoi(argv[16]);

  // index.BatchSearchCpp(pquery, dim, numQuery, topk, para, answer_ids,
  // answer_dists); recall<float, uint32_t>(ground_truth_file, answer_file,
  // metric_type, true, false);

  if (argv[1] == std::string("float")) {
    // search<float, float>(topk, radius, para);
  } else if (argv[1] == std::string("uint8")) {
    search<uint8_t, uint32_t>(topk, radius, para);
  } else if (argv[1] == std::string("int8")) {
    // search<int8_t, int32_t>(topk, radius, para);
  }

  rc.ElapseFromBegin(" totally done.");
  return 0;
}
