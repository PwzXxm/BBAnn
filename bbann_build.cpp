#include "lib/bbannlib2.h"
#include "util/constants.h"
#include "util/utils_inline.h"

int main(int argc, char **argv) {
  if (argc != 11) {
    std::cout << "Usage: << " << argv[0] << " data_type(float or uint8 or int8)"
              << " binary raw data file"
              << " index output path"
              << " hnsw.M"
              << " hnsw.efConstruction"
              << " metric type(L2 or IP)"
              << " K1"
              << " page per block"
              << " vector_use_sq (0 or 1)"
              << " use_hnsw_sq (0 or 1)"
              << std::endl;
    return 1;
  }

  bbann::BBAnnParameters para;

  // parse parameters
  std::string raw_data_bin_file(argv[2]);
  std::string output_path(argv[3]);
  if ('/' != *output_path.rbegin())
    output_path += '/';

  para.dataFilePath = raw_data_bin_file;
  para.indexPrefixPath = output_path;
  para.hnswM = std::stoi(argv[4]);
  para.hnswefC = std::stoi(argv[5]);
  para.metric = bbann::util::get_metric_type_by_name(std::string(argv[6]));
  para.K1 = std::stoi(argv[7]);
  para.blockSize = std::stoul(argv[8]) * PAGESIZE;
  para.vector_use_sq = std::stoi(argv[9]) == 0 ? false : true;
  para.use_hnsw_sq = std::stoi(argv[10]) == 0 ? false : true;

  if (argv[1] == std::string("float")) {
    bbann::BBAnnIndex2<float, float> index(para.metric);
    index.BuildIndex(para);
  } else if (argv[1] == std::string("uint8")) {
    bbann::BBAnnIndex2<uint8_t, uint32_t> index(para.metric);
    index.BuildIndex(para);
  } else if (argv[1] == std::string("int8")) {
    bbann::BBAnnIndex2<int8_t, int32_t> index(para.metric);
    index.BuildIndex(para);
  }
  return 0;
}
