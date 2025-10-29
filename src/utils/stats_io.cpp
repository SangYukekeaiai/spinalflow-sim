// All comments are in English.
#include "utils/stats_io.hpp"

#include <algorithm>
#include <cctype>
#include <filesystem>
#include <fstream>
#include <stdexcept>

namespace sf {

namespace {

void EnsureParentDir(const std::filesystem::path& path) {
  const auto dir = path.parent_path();
  if (!dir.empty()) {
    std::filesystem::create_directories(dir);
  }
}

std::string KindToStringInternal(LayerKind kind) {
  switch (kind) {
    case LayerKind::kConv: return "conv";
    case LayerKind::kFC:   return "fc";
    default:               return "unknown";
  }
}

} // namespace

std::string SanitizeName(const std::string& input) {
  std::string out;
  out.reserve(input.size());
  for (char ch : input) {
    const unsigned char uch = static_cast<unsigned char>(ch);
    if (std::isalnum(uch) || ch == '_' || ch == '-') {
      out.push_back(ch);
    } else {
      out.push_back('_');
    }
  }
  if (out.empty()) {
    out = "unnamed";
  }
  return out;
}

const char* LayerKindToString(LayerKind kind) {
  static const char* kConv = "conv";
  static const char* kFC   = "fc";
  static const char* kUnknown = "unknown";
  switch (kind) {
    case LayerKind::kConv: return kConv;
    case LayerKind::kFC:   return kFC;
    default:               return kUnknown;
  }
}

std::filesystem::path BuildStageCyclesCsvPath(const std::string& repo_name,
                                              const std::string& model_name) {
  return std::filesystem::path("stats") / repo_name / model_name / "timing" / "stage_cycles.csv";
}

void WriteStageCyclesCsv(const std::string& repo_name,
                         const std::string& model_name,
                         const std::vector<LayerStageRecord>& rows) {
  const auto csv_path = BuildStageCyclesCsvPath(repo_name, model_name);
  EnsureParentDir(csv_path);
  std::ofstream ofs(csv_path, std::ios::out | std::ios::trunc);
  if (!ofs) {
    throw std::runtime_error("Failed to open stage_cycles CSV at " + csv_path.string());
  }
  ofs << "layer_id,layer_name,kind,load_cycles,compute_cycles,store_cycles\n";
  for (const auto& row : rows) {
    ofs << row.layer_id << ','
        << SanitizeName(row.layer_name) << ','
        << KindToStringInternal(row.kind) << ','
        << row.cycles.load_cycles << ','
        << row.cycles.compute_cycles << ','
        << row.cycles.store_cycles << '\n';
  }
}

} // namespace sf
