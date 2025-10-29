// All comments are in English.
#pragma once

#include <filesystem>
#include <string>
#include <vector>

#include "utils/stats_types.hpp"

namespace sf {

std::string SanitizeName(const std::string& input);
const char* LayerKindToString(LayerKind kind);

std::filesystem::path BuildStageCyclesCsvPath(const std::string& repo_name,
                                              const std::string& model_name);
void WriteStageCyclesCsv(const std::string& repo_name,
                         const std::string& model_name,
                         const std::vector<LayerStageRecord>& rows);

} // namespace sf
