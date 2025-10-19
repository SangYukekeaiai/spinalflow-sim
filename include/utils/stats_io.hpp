// All comments are in English.
#pragma once

#include <filesystem>
#include <string>
#include <unordered_map>
#include <vector>

#include "utils/stats_types.hpp"
#include "arch/cache/cache.hpp"

namespace sf {

// String helpers
std::string SanitizeName(const std::string& input);
const char* LayerKindToString(LayerKind kind);
const char* EvictionPolicyToString(sf::arch::cache::EvictionPolicy policy);

// Path builders for certain CSVs
std::filesystem::path BuildSramAccessCsvPath(const std::string& repo_name,
                                             const std::string& model_name);
std::filesystem::path BuildSramCapacityCsvPath(const std::string& repo_name,
                                               const std::string& model_name);
std::filesystem::path BuildStageCsvPath(const std::string& repo_name,
                                        const std::string& model_name);
std::filesystem::path BuildLayerTablesDir(const std::string& repo_name,
                                          const std::string& model_name);

// CSV writers
void WriteReuseDistributionCsv(const std::filesystem::path& csv_path,
                               const std::unordered_map<std::uint64_t, std::uint64_t>& histogram);

// Writes per-set unique demand line counts (0..num_sets-1)
void WritePerSetUniqueDemandLinesCsv(const std::filesystem::path& csv_path,
                                     int num_sets,
                                     const std::unordered_map<int, std::uint64_t>& counts);

void WriteStageCyclesCsv(const std::string& repo_name,
                         const std::string& model_name,
                         const std::vector<LayerStageRecord>& rows);

void WriteSramAccessCsv(const std::string& repo_name,
                        const std::string& model_name,
                        const std::vector<LayerStageRecord>& rows);

void WriteSramCapacityCsv(const std::string& repo_name,
                          const std::string& model_name,
                          const std::vector<LayerStageRecord>& rows);

void WritePerLayerSramTables(const std::string& repo_name,
                             const std::string& model_name,
                             const std::vector<LayerStageRecord>& rows);

// Cache CSV helpers (extracted from simulation)
// - Writes per-configuration CSVs (model-level + per-layer) and reuse distributions
// - Produces aggregated rows for later summary CSVs
void WriteCacheConfigCsvs(const std::string& repo_name,
                          const std::string& model_name,
                          const std::string& policy_tag,
                          int cache_ways,
                          int prefetch_depth,
                          std::size_t cache_size_kb,
                          const std::vector<LayerStageRecord>& stage_rows,
                          bool write_stats_csv,
                          bool write_reuse_distribution_csv,
                          bool write_visit_count_distribution_csv,
                          bool write_per_set_unique_csv,
                          bool single_layer_run,
                          bool is_lru_policy,
                          std::vector<std::pair<int, CacheTotalsRow>>* per_layer_rows_out,
                          CacheTotalsRow* model_row_out);

// Writes aggregated cache_totals_*.csv (model-level and per-layer) across cache sizes
void WriteAggregatedCacheTotalsCsvs(const std::string& repo_name,
                                    const std::string& model_name,
                                    const std::string& policy_tag,
                                    int cache_ways,
                                    int prefetch_depth,
                                    bool is_lru_policy,
                                    bool write_stats_csv,
                                    bool single_layer_run,
                                    const std::vector<CacheTotalsRow>& cache_total_rows,
                                    const std::unordered_map<int, std::vector<CacheTotalsRow>>& per_layer_totals_rows);

} // namespace sf
