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

// Writes scoreboard distribution per layer: rows of (score, num_channels, channel_ids)
void WriteScoreboardScoresCsv(const std::filesystem::path& csv_path,
                              const std::unordered_map<int, int>& scoreboard_scores);

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

// ----------------------------------------------------------------------------
// Cache step/timestep CSVs (moved from CacheSim)
// These helpers keep the simulator code clean and centralize path logic.
// Scoreboard step CSVs respect the cache trace toggle. The ts_duration CSVs
// are controlled independently via CacheConfig::ts_duration_enabled and can be
// written even when tracing is disabled.

// Emits a per-step scoreboard snapshot CSV for step t if enabled.
// The CSV groups channels by score with columns: score,num_channels,channel_ids
// - Output path is stats/<repo>/<model>/layer<layer_id>/scoreboard_steps/ts<t>.csv
// - Uses 1-based timestep numbering in filenames (ts1.csv, ts2.csv, ...)
// - Does not throw; silently returns on any failure.
void WriteScoreboardStepCsvIfEnabled(const sf::arch::cache::CacheConfig& cfg,
                                     int layer_id,
                                     int t,
                                     const std::unordered_map<int, int>& scores);

// Emits per-layer timestep access CSV if enabled (independent of trace):
// - Writes a single CSV under stats/<repo>/<model>/ts_duration/layer_<id>.csv
// - Matrix format: rows per output_spine_id, columns t0..tN plus an avg row
// - Does not throw; silently returns on any failure.
// Also writes/updates a JSON mapping file `layer_dims.json` under the same
// ts_duration directory with per-layer dims for default plot annotations.
void WriteLayerTimestepAccessCsvsIfEnabled(
    const sf::arch::cache::CacheConfig& cfg,
    const std::unordered_map<int, std::unordered_map<int, std::uint64_t>>& per_site_step_access_counts,
    int max_timestep_observed,
    int layer_id_for_paths,
    int Cin, int Hin, int Win);

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
                          bool write_scoreboard_csv,
                          bool write_visit_count_distribution_csv,
                          bool write_per_set_unique_csv,
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
                                    const std::vector<CacheTotalsRow>& cache_total_rows,
                                    const std::unordered_map<int, std::vector<CacheTotalsRow>>& per_layer_totals_rows);

} // namespace sf
