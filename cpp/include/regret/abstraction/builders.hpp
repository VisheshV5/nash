#pragma once

#include <cstdint>
#include <vector>

// Offline builders for the card abstraction: feature samples for fitting centroids, and full
// bucket tables indexed by the suit-isomorphic (hole, board) index of each street.
namespace regret::abstraction {

// Preflop class (0..168, the {2} isomorphism index) of every hole index.
std::vector<std::uint8_t> preflop_class_of_hole();

// Monte Carlo all-in equity of each preflop class vs a random hand (`samples` runouts each).
std::vector<double> preflop_class_equity(int samples, std::uint64_t seed, int threads);

// OCHS features of every hole on `boards` random river boards: boards * 1081 rows of k floats.
std::vector<float> sample_river_ochs(int boards, std::uint64_t seed,
                                     const std::vector<std::uint8_t>& cluster_of, int k,
                                     int threads);

// River-equity histograms of every hole on `boards` random turn boards: boards * 1128 rows of
// `bins` counts.
std::vector<std::uint8_t> sample_turn_histograms(int boards, std::uint64_t seed, int bins,
                                                 int threads);

struct FlopFeatures {
  std::vector<std::uint8_t> counts;  // rows * k_turn: turn buckets hit, ordered by turn rank
  std::vector<float> weights;        // rows: how many raw deals each row stands for
  std::int64_t rows = 0;
};

// Potential-aware flop features for every hole on every suit-distinct flop, using the turn
// table and turn_rank[bucket] (turn buckets ordered by mean equity).
FlopFeatures flop_features(const std::vector<std::uint16_t>& turn_table,
                           const std::vector<std::uint16_t>& turn_rank, int threads);

// Full tables. Centroids are row-major (buckets x dims) in the same feature space as fitting:
// river OCHS values; turn and flop CDFs of their histograms.
std::vector<std::uint16_t> build_river_table(const std::vector<std::uint8_t>& cluster_of, int k,
                                             const std::vector<float>& centroids, int threads);
std::vector<std::uint16_t> build_turn_table(int bins, const std::vector<float>& centroids,
                                            int threads);
std::vector<std::uint16_t> build_flop_table(const std::vector<std::uint16_t>& turn_table,
                                            const std::vector<std::uint16_t>& turn_rank,
                                            const std::vector<float>& centroids, int threads);

}  // namespace regret::abstraction
