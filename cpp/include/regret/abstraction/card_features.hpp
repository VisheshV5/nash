#pragma once

#include <array>
#include <cstdint>
#include <vector>

#include "regret/cards.hpp"

// Hand-strength features for card abstraction (ROADMAP M3).
//
//   river: OCHS, equity vs each of k opponent-hand clusters (Johanson et al. 2013)
//   turn:  histogram of river equity (vs a random hand) over the 46 river cards
//   flop:  histogram of turn *buckets* over the 47 turn cards (potential-aware), with turn
//          buckets ordered by mean equity so a CDF over them is meaningful
//
// Clustering uses L2 distance on the CDFs of the histograms. For one-dimensional histograms
// that's the Cramér distance, a close cousin of the earth mover's distance the literature uses,
// and it lets k-means run as plain matrix math.
//
// All-hands-at-once routines work one board at a time: evaluate every hole on the board once,
// sort, and read each hole's wins and ties off running counts (minus hands sharing a card with
// it). That turns ~1,000 evaluations per hand into ~1 per hand.
namespace regret::abstraction {

inline constexpr int kNumHoles = 1326;  // C(52, 2)

struct HolePair {
  Card lo, hi;
};

const std::array<HolePair, kNumHoles>& hole_pairs();
inline int hole_index(Card a, Card b) {
  if (a > b) std::swap(a, b);
  return b * (b - 1) / 2 + a;
}

// ---- all holes on one board. Holes that share a card with the board get -1 / are skipped.

// Equity vs a uniformly random opponent hole (ties count half), for every hole.
void river_equities(const Card* board5, float* equity);

// Equity vs each opponent cluster. `cluster_of` maps hole index -> cluster in [0, k).
// `ochs` is kNumHoles * k (row-major by hole).
void river_ochs(const Card* board5, const std::uint8_t* cluster_of, int k, float* ochs);

// River-equity histograms (counts over `bins` equal-width bins of [0, 1]) for every hole on a
// 4-card board: kNumHoles * bins counts, each valid row summing to 46.
void turn_histograms(const Card* board4, int bins, std::uint8_t* counts);

// ---- one hand (runtime bucketing)

void river_ochs_one(Card h0, Card h1, const Card* board5, const std::uint8_t* cluster_of, int k,
                    float* out);
void turn_histogram_one(Card h0, Card h1, const Card* board4, int bins, std::uint8_t* out);

// ---- equity vs a range (evaluation bots, LBR)

// Showdown equity of (h0, h1) against an opponent range `weights` (kNumHoles entries; blocked
// holes are ignored), averaged over the rest of the board: every runout when the board has 3+
// cards, else `samples` random boards. Returns 0.5 if the range is empty.
double equity_vs_range(Card h0, Card h1, const Card* board, int board_size, const float* weights,
                       int samples, std::uint64_t seed);

// ---- helpers

// Squared-L2 nearest centroid of `x` (dims) among `k` centroids (row-major k * dims).
int nearest(const float* x, const float* centroids, int k, int dims);

// Counts -> cumulative distribution (divided by the row total).
void to_cdf(const std::uint8_t* counts, int n, float* cdf);

}  // namespace regret::abstraction
