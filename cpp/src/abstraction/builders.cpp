#include "regret/abstraction/builders.hpp"

#include <algorithm>
#include <stdexcept>
#include <string>

#include "regret/abstraction/card_features.hpp"
#include "regret/cfr/rng.hpp"
#include "regret/evaluator.hpp"
#include "regret/isomorphism.hpp"
#include "regret/parallel.hpp"

namespace regret::abstraction {
namespace {

constexpr std::uint16_t kUnset = 0xFFFF;

// `n` distinct random cards not in `used`.
void deal(Rng& rng, CardMask used, Card* out, int n) {
  for (int i = 0; i < n;) {
    const Card c = static_cast<Card>(rng.below(kNumCards));
    if (used & card_bit(c)) continue;
    used |= card_bit(c);
    out[i++] = c;
  }
}

CardMask mask_of(const Card* cards, int n) {
  CardMask m = 0;
  for (int i = 0; i < n; ++i) m |= card_bit(cards[i]);
  return m;
}

// Relaxed store: several boards may write the same (equal) value to one table cell.
inline void store(std::uint16_t* p, std::uint16_t v) { __atomic_store_n(p, v, __ATOMIC_RELAXED); }

void check_complete(const std::vector<std::uint16_t>& table, const char* name) {
  if (std::find(table.begin(), table.end(), kUnset) != table.end()) {
    throw std::logic_error(std::string(name) + " table has unfilled entries");
  }
}

int dims_of(const std::vector<float>& centroids, int dims, const char* name) {
  if (dims <= 0 || centroids.empty() || centroids.size() % static_cast<std::size_t>(dims)) {
    throw std::invalid_argument(std::string(name) + ": centroid array doesn't match dims");
  }
  const auto k = static_cast<std::int64_t>(centroids.size() / static_cast<std::size_t>(dims));
  if (k > kUnset) throw std::invalid_argument(std::string(name) + ": too many buckets");
  return static_cast<int>(k);
}

}  // namespace

std::vector<std::uint8_t> preflop_class_of_hole() {
  const HandIndexer pre({2});
  std::vector<std::uint8_t> out(kNumHoles);
  for (int h = 0; h < kNumHoles; ++h) {
    const Card cards[2] = {hole_pairs()[h].lo, hole_pairs()[h].hi};
    out[h] = static_cast<std::uint8_t>(pre.index(cards));
  }
  return out;
}

std::vector<double> preflop_class_equity(int samples, std::uint64_t seed, int threads) {
  const HandIndexer pre({2});
  std::vector<double> out(pre.size());
  parallel_for(static_cast<std::int64_t>(pre.size()), threads, [&](std::int64_t c) {
    Card hero[2];
    pre.unindex(static_cast<std::uint64_t>(c), hero);
    Rng rng = Rng::stream(seed, static_cast<std::uint64_t>(c));
    const CardMask mine = mask_of(hero, 2);
    double won = 0;
    for (int s = 0; s < samples; ++s) {
      Card rest[7];  // opponent's 2 + board 5
      deal(rng, mine, rest, 7);
      const CardMask board = mask_of(rest + 2, 5);
      const HandValue me = evaluate(board | mine);
      const HandValue them = evaluate(board | mask_of(rest, 2));
      won += me > them ? 1.0 : (me == them ? 0.5 : 0.0);
    }
    out[static_cast<std::size_t>(c)] = won / samples;
  });
  return out;
}

std::vector<float> sample_river_ochs(int boards, std::uint64_t seed,
                                     const std::vector<std::uint8_t>& cluster_of, int k,
                                     int threads) {
  if (cluster_of.size() != kNumHoles) throw std::invalid_argument("cluster_of needs 1326 entries");
  constexpr int kRows = 1081;  // holes disjoint from a 5-card board
  std::vector<float> out(static_cast<std::size_t>(boards) * kRows * k);
  parallel_for(boards, threads, [&](std::int64_t b) {
    Rng rng = Rng::stream(seed, static_cast<std::uint64_t>(b));
    Card board[5];
    deal(rng, 0, board, 5);
    std::vector<float> ochs(static_cast<std::size_t>(kNumHoles) * k);
    river_ochs(board, cluster_of.data(), k, ochs.data());
    float* dst = out.data() + static_cast<std::size_t>(b) * kRows * k;
    for (int h = 0; h < kNumHoles; ++h) {
      if (ochs[static_cast<std::size_t>(h) * k] < 0) continue;
      std::copy_n(ochs.data() + static_cast<std::size_t>(h) * k, k, dst);
      dst += k;
    }
  });
  return out;
}

std::vector<std::uint8_t> sample_turn_histograms(int boards, std::uint64_t seed, int bins,
                                                 int threads) {
  constexpr int kRows = 1128;  // holes disjoint from a 4-card board
  std::vector<std::uint8_t> out(static_cast<std::size_t>(boards) * kRows * bins);
  parallel_for(boards, threads, [&](std::int64_t b) {
    Rng rng = Rng::stream(seed, static_cast<std::uint64_t>(b));
    Card board[4];
    deal(rng, 0, board, 4);
    const CardMask bm = mask_of(board, 4);
    std::vector<std::uint8_t> counts(static_cast<std::size_t>(kNumHoles) * bins);
    turn_histograms(board, bins, counts.data());
    std::uint8_t* dst = out.data() + static_cast<std::size_t>(b) * kRows * bins;
    for (int h = 0; h < kNumHoles; ++h) {
      if (bm & (card_bit(hole_pairs()[h].lo) | card_bit(hole_pairs()[h].hi))) continue;
      std::copy_n(counts.data() + static_cast<std::size_t>(h) * bins, bins, dst);
      dst += bins;
    }
  });
  return out;
}

FlopFeatures flop_features(const std::vector<std::uint16_t>& turn_table,
                           const std::vector<std::uint16_t>& turn_rank, int threads) {
  const HandIndexer flops({3}), turn_idx({2, 4});
  if (turn_table.size() != turn_idx.size()) throw std::invalid_argument("wrong turn table size");
  const int k_turn = static_cast<int>(turn_rank.size());
  constexpr int kRows = 1176;  // holes disjoint from a 3-card board

  // How many raw flops map to each suit-distinct flop.
  std::vector<float> multiplicity(flops.size(), 0.0f);
  for (int a = 0; a < kNumCards; ++a)
    for (int b = a + 1; b < kNumCards; ++b)
      for (int c = b + 1; c < kNumCards; ++c) {
        const Card f[3] = {static_cast<Card>(a), static_cast<Card>(b), static_cast<Card>(c)};
        multiplicity[flops.index(f)] += 1.0f;
      }

  FlopFeatures out;
  out.rows = static_cast<std::int64_t>(flops.size()) * kRows;
  out.counts.assign(static_cast<std::size_t>(out.rows) * k_turn, 0);
  out.weights.assign(static_cast<std::size_t>(out.rows), 0.0f);
  parallel_for(static_cast<std::int64_t>(flops.size()), threads, [&](std::int64_t f) {
    Card cards[6];  // hole 2, flop 3, turn 1
    flops.unindex(static_cast<std::uint64_t>(f), cards + 2);
    const CardMask board = mask_of(cards + 2, 3);
    std::int64_t row = f * kRows;
    for (int h = 0; h < kNumHoles; ++h) {
      cards[0] = hole_pairs()[h].lo;
      cards[1] = hole_pairs()[h].hi;
      const CardMask hm = card_bit(cards[0]) | card_bit(cards[1]);
      if (hm & board) continue;
      std::uint8_t* dst = out.counts.data() + static_cast<std::size_t>(row) * k_turn;
      for (int t = 0; t < kNumCards; ++t) {
        if ((board | hm) & card_bit(static_cast<Card>(t))) continue;
        cards[5] = static_cast<Card>(t);
        const std::uint16_t bucket = turn_table[turn_idx.index(cards)];
        ++dst[turn_rank.at(bucket)];
      }
      out.weights[static_cast<std::size_t>(row)] = multiplicity[static_cast<std::size_t>(f)];
      ++row;
    }
  });
  return out;
}

std::vector<std::uint16_t> build_river_table(const std::vector<std::uint8_t>& cluster_of, int k,
                                             const std::vector<float>& centroids, int threads) {
  const int buckets = dims_of(centroids, k, "river");
  const HandIndexer boards({5}), idx({2, 5});
  std::vector<std::uint16_t> table(idx.size(), kUnset);
  parallel_for(static_cast<std::int64_t>(boards.size()), threads, [&](std::int64_t b) {
    Card cards[7];
    boards.unindex(static_cast<std::uint64_t>(b), cards + 2);
    std::vector<float> ochs(static_cast<std::size_t>(kNumHoles) * k);
    river_ochs(cards + 2, cluster_of.data(), k, ochs.data());
    for (int h = 0; h < kNumHoles; ++h) {
      const float* x = ochs.data() + static_cast<std::size_t>(h) * k;
      if (x[0] < 0) continue;
      cards[0] = hole_pairs()[h].lo;
      cards[1] = hole_pairs()[h].hi;
      store(&table[idx.index(cards)],
            static_cast<std::uint16_t>(nearest(x, centroids.data(), buckets, k)));
    }
  });
  check_complete(table, "river");
  return table;
}

std::vector<std::uint16_t> build_turn_table(int bins, const std::vector<float>& centroids,
                                            int threads) {
  const int buckets = dims_of(centroids, bins, "turn");
  const HandIndexer boards({4}), idx({2, 4});
  std::vector<std::uint16_t> table(idx.size(), kUnset);
  parallel_for(static_cast<std::int64_t>(boards.size()), threads, [&](std::int64_t b) {
    Card cards[6];
    boards.unindex(static_cast<std::uint64_t>(b), cards + 2);
    const CardMask bm = mask_of(cards + 2, 4);
    std::vector<std::uint8_t> counts(static_cast<std::size_t>(kNumHoles) * bins);
    turn_histograms(cards + 2, bins, counts.data());
    std::vector<float> cdf(bins);
    for (int h = 0; h < kNumHoles; ++h) {
      cards[0] = hole_pairs()[h].lo;
      cards[1] = hole_pairs()[h].hi;
      if (bm & (card_bit(cards[0]) | card_bit(cards[1]))) continue;
      to_cdf(counts.data() + static_cast<std::size_t>(h) * bins, bins, cdf.data());
      store(&table[idx.index(cards)],
            static_cast<std::uint16_t>(nearest(cdf.data(), centroids.data(), buckets, bins)));
    }
  });
  check_complete(table, "turn");
  return table;
}

std::vector<std::uint16_t> build_flop_table(const std::vector<std::uint16_t>& turn_table,
                                            const std::vector<std::uint16_t>& turn_rank,
                                            const std::vector<float>& centroids, int threads) {
  const int k_turn = static_cast<int>(turn_rank.size());
  const int buckets = dims_of(centroids, k_turn, "flop");
  const FlopFeatures feats = flop_features(turn_table, turn_rank, threads);
  const HandIndexer flops({3}), idx({2, 3});
  std::vector<std::uint16_t> table(idx.size(), kUnset);
  constexpr int kRows = 1176;
  parallel_for(static_cast<std::int64_t>(flops.size()), threads, [&](std::int64_t f) {
    Card cards[5];
    flops.unindex(static_cast<std::uint64_t>(f), cards + 2);
    const CardMask bm = mask_of(cards + 2, 3);
    std::vector<float> cdf(k_turn);
    std::int64_t row = f * kRows;
    for (int h = 0; h < kNumHoles; ++h) {
      cards[0] = hole_pairs()[h].lo;
      cards[1] = hole_pairs()[h].hi;
      if (bm & (card_bit(cards[0]) | card_bit(cards[1]))) continue;
      to_cdf(feats.counts.data() + static_cast<std::size_t>(row) * k_turn, k_turn, cdf.data());
      store(&table[idx.index(cards)],
            static_cast<std::uint16_t>(nearest(cdf.data(), centroids.data(), buckets, k_turn)));
      ++row;
    }
  });
  check_complete(table, "flop");
  return table;
}

}  // namespace regret::abstraction
