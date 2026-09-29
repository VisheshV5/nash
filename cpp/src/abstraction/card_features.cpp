#include "regret/abstraction/card_features.hpp"

#include <algorithm>
#include <cstring>

#include "regret/evaluator.hpp"

namespace regret::abstraction {
namespace {

std::array<HolePair, kNumHoles> make_pairs() {
  std::array<HolePair, kNumHoles> p{};
  for (int b = 1; b < kNumCards; ++b) {
    for (int a = 0; a < b; ++a) {
      p[hole_index(static_cast<Card>(a), static_cast<Card>(b))] = {static_cast<Card>(a),
                                                                   static_cast<Card>(b)};
    }
  }
  return p;
}

CardMask mask_of(const Card* cards, int n) {
  CardMask m = 0;
  for (int i = 0; i < n; ++i) m |= card_bit(cards[i]);
  return m;
}

struct Ranked {
  HandValue value;
  std::int16_t hole;
};

// Holes disjoint from `board`, sorted by hand value on that board. Returns the count.
int rank_holes(CardMask board, Ranked* out) {
  const auto& pairs = hole_pairs();
  int n = 0;
  for (int h = 0; h < kNumHoles; ++h) {
    const CardMask hm = card_bit(pairs[h].lo) | card_bit(pairs[h].hi);
    if (hm & board) continue;
    out[n++] = {evaluate(board | hm), static_cast<std::int16_t>(h)};
  }
  std::sort(out, out + n, [](const Ranked& x, const Ranked& y) { return x.value < y.value; });
  return n;
}

// Sweeps holes in ascending strength. For each hole it hands `emit` the number of opponent
// holes (not sharing a card with it) that are weaker and that tie, overall and per cluster.
template <class Emit>
void sweep(const Ranked* ranked, int n, const std::uint8_t* cluster_of, int k, Emit&& emit) {
  const auto& pairs = hole_pairs();
  // Running counts of already-passed (weaker) holes: total, per card, per cluster, per
  // (card, cluster). Group counts are the same for the current tie group.
  int below = 0;
  int below_card[kNumCards] = {};
  std::vector<int> below_k(k, 0), below_card_k(static_cast<std::size_t>(kNumCards) * k, 0);
  // Totals over all holes on the board, for per-cluster denominators.
  std::vector<int> all_k(k, 0), all_card_k(static_cast<std::size_t>(kNumCards) * k, 0);
  if (cluster_of) {
    for (int i = 0; i < n; ++i) {
      const int h = ranked[i].hole;
      const int c = cluster_of[h];
      ++all_k[c];
      ++all_card_k[pairs[h].lo * k + c];
      ++all_card_k[pairs[h].hi * k + c];
    }
  }

  std::vector<int> group_k(k), group_card_k(static_cast<std::size_t>(kNumCards) * k);
  int group_card[kNumCards];
  for (int start = 0; start < n;) {
    int end = start;
    while (end < n && ranked[end].value == ranked[start].value) ++end;
    const int group = end - start;
    std::memset(group_card, 0, sizeof(group_card));
    if (cluster_of) {
      std::fill(group_k.begin(), group_k.end(), 0);
      std::fill(group_card_k.begin(), group_card_k.end(), 0);
    }
    for (int i = start; i < end; ++i) {
      const int h = ranked[i].hole;
      ++group_card[pairs[h].lo];
      ++group_card[pairs[h].hi];
      if (cluster_of) {
        const int c = cluster_of[h];
        ++group_k[c];
        ++group_card_k[pairs[h].lo * k + c];
        ++group_card_k[pairs[h].hi * k + c];
      }
    }
    for (int i = start; i < end; ++i) {
      const int h = ranked[i].hole;
      const int a = pairs[h].lo;
      const int b = pairs[h].hi;
      // Hands sharing a card with h: those containing a, plus those containing b, minus h.
      const int weaker = below - below_card[a] - below_card[b];
      const int ties = group - group_card[a] - group_card[b] + 1;
      emit(h, weaker, ties, [&](int c, int* w, int* t, int* total) {
        const int self = cluster_of[h] == c ? 1 : 0;
        *w = below_k[c] - below_card_k[a * k + c] - below_card_k[b * k + c];
        *t = group_k[c] - group_card_k[a * k + c] - group_card_k[b * k + c] + self;
        *total = all_k[c] - all_card_k[a * k + c] - all_card_k[b * k + c] + self;
      });
    }
    below += group;
    for (int c = 0; c < kNumCards; ++c) below_card[c] += group_card[c];
    if (cluster_of) {
      for (int c = 0; c < k; ++c) below_k[c] += group_k[c];
      for (std::size_t i = 0; i < group_card_k.size(); ++i) below_card_k[i] += group_card_k[i];
    }
    start = end;
  }
}

}  // namespace

const std::array<HolePair, kNumHoles>& hole_pairs() {
  static const std::array<HolePair, kNumHoles> pairs = make_pairs();
  return pairs;
}

void river_equities(const Card* board5, float* equity) {
  std::fill(equity, equity + kNumHoles, -1.0f);
  Ranked ranked[kNumHoles];
  const int n = rank_holes(mask_of(board5, 5), ranked);
  // Every hole on a 5-card board faces 47C2 - 91 = 990 possible opponent holes.
  const float opponents = static_cast<float>(n - (2 * (kNumCards - 5 - 1) - 1));
  sweep(ranked, n, nullptr, 0, [&](int h, int weaker, int ties, auto&&) {
    equity[h] = (static_cast<float>(weaker) + 0.5f * static_cast<float>(ties)) / opponents;
  });
}

void river_ochs(const Card* board5, const std::uint8_t* cluster_of, int k, float* ochs) {
  std::fill(ochs, ochs + static_cast<std::size_t>(kNumHoles) * k, -1.0f);
  Ranked ranked[kNumHoles];
  const int n = rank_holes(mask_of(board5, 5), ranked);
  sweep(ranked, n, cluster_of, k, [&](int h, int, int, auto&& per_cluster) {
    for (int c = 0; c < k; ++c) {
      int w, t, total;
      per_cluster(c, &w, &t, &total);
      ochs[static_cast<std::size_t>(h) * k + c] =
          total > 0 ? (static_cast<float>(w) + 0.5f * static_cast<float>(t)) / total : 0.5f;
    }
  });
}

void turn_histograms(const Card* board4, int bins, std::uint8_t* counts) {
  std::memset(counts, 0, static_cast<std::size_t>(kNumHoles) * bins);
  const CardMask board_mask = mask_of(board4, 4);
  float eq[kNumHoles];
  Card board5[5] = {board4[0], board4[1], board4[2], board4[3], 0};
  for (int r = 0; r < kNumCards; ++r) {
    if (board_mask & card_bit(static_cast<Card>(r))) continue;
    board5[4] = static_cast<Card>(r);
    river_equities(board5, eq);
    for (int h = 0; h < kNumHoles; ++h) {
      if (eq[h] < 0) continue;
      const int bin = std::min(bins - 1, static_cast<int>(eq[h] * static_cast<float>(bins)));
      ++counts[static_cast<std::size_t>(h) * bins + bin];
    }
  }
}

void river_ochs_one(Card h0, Card h1, const Card* board5, const std::uint8_t* cluster_of, int k,
                    float* out) {
  const CardMask board = mask_of(board5, 5);
  const CardMask mine = card_bit(h0) | card_bit(h1);
  const HandValue me = evaluate(board | mine);
  std::vector<double> won(k, 0.0);
  std::vector<int> total(k, 0);
  const auto& pairs = hole_pairs();
  for (int h = 0; h < kNumHoles; ++h) {
    const CardMask hm = card_bit(pairs[h].lo) | card_bit(pairs[h].hi);
    if (hm & (board | mine)) continue;
    const HandValue v = evaluate(board | hm);
    const int c = cluster_of[h];
    ++total[c];
    won[c] += me > v ? 1.0 : (me == v ? 0.5 : 0.0);
  }
  for (int c = 0; c < k; ++c) out[c] = total[c] ? static_cast<float>(won[c] / total[c]) : 0.5f;
}

void turn_histogram_one(Card h0, Card h1, const Card* board4, int bins, std::uint8_t* out) {
  std::memset(out, 0, static_cast<std::size_t>(bins));
  const CardMask board = mask_of(board4, 4);
  const CardMask mine = card_bit(h0) | card_bit(h1);
  const auto& pairs = hole_pairs();
  for (int r = 0; r < kNumCards; ++r) {
    const CardMask rm = card_bit(static_cast<Card>(r));
    if (rm & (board | mine)) continue;
    const CardMask b5 = board | rm;
    const HandValue me = evaluate(b5 | mine);
    int opponents = 0;
    double won = 0;
    for (int h = 0; h < kNumHoles; ++h) {
      const CardMask hm = card_bit(pairs[h].lo) | card_bit(pairs[h].hi);
      if (hm & (b5 | mine)) continue;
      const HandValue v = evaluate(b5 | hm);
      ++opponents;
      won += me > v ? 1.0 : (me == v ? 0.5 : 0.0);
    }
    const float eq = static_cast<float>(won / opponents);
    ++out[std::min(bins - 1, static_cast<int>(eq * static_cast<float>(bins)))];
  }
}

int nearest(const float* x, const float* centroids, int k, int dims) {
  int best = 0;
  float best_d = 3.4e38f;
  for (int c = 0; c < k; ++c) {
    const float* y = centroids + static_cast<std::size_t>(c) * dims;
    float d = 0;
    for (int i = 0; i < dims; ++i) {
      const float t = x[i] - y[i];
      d += t * t;
    }
    if (d < best_d) {
      best_d = d;
      best = c;
    }
  }
  return best;
}

void to_cdf(const std::uint8_t* counts, int n, float* cdf) {
  int total = 0;
  for (int i = 0; i < n; ++i) total += counts[i];
  int run = 0;
  for (int i = 0; i < n; ++i) {
    run += counts[i];
    cdf[i] = total ? static_cast<float>(run) / static_cast<float>(total) : 0.0f;
  }
}

}  // namespace regret::abstraction
