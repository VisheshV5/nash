#include "regret/isomorphism.hpp"

#include <algorithm>
#include <stdexcept>
#include <string>

namespace regret {
namespace {

__extension__ typedef unsigned __int128 u128;  // NOLINT: GCC/Clang extension, fine on our targets

// n choose k for the (small k) sizes used here. Exact: each step divides evenly.
std::uint64_t choose(std::uint64_t n, std::uint64_t k) {
  if (k > n) return 0;
  u128 r = 1;
  for (std::uint64_t j = 1; j <= k; ++j) {
    r = r * (n - k + j) / j;
    if (r > ~std::uint64_t{0}) throw std::overflow_error("HandIndexer: binomial overflow");
  }
  return static_cast<std::uint64_t>(r);
}

struct SmallChoose {
  std::uint32_t c[kNumRanks + 1][kNumRanks + 1] = {};
  SmallChoose() {
    for (int n = 0; n <= kNumRanks; ++n) {
      c[n][0] = 1;
      for (int k = 1; k <= n; ++k) c[n][k] = c[n - 1][k - 1] + (k < n ? c[n - 1][k] : 0);
    }
  }
};
const SmallChoose kChoose;

inline int popcount(std::uint32_t x) { return __builtin_popcount(x); }

// Colex index of `set` (a subset of the ranks not in `used`), counting positions among the
// unused ranks only.
std::uint32_t rank_set_index(std::uint32_t set, std::uint32_t used) {
  std::uint32_t idx = 0;
  int pos = 0;
  int j = 1;
  for (int r = 0; r < kNumRanks; ++r) {
    if (used >> r & 1) continue;
    if (set >> r & 1) idx += kChoose.c[pos][j++];
    ++pos;
  }
  return idx;
}

// Inverse of rank_set_index for a set of `count` ranks.
std::uint32_t rank_set_from_index(std::uint32_t idx, int count, std::uint32_t used) {
  std::uint32_t positions = 0;
  for (int j = count; j >= 1; --j) {
    int p = kNumRanks - 1;
    while (kChoose.c[p][j] > idx) --p;
    idx -= kChoose.c[p][j];
    positions |= 1u << p;
  }
  std::uint32_t set = 0;
  int pos = 0;
  for (int r = 0; r < kNumRanks; ++r) {
    if (used >> r & 1) continue;
    if (positions >> pos & 1) set |= 1u << r;
    ++pos;
  }
  return set;
}

}  // namespace

HandIndexer::HandIndexer(std::vector<int> cards_per_round)
    : cards_per_round_(std::move(cards_per_round)) {
  if (cards_per_round_.empty() || cards_per_round_.size() > kMaxRounds) {
    throw std::invalid_argument("HandIndexer: need 1.." + std::to_string(kMaxRounds) + " rounds");
  }
  for (int n : cards_per_round_) {
    if (n < 1 || n > kMaxCardsPerRound) {
      throw std::invalid_argument("HandIndexer: each round needs 1.." +
                                  std::to_string(kMaxCardsPerRound) + " cards");
    }
    total_cards_ += n;
  }
  if (total_cards_ > kNumCards) throw std::invalid_argument("HandIndexer: more than 52 cards");
  enumerate_configs();
}

std::uint32_t HandIndexer::pack(const Shape& s) const {
  std::uint32_t p = 0;
  for (int r = 0; r < kMaxRounds; ++r) p = (p << 3) | s[r];
  return p;
}

std::uint64_t HandIndexer::key_space(const Shape& s) const {
  std::uint64_t n = 1;
  int used = 0;
  for (int r = 0; r < rounds(); ++r) {
    n *= kChoose.c[kNumRanks - used][s[r]];
    used += s[r];
  }
  return n;
}

std::uint64_t HandIndexer::suit_key(const Shape& s,
                                    const std::array<std::uint32_t, kMaxRounds>& sets) const {
  std::uint64_t key = 0;
  std::uint32_t used = 0;
  for (int r = 0; r < rounds(); ++r) {
    const int avail = kNumRanks - popcount(used);
    key = key * kChoose.c[avail][s[r]] + rank_set_index(sets[r], used);
    used |= sets[r];
  }
  return key;
}

void HandIndexer::suit_sets(const Shape& s, std::uint64_t key,
                            std::array<std::uint32_t, kMaxRounds>& sets) const {
  std::array<std::uint32_t, kMaxRounds> idx{};
  std::array<std::uint32_t, kMaxRounds> radix{};
  int used_count = 0;
  for (int r = 0; r < rounds(); ++r) {
    radix[r] = kChoose.c[kNumRanks - used_count][s[r]];
    used_count += s[r];
  }
  for (int r = rounds() - 1; r >= 0; --r) {
    idx[r] = static_cast<std::uint32_t>(key % radix[r]);
    key /= radix[r];
  }
  std::uint32_t used = 0;
  for (int r = 0; r < rounds(); ++r) {
    sets[r] = rank_set_from_index(idx[r], s[r], used);
    used |= sets[r];
  }
}

void HandIndexer::enumerate_configs() {
  // Every shape a single suit can have, sorted by descending pack().
  std::vector<Shape> shapes;
  Shape cur{};
  auto rec = [&](auto&& self, int r, int used) -> void {
    if (r == rounds()) {
      shapes.push_back(cur);
      return;
    }
    for (int c = 0; c <= std::min(cards_per_round_[r], kNumRanks - used); ++c) {
      cur[r] = static_cast<std::uint8_t>(c);
      self(self, r + 1, used + c);
    }
    cur[r] = 0;
  };
  rec(rec, 0, 0);
  std::sort(shapes.begin(), shapes.end(),
            [&](const Shape& a, const Shape& b) { return pack(a) > pack(b); });

  // Non-increasing 4-tuples of shapes whose per-round totals match.
  const int n = static_cast<int>(shapes.size());
  std::array<int, kNumSuits> pick{};
  auto choose_suit = [&](auto&& self, int suit, int from) -> void {
    if (suit == kNumSuits) {
      for (int r = 0; r < rounds(); ++r) {
        int total = 0;
        for (int i : pick) total += shapes[i][r];
        if (total != cards_per_round_[r]) return;
      }
      Config cfg{size_, 1, {}};
      std::uint64_t packed = 0;
      for (int i = 0; i < kNumSuits; ++i) {
        const Shape& sh = shapes[pick[i]];
        packed = (packed << 12) | pack(sh);
        if (!cfg.groups.empty() && cfg.groups.back().shape == sh) {
          ++cfg.groups.back().count;
        } else {
          cfg.groups.push_back({sh, 1, key_space(sh), 0});
        }
      }
      for (Group& g : cfg.groups) {
        g.combos = choose(g.keys + g.count - 1, g.count);
        cfg.size *= g.combos;
      }
      config_by_shapes_.emplace(packed, static_cast<int>(configs_.size()));
      size_ += cfg.size;
      configs_.push_back(std::move(cfg));
      return;
    }
    for (int i = from; i < n; ++i) {
      pick[suit] = i;
      self(self, suit + 1, i);
    }
  };
  choose_suit(choose_suit, 0, 0);
}

std::uint64_t HandIndexer::index(const Card* cards) const {
  std::array<std::array<std::uint32_t, kMaxRounds>, kNumSuits> sets{};
  for (int r = 0, i = 0; r < rounds(); ++r) {
    for (int j = 0; j < cards_per_round_[r]; ++j, ++i) {
      sets[suit_of(cards[i])][r] |= 1u << rank_of(cards[i]);
    }
  }

  struct SuitInfo {
    std::uint32_t pack;
    std::uint64_t key;
  };
  std::array<SuitInfo, kNumSuits> info{};
  for (int s = 0; s < kNumSuits; ++s) {
    Shape shape{};
    for (int r = 0; r < rounds(); ++r) shape[r] = static_cast<std::uint8_t>(popcount(sets[s][r]));
    info[s] = {pack(shape), suit_key(shape, sets[s])};
  }
  std::sort(info.begin(), info.end(), [](const SuitInfo& a, const SuitInfo& b) {
    return a.pack != b.pack ? a.pack > b.pack : a.key > b.key;
  });

  std::uint64_t packed = 0;
  for (const SuitInfo& si : info) packed = (packed << 12) | si.pack;
  const auto it = config_by_shapes_.find(packed);
  if (it == config_by_shapes_.end()) {
    throw std::invalid_argument("HandIndexer::index: cards don't match the round sizes");
  }
  const Config& cfg = configs_[it->second];

  std::uint64_t rem = 0;
  int s = 0;
  for (const Group& g : cfg.groups) {
    // The group's keys are descending in `info`; walk them ascending for the multiset index.
    std::uint64_t midx = 0;
    for (int i = 1; i <= g.count; ++i) {
      midx += choose(info[s + g.count - i].key + i - 1, i);
    }
    rem = rem * g.combos + midx;
    s += g.count;
  }
  return cfg.offset + rem;
}

void HandIndexer::unindex(std::uint64_t idx, Card* out) const {
  if (idx >= size_) throw std::out_of_range("HandIndexer::unindex: index out of range");
  const auto it = std::upper_bound(configs_.begin(), configs_.end(), idx,
                                   [](std::uint64_t v, const Config& c) { return v < c.offset; });
  const Config& cfg = *std::prev(it);
  std::uint64_t rem = idx - cfg.offset;

  std::array<std::uint64_t, kNumSuits> keys{};
  std::array<const Shape*, kNumSuits> shapes{};
  int s = kNumSuits;
  for (auto g = cfg.groups.rbegin(); g != cfg.groups.rend(); ++g) {
    std::uint64_t midx = rem % g->combos;
    rem /= g->combos;
    s -= g->count;
    // Decode a multiset of `count` keys: largest b with C(b, i) <= midx, for i = count..1.
    for (int i = g->count; i >= 1; --i) {
      std::uint64_t lo = i - 1;
      std::uint64_t hi = g->keys + g->count - 2;
      while (lo < hi) {
        const std::uint64_t mid = lo + (hi - lo + 1) / 2;
        if (choose(mid, i) <= midx) {
          lo = mid;
        } else {
          hi = mid - 1;
        }
      }
      midx -= choose(lo, i);
      // Ascending position i maps to descending suit slot count - i.
      keys[s + g->count - i] = lo - (i - 1);
      shapes[s + g->count - i] = &g->shape;
    }
  }

  std::array<std::vector<Card>, kMaxRounds> by_round;
  for (int suit = 0; suit < kNumSuits; ++suit) {
    std::array<std::uint32_t, kMaxRounds> sets{};
    suit_sets(*shapes[suit], keys[suit], sets);
    for (int r = 0; r < rounds(); ++r) {
      for (int rank = 0; rank < kNumRanks; ++rank) {
        if (sets[r] >> rank & 1) by_round[r].push_back(make_card(rank, suit));
      }
    }
  }
  int i = 0;
  for (int r = 0; r < rounds(); ++r) {
    std::sort(by_round[r].begin(), by_round[r].end());
    for (Card c : by_round[r]) out[i++] = c;
  }
}

}  // namespace regret
