// Exhaustive checks that are too slow for Python. Run: regret_tests [--slow]
#include <algorithm>
#include <array>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <set>
#include <vector>

#include "regret/cards.hpp"
#include "regret/evaluator.hpp"
#include "regret/isomorphism.hpp"

using namespace regret;

namespace {

int g_failures = 0;

#define CHECK(cond)                                                        \
  do {                                                                     \
    if (!(cond)) {                                                         \
      std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond); \
      ++g_failures;                                                        \
    }                                                                      \
  } while (0)

#define CHECK_EQ(a, b)                                                                          \
  do {                                                                                          \
    const auto va = (a);                                                                        \
    const auto vb = (b);                                                                        \
    if (!(va == vb)) {                                                                          \
      std::fprintf(stderr, "FAIL %s:%d: %s == %s (%llu vs %llu)\n", __FILE__, __LINE__, #a, #b, \
                   static_cast<unsigned long long>(va), static_cast<unsigned long long>(vb));   \
      ++g_failures;                                                                             \
    }                                                                                           \
  } while (0)

struct Rng {  // xorshift64*; tests only
  std::uint64_t s;
  std::uint64_t next() {
    s ^= s >> 12;
    s ^= s << 25;
    s ^= s >> 27;
    return s * 2685821657736338717ULL;
  }
  int below(int n) { return static_cast<int>(next() % static_cast<std::uint64_t>(n)); }
};

// k distinct random cards.
void deal(Rng& rng, Card* out, int k) {
  std::array<Card, kNumCards> deck{};
  for (int i = 0; i < kNumCards; ++i) deck[i] = static_cast<Card>(i);
  for (int i = 0; i < k; ++i) {
    std::swap(deck[i], deck[i + rng.below(kNumCards - i)]);
    out[i] = deck[i];
  }
}

double seconds_since(std::chrono::steady_clock::time_point t0) {
  return std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
}

void test_five_card_exhaustive() {
  std::array<std::uint64_t, kNumHandCategories> counts{};
  std::set<HandValue> distinct;
  Card c[5];
  for (c[0] = 0; c[0] < 52; ++c[0])
    for (c[1] = c[0] + 1; c[1] < 52; ++c[1])
      for (c[2] = c[1] + 1; c[2] < 52; ++c[2])
        for (c[3] = c[2] + 1; c[3] < 52; ++c[3])
          for (c[4] = c[3] + 1; c[4] < 52; ++c[4]) {
            const HandValue v = evaluate(c, 5);
            ++counts[static_cast<int>(category_of(v))];
            distinct.insert(v);
          }
  const std::array<std::uint64_t, kNumHandCategories> expected = {
      1302540, 1098240, 123552, 54912, 10200, 5108, 3744, 624, 40};
  for (int i = 0; i < kNumHandCategories; ++i) CHECK_EQ(counts[i], expected[i]);
  CHECK_EQ(distinct.size(), std::size_t{7462});
}

void test_seven_card_exhaustive() {
  std::array<std::uint64_t, kNumHandCategories> counts{};
  const auto t0 = std::chrono::steady_clock::now();
  CardMask m[8] = {};
  int a[7];
  for (a[0] = 0; a[0] < 52; ++a[0]) {
    m[1] = m[0] | card_bit(a[0]);
    for (a[1] = a[0] + 1; a[1] < 52; ++a[1]) {
      m[2] = m[1] | card_bit(a[1]);
      for (a[2] = a[1] + 1; a[2] < 52; ++a[2]) {
        m[3] = m[2] | card_bit(a[2]);
        for (a[3] = a[2] + 1; a[3] < 52; ++a[3]) {
          m[4] = m[3] | card_bit(a[3]);
          for (a[4] = a[3] + 1; a[4] < 52; ++a[4]) {
            m[5] = m[4] | card_bit(a[4]);
            for (a[5] = a[4] + 1; a[5] < 52; ++a[5]) {
              m[6] = m[5] | card_bit(a[5]);
              for (a[6] = a[5] + 1; a[6] < 52; ++a[6]) {
                ++counts[static_cast<int>(category_of(evaluate(m[6] | card_bit(a[6]))))];
              }
            }
          }
        }
      }
    }
  }
  const std::array<std::uint64_t, kNumHandCategories> expected = {
      23294460, 58627800, 31433400, 6461620, 6180020, 4047644, 3473184, 224848, 41584};
  for (int i = 0; i < kNumHandCategories; ++i) CHECK_EQ(counts[i], expected[i]);
  std::printf("  7-card exhaustive (133,784,560 hands): %.2fs\n", seconds_since(t0));
}

void test_seven_equals_best_five() {
  Rng rng{0x9E3779B97F4A7C15ULL};
  Card c[7];
  Card sub[5];
  for (int trial = 0; trial < 200000; ++trial) {
    deal(rng, c, 7);
    HandValue best = 0;
    for (int skip1 = 0; skip1 < 7; ++skip1)
      for (int skip2 = skip1 + 1; skip2 < 7; ++skip2) {
        int k = 0;
        for (int i = 0; i < 7; ++i)
          if (i != skip1 && i != skip2) sub[k++] = c[i];
        best = std::max(best, evaluate(sub, 5));
      }
    CHECK_EQ(evaluate(c, 7), best);
    if (g_failures) return;
  }
}

void bench_evaluator() {
  constexpr int kHands = 10'000'000;
  Rng rng{42};
  std::vector<Card> cards(static_cast<std::size_t>(kHands) * 7);
  for (int i = 0; i < kHands; ++i) deal(rng, &cards[static_cast<std::size_t>(i) * 7], 7);
  const auto t0 = std::chrono::steady_clock::now();
  std::uint64_t sink = 0;
  for (int i = 0; i < kHands; ++i) sink += evaluate(&cards[static_cast<std::size_t>(i) * 7], 7);
  const double s = seconds_since(t0);
  std::printf("  evaluate(7 cards): %.1fM hands/s on one thread (checksum %llu)\n",
              kHands / s / 1e6, static_cast<unsigned long long>(sink));
}

void test_indexer_sizes() {
  // Board as one set (what the card abstraction uses).
  CHECK_EQ(HandIndexer({2}).size(), std::uint64_t{169});
  CHECK_EQ(HandIndexer({2, 3}).size(), std::uint64_t{1286792});
  CHECK_EQ(HandIndexer({2, 4}).size(), std::uint64_t{13960050});
  CHECK_EQ(HandIndexer({2, 5}).size(), std::uint64_t{123156254});
  // Board split by street (Waugh 2013's per-round counts).
  CHECK_EQ(HandIndexer({2, 3, 1}).size(), std::uint64_t{55190538});
  CHECK_EQ(HandIndexer({2, 3, 1, 1}).size(), std::uint64_t{2428287420});
}

// Every raw deal maps into range, and every index is hit. Together with suit-permutation
// invariance this proves index classes == isomorphism classes.
void test_surjective(const HandIndexer& h) {
  std::vector<std::uint8_t> hit(h.size(), 0);
  std::uint64_t deals = 0;
  const int n = h.total_cards();
  std::vector<Card> cards(n);
  // Enumerate ordered-by-round deals: each round is a combination from the remaining deck.
  auto rec = [&](auto&& self, int round, int pos, CardMask used) -> void {
    if (round == h.rounds()) {
      const std::uint64_t idx = h.index(cards.data());
      CHECK(idx < h.size());
      if (idx < h.size()) hit[idx] = 1;
      ++deals;
      return;
    }
    const int k = h.cards_in_round(round);
    auto combo = [&](auto&& cself, int j, int from) -> void {
      if (j == k) {
        CardMask u = used;
        for (int t = 0; t < k; ++t) u |= card_bit(cards[pos + t]);
        self(self, round + 1, pos + k, u);
        return;
      }
      for (int c = from; c < kNumCards; ++c) {
        if (used & card_bit(static_cast<Card>(c))) continue;
        cards[pos + j] = static_cast<Card>(c);
        cself(cself, j + 1, c + 1);
      }
    };
    combo(combo, 0, 0);
  };
  rec(rec, 0, 0, 0);
  CHECK_EQ(static_cast<std::uint64_t>(std::count(hit.begin(), hit.end(), 1)), h.size());
  std::printf("  surjective over %llu deals\n", static_cast<unsigned long long>(deals));
}

void test_round_trip(const HandIndexer& h, std::uint64_t stride) {
  const auto t0 = std::chrono::steady_clock::now();
  std::vector<Card> cards(h.total_cards());
  for (std::uint64_t i = 0; i < h.size(); i += stride) {
    h.unindex(i, cards.data());
    const std::uint64_t back = h.index(cards.data());
    if (back != i) {
      CHECK_EQ(back, i);
      return;
    }
  }
  std::printf("  round trip of %llu indices: %.2fs\n",
              static_cast<unsigned long long>((h.size() + stride - 1) / stride), seconds_since(t0));
}

void test_suit_permutation_invariance(const HandIndexer& h) {
  std::array<int, 4> perm = {0, 1, 2, 3};
  std::vector<std::array<int, 4>> perms;
  do perms.push_back(perm);
  while (std::next_permutation(perm.begin(), perm.end()));

  Rng rng{7};
  const int n = h.total_cards();
  std::vector<Card> cards(n);
  std::vector<Card> permuted(n);
  for (int trial = 0; trial < 20000; ++trial) {
    deal(rng, cards.data(), n);
    const std::uint64_t idx = h.index(cards.data());
    for (const auto& p : perms) {
      for (int i = 0; i < n; ++i) permuted[i] = make_card(rank_of(cards[i]), p[suit_of(cards[i])]);
      // Shuffle within rounds too: order inside a round must not matter.
      for (int r = 0, pos = 0; r < h.rounds(); pos += h.cards_in_round(r), ++r) {
        std::reverse(permuted.begin() + pos, permuted.begin() + pos + h.cards_in_round(r));
      }
      if (h.index(permuted.data()) != idx) {
        CHECK_EQ(h.index(permuted.data()), idx);
        return;
      }
    }
  }
}

}  // namespace

int main(int argc, char** argv) {
  const bool slow = argc > 1 && std::strcmp(argv[1], "--slow") == 0;

  std::printf("evaluator\n");
  test_five_card_exhaustive();
  test_seven_card_exhaustive();
  test_seven_equals_best_five();
  bench_evaluator();

  std::printf("isomorphism\n");
  test_indexer_sizes();
  const HandIndexer preflop({2}), flop({2, 3}), turn({2, 4}), river({2, 5});
  const HandIndexer turn_split({2, 3, 1}), river_split({2, 3, 1, 1});
  test_surjective(preflop);
  test_surjective(flop);
  if (slow) test_surjective(turn);
  for (const HandIndexer* h : {&preflop, &flop, &turn, &river, &turn_split, &river_split}) {
    test_suit_permutation_invariance(*h);
  }
  test_round_trip(preflop, 1);
  test_round_trip(flop, 1);
  test_round_trip(turn, slow ? 1 : 13);
  test_round_trip(river, slow ? 1 : 97);
  test_round_trip(turn_split, slow ? 1 : 97);
  test_round_trip(river_split, slow ? 97 : 9973);  // 2.4B indices; full pass ~40 min

  std::printf(g_failures ? "%d FAILURE(S)\n" : "all passed\n", g_failures);
  return g_failures ? 1 : 0;
}
