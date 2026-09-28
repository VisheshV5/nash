#include "regret/evaluator.hpp"

namespace regret {
namespace {

inline int popcount(std::uint32_t x) { return __builtin_popcount(x); }
inline int top_rank(std::uint32_t mask) { return 31 - __builtin_clz(mask); }  // mask != 0

// Top rank of the best straight in a 13-bit rank mask, or -1. The wheel (A-2-3-4-5) has top
// rank 3 (the five).
inline int straight_top(std::uint32_t ranks) {
  // Bit j of `low` stands for rank j - 1, with bit 0 being the ace played low.
  const std::uint32_t low = (ranks << 1) | ((ranks >> 12) & 1);
  const std::uint32_t runs = low & (low >> 1) & (low >> 2) & (low >> 3) & (low >> 4);
  return runs ? top_rank(runs) + 3 : -1;
}

class Packer {
 public:
  explicit Packer(HandCategory c) : value_(static_cast<HandValue>(c) << 20) {}

  void rank(int r) {
    shift_ -= 4;
    value_ |= static_cast<HandValue>(r) << shift_;
  }

  // Appends the highest `n` ranks of `mask`.
  void top(std::uint32_t mask, int n) {
    for (int i = 0; i < n && mask; ++i) {
      const int r = top_rank(mask);
      rank(r);
      mask &= ~(1u << r);
    }
  }

  HandValue value() const { return value_; }

 private:
  HandValue value_;
  int shift_ = 20;
};

}  // namespace

HandValue evaluate(CardMask cards) {
  const std::uint32_t c = suit_ranks(cards, 0);
  const std::uint32_t d = suit_ranks(cards, 1);
  const std::uint32_t h = suit_ranks(cards, 2);
  const std::uint32_t s = suit_ranks(cards, 3);

  const std::uint32_t ranks = c | d | h | s;
  const std::uint32_t pairs_up = (c & d) | (c & h) | (c & s) | (d & h) | (d & s) | (h & s);
  const std::uint32_t trips_up = (c & d & h) | (c & d & s) | (c & h & s) | (d & h & s);
  const std::uint32_t quads = c & d & h & s;
  const std::uint32_t trips = trips_up & ~quads;
  const std::uint32_t pairs = pairs_up & ~trips_up;

  std::uint32_t flush = 0;
  for (const std::uint32_t suit : {c, d, h, s}) {
    if (popcount(suit) >= 5) flush = suit;
  }

  if (flush) {
    const int sf = straight_top(flush);
    if (sf >= 0) {
      Packer p(HandCategory::kStraightFlush);
      p.rank(sf);
      return p.value();
    }
  }
  if (quads) {
    Packer p(HandCategory::kFourOfAKind);
    const int q = top_rank(quads);
    p.rank(q);
    p.top(ranks & ~(1u << q), 1);
    return p.value();
  }
  if (trips && (popcount(trips) >= 2 || pairs)) {
    Packer p(HandCategory::kFullHouse);
    const int t = top_rank(trips);
    p.rank(t);
    p.top((trips & ~(1u << t)) | pairs, 1);
    return p.value();
  }
  if (flush) {
    Packer p(HandCategory::kFlush);
    p.top(flush, 5);
    return p.value();
  }
  if (const int st = straight_top(ranks); st >= 0) {
    Packer p(HandCategory::kStraight);
    p.rank(st);
    return p.value();
  }
  if (trips) {
    Packer p(HandCategory::kThreeOfAKind);
    p.rank(top_rank(trips));
    p.top(ranks & ~trips, 2);
    return p.value();
  }
  if (popcount(pairs) >= 2) {
    Packer p(HandCategory::kTwoPair);
    const int hi = top_rank(pairs);
    const int lo = top_rank(pairs & ~(1u << hi));
    p.rank(hi);
    p.rank(lo);
    p.top(ranks & ~(1u << hi) & ~(1u << lo), 1);
    return p.value();
  }
  if (pairs) {
    Packer p(HandCategory::kPair);
    p.rank(top_rank(pairs));
    p.top(ranks & ~pairs, 3);
    return p.value();
  }
  Packer p(HandCategory::kHighCard);
  p.top(ranks, 5);
  return p.value();
}

HandValue evaluate(const Card* cards, int n) {
  CardMask m = 0;
  for (int i = 0; i < n; ++i) m |= card_bit(cards[i]);
  return evaluate(m);
}

}  // namespace regret
