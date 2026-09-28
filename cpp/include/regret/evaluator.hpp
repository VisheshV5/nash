#pragma once

#include <cstdint>

#include "regret/cards.hpp"

namespace regret {

// Strength of a 5- to 7-card hand. Higher is better; equal values tie.
// Layout: category << 20 | five 4-bit rank slots (most significant first) that break ties
// within the category. Unused slots are zero.
using HandValue = std::uint32_t;

enum class HandCategory : std::uint8_t {
  kHighCard = 0,
  kPair,
  kTwoPair,
  kThreeOfAKind,
  kStraight,
  kFlush,
  kFullHouse,
  kFourOfAKind,
  kStraightFlush,
};

inline constexpr int kNumHandCategories = 9;

constexpr HandCategory category_of(HandValue v) { return static_cast<HandCategory>(v >> 20); }

// Best 5-card hand among the cards in `cards` (5 to 7 cards).
HandValue evaluate(CardMask cards);

// Same, for an array of `n` distinct cards (5 <= n <= 7).
HandValue evaluate(const Card* cards, int n);

}  // namespace regret
