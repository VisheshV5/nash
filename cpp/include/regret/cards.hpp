#pragma once

#include <cstdint>
#include <optional>
#include <string>
#include <string_view>

namespace regret {

// A card is rank * 4 + suit, so 0..51.
// Ranks 0..12 are 2..A. Suits 0..3 are c, d, h, s.
using Card = std::uint8_t;

inline constexpr int kNumRanks = 13;
inline constexpr int kNumSuits = 4;
inline constexpr int kNumCards = 52;

inline constexpr char kRankChars[] = "23456789TJQKA";
inline constexpr char kSuitChars[] = "cdhs";

constexpr int rank_of(Card c) { return c >> 2; }
constexpr int suit_of(Card c) { return c & 3; }
constexpr Card make_card(int rank, int suit) { return static_cast<Card>(rank * 4 + suit); }

// A set of cards as 4 lanes of 16 bits, one lane per suit: bit (suit * 16 + rank).
// Lanes make per-suit rank masks a shift and a mask away.
using CardMask = std::uint64_t;

inline constexpr std::uint32_t kRankMaskAll = 0x1FFF;

constexpr CardMask card_bit(Card c) { return CardMask{1} << (suit_of(c) * 16 + rank_of(c)); }

constexpr std::uint32_t suit_ranks(CardMask m, int suit) {
  return static_cast<std::uint32_t>(m >> (suit * 16)) & kRankMaskAll;
}

// Parses "As", "Td", "2c". Rank is case-insensitive; suit must be one of "cdhs" (either case).
inline std::optional<Card> parse_card(std::string_view s) {
  if (s.size() != 2) return std::nullopt;
  const std::string_view ranks = kRankChars;
  const std::string_view suits = kSuitChars;
  char r = s[0];
  char su = s[1];
  if (r >= 'a' && r <= 'z') r = static_cast<char>(r - 'a' + 'A');
  if (su >= 'A' && su <= 'Z') su = static_cast<char>(su - 'A' + 'a');
  const auto ri = ranks.find(r);
  const auto si = suits.find(su);
  if (ri == std::string_view::npos || si == std::string_view::npos) return std::nullopt;
  return make_card(static_cast<int>(ri), static_cast<int>(si));
}

inline std::string card_to_string(Card c) {
  return {kRankChars[rank_of(c)], kSuitChars[suit_of(c)]};
}

}  // namespace regret
