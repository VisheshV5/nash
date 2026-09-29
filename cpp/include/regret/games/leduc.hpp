#pragma once

#include <array>
#include <cstdint>

#include "regret/cfr/rng.hpp"

namespace regret::games {

// Two-player Leduc hold'em: 6 cards (J, Q, K in two suits), ante 1 each, one private card each,
// then one public card. Two limit betting rounds (bet 2, then 4), at most 2 bets per round,
// player 0 acts first in each round. A pair with the board wins, otherwise the higher card;
// equal ranks split.
//
// Actions: 0 = fold (only when facing a bet), 1 = check/call, 2 = bet/raise. Infosets see
// legal actions in that order, skipping illegal ones.
class Leduc {
 public:
  struct State {
    std::array<std::uint8_t, 2> rank{};  // private card ranks 0..2
    std::uint8_t board = 0;              // public card rank, revealed in round 1
    std::uint8_t round = 0;
    std::uint8_t to_act = 0;
    std::uint8_t bets = 0;   // bets/raises this round
    std::uint8_t acted = 0;  // actions this round
    std::array<std::uint8_t, 2> contrib{1, 1};
    std::int8_t folded = -1;
    bool done = false;
    std::uint32_t history = 0;  // 2 bits per action, both rounds
    std::uint8_t len = 0;
  };

  int num_players() const { return 2; }

  // Ordered (p0 card, p1 card, board card) from 6 cards: 120 equally likely deals.
  int num_deals() const { return 120; }
  State deal(int index) const {
    int deck[6] = {0, 1, 2, 3, 4, 5};  // card c has rank c / 2
    int remaining = 6;
    int picked[3];
    for (int k = 0; k < 3; ++k) {
      const int pick = index % remaining;
      index /= remaining;
      picked[k] = deck[pick];
      for (int j = pick; j + 1 < remaining; ++j) deck[j] = deck[j + 1];
      --remaining;
    }
    State s;
    s.rank = {static_cast<std::uint8_t>(picked[0] / 2), static_cast<std::uint8_t>(picked[1] / 2)};
    s.board = static_cast<std::uint8_t>(picked[2] / 2);
    return s;
  }

  State sample_deal(Rng& rng) const { return deal(static_cast<int>(rng.below(num_deals()))); }

  bool is_terminal(const State& s) const { return s.done; }
  int current_player(const State& s) const { return s.to_act; }

  bool facing_bet(const State& s) const { return s.contrib[s.to_act] < s.contrib[1 - s.to_act]; }

  // Legal actions in order; returns how many.
  int legal(const State& s, int* out) const {
    int n = 0;
    if (facing_bet(s)) out[n++] = 0;
    out[n++] = 1;
    if (s.bets < 2) out[n++] = 2;
    return n;
  }
  int num_actions(const State& s) const {
    int buf[3];
    return legal(s, buf);
  }

  // `index` is a position in legal(s).
  void apply(State& s, int index) const {
    int acts[3];
    legal(s, acts);
    const int a = acts[index];
    s.history |= static_cast<std::uint32_t>(a) << (2 * s.len);
    ++s.len;
    ++s.acted;
    const int p = s.to_act;
    const int bet_size = s.round == 0 ? 2 : 4;
    if (a == 0) {
      s.folded = static_cast<std::int8_t>(p);
      s.done = true;
      return;
    }
    if (a == 2) {
      s.contrib[p] = static_cast<std::uint8_t>(s.contrib[1 - p] + bet_size);
      ++s.bets;
      s.to_act = static_cast<std::uint8_t>(1 - p);
      return;
    }
    // Check or call.
    const bool was_call = s.contrib[p] < s.contrib[1 - p];
    s.contrib[p] = s.contrib[1 - p];
    if (was_call || s.acted >= 2) {  // round over
      if (s.round == 1) {
        s.done = true;
      } else {
        s.round = 1;
        s.to_act = 0;
        s.bets = 0;
        s.acted = 0;
      }
      return;
    }
    s.to_act = static_cast<std::uint8_t>(1 - p);
  }

  std::uint64_t infoset_key(const State& s) const {
    const std::uint64_t board = s.round == 1 ? s.board + 1u : 0u;
    return std::uint64_t{s.rank[s.to_act]} | (board << 2) | (std::uint64_t{s.len} << 4) |
           (std::uint64_t{s.history} << 8);
  }

  double utility(const State& s, int p) const {
    const int o = 1 - p;
    if (s.folded >= 0) return s.folded == p ? -double(s.contrib[p]) : double(s.contrib[o]);
    const int mine = strength(s.rank[p], s.board);
    const int theirs = strength(s.rank[o], s.board);
    if (mine == theirs) return 0.0;
    return mine > theirs ? double(s.contrib[o]) : -double(s.contrib[p]);
  }

 private:
  static int strength(int rank, int board) { return rank == board ? 10 + rank : rank; }
};

}  // namespace regret::games
