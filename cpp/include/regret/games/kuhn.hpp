#pragma once

#include <array>
#include <cstdint>
#include <stdexcept>

namespace regret::games {

// N-player Kuhn poker (N = 2 or 3): N + 1 cards, ante 1, one betting round with a single bet of
// 1. Players act in order; before any bet each may check (0) or bet (1). After a bet, each other
// player in turn calls (1) or folds (0). Highest card among players still in wins the pot.
// For N = 2 the game value is -1/18 for player 0.
class Kuhn {
 public:
  static constexpr int kMaxPlayers = 3;

  struct State {
    std::array<std::uint8_t, kMaxPlayers> card{};
    std::array<std::uint8_t, kMaxPlayers> in{};  // 1 if the player called/bet (after a bet)
    std::uint8_t len = 0;                        // actions taken
    std::uint8_t history = 0;                    // action bits, first action in bit 0
    std::int8_t bettor = -1;
    std::uint8_t to_act = 0;
    std::uint8_t responses = 0;  // players who have answered the bet
  };

  explicit Kuhn(int players) : n_(players) {
    if (players < 2 || players > kMaxPlayers) throw std::invalid_argument("Kuhn: 2 or 3 players");
  }

  int num_players() const { return n_; }

  // Ordered deals of N distinct cards from N + 1, all equally likely.
  int num_deals() const { return n_ == 2 ? 6 : 24; }
  State deal(int index) const {
    State s;
    std::array<std::uint8_t, kMaxPlayers + 1> deck{};
    for (int i = 0; i <= n_; ++i) deck[i] = static_cast<std::uint8_t>(i);
    int remaining = n_ + 1;
    for (int p = 0; p < n_; ++p) {
      const int pick = index % remaining;
      index /= remaining;
      s.card[p] = deck[pick];
      for (int j = pick; j + 1 < remaining; ++j) deck[j] = deck[j + 1];
      --remaining;
    }
    return s;
  }

  bool is_terminal(const State& s) const {
    if (s.bettor < 0) return s.len == n_;
    return s.responses == n_ - 1;
  }
  int current_player(const State& s) const { return s.to_act; }
  int num_actions(const State&) const { return 2; }

  void apply(State& s, int a) const {
    s.history |= static_cast<std::uint8_t>(a << s.len);
    ++s.len;
    if (s.bettor < 0) {
      if (a == 1) {
        s.bettor = static_cast<std::int8_t>(s.to_act);
        s.in[s.to_act] = 1;
      }
    } else {
      s.in[s.to_act] = static_cast<std::uint8_t>(a);
      ++s.responses;
    }
    s.to_act = static_cast<std::uint8_t>((s.to_act + 1) % n_);
  }

  std::uint64_t infoset_key(const State& s) const {
    return static_cast<std::uint64_t>(s.card[s.to_act]) | (std::uint64_t{s.len} << 4) |
           (std::uint64_t{s.history} << 8);
  }

  double utility(const State& s, int p) const {
    double pot = n_;
    int winner = -1;
    for (int i = 0; i < n_; ++i) {
      const bool contending = s.bettor < 0 || s.in[i];
      if (s.bettor >= 0 && s.in[i]) pot += 1;
      if (contending && (winner < 0 || s.card[i] > s.card[winner])) winner = i;
    }
    const double paid = 1 + (s.bettor >= 0 && s.in[p] ? 1 : 0);
    return (winner == p ? pot : 0.0) - paid;
  }

 private:
  int n_;
};

}  // namespace regret::games
