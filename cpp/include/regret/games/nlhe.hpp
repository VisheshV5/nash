#pragma once

#include <array>
#include <cstdint>
#include <memory>
#include <vector>

#include "regret/abstraction/actions.hpp"
#include "regret/cfr/rng.hpp"
#include "regret/engine.hpp"
#include "regret/isomorphism.hpp"

namespace regret::games {

// Card bucket tables, indexed by the suit-isomorphic (hole, board) index for {2,3}, {2,4}, {2,5}.
// Preflop buckets are the 169 classes (the {2} index) and need no table.
struct NlheTables {
  std::vector<std::uint16_t> flop, turn, river;
};

// No-limit hold'em for MCCFR: the real betting engine, restricted to the abstract actions, with
// card buckets for information sets.
//
// All cards (holes and the full board) are dealt at the root and every player's bucket for every
// street is looked up once, so traversal never touches the card tables. An infoset is
// (bucket of the player to act on this street, hash of the abstract action indices so far); the
// action sequence determines the street and who acts. Utilities are in big blinds.
class Nlhe {
 public:
  static constexpr int kMaxActions = 16;

  struct State {
    HandState h;
    std::array<Action, kMaxActions> acts{};
    std::uint8_t num_acts = 0;
    std::array<std::array<std::uint16_t, 4>, kMaxPlayers> bucket{};
    std::array<std::array<Card, 2>, kMaxPlayers> hole{};
    std::array<Card, 5> board{};
    std::uint64_t history = kHistoryRoot;
  };

  static constexpr std::uint64_t kHistoryRoot = 0x5EED0F7A11C0FFEEULL;
  static std::uint64_t extend_history(std::uint64_t history, int action_index) {
    return splitmix64(history ^
                      (0x9E3779B97F4A7C15ULL * static_cast<std::uint64_t>(action_index + 1)));
  }
  static std::uint64_t infoset_key(std::uint64_t history, int bucket) {
    return splitmix64(history + 0xD1B54A32D192ED03ULL * static_cast<std::uint64_t>(bucket + 1));
  }

  Nlhe(TableRules rules, abstraction::ActionRules actions,
       std::shared_ptr<const NlheTables> tables);

  int num_players() const { return static_cast<int>(rules_.starting_stacks.size()); }
  const abstraction::ActionRules& actions() const { return actions_; }
  const TableRules& rules() const { return rules_; }

  // Bucket of `hole` on `board` (0, 3, 4 or 5 cards).
  int bucket(const Card* hole, const Card* board, int board_size) const;

  State sample_deal(Rng& rng) const;
  bool is_terminal(const State& s) const { return s.h.is_terminal(); }
  int current_player(const State& s) const { return s.h.to_act(); }
  int num_actions(const State& s) const { return s.num_acts; }
  void apply(State& s, int action_index) const;
  std::uint64_t infoset_key(const State& s) const {
    const int p = s.h.to_act();
    return infoset_key(s.history, s.bucket[p][static_cast<int>(s.h.street())]);
  }
  double utility(const State& s, int player) const;

 private:
  void refresh_actions(State& s) const;

  TableRules rules_;
  abstraction::ActionRules actions_;
  std::shared_ptr<const NlheTables> tables_;
  HandIndexer preflop_{{2}}, flop_{{2, 3}}, turn_{{2, 4}}, river_{{2, 5}};
};

}  // namespace regret::games
