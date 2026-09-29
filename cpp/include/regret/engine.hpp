#pragma once

#include <array>
#include <cstdint>
#include <vector>

#include "regret/cards.hpp"

namespace regret {

// Integer chips. The Python layer maps big blinds to chips (100 chips = 1bb by default).
using Chips = std::int64_t;

inline constexpr int kMinPlayers = 2;
inline constexpr int kMaxPlayers = 6;

enum class Street : std::uint8_t { kPreflop = 0, kFlop, kTurn, kRiver };
enum class PlayerStatus : std::uint8_t { kActive = 0, kFolded, kAllIn };
enum class ActionType : std::uint8_t { kFold = 0, kCheckCall, kBetRaise };

struct Action {
  ActionType type;
  Chips amount = 0;  // kBetRaise only: the player's total bet this street after the action

  static Action fold() { return {ActionType::kFold, 0}; }
  static Action check_call() { return {ActionType::kCheckCall, 0}; }
  static Action bet_raise_to(Chips to) { return {ActionType::kBetRaise, to}; }
};

struct TableRules {
  std::vector<Chips> starting_stacks;  // one per seat; size is the number of players
  Chips small_blind;
  Chips big_blind;
};

struct LegalActions {
  bool fold = false;  // only when facing a bet
  bool check_call = false;
  Chips call_amount = 0;  // chips added by check/call (0 = check)
  bool bet_raise = false;
  Chips min_raise_to = 0;  // may equal max_raise_to (all-in for less than a full raise)
  Chips max_raise_to = 0;  // all-in
};

struct Pot {
  Chips amount;
  std::vector<int> eligible;  // seats that can win it, ascending
};

struct HistoryEntry {
  Street street;
  int seat;
  Action action;
};

// No-limit hold'em betting for 2-6 players, independent of the cards.
//
// Seats: 0 = SB, 1 = BB, then clockwise; the button is seat n-1, except heads-up, where the
// button is seat 0 (the SB). Preflop, the seat after the BB acts first (the SB heads-up);
// postflop, the first seat after the button acts first (the BB heads-up).
//
// Rules follow PokerKit's no-limit engine, which the tests fuzz against:
//   * min raise-to = current bet + max(largest raise increment this street, big blind);
//     anyone may go all-in for less;
//   * an all-in raise smaller than the largest increment doesn't reopen raising for players who
//     already acted, unless consecutive short all-ins add up to a full increment;
//   * nobody may bet or raise when no other player could call more than the current bet;
//   * a street's action skips players nobody can bet against (effective stack of zero);
//   * uncalled chips are returned; side pots with identical eligible players are merged;
//   * at showdown, hands that win nothing are mucked and pots left with the same contenders
//     are combined before splitting; odd chips go to the first winner clockwise from the
//     button.
//
// The state is a flat value type (no heap allocation unless history is recorded), because CFR
// copies it at every node.
class HandState {
 public:
  explicit HandState(const TableRules& rules, bool record_history = true);

  int num_players() const { return n_; }
  int button() const { return n_ == 2 ? 0 : n_ - 1; }
  Chips small_blind() const { return small_blind_; }
  Chips big_blind() const { return big_blind_; }
  Chips starting_stack(int seat) const { return starting_[seat]; }

  Street street() const { return street_; }
  bool is_terminal() const { return terminal_; }
  // Terminal with 2+ players left: needs a showdown (board may need running out).
  bool is_showdown() const { return terminal_ && non_folded() >= 2; }
  int to_act() const { return queue_size_ == 0 || terminal_ ? -1 : queue_[queue_head_]; }

  Chips stack(int seat) const { return stacks_[seat]; }
  Chips bet(int seat) const { return bets_[seat]; }                 // this street
  Chips contributed(int seat) const { return contributed_[seat]; }  // whole hand, incl. bet
  PlayerStatus status(int seat) const { return statuses_[seat]; }
  Chips current_bet() const;
  Chips pot() const;  // everything put in so far, including this street's bets
  int non_folded() const;
  // Bets and raises so far on this street (the blinds don't count).
  int raises_this_street() const { return raises_this_street_; }
  // Empty unless constructed with record_history.
  const std::vector<HistoryEntry>& history() const { return history_; }

  LegalActions legal_actions() const;
  // Empty if legal, else a human-readable reason.
  const char* why_illegal(const Action& a) const;
  void apply(const Action& a);  // throws std::invalid_argument if illegal

  // Main pot then side pots, built from total contributions.
  std::vector<Pot> pots() const;
  // Net chips won (+) or lost (-) per seat. `hole` holds 2 cards per seat (folded seats may
  // hold anything); `board` must be complete when there's a showdown.
  std::vector<Chips> payoffs(const std::vector<std::array<Card, 2>>& hole,
                             const std::array<Card, 5>& board) const;

 private:
  bool can_raise(int seat) const;
  Chips effective_stack(int seat) const;
  void pop_actor();
  void end_betting_round();
  void start_street(Street s);
  void queue_clear() { queue_head_ = queue_size_ = 0; }
  void queue_push(int seat) {
    queue_[(queue_head_ + queue_size_++) % kMaxPlayers] = static_cast<std::int8_t>(seat);
  }

  using Seats = std::array<Chips, kMaxPlayers>;
  int n_ = 0;
  Chips small_blind_ = 0;
  Chips big_blind_ = 0;
  Street street_ = Street::kPreflop;
  bool terminal_ = false;
  bool record_history_ = true;
  Seats starting_{};
  Seats stacks_{};
  Seats bets_{};
  Seats contributed_{};
  std::array<PlayerStatus, kMaxPlayers> statuses_{};  // seats >= n are kFolded
  std::array<bool, kMaxPlayers> acted_{};
  std::array<std::int8_t, kMaxPlayers> queue_{};  // seats still to act this round (ring)
  std::int8_t queue_head_ = 0;
  std::int8_t queue_size_ = 0;
  std::int8_t raises_this_street_ = 0;
  Chips raise_increment_ = 0;
  Chips short_all_in_sum_ = 0;  // consecutive short all-in raises since the last full one
  std::vector<HistoryEntry> history_;
};

}  // namespace regret
