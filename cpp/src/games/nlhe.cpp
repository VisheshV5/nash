#include "regret/games/nlhe.hpp"

#include <algorithm>
#include <stdexcept>
#include <string>

#include "regret/evaluator.hpp"

namespace regret::games {

Nlhe::Nlhe(TableRules rules, abstraction::ActionRules actions,
           std::shared_ptr<const NlheTables> tables)
    : rules_(std::move(rules)), actions_(std::move(actions)), tables_(std::move(tables)) {
  if (!tables_ || tables_->flop.size() != flop_.size() || tables_->turn.size() != turn_.size() ||
      tables_->river.size() != river_.size()) {
    throw std::invalid_argument("Nlhe: bucket tables have the wrong sizes");
  }
  HandState probe(rules_);  // validates the table rules
}

int Nlhe::bucket(const Card* hole, const Card* board, int board_size) const {
  Card cards[7] = {hole[0], hole[1]};
  for (int i = 0; i < board_size; ++i) cards[2 + i] = board[i];
  switch (board_size) {
    case 0:
      return static_cast<int>(preflop_.index(cards));
    case 3:
      return tables_->flop[flop_.index(cards)];
    case 4:
      return tables_->turn[turn_.index(cards)];
    case 5:
      return tables_->river[river_.index(cards)];
    default:
      throw std::invalid_argument("Nlhe::bucket: board must have 0, 3, 4 or 5 cards");
  }
}

Nlhe::State Nlhe::sample_deal(Rng& rng) const {
  State s{HandState(rules_, /*record_history=*/false)};
  const int n = num_players();
  CardMask used = 0;
  auto draw = [&] {
    for (;;) {
      const Card c = static_cast<Card>(rng.below(kNumCards));
      if (!(used & card_bit(c))) {
        used |= card_bit(c);
        return c;
      }
    }
  };
  for (int p = 0; p < n; ++p) s.hole[p] = {draw(), draw()};
  for (Card& c : s.board) c = draw();
  for (int p = 0; p < n; ++p) {
    Card seven[7] = {s.hole[p][0], s.hole[p][1]};
    std::copy(s.board.begin(), s.board.end(), seven + 2);
    s.value[p] = evaluate(seven, 7);
  }
  static constexpr int kBoardSize[4] = {0, 3, 4, 5};
  for (int p = 0; p < n; ++p) {
    for (int street = 0; street < 4; ++street) {
      s.bucket[p][street] =
          static_cast<std::uint16_t>(bucket(s.hole[p].data(), s.board.data(), kBoardSize[street]));
    }
  }
  refresh_actions(s);
  return s;
}

void Nlhe::refresh_actions(State& s) const {
  s.num_acts =
      s.h.is_terminal()
          ? 0
          : static_cast<std::uint8_t>(abstraction::abstract_actions(s.h, actions_, s.acts.data()));
}

void Nlhe::apply(State& s, int action_index) const {
  s.h.apply(s.acts[action_index]);
  s.history = extend_history(s.history, action_index);
  refresh_actions(s);
}

double Nlhe::utility(const State& s, int player) const {
  const double bb = static_cast<double>(rules_.big_blind);
  if (num_players() == 2) {
    // Heads-up: the loser pays what both put in up to the smaller contribution (any excess is
    // an uncalled bet and comes back); a fold loses the folder's whole contribution.
    const int o = 1 - player;
    const Chips mine = s.h.contributed(player), theirs = s.h.contributed(o);
    if (s.h.status(player) == PlayerStatus::kFolded) return -static_cast<double>(mine) / bb;
    if (s.h.status(o) == PlayerStatus::kFolded) return static_cast<double>(theirs) / bb;
    const double stake = static_cast<double>(std::min(mine, theirs)) / bb;
    if (s.value[player] == s.value[o]) return 0.0;
    return s.value[player] > s.value[o] ? stake : -stake;
  }
  std::vector<std::array<Card, 2>> hole(s.hole.begin(), s.hole.begin() + num_players());
  const std::vector<Chips> pay = s.h.payoffs(hole, s.board);
  return static_cast<double>(pay[player]) / static_cast<double>(rules_.big_blind);
}

}  // namespace regret::games
