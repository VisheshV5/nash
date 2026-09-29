#include "regret/games/nlhe.hpp"

#include <stdexcept>
#include <string>

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
  s.num_acts = 0;
  if (s.h.is_terminal()) return;
  const std::vector<Action> acts = abstraction::abstract_actions(s.h, actions_);
  if (acts.size() > kMaxActions) throw std::logic_error("Nlhe: too many abstract actions");
  for (const Action& a : acts) s.acts[s.num_acts++] = a;
}

void Nlhe::apply(State& s, int action_index) const {
  s.h.apply(s.acts[action_index]);
  s.history = extend_history(s.history, action_index);
  refresh_actions(s);
}

double Nlhe::utility(const State& s, int player) const {
  std::vector<std::array<Card, 2>> hole(s.hole.begin(), s.hole.begin() + num_players());
  const std::vector<Chips> pay = s.h.payoffs(hole, s.board);
  return static_cast<double>(pay[player]) / static_cast<double>(rules_.big_blind);
}

}  // namespace regret::games
