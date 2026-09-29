#include "regret/engine.hpp"

#include <algorithm>
#include <numeric>
#include <stdexcept>
#include <string>

#include "regret/evaluator.hpp"

namespace regret {

HandState::HandState(const TableRules& rules, bool record_history)
    : small_blind_(rules.small_blind),
      big_blind_(rules.big_blind),
      record_history_(record_history) {
  const int n = static_cast<int>(rules.starting_stacks.size());
  if (n < kMinPlayers || n > kMaxPlayers) {
    throw std::invalid_argument("HandState: need 2-6 players, got " + std::to_string(n));
  }
  if (rules.small_blind <= 0 || rules.big_blind <= rules.small_blind) {
    throw std::invalid_argument("HandState: need 0 < small blind < big blind");
  }
  for (Chips s : rules.starting_stacks) {
    if (s <= 0) throw std::invalid_argument("HandState: starting stacks must be positive");
  }

  n_ = n;
  statuses_.fill(PlayerStatus::kFolded);
  for (int i = 0; i < n; ++i) {
    starting_[i] = stacks_[i] = rules.starting_stacks[i];
    statuses_[i] = PlayerStatus::kActive;
  }

  const Chips blinds[2] = {rules.small_blind, rules.big_blind};
  for (int seat = 0; seat < 2; ++seat) {
    const Chips post = std::min(blinds[seat], stacks_[seat]);
    stacks_[seat] -= post;
    bets_[seat] = post;
    contributed_[seat] = post;
    if (stacks_[seat] == 0) statuses_[seat] = PlayerStatus::kAllIn;
  }
  start_street(Street::kPreflop);
}

Chips HandState::current_bet() const { return *std::max_element(bets_.begin(), bets_.end()); }

Chips HandState::pot() const {
  return std::accumulate(contributed_.begin(), contributed_.end(), Chips{0});
}

int HandState::non_folded() const {
  return static_cast<int>(std::count_if(statuses_.begin(), statuses_.end(),
                                        [](PlayerStatus s) { return s != PlayerStatus::kFolded; }));
}

void HandState::start_street(Street s) {
  street_ = s;
  acted_.fill(false);
  raise_increment_ = 0;
  short_all_in_sum_ = 0;
  raises_this_street_ = 0;

  const int n = num_players();
  const int opener = s == Street::kPreflop ? 2 % n : (button() + 1) % n;
  queue_clear();
  for (int k = 0; k < n; ++k) {
    const int seat = (opener + k) % n;
    if (statuses_[seat] == PlayerStatus::kActive && effective_stack(seat) > 0) queue_push(seat);
  }
  // Nobody to act, or a lone player who already matches the bet (e.g. everyone else all-in).
  if (queue_size_ == 0 || (queue_size_ == 1 && bets_[queue_[queue_head_]] >= current_bet())) {
    end_betting_round();
  }
}

void HandState::end_betting_round() {
  queue_clear();
  bets_.fill(0);

  const int with_chips =
      static_cast<int>(std::count_if(statuses_.begin(), statuses_.end(),
                                     [](PlayerStatus s) { return s == PlayerStatus::kActive; }));
  if (non_folded() <= 1 || street_ == Street::kRiver || with_chips <= 1) {
    terminal_ = true;  // uncontested, or showdown (running out the board if needed)
    return;
  }
  start_street(static_cast<Street>(static_cast<int>(street_) + 1));
}

Chips HandState::effective_stack(int seat) const {
  // What the seat can still lose: capped by the second-largest total (bet + stack) among players
  // still in the hand.
  Chips largest = 0;
  Chips second = 0;
  for (int i = 0; i < num_players(); ++i) {
    if (statuses_[i] == PlayerStatus::kFolded) continue;
    const Chips total = bets_[i] + stacks_[i];
    if (total > largest) {
      second = largest;
      largest = total;
    } else if (total > second) {
      second = total;
    }
  }
  return std::min(stacks_[seat], std::max(Chips{0}, second - bets_[seat]));
}

bool HandState::can_raise(int seat) const {
  const Chips cur = current_bet();
  if (short_all_in_sum_ > 0 && acted_[seat] && short_all_in_sum_ < raise_increment_) {
    return false;  // short all-in didn't reopen the action
  }
  if (stacks_[seat] <= cur - bets_[seat]) return false;  // can only call (all-in) anyway
  for (int i = 0; i < num_players(); ++i) {
    if (i != seat && statuses_[i] != PlayerStatus::kFolded && stacks_[i] + bets_[i] > cur) {
      return true;
    }
  }
  return false;  // nobody left who could call more than the current bet
}

LegalActions HandState::legal_actions() const {
  LegalActions la;
  const int seat = to_act();
  if (seat < 0) return la;
  const Chips cur = current_bet();
  la.fold = bets_[seat] < cur;
  la.check_call = true;
  la.call_amount = std::min(cur - bets_[seat], stacks_[seat]);
  if (can_raise(seat)) {
    la.bet_raise = true;
    la.max_raise_to = stacks_[seat] + bets_[seat];
    la.min_raise_to = std::min(la.max_raise_to, cur + std::max(raise_increment_, big_blind_));
  }
  return la;
}

const char* HandState::why_illegal(const Action& a) const {
  if (terminal_) return "the hand is over";
  if (to_act() < 0) return "no player to act";
  const LegalActions la = legal_actions();
  switch (a.type) {
    case ActionType::kFold:
      return la.fold ? nullptr : "cannot fold when not facing a bet (check instead)";
    case ActionType::kCheckCall:
      return nullptr;
    case ActionType::kBetRaise:
      if (!la.bet_raise) return "betting or raising is not allowed here";
      if (a.amount < la.min_raise_to) return "bet/raise is below the minimum";
      if (a.amount > la.max_raise_to) return "bet/raise is more than the player's stack";
      return nullptr;
  }
  return "unknown action type";
}

void HandState::pop_actor() {
  acted_[queue_[queue_head_]] = true;
  queue_head_ = static_cast<std::int8_t>((queue_head_ + 1) % kMaxPlayers);
  --queue_size_;
}

void HandState::apply(const Action& a) {
  if (const char* why = why_illegal(a)) throw std::invalid_argument(why);
  const int seat = to_act();
  if (record_history_) history_.push_back({street_, seat, a});

  switch (a.type) {
    case ActionType::kFold:
      statuses_[seat] = PlayerStatus::kFolded;
      pop_actor();
      break;

    case ActionType::kCheckCall: {
      const Chips amount = std::min(current_bet() - bets_[seat], stacks_[seat]);
      stacks_[seat] -= amount;
      bets_[seat] += amount;
      contributed_[seat] += amount;
      if (stacks_[seat] == 0) statuses_[seat] = PlayerStatus::kAllIn;
      pop_actor();
      break;
    }

    case ActionType::kBetRaise: {
      const Chips prev = current_bet();
      const Chips delta = a.amount - bets_[seat];
      stacks_[seat] -= delta;
      bets_[seat] = a.amount;
      contributed_[seat] += delta;
      pop_actor();
      ++raises_this_street_;
      if (stacks_[seat] == 0) statuses_[seat] = PlayerStatus::kAllIn;

      // Everyone else with chips acts again.
      queue_clear();
      const int n = num_players();
      for (int k = 1; k < n; ++k) {
        const int s = (seat + k) % n;
        if (statuses_[s] == PlayerStatus::kActive) queue_push(s);
      }

      const Chips increment = a.amount - prev;
      if (increment >= raise_increment_) {  // full raise: reopens raising for everyone
        acted_.fill(false);
        acted_[seat] = true;
      }
      raise_increment_ = std::max(raise_increment_, increment);
      short_all_in_sum_ = stacks_[seat] > 0 ? 0 : short_all_in_sum_ + increment;
      if (short_all_in_sum_ >= raise_increment_) short_all_in_sum_ = 0;
      break;
    }
  }

  if (queue_size_ == 0 || non_folded() <= 1) end_betting_round();
}

std::vector<Pot> HandState::pots() const {
  std::vector<Chips> levels;
  for (int i = 0; i < num_players(); ++i) {
    if (contributed_[i] > 0) levels.push_back(contributed_[i]);
  }
  std::sort(levels.begin(), levels.end());
  levels.erase(std::unique(levels.begin(), levels.end()), levels.end());

  std::vector<Pot> pots;
  Chips prev = 0;
  for (Chips level : levels) {
    Pot pot{0, {}};
    for (int i = 0; i < num_players(); ++i) {
      if (contributed_[i] >= level) {
        pot.amount += level - prev;
        if (statuses_[i] != PlayerStatus::kFolded) pot.eligible.push_back(i);
      }
    }
    if (!pots.empty() && pots.back().eligible == pot.eligible) {
      pots.back().amount += pot.amount;
    } else {
      pots.push_back(std::move(pot));
    }
    prev = level;
  }
  return pots;
}

std::vector<Chips> HandState::payoffs(const std::vector<std::array<Card, 2>>& hole,
                                      const std::array<Card, 5>& board) const {
  if (!terminal_) throw std::logic_error("payoffs: the hand isn't over");
  const int n = num_players();
  std::vector<Chips> out(n);
  for (int i = 0; i < n; ++i) out[i] = -contributed_[i];

  if (non_folded() == 1) {
    const int winner =
        static_cast<int>(std::find_if(statuses_.begin(), statuses_.end(),
                                      [](PlayerStatus s) { return s != PlayerStatus::kFolded; }) -
                         statuses_.begin());
    out[winner] += pot();
    return out;
  }

  if (static_cast<int>(hole.size()) != n) {
    throw std::invalid_argument("payoffs: need hole cards for every seat");
  }
  CardMask board_mask = 0;
  for (Card c : board) {
    if (c >= kNumCards || (board_mask & card_bit(c))) {
      throw std::invalid_argument("payoffs: invalid or duplicate board card");
    }
    board_mask |= card_bit(c);
  }
  std::vector<HandValue> value(n, 0);
  CardMask seen = board_mask;
  for (int i = 0; i < n; ++i) {
    if (statuses_[i] == PlayerStatus::kFolded) continue;
    CardMask m = board_mask;
    for (Card c : hole[i]) {
      if (c >= kNumCards || (seen & card_bit(c))) {
        throw std::invalid_argument("payoffs: invalid or duplicate hole card");
      }
      seen |= card_bit(c);
      m |= card_bit(c);
    }
    value[i] = evaluate(m);
  }

  // Order for odd chips: clockwise from the button.
  std::vector<int> order;
  for (int k = 1; k <= n; ++k) order.push_back((button() + k) % n);

  auto best_of = [&](const std::vector<int>& seats) {
    std::vector<int> winners;
    HandValue best = 0;
    for (int i : seats) best = std::max(best, value[i]);
    for (int i : order) {
      if (std::find(seats.begin(), seats.end(), i) != seats.end() && value[i] == best) {
        winners.push_back(i);
      }
    }
    return winners;
  };

  // Like PokerKit, muck hands that win nothing, then combine neighbouring pots that have the
  // same remaining contenders before splitting. This only changes where odd chips go.
  const std::vector<Pot> layered = pots();
  std::vector<bool> contending(n, false);
  for (const Pot& pot : layered) {
    if (pot.eligible.empty()) throw std::logic_error("payoffs: pot with no eligible player");
    for (int w : best_of(pot.eligible)) contending[w] = true;
  }
  std::vector<Pot> merged;
  for (const Pot& pot : layered) {
    Pot p{pot.amount, {}};
    for (int i : pot.eligible) {
      if (contending[i]) p.eligible.push_back(i);
    }
    if (!merged.empty() && merged.back().eligible == p.eligible) {
      merged.back().amount += p.amount;
    } else {
      merged.push_back(std::move(p));
    }
  }

  for (const Pot& pot : merged) {
    const std::vector<int> winners = best_of(pot.eligible);
    const Chips share = pot.amount / static_cast<Chips>(winners.size());
    const Chips odd = pot.amount % static_cast<Chips>(winners.size());
    for (int w : winners) out[w] += share;
    out[winners.front()] += odd;
  }
  return out;
}

}  // namespace regret
