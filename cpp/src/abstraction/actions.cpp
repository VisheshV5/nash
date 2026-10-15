#include "regret/abstraction/actions.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace regret::abstraction {
namespace {

// Last to act postflop among players still in the hand.
bool in_position(const HandState& s, int seat) {
  const int n = s.num_players();
  auto pos = [&](int x) { return (x - (s.button() + 1) + n) % n; };
  for (int o = 0; o < n; ++o) {
    if (o != seat && s.status(o) != PlayerStatus::kFolded && pos(o) > pos(seat)) return false;
  }
  return true;
}

}  // namespace

const PostflopSizing& ActionRules::postflop_for(int players) const {
  const PostflopSizing* best = nullptr;
  int best_key = -1;
  for (const auto& [key, sizing] : postflop) {
    if (key <= players && key > best_key) {
      best = &sizing;
      best_key = key;
    }
  }
  if (!best) throw std::invalid_argument("ActionRules: no postflop sizing for this many players");
  return *best;
}

double pot_fraction(const HandState& s, Chips to) {
  const int seat = s.to_act();
  const Chips cur = s.current_bet();
  const Chips call = cur - (seat >= 0 ? s.bet(seat) : 0);
  return static_cast<double>(to - cur) / static_cast<double>(s.pot() + call);
}

int abstract_actions(const HandState& s, const ActionRules& rules, Action* out) {
  int n = 0;
  const int seat = s.to_act();
  if (seat < 0) return 0;
  const LegalActions la = s.legal_actions();
  if (la.fold) out[n++] = Action::fold();
  out[n++] = Action::check_call();
  if (!la.bet_raise) return n;

  const Chips cur = s.current_bet();
  const int raises = s.raises_this_street();
  const std::vector<double>* sizing = nullptr;  // raise-to targets come from this list
  double scale = 0, offset = 0;                 // target = offset + scale * value
  bool all_in = true;
  if (s.street() == Street::kPreflop) {
    const PreflopSizing& p = rules.preflop;
    all_in = p.all_in;
    if (raises < p.max_raises) {
      if (raises == 0) {
        sizing = &p.open_bb;
        scale = static_cast<double>(rules.chips_per_bb);
      } else {
        sizing = raises == 1 ? (in_position(s, seat) ? &p.reraise_x_ip : &p.reraise_x_oop)
                             : &p.four_bet_x;
        scale = static_cast<double>(cur);
      }
    }
  } else {
    const PostflopSizing& p = rules.postflop_for(s.non_folded());
    all_in = p.all_in;
    if (raises < p.max_raises) {
      sizing = raises == 0 ? &p.bet_pot : &p.raise_pot;
      offset = static_cast<double>(cur);
      scale = static_cast<double>(s.pot() + (cur - s.bet(seat)));
    }
  }

  Chips sizes[kMaxAbstractActions];
  int k = 0;
  if (sizing) {
    for (double v : *sizing) {
      const Chips to = std::clamp(static_cast<Chips>(std::llround(offset + scale * v)),
                                  la.min_raise_to, la.max_raise_to);
      if (to < la.max_raise_to && k < kMaxAbstractActions - 3) sizes[k++] = to;  // all-in below
    }
  }
  std::sort(sizes, sizes + k);
  k = static_cast<int>(std::unique(sizes, sizes + k) - sizes);
  for (int i = 0; i < k; ++i) out[n++] = Action::bet_raise_to(sizes[i]);
  if (all_in || k == 0) out[n++] = Action::bet_raise_to(la.max_raise_to);
  return n;
}

std::vector<Action> abstract_actions(const HandState& s, const ActionRules& rules) {
  Action buf[kMaxAbstractActions];
  const int n = abstract_actions(s, rules, buf);
  return {buf, buf + n};
}

double pseudo_harmonic(double a, double b, double x) {
  if (b <= a) return 1.0;
  const double p = ((b - x) * (1.0 + a)) / ((b - a) * (1.0 + x));
  return std::clamp(p, 0.0, 1.0);
}

std::vector<std::pair<int, double>> translate(const HandState& s, const ActionRules& rules,
                                              const Action& actual) {
  if (const char* why = s.why_illegal(actual)) throw std::invalid_argument(why);
  const std::vector<Action> acts = abstract_actions(s, rules);
  auto index_of = [&](ActionType t) {
    for (int i = 0; i < static_cast<int>(acts.size()); ++i) {
      if (acts[i].type == t) return i;
    }
    throw std::logic_error("translate: abstract action missing");
  };
  if (actual.type != ActionType::kBetRaise) return {{index_of(actual.type), 1.0}};

  // Exact match first (including all-in).
  for (int i = 0; i < static_cast<int>(acts.size()); ++i) {
    if (acts[i].type == ActionType::kBetRaise && acts[i].amount == actual.amount) return {{i, 1.0}};
  }
  return translate_bet(s, rules, pot_fraction(s, actual.amount),
                       actual.amount == s.legal_actions().max_raise_to);
}

std::vector<std::pair<int, double>> translate_bet(const HandState& s, const ActionRules& rules,
                                                  double x, bool all_in) {
  const std::vector<Action> acts = abstract_actions(s, rules);
  if (acts.empty()) throw std::invalid_argument("translate_bet: no player to act");
  const LegalActions la = s.legal_actions();
  int last_raise = -1;
  for (int i = 0; i < static_cast<int>(acts.size()); ++i) {
    if (acts[i].type == ActionType::kBetRaise) last_raise = i;
  }
  const int call = acts[0].type == ActionType::kFold ? 1 : 0;
  if (last_raise < 0) return {{call, 1.0}};  // no raising in this abstract state: treat as a call
  if (all_in && acts[last_raise].amount == la.max_raise_to) return {{last_raise, 1.0}};
  // Neighbours by pot fraction; check/call is size 0.
  int lo = call;
  double lo_f = 0.0;
  for (int i = 0; i < static_cast<int>(acts.size()); ++i) {
    if (acts[i].type != ActionType::kBetRaise) continue;
    const double f = pot_fraction(s, acts[i].amount);
    if (f <= x) {
      lo = i;
      lo_f = f;
    } else {
      const double p = pseudo_harmonic(lo_f, f, x);
      if (p >= 1.0) return {{lo, 1.0}};
      if (p <= 0.0) return {{i, 1.0}};
      return {{lo, p}, {i, 1.0 - p}};
    }
  }
  return {{lo, 1.0}};  // bigger than every abstract size: the largest (all-in if present)
}

}  // namespace regret::abstraction
