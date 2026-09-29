#include "regret/abstraction/actions.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace regret::abstraction {
namespace {

int raises_this_street(const HandState& s) {
  int n = 0;
  for (const HistoryEntry& e : s.history()) {
    if (e.street == s.street() && e.action.type == ActionType::kBetRaise) ++n;
  }
  return n;
}

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

std::vector<Action> abstract_actions(const HandState& s, const ActionRules& rules) {
  std::vector<Action> out;
  const int seat = s.to_act();
  if (seat < 0) return out;
  const LegalActions la = s.legal_actions();
  if (la.fold) out.push_back(Action::fold());
  out.push_back(Action::check_call());
  if (!la.bet_raise) return out;

  const Chips cur = s.current_bet();
  const int raises = raises_this_street(s);
  std::vector<double> targets;  // raise-to amounts, in chips
  bool all_in = true;
  if (s.street() == Street::kPreflop) {
    const PreflopSizing& p = rules.preflop;
    all_in = p.all_in;
    if (raises < p.max_raises) {
      if (raises == 0) {
        for (double bb : p.open_bb) targets.push_back(bb * static_cast<double>(rules.chips_per_bb));
      } else {
        const auto& mults =
            raises == 1 ? (in_position(s, seat) ? p.reraise_x_ip : p.reraise_x_oop) : p.four_bet_x;
        for (double x : mults) targets.push_back(x * static_cast<double>(cur));
      }
    }
  } else {
    const PostflopSizing& p = rules.postflop_for(s.non_folded());
    all_in = p.all_in;
    if (raises < p.max_raises) {
      const double pot_after_call = static_cast<double>(s.pot() + (cur - s.bet(seat)));
      for (double f : raises == 0 ? p.bet_pot : p.raise_pot) {
        targets.push_back(static_cast<double>(cur) + f * pot_after_call);
      }
    }
  }

  std::vector<Chips> sizes;
  for (double t : targets) {
    const Chips to =
        std::clamp(static_cast<Chips>(std::llround(t)), la.min_raise_to, la.max_raise_to);
    if (to < la.max_raise_to) sizes.push_back(to);  // all-in is added once, below
  }
  std::sort(sizes.begin(), sizes.end());
  sizes.erase(std::unique(sizes.begin(), sizes.end()), sizes.end());
  for (Chips to : sizes) out.push_back(Action::bet_raise_to(to));
  if (all_in || sizes.empty()) out.push_back(Action::bet_raise_to(la.max_raise_to));
  return out;
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
  // Neighbours by pot fraction; check/call is size 0.
  const double x = pot_fraction(s, actual.amount);
  int lo = index_of(ActionType::kCheckCall);
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
