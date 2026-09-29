#pragma once

#include <utility>
#include <vector>

#include "regret/engine.hpp"

// Action abstraction (ROADMAP D5): the discrete bets the blueprint considers at each decision,
// and translation of any real bet onto them.
namespace regret::abstraction {

struct PreflopSizing {
  std::vector<double> open_bb;        // first raise: total amount, in big blinds
  std::vector<double> reraise_x_ip;   // second raise, in position: multiple of the bet faced
  std::vector<double> reraise_x_oop;  // second raise, out of position
  std::vector<double> four_bet_x;     // third raise and later
  bool all_in = true;
  int max_raises = 4;  // sized raises stop here (all-in stays available)
};

struct PostflopSizing {
  std::vector<double> bet_pot;    // first bet: fraction of the pot
  std::vector<double> raise_pot;  // raises: fraction of the pot after calling
  bool all_in = true;
  int max_raises = 3;  // bets + raises per street
};

struct ActionRules {
  Chips chips_per_bb = 100;
  PreflopSizing preflop;
  // (minimum players in the hand, sizing); the rule with the largest key <= players applies.
  std::vector<std::pair<int, PostflopSizing>> postflop;

  const PostflopSizing& postflop_for(int players) const;
};

// Abstract actions at the current decision, all legal in `s`, in order: fold (only when facing a
// bet), check/call, then bets/raises by increasing size, all-in last. Sizes are rounded to chips,
// clamped to the legal range, and deduplicated.
std::vector<Action> abstract_actions(const HandState& s, const ActionRules& rules);

// A bet/raise to `to` as a fraction of the pot after calling: (to - current bet) / (pot + call).
double pot_fraction(const HandState& s, Chips to);

// Pseudo-harmonic mapping (Ganzfried & Sandholm 2013): the probability of mapping a bet of size
// x to the smaller neighbour a rather than b, all as pot fractions, a <= x <= b.
double pseudo_harmonic(double a, double b, double x);

// Distribution over indices of abstract_actions(s) for a real action. Fold and check/call map to
// themselves; a bet/raise maps to its two neighbouring sizes (check/call counts as size 0), or to
// all-in when it's at least as big as the largest.
std::vector<std::pair<int, double>> translate(const HandState& s, const ActionRules& rules,
                                              const Action& actual);

// The same, for an action taken in a *different* state (e.g. the real game while `s` is the
// abstract game): the bet is given by its pot fraction, and an all-in maps to all-in.
std::vector<std::pair<int, double>> translate_bet(const HandState& s, const ActionRules& rules,
                                                  double pot_fraction, bool all_in);

}  // namespace regret::abstraction
