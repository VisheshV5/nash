#pragma once

#include <array>
#include <cstdint>
#include <vector>

#include "regret/cards.hpp"
#include "regret/engine.hpp"

// Real-time search (ROADMAP M5): re-solve the current betting round heads-up with exact hands.
//
// The subgame starts at the beginning of the current street, from both players' ranges (a weight
// for each of the 1,326 hole-card pairs, normally the blueprint's Bayesian beliefs). Its betting
// tree uses a richer bet-size menu than the blueprint and always contains the real actions
// already taken on this street, so the current decision is a node in the tree.
//
// Solving is vector CFR+ over the public tree: every node carries one value per hole-card pair,
// so a single pass handles all hands at once. Fold and showdown values come from running sums
// over the opponent's range with card removal (and, at showdown, hands sorted by strength), so a
// terminal costs O(1,326) rather than O(1,326^2).
//
// v1 solves the river exactly (to showdown). Earlier streets are left to the blueprint.
namespace regret::search {

inline constexpr int kHands = 1326;
using Range = std::array<float, kHands>;

struct SearchActions {
  std::vector<double> bet_pot = {0.33, 0.5, 0.75, 1.0, 1.5};  // first bet, fraction of pot
  std::vector<double> raise_pot = {0.5, 1.0};  // raises, fraction of pot after calling
  int max_raises = 3;                          // bets + raises per street
  bool all_in = true;
};

struct RiverProblem {
  HandState start{TableRules{{1, 1}, 1, 2}};  // heads-up, at the start of the river betting round
  std::array<Card, 5> board{};
  std::vector<Action> path;      // real river actions so far (forced into the tree)
  std::array<Range, 2> range{};  // per seat; blocked hands are ignored
  SearchActions actions;
  int min_iterations = 50;
  int max_iterations = 2000;
  double max_seconds = 0.8;
  int threads = 1;  // > 1 solves the subtrees under the root decision in parallel
  // Iteration t's strategy enters the average with weight t^average_power. Higher powers
  // forget the poor early iterations faster (Discounted CFR uses 2).
  double average_power = 2.0;
};

struct RiverSolution {
  std::vector<Action> actions;  // at the current decision (after `path`)
  int player = -1;              // seat to act there
  std::vector<float> strategy;  // actions.size() x kHands, average strategy (rows sum to 1)
  std::vector<float> values;    // kHands: the acting player's counterfactual value per hand
                                // (chips, per unit of opponent range mass)
  int iterations = 0;
  double seconds = 0;
  int nodes = 0;
  double exploitability = 0;  // chips per hand, of the whole subgame strategy
};

RiverSolution solve_river(const RiverProblem& problem);

}  // namespace regret::search
