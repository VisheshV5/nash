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
// Solving starts on the turn or the river and always runs to the end of the hand. From the turn,
// the river card is a chance node: one river subtree per possible card (with the river's own,
// smaller bet menu), and a hand that holds the river card takes no part in that subtree. The
// flop and preflop are left to the blueprint.
namespace regret::search {

inline constexpr int kHands = 1326;
using Range = std::array<float, kHands>;

struct SearchActions {
  std::vector<double> bet_pot = {0.33, 0.5, 0.75, 1.0, 1.5};  // first bet, fraction of pot
  std::vector<double> raise_pot = {0.5, 1.0};  // raises, fraction of pot after calling
  int max_raises = 3;                          // bets + raises per street
  bool all_in = true;
};

struct SubgameProblem {
  HandState start{TableRules{{1, 1}, 1, 2}};  // heads-up, at the start of a turn or river round
  std::vector<Card> board;                    // 4 cards (turn) or 5 (river)
  std::vector<Action> path;      // real actions on this street so far (forced into the tree)
  std::array<Range, 2> range{};  // per seat; blocked hands are ignored
  SearchActions actions;         // the start street's bet menu
  SearchActions river_actions;   // the river's, when solving from the turn
  int min_iterations = 50;
  int max_iterations = 2000;
  double max_seconds = 0.8;
  // > 1: from the river, the subtrees under the root decision are solved in parallel; from the
  // turn, the river cards under each chance node are.
  int threads = 1;
  bool exploitability = true;  // measure the result's exploitability (one more pass or two)
  // Iteration t's strategy enters the average with weight t^average_power. Higher powers
  // forget the poor early iterations faster (Discounted CFR uses 2).
  double average_power = 2.0;
};

struct SubgameSolution {
  std::vector<Action> actions;  // at the current decision (after `path`)
  int player = -1;              // seat to act there
  std::vector<float> strategy;  // actions.size() x kHands, average strategy (rows sum to 1)
  std::vector<float> values;    // kHands: the acting player's counterfactual value per hand
                                // (chips, per unit of opponent range mass)
  int iterations = 0;
  double seconds = 0;
  int nodes = 0;
  double exploitability = -1;  // chips per hand, of the whole subgame strategy (-1: not measured)
};

SubgameSolution solve_subgame(const SubgameProblem& problem);

}  // namespace regret::search
