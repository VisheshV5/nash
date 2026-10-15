#include "regret/search/subgame.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <stdexcept>

#include "regret/abstraction/card_features.hpp"
#include "regret/evaluator.hpp"

namespace regret::search {
namespace {

using Vec = std::vector<float>;

struct Node {
  enum Kind : std::uint8_t { kDecision, kFold, kShowdown } kind = kDecision;
  int player = -1;  // decision: seat to act; fold: the seat that folded
  std::vector<int> children;
  std::vector<Action> actions;
  Chips contributed[2] = {0, 0};
  std::size_t offset = 0;  // decision: start of its actions x kHands block
};

bool same(const Action& a, const Action& b) {
  return a.type == b.type && (a.type != ActionType::kBetRaise || a.amount == b.amount);
}

std::vector<Action> search_actions(const HandState& s, const SearchActions& sa,
                                   const Action* forced) {
  std::vector<Action> out;
  const LegalActions la = s.legal_actions();
  if (la.fold) out.push_back(Action::fold());
  out.push_back(Action::check_call());
  std::vector<Chips> sizes;
  if (la.bet_raise) {
    const int seat = s.to_act();
    const Chips cur = s.current_bet();
    if (s.raises_this_street() < sa.max_raises) {
      const double pot_after_call = static_cast<double>(s.pot() + (cur - s.bet(seat)));
      for (double f : s.raises_this_street() == 0 ? sa.bet_pot : sa.raise_pot) {
        const Chips to = std::clamp(static_cast<Chips>(std::llround(cur + f * pot_after_call)),
                                    la.min_raise_to, la.max_raise_to);
        sizes.push_back(to);
      }
    }
    if (sa.all_in || sizes.empty()) sizes.push_back(la.max_raise_to);
  }
  if (forced && forced->type == ActionType::kBetRaise) sizes.push_back(forced->amount);
  std::sort(sizes.begin(), sizes.end());
  sizes.erase(std::unique(sizes.begin(), sizes.end()), sizes.end());
  for (Chips to : sizes) out.push_back(Action::bet_raise_to(to));
  if (forced && forced->type == ActionType::kFold && !la.fold) {
    throw std::invalid_argument("search: the real history folds where folding isn't legal");
  }
  return out;
}

class RiverSolver {
 public:
  explicit RiverSolver(const RiverProblem& p) : p_(p) {
    if (p.start.num_players() != 2) throw std::invalid_argument("search: heads-up only");
    if (p.start.street() != Street::kRiver) {
      throw std::invalid_argument("search: the start state must be at the start of the river");
    }
    CardMask board = 0;
    for (Card c : p.board) {
      if (board & card_bit(c)) throw std::invalid_argument("search: duplicate board card");
      board |= card_bit(c);
    }
    const auto& pairs = abstraction::hole_pairs();
    std::vector<std::pair<HandValue, int>> ranked;
    for (int h = 0; h < kHands; ++h) {
      const CardMask hm = card_bit(pairs[h].lo) | card_bit(pairs[h].hi);
      valid_[h] = !(hm & board);
      if (valid_[h]) ranked.emplace_back(evaluate(board | hm), h);
      for (int s = 0; s < 2; ++s) range_[s][h] = valid_[h] ? std::max(0.0f, p.range[s][h]) : 0.0f;
    }
    std::sort(ranked.begin(), ranked.end());
    for (std::size_t i = 0; i < ranked.size(); ++i) {
      if (i == 0 || ranked[i].first != ranked[i - 1].first) group_start_.push_back(order_.size());
      order_.push_back(ranked[i].second);
    }
    group_start_.push_back(order_.size());

    HandState start = p.start;
    build(start, 0, true);
    scratch_.resize(static_cast<std::size_t>(max_depth_) + 1);
    for (Scratch& sc : scratch_) {
      sc.sigma.resize(kMaxActions * kHands);
      sc.reach.resize(kHands);
      sc.child.resize(kMaxActions * kHands);
    }
    if (current_ < 0) throw std::invalid_argument("search: the real river actions end the hand");
    regret_.assign(slots_, 0.0f);
    sum_.assign(slots_, 0.0f);
  }

  RiverSolution solve() {
    const auto t0 = std::chrono::steady_clock::now();
    auto elapsed = [&] {
      return std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    };
    int it = 0;
    while (it < p_.max_iterations && (it < p_.min_iterations || elapsed() < p_.max_seconds)) {
      ++it;
      for (int player = 0; player < 2; ++player) {
        cfr(0, player, range_[player].data(), range_[1 - player].data(), static_cast<float>(it), 0,
            root_value_.data());
      }
    }
    RiverSolution out;
    const Node& cur = nodes_[current_];
    out.actions = cur.actions;
    out.player = cur.player;
    out.strategy.resize(cur.actions.size() * kHands);
    for (int h = 0; h < kHands; ++h) {
      float total = 0;
      for (std::size_t a = 0; a < cur.actions.size(); ++a)
        total += sum_[cur.offset + a * kHands + h];
      for (std::size_t a = 0; a < cur.actions.size(); ++a) {
        out.strategy[a * kHands + h] = total > 0 ? sum_[cur.offset + a * kHands + h] / total
                                                 : 1.0f / static_cast<float>(cur.actions.size());
      }
    }
    {
      Vec own(range_[cur.player].begin(), range_[cur.player].end());
      Vec opp(range_[1 - cur.player].begin(), range_[1 - cur.player].end());
      captured_.clear();
      evaluate_avg(0, cur.player, opp, /*best_response=*/false);
      out.values = captured_;
    }
    out.iterations = it;
    out.seconds = elapsed();
    out.nodes = static_cast<int>(nodes_.size());
    out.exploitability = exploitability();
    return out;
  }

 private:
  int build(HandState& s, std::size_t path_index, bool on_path, int depth = 0) {
    max_depth_ = std::max(max_depth_, depth);
    const int id = static_cast<int>(nodes_.size());
    nodes_.emplace_back();
    if (s.is_terminal()) {
      Node& n = nodes_[id];
      n.contributed[0] = s.contributed(0);
      n.contributed[1] = s.contributed(1);
      if (s.non_folded() == 1) {
        n.kind = Node::kFold;
        n.player = s.status(0) == PlayerStatus::kFolded ? 0 : 1;
      } else {
        n.kind = Node::kShowdown;
      }
      return id;
    }
    if (on_path && path_index == p_.path.size()) current_ = id;
    const Action* forced = on_path && path_index < p_.path.size() ? &p_.path[path_index] : nullptr;
    std::vector<Action> acts = search_actions(s, p_.actions, forced);
    if (acts.size() > kMaxActions) throw std::logic_error("search: too many actions at a node");
    nodes_[id].kind = Node::kDecision;
    nodes_[id].player = s.to_act();
    nodes_[id].actions = acts;
    nodes_[id].offset = slots_;
    slots_ += acts.size() * kHands;
    std::vector<int> kids;
    for (const Action& a : acts) {
      HandState child = s;
      child.apply(a);
      kids.push_back(build(child, path_index + 1, forced && same(a, *forced), depth + 1));
    }
    nodes_[id].children = std::move(kids);
    return id;
  }

  void terminal(const Node& n, int player, const float* opp, float* v) const {
    std::fill(v, v + kHands, 0.0f);
    const auto& pairs = abstraction::hole_pairs();
    float total = 0;
    float by_card[kNumCards] = {};
    for (int j = 0; j < kHands; ++j) {
      if (opp[j] == 0) continue;
      total += opp[j];
      by_card[pairs[j].lo] += opp[j];
      by_card[pairs[j].hi] += opp[j];
    }
    if (n.kind == Node::kFold) {
      const float payoff = n.player == player ? -static_cast<float>(n.contributed[player])
                                              : static_cast<float>(n.contributed[1 - player]);
      for (int h = 0; h < kHands; ++h) {
        if (valid_[h])
          v[h] = payoff * (total - by_card[pairs[h].lo] - by_card[pairs[h].hi] + opp[h]);
      }
      return;
    }
    const float stake = static_cast<float>(std::min(n.contributed[0], n.contributed[1]));
    float below = 0;
    float below_card[kNumCards] = {};
    float group_card[kNumCards] = {};  // only entries touched by the current group are non-zero
    for (std::size_t g = 0; g + 1 < group_start_.size(); ++g) {
      const std::size_t lo = group_start_[g], hi = group_start_[g + 1];
      float group = 0;
      for (std::size_t i = lo; i < hi; ++i) {
        const int h = order_[i];
        group += opp[h];
        group_card[pairs[h].lo] += opp[h];
        group_card[pairs[h].hi] += opp[h];
      }
      for (std::size_t i = lo; i < hi; ++i) {
        const int h = order_[i];
        const int a = pairs[h].lo, b = pairs[h].hi;
        const float compatible = total - by_card[a] - by_card[b] + opp[h];
        const float less = below - below_card[a] - below_card[b];
        const float tie = group - group_card[a] - group_card[b] + opp[h];
        v[h] = stake * (less - (compatible - less - tie));
      }
      below += group;
      for (std::size_t i = lo; i < hi; ++i) {
        const int h = order_[i];
        for (int c : {static_cast<int>(pairs[h].lo), static_cast<int>(pairs[h].hi)}) {
          below_card[c] += group_card[c];
          group_card[c] = 0;
        }
      }
    }
  }

  Vec terminal(const Node& n, int player, const Vec& opp) const {
    Vec v(kHands);
    terminal(n, player, opp.data(), v.data());
    return v;
  }

  struct Scratch {
    std::vector<float> sigma, reach, child;
  };
  // One set of buffers per tree depth, all allocated before solving: recursion holds pointers
  // into shallower levels, so the pool must never reallocate.
  Scratch& scratch(int depth) { return scratch_[static_cast<std::size_t>(depth)]; }

  // CFR+ pass for `player`: writes counterfactual values of player's hands at `id` into `value`.
  void cfr(int id, int player, const float* own, const float* opp, float weight, int depth,
           float* value) {
    const Node& n = nodes_[id];
    if (n.kind != Node::kDecision) {
      terminal(n, player, opp, value);
      return;
    }
    const std::size_t k = n.actions.size();
    Scratch& sc = scratch(depth);
    float* sigma = sc.sigma.data();
    for (int h = 0; h < kHands; ++h) {
      float total = 0;
      for (std::size_t a = 0; a < k; ++a) {
        const float r = std::max(0.0f, regret_[n.offset + a * kHands + h]);
        sigma[a * kHands + h] = r;
        total += r;
      }
      const float inv = total > 0 ? 1.0f / total : 0.0f;
      for (std::size_t a = 0; a < k; ++a) {
        sigma[a * kHands + h] =
            total > 0 ? sigma[a * kHands + h] * inv : 1.0f / static_cast<float>(k);
      }
    }
    std::fill(value, value + kHands, 0.0f);
    float* reach = sc.reach.data();
    if (n.player == player) {
      for (std::size_t a = 0; a < k; ++a) {
        float* cv = sc.child.data() + a * kHands;
        for (int h = 0; h < kHands; ++h) reach[h] = own[h] * sigma[a * kHands + h];
        cfr(n.children[a], player, reach, opp, weight, depth + 1, cv);
        for (int h = 0; h < kHands; ++h) value[h] += sigma[a * kHands + h] * cv[h];
      }
      for (std::size_t a = 0; a < k; ++a) {
        const float* cv = sc.child.data() + a * kHands;
        float* r = &regret_[n.offset + a * kHands];
        float* s = &sum_[n.offset + a * kHands];
        for (int h = 0; h < kHands; ++h) {
          r[h] = std::max(0.0f, r[h] + cv[h] - value[h]);
          s[h] += weight * own[h] * sigma[a * kHands + h];
        }
      }
    } else {
      float* cv = sc.child.data();
      for (std::size_t a = 0; a < k; ++a) {
        for (int h = 0; h < kHands; ++h) reach[h] = opp[h] * sigma[a * kHands + h];
        cfr(n.children[a], player, own, reach, weight, depth + 1, cv);
        for (int h = 0; h < kHands; ++h) value[h] += cv[h];
      }
    }
  }

  float avg(const Node& n, std::size_t a, int h) const {
    float total = 0;
    for (std::size_t b = 0; b < n.actions.size(); ++b) total += sum_[n.offset + b * kHands + h];
    return total > 0 ? sum_[n.offset + a * kHands + h] / total
                     : 1.0f / static_cast<float>(n.actions.size());
  }

  // Values of `player`'s hands when both play the average strategy, or when `player`
  // best-responds. Captures the values at the current decision node.
  Vec evaluate_avg(int id, int player, const Vec& opp, bool best_response) {
    const Node& n = nodes_[id];
    Vec value(kHands,
              best_response && n.kind == Node::kDecision && n.player == player ? -3.4e38f : 0.0f);
    if (n.kind != Node::kDecision) {
      value = terminal(n, player, opp);
    } else if (n.player == player) {
      for (std::size_t a = 0; a < n.actions.size(); ++a) {
        const Vec v = evaluate_avg(n.children[a], player, opp, best_response);
        for (int h = 0; h < kHands; ++h) {
          if (best_response) {
            value[h] = std::max(value[h], v[h]);
          } else {
            value[h] += avg(n, a, h) * v[h];
          }
        }
      }
    } else {
      for (std::size_t a = 0; a < n.actions.size(); ++a) {
        Vec reach(kHands);
        for (int h = 0; h < kHands; ++h) reach[h] = opp[h] * avg(n, a, h);
        const Vec v = evaluate_avg(n.children[a], player, reach, best_response);
        for (int h = 0; h < kHands; ++h) value[h] += v[h];
      }
    }
    if (id == current_ && !best_response && n.player == player) captured_ = value;
    return value;
  }

  double exploitability() {
    double br = 0, mass = 0;
    for (int player = 0; player < 2; ++player) {
      Vec opp(range_[1 - player].begin(), range_[1 - player].end());
      const Vec v = evaluate_avg(0, player, opp, /*best_response=*/true);
      for (int h = 0; h < kHands; ++h) br += range_[player][h] * v[h];
    }
    // Joint mass of compatible (hand, hand) pairs, to turn totals into per-hand values.
    const Range& opp = range_[1];
    float total = 0;
    float by_card[kNumCards] = {};
    const auto& pairs = abstraction::hole_pairs();
    for (int j = 0; j < kHands; ++j) {
      total += opp[j];
      by_card[pairs[j].lo] += opp[j];
      by_card[pairs[j].hi] += opp[j];
    }
    for (int h = 0; h < kHands; ++h) {
      mass += range_[0][h] * (total - by_card[pairs[h].lo] - by_card[pairs[h].hi] + opp[h]);
    }
    return mass > 0 ? br / mass / 2 : 0.0;
  }

  const RiverProblem& p_;
  std::array<bool, kHands> valid_{};
  std::array<Range, 2> range_{};
  std::vector<int> order_;                // valid holes by ascending strength
  std::vector<std::size_t> group_start_;  // tie groups in order_
  std::vector<Node> nodes_;
  std::size_t slots_ = 0;
  int current_ = -1;
  std::vector<float> regret_, sum_;
  Vec captured_;
  std::vector<Scratch> scratch_;
  Vec root_value_ = Vec(kHands);
  static constexpr std::size_t kMaxActions = 16;
  int max_depth_ = 0;
};

}  // namespace

RiverSolution solve_river(const RiverProblem& problem) { return RiverSolver(problem).solve(); }

}  // namespace regret::search
