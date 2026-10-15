#include "regret/search/subgame.hpp"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <functional>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <thread>

#include "regret/abstraction/card_features.hpp"
#include "regret/evaluator.hpp"

namespace regret::search {
namespace {

using Vec = std::vector<float>;

struct Node {
  enum Kind : std::uint8_t { kDecision, kFold, kShowdown, kChance } kind = kDecision;
  int player = -1;  // decision: seat to act; fold: the seat that folded
  int board = 0;    // index into the solver's boards
  std::vector<int> children;
  std::vector<Action> actions;
  Chips contributed[2] = {0, 0};
  std::size_t offset = 0;  // decision: start of its actions x kHands block
};

// One board the subtree below a node plays on: the start board, or that plus a river card.
struct Board {
  CardMask mask = 0;
  Card river = 0;  // the card the chance node dealt (river boards only)
  std::array<bool, kHands> valid{};
  std::vector<int> order;                // valid holes by ascending strength (5-card boards)
  std::vector<std::size_t> group_start;  // tie groups in order
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

bool holds(int hand, Card c) {
  const auto& pair = abstraction::hole_pairs()[hand];
  return pair.lo == c || pair.hi == c;
}

// A fixed set of worker threads for the chance nodes, so that a solve doesn't start threads on
// every visit. run() hands out indices to the workers and the calling thread (worker 0).
class Workers {
 public:
  using Job = std::function<void(int index, int worker)>;

  explicit Workers(int extra) {
    for (int w = 1; w <= extra; ++w) threads_.emplace_back([this, w] { loop(w); });
  }
  ~Workers() {
    {
      std::lock_guard<std::mutex> lock(mu_);
      stop_ = true;
    }
    wake_.notify_all();
    for (auto& t : threads_) t.join();
  }
  Workers(const Workers&) = delete;
  Workers& operator=(const Workers&) = delete;

  void run(int count, const Job& job) {
    {
      std::lock_guard<std::mutex> lock(mu_);
      job_ = &job;
      count_ = count;
      next_.store(0);
      busy_ = static_cast<int>(threads_.size());
      ++generation_;
    }
    wake_.notify_all();
    for (int i; (i = next_.fetch_add(1)) < count;) job(i, 0);
    std::unique_lock<std::mutex> lock(mu_);
    done_.wait(lock, [&] { return busy_ == 0; });
  }

 private:
  void loop(int worker) {
    std::uint64_t seen = 0;
    for (;;) {
      const Job* job;
      int count;
      {
        std::unique_lock<std::mutex> lock(mu_);
        wake_.wait(lock, [&] { return stop_ || generation_ != seen; });
        if (stop_) return;
        seen = generation_;
        job = job_;
        count = count_;
      }
      for (int i; (i = next_.fetch_add(1)) < count;) (*job)(i, worker);
      std::lock_guard<std::mutex> lock(mu_);
      if (--busy_ == 0) done_.notify_one();
    }
  }

  std::vector<std::thread> threads_;
  std::mutex mu_;
  std::condition_variable wake_, done_;
  const Job* job_ = nullptr;
  int count_ = 0;
  std::atomic<int> next_{0};
  int busy_ = 0;
  std::uint64_t generation_ = 0;
  bool stop_ = false;
};

class Solver {
 public:
  explicit Solver(const SubgameProblem& p) : p_(p) {
    if (p.start.num_players() != 2) throw std::invalid_argument("search: heads-up only");
    const bool turn = p.start.street() == Street::kTurn;
    if (!turn && p.start.street() != Street::kRiver) {
      throw std::invalid_argument(
          "search: the start state must be at the start of the turn or river");
    }
    if (p.board.size() != (turn ? 4u : 5u)) {
      throw std::invalid_argument(turn ? "search: the turn needs a 4-card board"
                                       : "search: the river needs a 5-card board");
    }
    CardMask board = 0;
    for (Card c : p.board) {
      if (c >= kNumCards || (board & card_bit(c))) {
        throw std::invalid_argument("search: bad or duplicate board card");
      }
      board |= card_bit(c);
    }
    boards_.push_back(make_board(board, 0));
    if (turn) {
      for (int c = 0; c < kNumCards; ++c) {
        const Card card = static_cast<Card>(c);
        if (!(board & card_bit(card))) boards_.push_back(make_board(board | card_bit(card), card));
      }
    }
    for (int s = 0; s < 2; ++s) {
      for (int h = 0; h < kHands; ++h) {
        range_[s][h] = boards_[0].valid[h] ? std::max(0.0f, p.range[s][h]) : 0.0f;
      }
    }

    HandState start = p.start;
    build(start, 0, true, 0, 0);
    if (current_ < 0) throw std::invalid_argument("search: the real actions end the hand");

    // Threads: from the river, one per root action (each with its own scratch pool, 1..k);
    // from the turn, a fixed set of workers shared by the chance nodes (pools 0..threads-1).
    const int threads = std::max(1, p.threads);
    std::size_t pools = 1;
    if (threads > 1 && turn) {
      workers_ = std::make_unique<Workers>(threads - 1);
      pools = static_cast<std::size_t>(threads);
    } else if (threads > 1) {
      root_parallel_ = true;
      pools = nodes_[0].kind == Node::kDecision ? nodes_[0].actions.size() + 1 : 1;
    }
    pools_.resize(pools);
    for (auto& pool : pools_) {
      pool.resize(static_cast<std::size_t>(max_depth_) + 1);
      for (Scratch& sc : pool) {
        sc.sigma.resize(kMaxActions * kHands);
        sc.reach.resize(kHands);
        sc.child.resize(kMaxActions * kHands);
        sc.own.resize(kHands);
        sc.opp.resize(kHands);
      }
    }
    if (turn) {
      chance_values_.resize(static_cast<std::size_t>(max_depth_) + 1);
      for (auto& buf : chance_values_) buf.resize((boards_.size() - 1) * kHands);
    }
    regret_.assign(slots_, 0.0f);
    sum_.assign(slots_, 0.0f);
  }

  SubgameSolution solve() {
    const auto t0 = std::chrono::steady_clock::now();
    auto elapsed = [&] {
      return std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    };
    int it = 0;
    while (it < p_.max_iterations && (it < p_.min_iterations || elapsed() < p_.max_seconds)) {
      ++it;
      for (int player = 0; player < 2; ++player) {
        cfr(0, player, range_[player].data(), range_[1 - player].data(),
            static_cast<float>(std::pow(static_cast<double>(it), p_.average_power)), 0,
            root_value_.data());
      }
    }
    SubgameSolution out;
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
      Vec opp(range_[1 - cur.player].begin(), range_[1 - cur.player].end());
      captured_.clear();
      evaluate_avg(0, cur.player, opp, /*best_response=*/false);
      out.values = captured_;
    }
    out.iterations = it;
    out.seconds = elapsed();
    out.nodes = static_cast<int>(nodes_.size());
    if (p_.exploitability) out.exploitability = exploitability();
    return out;
  }

 private:
  static Board make_board(CardMask mask, Card river) {
    Board b;
    b.mask = mask;
    b.river = river;
    const auto& pairs = abstraction::hole_pairs();
    std::vector<std::pair<HandValue, int>> ranked;
    const bool full = __builtin_popcountll(mask) == 5;
    for (int h = 0; h < kHands; ++h) {
      const CardMask hm = card_bit(pairs[h].lo) | card_bit(pairs[h].hi);
      b.valid[h] = !(hm & mask);
      if (b.valid[h] && full) ranked.emplace_back(evaluate(mask | hm), h);
    }
    std::sort(ranked.begin(), ranked.end());
    for (std::size_t i = 0; i < ranked.size(); ++i) {
      if (i == 0 || ranked[i].first != ranked[i - 1].first) b.group_start.push_back(b.order.size());
      b.order.push_back(ranked[i].second);
    }
    b.group_start.push_back(b.order.size());
    return b;
  }

  // From the turn, reaching the river (or a turn all-in showdown) deals the river card.
  bool needs_card(const HandState& s, int board) const {
    if (__builtin_popcountll(boards_[board].mask) == 5) return false;
    return s.is_terminal() ? s.non_folded() > 1 : s.street() == Street::kRiver;
  }

  int build(HandState& s, std::size_t path_index, bool on_path, int depth, int board) {
    max_depth_ = std::max(max_depth_, depth);
    const int id = static_cast<int>(nodes_.size());
    nodes_.emplace_back();
    nodes_[id].board = board;
    if (needs_card(s, board)) {
      nodes_[id].kind = Node::kChance;
      std::vector<int> kids;
      for (std::size_t b = 1; b < boards_.size(); ++b) {
        kids.push_back(build(s, path_index, false, depth + 1, static_cast<int>(b)));
      }
      nodes_[id].children = std::move(kids);
      return id;
    }
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
    const bool river_menu = board > 0;
    std::vector<Action> acts =
        search_actions(s, river_menu ? p_.river_actions : p_.actions, forced);
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
      kids.push_back(build(child, path_index + 1, forced && same(a, *forced), depth + 1, board));
    }
    nodes_[id].children = std::move(kids);
    return id;
  }

  void terminal(const Node& n, int player, const float* opp, float* v) const {
    std::fill(v, v + kHands, 0.0f);
    const Board& bd = boards_[n.board];
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
        if (bd.valid[h])
          v[h] = payoff * (total - by_card[pairs[h].lo] - by_card[pairs[h].hi] + opp[h]);
      }
      return;
    }
    const float stake = static_cast<float>(std::min(n.contributed[0], n.contributed[1]));
    float below = 0;
    float below_card[kNumCards] = {};
    float group_card[kNumCards] = {};  // only entries touched by the current group are non-zero
    for (std::size_t g = 0; g + 1 < bd.group_start.size(); ++g) {
      const std::size_t lo = bd.group_start[g], hi = bd.group_start[g + 1];
      float group = 0;
      for (std::size_t i = lo; i < hi; ++i) {
        const int h = bd.order[i];
        group += opp[h];
        group_card[pairs[h].lo] += opp[h];
        group_card[pairs[h].hi] += opp[h];
      }
      for (std::size_t i = lo; i < hi; ++i) {
        const int h = bd.order[i];
        const int a = pairs[h].lo, b = pairs[h].hi;
        const float compatible = total - by_card[a] - by_card[b] + opp[h];
        const float less = below - below_card[a] - below_card[b];
        const float tie = group - group_card[a] - group_card[b] + opp[h];
        v[h] = stake * (less - (compatible - less - tie));
      }
      below += group;
      for (std::size_t i = lo; i < hi; ++i) {
        const int h = bd.order[i];
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

  // Every compatible pair of holes on the turn leaves 44 river cards.
  static constexpr float kRiverCards = 44.0f;

  struct Scratch {
    std::vector<float> sigma, reach, child;
    std::vector<float> own, opp;  // chance nodes: one river card's reaches
  };
  // One set of buffers per tree depth, all allocated before solving: recursion holds pointers
  // into shallower levels, so the pool must never reallocate.
  Scratch& scratch(int depth, int pool) {
    return pools_[static_cast<std::size_t>(pool)][static_cast<std::size_t>(depth)];
  }

  // A chance node's river cards: each child sees both ranges without the hands that hold its
  // card; values are averaged over the cards. Each card's values get their own buffer and are
  // summed in card order, so the result doesn't depend on which worker ran which card.
  void chance(const Node& n, int player, const float* own, const float* opp, float weight,
              int depth, float* value, int pool) {
    float* out = chance_values_[static_cast<std::size_t>(depth)].data();
    auto one = [&](int i, int w) {
      const int pl = workers_ ? w : pool;
      Scratch& sc = scratch(depth, pl);
      const Card c = boards_[nodes_[n.children[i]].board].river;
      for (int h = 0; h < kHands; ++h) {
        const bool blocked = holds(h, c);
        sc.own[h] = blocked ? 0.0f : own[h];
        sc.opp[h] = blocked ? 0.0f : opp[h];
      }
      cfr(n.children[i], player, sc.own.data(), sc.opp.data(), weight, depth + 1,
          out + static_cast<std::size_t>(i) * kHands, pl);
    };
    const int cards = static_cast<int>(n.children.size());
    if (workers_ && pool == 0) {
      workers_->run(cards, one);
    } else {
      for (int i = 0; i < cards; ++i) one(i, pool);
    }
    std::fill(value, value + kHands, 0.0f);
    for (int i = 0; i < cards; ++i) {
      const float* v = out + static_cast<std::size_t>(i) * kHands;
      for (int h = 0; h < kHands; ++h) value[h] += v[h];
    }
    for (int h = 0; h < kHands; ++h) value[h] /= kRiverCards;
  }

  // CFR+ pass for `player`: writes counterfactual values of player's hands at `id` into `value`.
  void cfr(int id, int player, const float* own, const float* opp, float weight, int depth,
           float* value, int pool = 0) {
    const Node& n = nodes_[id];
    if (n.kind == Node::kChance) {
      chance(n, player, own, opp, weight, depth, value, pool);
      return;
    }
    if (n.kind != Node::kDecision) {
      terminal(n, player, opp, value);
      return;
    }
    const std::size_t k = n.actions.size();
    Scratch& sc = scratch(depth, pool);
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
    if (depth == 0 && root_parallel_ && k > 1) {
      // River root: each action's subtree touches only its own nodes, so solve them side by side.
      const bool mine = n.player == player;
      std::vector<std::thread> pool_threads;
      for (std::size_t a = 0; a < k; ++a) {
        pool_threads.emplace_back([&, a] {
          const int child_pool = static_cast<int>(a) + 1;
          float* r = scratch(0, child_pool).reach.data();
          const float* base = mine ? own : opp;
          for (int h = 0; h < kHands; ++h) r[h] = base[h] * sigma[a * kHands + h];
          cfr(n.children[a], player, mine ? r : own, mine ? opp : r, weight, 1,
              sc.child.data() + a * kHands, child_pool);
        });
      }
      for (auto& t : pool_threads) t.join();
      for (std::size_t a = 0; a < k; ++a) {
        const float* cv = sc.child.data() + a * kHands;
        for (int h = 0; h < kHands; ++h) value[h] += mine ? sigma[a * kHands + h] * cv[h] : cv[h];
      }
      if (mine) {
        for (std::size_t a = 0; a < k; ++a) {
          const float* cv = sc.child.data() + a * kHands;
          float* r = &regret_[n.offset + a * kHands];
          float* s = &sum_[n.offset + a * kHands];
          for (int h = 0; h < kHands; ++h) {
            r[h] = std::max(0.0f, r[h] + cv[h] - value[h]);
            s[h] += weight * own[h] * sigma[a * kHands + h];
          }
        }
      }
      return;
    }
    float* reach = sc.reach.data();
    if (n.player == player) {
      for (std::size_t a = 0; a < k; ++a) {
        float* cv = sc.child.data() + a * kHands;
        for (int h = 0; h < kHands; ++h) reach[h] = own[h] * sigma[a * kHands + h];
        cfr(n.children[a], player, reach, opp, weight, depth + 1, cv, pool);
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
        cfr(n.children[a], player, own, reach, weight, depth + 1, cv, pool);
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
    if (n.kind == Node::kChance) {
      for (int child : n.children) {
        const Card c = boards_[nodes_[child].board].river;
        Vec o = opp;
        for (int h = 0; h < kHands; ++h) {
          if (holds(h, c)) o[h] = 0;
        }
        const Vec v = evaluate_avg(child, player, o, best_response);
        for (int h = 0; h < kHands; ++h) {
          if (!holds(h, c)) value[h] += v[h] / kRiverCards;
        }
      }
    } else if (n.kind != Node::kDecision) {
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

  const SubgameProblem& p_;
  std::vector<Board> boards_;  // 0: the start board; from the turn, 1..48: one per river card
  std::array<Range, 2> range_{};
  std::vector<Node> nodes_;
  std::size_t slots_ = 0;
  int current_ = -1;
  std::vector<float> regret_, sum_;
  Vec captured_;
  std::vector<std::vector<Scratch>> pools_;
  std::vector<Vec> chance_values_;  // per depth: a chance node's values, one block per card
  std::unique_ptr<Workers> workers_;
  bool root_parallel_ = false;
  Vec root_value_ = Vec(kHands);
  static constexpr std::size_t kMaxActions = 16;
  int max_depth_ = 0;
};

}  // namespace

SubgameSolution solve_subgame(const SubgameProblem& problem) { return Solver(problem).solve(); }

}  // namespace regret::search
