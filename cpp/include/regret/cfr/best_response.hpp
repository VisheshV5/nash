#pragma once

#include <algorithm>
#include <cstdint>
#include <map>
#include <unordered_map>
#include <vector>

#include "regret/cfr/infoset_store.hpp"

namespace regret {

// Average strategy (normalized strategy sums), uniform where an infoset was never reached.
inline void average_strategy(const InfosetStore& store, std::uint64_t key, int n, double* out) {
  const InfosetStore::Slot slot = store.find(key);
  double total = 0;
  for (int a = 0; a < n; ++a) {
    out[a] = slot.data ? std::max(0.0f, slot.data[slot.num_actions + a].load()) : 0.0;
    total += out[a];
  }
  for (int a = 0; a < n; ++a) out[a] = total > 0 ? out[a] / total : 1.0 / n;
}

struct NashConvResult {
  std::vector<double> best_response;  // per player: value of a best response to the others
  std::vector<double> on_policy;      // per player: value when everyone plays the average
  double nash_conv = 0;               // sum over players of (best_response - on_policy)
};

// Exact best responses by full tree walk. Only for small games (Kuhn, Leduc): the tree is
// enumerated once per player.
template <class Game>
class BestResponse {
 public:
  using State = decltype(std::declval<Game>().deal(0));

  BestResponse(const Game& game, const InfosetStore& store) : game_(game), store_(store) {}

  NashConvResult nash_conv() {
    NashConvResult r;
    for (int p = 0; p < game_.num_players(); ++p) {
      br_player_ = p;
      br_action_.clear();
      r.on_policy.push_back(root_value(/*use_br=*/false));
      r.best_response.push_back(best_response_value());
      r.nash_conv += r.best_response.back() - r.on_policy.back();
    }
    return r;
  }

 private:
  struct Item {
    State state;
    double reach;  // chance and opponents' reach
  };

  double root_value(bool use_br) {
    double v = 0;
    for (int d = 0; d < game_.num_deals(); ++d) v += eval(game_.deal(d), use_br);
    return v / game_.num_deals();
  }

  // Groups the BR player's histories by infoset (with depth = own decisions so far), then picks
  // actions deepest first, so every evaluation below an infoset already knows the BR's choices.
  double best_response_value() {
    std::map<int, std::unordered_map<std::uint64_t, std::vector<Item>>, std::greater<>> by_depth;
    for (int d = 0; d < game_.num_deals(); ++d) {
      collect(game_.deal(d), 1.0 / game_.num_deals(), 0, by_depth);
    }
    for (auto& [depth, infosets] : by_depth) {
      for (auto& [key, items] : infosets) {
        const int n = game_.num_actions(items.front().state);
        int best = 0;
        double best_value = -1e300;
        for (int a = 0; a < n; ++a) {
          double v = 0;
          for (const Item& it : items) {
            State child = it.state;
            game_.apply(child, a);
            v += it.reach * eval(child, true);
          }
          if (v > best_value) {
            best_value = v;
            best = a;
          }
        }
        br_action_[key] = best;
      }
    }
    return root_value(true);
  }

  void collect(
      const State& s, double reach, int depth,
      std::map<int, std::unordered_map<std::uint64_t, std::vector<Item>>, std::greater<>>& out) {
    if (game_.is_terminal(s) || reach == 0) return;
    const int n = game_.num_actions(s);
    if (game_.current_player(s) == br_player_) {
      out[depth][game_.infoset_key(s)].push_back({s, reach});
      for (int a = 0; a < n; ++a) {
        State child = s;
        game_.apply(child, a);
        collect(child, reach, depth + 1, out);
      }
      return;
    }
    double sigma[InfosetStore::kMaxActions];
    average_strategy(store_, game_.infoset_key(s), n, sigma);
    for (int a = 0; a < n; ++a) {
      State child = s;
      game_.apply(child, a);
      collect(child, reach * sigma[a], depth, out);
    }
  }

  double eval(const State& s, bool use_br) {
    if (game_.is_terminal(s)) return game_.utility(s, br_player_);
    const int n = game_.num_actions(s);
    const std::uint64_t key = game_.infoset_key(s);
    if (use_br && game_.current_player(s) == br_player_) {
      State child = s;
      const auto it = br_action_.find(key);  // missing only if unreachable for opponents
      game_.apply(child, it == br_action_.end() ? 0 : it->second);
      return eval(child, use_br);
    }
    double sigma[InfosetStore::kMaxActions];
    average_strategy(store_, key, n, sigma);
    double v = 0;
    for (int a = 0; a < n; ++a) {
      if (sigma[a] == 0) continue;
      State child = s;
      game_.apply(child, a);
      v += sigma[a] * eval(child, use_br);
    }
    return v;
  }

  const Game& game_;
  const InfosetStore& store_;
  int br_player_ = 0;
  std::unordered_map<std::uint64_t, int> br_action_;
};

}  // namespace regret
