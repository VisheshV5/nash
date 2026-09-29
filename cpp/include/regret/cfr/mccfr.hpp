#pragma once

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <thread>
#include <vector>

#include "regret/cfr/infoset_store.hpp"
#include "regret/cfr/rng.hpp"

namespace regret {

struct CfrParams {
  std::uint64_t seed = 0;
  int threads = 1;
  // Linear CFR: every `discount_interval` iterations up to `lcfr_until`, multiply regrets and
  // strategy sums by t / (t + 1), where t counts the intervals so far. 0 disables.
  std::int64_t lcfr_until = 0;
  std::int64_t discount_interval = 1000;
  // Negative-regret pruning: after `prune_after` iterations, a `prune_probability` share of
  // iterations skip the traverser's actions with regret below `prune_threshold`.
  std::int64_t prune_after = -1;  // -1 disables
  double prune_probability = 0.95;
  float prune_threshold = -1e9f;
  float regret_floor = -1e30f;  // regrets never go below this, so pruned actions can recover
  std::size_t max_infosets = std::size_t{1} << 20;  // regret table capacity
};

// Regret matching over the infoset's regrets.
inline void regret_matching(const InfosetStore::Slot& slot, double* sigma) {
  double total = 0;
  for (int a = 0; a < slot.num_actions; ++a) {
    sigma[a] = std::max(0.0f, slot.data[a].load(std::memory_order_relaxed));
    total += sigma[a];
  }
  for (int a = 0; a < slot.num_actions; ++a) {
    sigma[a] = total > 0 ? sigma[a] / total : 1.0 / slot.num_actions;
  }
}

// N-player external-sampling Monte Carlo CFR (Lanctot et al. 2009) with the Pluribus
// additions: Linear CFR discounting and negative-regret pruning.
//
// Iteration i traverses for player i % N with randomness drawn only from (seed, i), so a
// single-threaded run is fully determined by the seed and the iteration range, and a resumed
// run matches an uninterrupted one bit for bit. With several threads, iterations run
// concurrently against shared regrets (atomic updates): statistically the same algorithm, but
// not bit-reproducible.
//
// Game concept: State; num_players(); sample_deal(Rng&) -> State (all chance at the root);
// is_terminal; current_player; num_actions; apply(State&, action index); infoset_key (for the
// player to act); utility(State, player). Exact best response additionally needs num_deals()
// and deal(i) enumerating equally likely deals (small games only).
template <class Game>
class Mccfr {
 public:
  Mccfr(Game game, CfrParams params)
      : game_(std::move(game)), params_(params), store_(params.max_infosets * 10 / 9 + 1) {}

  // Runs until `max_iterations` more iterations are done or `max_seconds` pass. Returns the
  // number of iterations run.
  std::int64_t run(std::int64_t max_iterations, double max_seconds) {
    using Clock = std::chrono::steady_clock;
    const Clock::time_point deadline =
        max_seconds >= 1e9 ? Clock::time_point::max()
                           : Clock::now() + std::chrono::duration_cast<Clock::duration>(
                                                std::chrono::duration<double>(max_seconds));
    const std::int64_t start = iteration_;
    const std::int64_t end = iteration_ + max_iterations;
    while (iteration_ < end && std::chrono::steady_clock::now() < deadline) {
      std::int64_t segment_end = end;
      const bool discounting = params_.lcfr_until > 0 && iteration_ < params_.lcfr_until;
      if (discounting) {
        const std::int64_t boundary =
            (iteration_ / params_.discount_interval + 1) * params_.discount_interval;
        segment_end = std::min(segment_end, boundary);
      }
      run_segment(segment_end, deadline);
      if (discounting && iteration_ % params_.discount_interval == 0 &&
          iteration_ <= params_.lcfr_until) {
        const double t = static_cast<double>(iteration_ / params_.discount_interval);
        const auto f = static_cast<float>(t / (t + 1));
        store_.scale(f, f);
      }
    }
    return iteration_ - start;
  }

  std::int64_t iteration() const { return iteration_; }
  void set_iteration(std::int64_t it) { iteration_ = it; }
  InfosetStore& store() { return store_; }
  const InfosetStore& store() const { return store_; }
  const Game& game() const { return game_; }

 private:
  void run_one(std::int64_t i) {
    Rng rng = Rng::stream(params_.seed, static_cast<std::uint64_t>(i));
    const int traverser = static_cast<int>(i % game_.num_players());
    const bool prune = params_.prune_after >= 0 && i >= params_.prune_after &&
                       rng.uniform() < params_.prune_probability;
    auto state = game_.sample_deal(rng);
    traverse(state, traverser, rng, prune);
  }

  void run_segment(std::int64_t segment_end, std::chrono::steady_clock::time_point deadline) {
    if (params_.threads <= 1) {
      while (iteration_ < segment_end && std::chrono::steady_clock::now() < deadline) {
        run_one(iteration_++);
      }
      return;
    }
    std::atomic<std::int64_t> next{iteration_};
    auto worker = [&] {
      while (std::chrono::steady_clock::now() < deadline) {
        const std::int64_t i = next.fetch_add(1);
        if (i >= segment_end) return;
        run_one(i);
      }
    };
    std::vector<std::thread> pool;
    for (int t = 0; t < params_.threads; ++t) pool.emplace_back(worker);
    for (auto& th : pool) th.join();
    // Every claimed iteration completed, so [start, min(next, end)) is done.
    iteration_ = std::min(next.load(), segment_end);
  }

  template <class State>
  double traverse(const State& s, int traverser, Rng& rng, bool prune) {
    if (game_.is_terminal(s)) return game_.utility(s, traverser);
    const int player = game_.current_player(s);
    const int n = game_.num_actions(s);
    const InfosetStore::Slot slot = store_.get_or_create(game_.infoset_key(s), n);
    double sigma[InfosetStore::kMaxActions];
    regret_matching(slot, sigma);

    if (player == traverser) {
      double value[InfosetStore::kMaxActions];
      bool explored[InfosetStore::kMaxActions];
      double expected = 0;
      for (int a = 0; a < n; ++a) {
        explored[a] =
            !(prune && slot.data[a].load(std::memory_order_relaxed) < params_.prune_threshold);
        if (!explored[a]) continue;
        State child = s;
        game_.apply(child, a);
        value[a] = traverse(child, traverser, rng, prune);
        expected += sigma[a] * value[a];
      }
      for (int a = 0; a < n; ++a) {
        if (!explored[a]) continue;
        std::atomic<float>& r = slot.data[a];
        atomic_add(r, static_cast<float>(value[a] - expected));
        if (r.load(std::memory_order_relaxed) < params_.regret_floor) {
          r.store(params_.regret_floor, std::memory_order_relaxed);
        }
      }
      return expected;
    }

    // Other player: accumulate the average strategy, then sample one action.
    for (int a = 0; a < n; ++a) atomic_add(slot.data[n + a], static_cast<float>(sigma[a]));
    double u = rng.uniform();
    int a = 0;
    while (a < n - 1 && u >= sigma[a]) u -= sigma[a++];
    State child = s;
    game_.apply(child, a);
    return traverse(child, traverser, rng, prune);
  }

  Game game_;
  CfrParams params_;
  InfosetStore store_;
  std::int64_t iteration_ = 0;
};

}  // namespace regret
