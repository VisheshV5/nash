#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "regret/cfr/best_response.hpp"
#include "regret/cfr/mccfr.hpp"
#include "regret/games/nlhe.hpp"

namespace regret {

// Type-erased MCCFR solver, so Python can drive any game through one interface.
class CfrSolver {
 public:
  virtual ~CfrSolver() = default;

  virtual std::int64_t run(std::int64_t max_iterations, double max_seconds) = 0;
  virtual std::int64_t iteration() const = 0;
  virtual std::size_t num_infosets() const = 0;
  virtual std::size_t memory_bytes() const = 0;
  virtual int num_players() const = 0;

  // Exact NashConv of the average strategy (small games only; throws otherwise).
  virtual NashConvResult nash_conv() const = 0;
  // Average strategy at one infoset; empty if never visited.
  virtual std::vector<double> strategy(std::uint64_t key) const = 0;

  // Whole average strategy in flat arrays: sorted keys, offsets (size + 1) into probs.
  struct StrategyArrays {
    std::vector<std::uint64_t> keys;
    std::vector<std::uint32_t> offsets;
    std::vector<float> probs;
  };
  virtual StrategyArrays export_strategy() const = 0;
  // (key, average strategy) for every visited infoset, sorted by key.
  virtual std::vector<std::pair<std::uint64_t, std::vector<double>>> average_strategy() const = 0;

  // Iteration count + regret store, as bytes. load() restores exactly.
  virtual std::string save() const = 0;
  virtual void load(const std::string& bytes) = 0;
};

// Games: "kuhn" (2 players), "kuhn3", "leduc".
std::unique_ptr<CfrSolver> make_solver(const std::string& game, const CfrParams& params);

// No-limit hold'em with the given table, action abstraction and card bucket tables.
std::unique_ptr<CfrSolver> make_nlhe_solver(const TableRules& rules,
                                            const abstraction::ActionRules& actions,
                                            std::shared_ptr<const games::NlheTables> tables,
                                            const CfrParams& params);

}  // namespace regret
