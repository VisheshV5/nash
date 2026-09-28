#include "regret/cfr/solver.hpp"

#include <cstring>
#include <stdexcept>

#include "regret/games/kuhn.hpp"
#include "regret/games/leduc.hpp"

namespace regret {
namespace {

constexpr char kMagic[8] = {'R', 'G', 'T', 'C', 'F', 'R', '0', '1'};

template <class Game>
class SolverImpl final : public CfrSolver {
 public:
  SolverImpl(Game game, const CfrParams& params) : mccfr_(std::move(game), params) {}

  std::int64_t run(std::int64_t max_iterations, double max_seconds) override {
    return mccfr_.run(max_iterations, max_seconds);
  }
  std::int64_t iteration() const override { return mccfr_.iteration(); }
  std::size_t num_infosets() const override { return mccfr_.store().size(); }
  std::size_t memory_bytes() const override { return mccfr_.store().memory_bytes(); }
  int num_players() const override { return mccfr_.game().num_players(); }

  NashConvResult nash_conv() const override {
    BestResponse<Game> br(mccfr_.game(), mccfr_.store());
    return br.nash_conv();
  }

  std::vector<std::pair<std::uint64_t, std::vector<double>>> average_strategy() const override {
    std::vector<std::pair<std::uint64_t, std::vector<double>>> out;
    for (std::uint64_t key : mccfr_.store().sorted_keys()) {
      const int n = mccfr_.store().find(key).num_actions;
      std::vector<double> probs(n);
      regret::average_strategy(mccfr_.store(), key, n, probs.data());
      out.emplace_back(key, std::move(probs));
    }
    return out;
  }

  std::string save() const override {
    std::string out(kMagic, sizeof(kMagic));
    const std::int64_t it = mccfr_.iteration();
    out.append(reinterpret_cast<const char*>(&it), sizeof(it));
    out += mccfr_.store().serialize();
    return out;
  }

  void load(const std::string& bytes) override {
    constexpr std::size_t header = sizeof(kMagic) + sizeof(std::int64_t);
    if (bytes.size() < header || std::memcmp(bytes.data(), kMagic, sizeof(kMagic)) != 0) {
      throw std::invalid_argument("CfrSolver: not a solver checkpoint");
    }
    std::int64_t it;
    std::memcpy(&it, bytes.data() + sizeof(kMagic), sizeof(it));
    mccfr_.store().deserialize(bytes.substr(header));
    mccfr_.set_iteration(it);
  }

 private:
  Mccfr<Game> mccfr_;
};

}  // namespace

std::unique_ptr<CfrSolver> make_solver(const std::string& game, const CfrParams& params) {
  if (game == "kuhn") return std::make_unique<SolverImpl<games::Kuhn>>(games::Kuhn(2), params);
  if (game == "kuhn3") return std::make_unique<SolverImpl<games::Kuhn>>(games::Kuhn(3), params);
  if (game == "leduc") return std::make_unique<SolverImpl<games::Leduc>>(games::Leduc(), params);
  throw std::invalid_argument("unknown game '" + game + "' (kuhn, kuhn3, leduc)");
}

}  // namespace regret
