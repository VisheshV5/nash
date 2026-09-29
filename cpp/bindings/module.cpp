#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <string>
#include <vector>

#include "regret/abstraction/actions.hpp"
#include "regret/abstraction/builders.hpp"
#include "regret/abstraction/card_features.hpp"
#include "regret/cards.hpp"
#include "regret/cfr/solver.hpp"
#include "regret/engine.hpp"
#include "regret/evaluator.hpp"
#include "regret/isomorphism.hpp"
#include "regret/version.hpp"

namespace py = pybind11;
using regret::Card;

namespace {

using CardArray = py::array_t<std::uint8_t, py::array::c_style | py::array::forcecast>;

Card checked_card(int c) {
  if (c < 0 || c >= regret::kNumCards) {
    throw py::value_error("card id must be in 0..51, got " + std::to_string(c));
  }
  return static_cast<Card>(c);
}

// Validates ids and uniqueness; returns the cards as a vector.
std::vector<Card> checked_cards(const std::vector<int>& ids) {
  std::vector<Card> cards;
  regret::CardMask seen = 0;
  for (int id : ids) {
    const Card c = checked_card(id);
    if (seen & regret::card_bit(c)) {
      throw py::value_error("duplicate card " + regret::card_to_string(c));
    }
    seen |= regret::card_bit(c);
    cards.push_back(c);
  }
  return cards;
}

// Checks a 2-D batch of card ids: shape (n, width), ids in range, no duplicates per row.
void check_batch(const CardArray& a, int width_lo, int width_hi) {
  if (a.ndim() != 2) throw py::value_error("expected a 2-D array of card ids");
  const auto width = a.shape(1);
  if (width < width_lo || width > width_hi) {
    throw py::value_error("expected " + std::to_string(width_lo) + ".." + std::to_string(width_hi) +
                          " cards per row, got " + std::to_string(width));
  }
  const auto r = a.unchecked<2>();
  for (py::ssize_t i = 0; i < a.shape(0); ++i) {
    regret::CardMask seen = 0;
    for (py::ssize_t j = 0; j < width; ++j) {
      const Card c = r(i, j);
      if (c >= regret::kNumCards)
        throw py::value_error("card id out of range in row " + std::to_string(i));
      if (seen & regret::card_bit(c)) {
        throw py::value_error("duplicate card in row " + std::to_string(i));
      }
      seen |= regret::card_bit(c);
    }
  }
}

}  // namespace

PYBIND11_MODULE(_core, m) {
  m.doc() = "Regret native core (engine, abstraction, CFR, search).";
  m.def("version", [] { return regret::kVersion; }, "Version string the extension was built with.");

  // ---- cards
  m.def(
      "parse_card",
      [](const std::string& s) {
        const auto c = regret::parse_card(s);
        if (!c) throw py::value_error("invalid card '" + s + "' (expected e.g. 'As', 'Td', '2c')");
        return static_cast<int>(*c);
      },
      py::arg("text"), "Parse a card like 'As' into its id (rank * 4 + suit).");
  m.def(
      "card_to_str", [](int c) { return regret::card_to_string(checked_card(c)); }, py::arg("card"),
      "Format a card id as text, e.g. 51 -> 'As'.");

  // ---- evaluator
  m.def(
      "evaluate",
      [](const std::vector<int>& ids) {
        if (ids.size() < 5 || ids.size() > 7) throw py::value_error("evaluate needs 5 to 7 cards");
        const auto cards = checked_cards(ids);
        return regret::evaluate(cards.data(), static_cast<int>(cards.size()));
      },
      py::arg("cards"), "Strength of the best 5-card hand (higher is better).");
  m.def(
      "evaluate_batch",
      [](const CardArray& a) {
        check_batch(a, 5, 7);
        const auto n = a.shape(0);
        const int width = static_cast<int>(a.shape(1));
        py::array_t<std::uint32_t> out(n);
        auto w = out.mutable_unchecked<1>();
        const std::uint8_t* data = a.data();
        {
          py::gil_scoped_release release;
          for (py::ssize_t i = 0; i < n; ++i) w(i) = regret::evaluate(data + i * width, width);
        }
        return out;
      },
      py::arg("cards"), "Evaluate each row of an (n, 5..7) uint8 array of card ids.");
  m.def(
      "hand_category", [](regret::HandValue v) { return static_cast<int>(regret::category_of(v)); },
      py::arg("value"), "Category (0 = high card .. 8 = straight flush) of a hand value.");

  // ---- isomorphism
  py::class_<regret::HandIndexer>(m, "HandIndexer", "Dense index of deals up to suit isomorphism.")
      .def(py::init<std::vector<int>>(), py::arg("cards_per_round"))
      .def_property_readonly("size", &regret::HandIndexer::size)
      .def_property_readonly("total_cards", &regret::HandIndexer::total_cards)
      .def_property_readonly("cards_per_round",
                             [](const regret::HandIndexer& h) {
                               std::vector<int> v;
                               for (int r = 0; r < h.rounds(); ++r)
                                 v.push_back(h.cards_in_round(r));
                               return v;
                             })
      .def(
          "index",
          [](const regret::HandIndexer& h, const std::vector<int>& ids) {
            if (static_cast<int>(ids.size()) != h.total_cards()) {
              throw py::value_error("expected " + std::to_string(h.total_cards()) + " cards");
            }
            const auto cards = checked_cards(ids);
            return h.index(cards.data());
          },
          py::arg("cards"))
      .def(
          "unindex",
          [](const regret::HandIndexer& h, std::uint64_t idx) {
            if (idx >= h.size()) throw py::index_error("index out of range");
            std::vector<Card> out(h.total_cards());
            h.unindex(idx, out.data());
            return std::vector<int>(out.begin(), out.end());
          },
          py::arg("index"))
      .def(
          "index_batch",
          [](const regret::HandIndexer& h, const CardArray& a) {
            check_batch(a, h.total_cards(), h.total_cards());
            const auto n = a.shape(0);
            py::array_t<std::uint64_t> out(n);
            auto w = out.mutable_unchecked<1>();
            const std::uint8_t* data = a.data();
            {
              py::gil_scoped_release release;
              for (py::ssize_t i = 0; i < n; ++i) w(i) = h.index(data + i * h.total_cards());
            }
            return out;
          },
          py::arg("cards"), "Index each row of an (n, total_cards) uint8 array.")
      .def(
          "unindex_batch",
          [](const regret::HandIndexer& h, const py::array_t<std::uint64_t>& idx) {
            const auto r = idx.unchecked<1>();
            const auto n = idx.shape(0);
            for (py::ssize_t i = 0; i < n; ++i) {
              if (r(i) >= h.size()) throw py::index_error("index out of range");
            }
            py::array_t<std::uint8_t> out({n, static_cast<py::ssize_t>(h.total_cards())});
            std::uint8_t* data = out.mutable_data();
            {
              py::gil_scoped_release release;
              for (py::ssize_t i = 0; i < n; ++i) h.unindex(r(i), data + i * h.total_cards());
            }
            return out;
          },
          py::arg("indices"), "Canonical representatives, as an (n, total_cards) uint8 array.");

  // ---- engine
  py::enum_<regret::Street>(m, "Street")
      .value("PREFLOP", regret::Street::kPreflop)
      .value("FLOP", regret::Street::kFlop)
      .value("TURN", regret::Street::kTurn)
      .value("RIVER", regret::Street::kRiver);
  py::enum_<regret::PlayerStatus>(m, "PlayerStatus")
      .value("ACTIVE", regret::PlayerStatus::kActive)
      .value("FOLDED", regret::PlayerStatus::kFolded)
      .value("ALL_IN", regret::PlayerStatus::kAllIn);
  py::enum_<regret::ActionType>(m, "ActionType")
      .value("FOLD", regret::ActionType::kFold)
      .value("CHECK_CALL", regret::ActionType::kCheckCall)
      .value("BET_RAISE", regret::ActionType::kBetRaise);

  py::class_<regret::Action>(m, "Action")
      .def(py::init<regret::ActionType, regret::Chips>(), py::arg("type"), py::arg("amount") = 0)
      .def_readonly("type", &regret::Action::type)
      .def_readonly("amount", &regret::Action::amount)
      .def_static("fold", &regret::Action::fold)
      .def_static("check_call", &regret::Action::check_call)
      .def_static("bet_raise_to", &regret::Action::bet_raise_to, py::arg("amount"))
      .def("__repr__", [](const regret::Action& a) {
        switch (a.type) {
          case regret::ActionType::kFold:
            return std::string("Action.fold()");
          case regret::ActionType::kCheckCall:
            return std::string("Action.check_call()");
          default:
            return "Action.bet_raise_to(" + std::to_string(a.amount) + ")";
        }
      });

  py::class_<regret::LegalActions>(m, "LegalActions")
      .def_readonly("fold", &regret::LegalActions::fold)
      .def_readonly("check_call", &regret::LegalActions::check_call)
      .def_readonly("call_amount", &regret::LegalActions::call_amount)
      .def_readonly("bet_raise", &regret::LegalActions::bet_raise)
      .def_readonly("min_raise_to", &regret::LegalActions::min_raise_to)
      .def_readonly("max_raise_to", &regret::LegalActions::max_raise_to);

  py::class_<regret::Pot>(m, "Pot")
      .def_readonly("amount", &regret::Pot::amount)
      .def_readonly("eligible", &regret::Pot::eligible);

  py::class_<regret::HandState>(m, "HandState", "No-limit hold'em betting state for 2-6 seats.")
      .def(py::init([](std::vector<regret::Chips> stacks, regret::Chips sb, regret::Chips bb) {
             return regret::HandState(regret::TableRules{std::move(stacks), sb, bb});
           }),
           py::arg("starting_stacks"), py::arg("small_blind"), py::arg("big_blind"))
      .def_property_readonly("num_players", &regret::HandState::num_players)
      .def_property_readonly("button", &regret::HandState::button)
      .def_property_readonly("street", &regret::HandState::street)
      .def_property_readonly("is_terminal", &regret::HandState::is_terminal)
      .def_property_readonly("is_showdown", &regret::HandState::is_showdown)
      .def_property_readonly("to_act", &regret::HandState::to_act)
      .def_property_readonly("current_bet", &regret::HandState::current_bet)
      .def_property_readonly("pot", &regret::HandState::pot)
      .def_property_readonly("non_folded", &regret::HandState::non_folded)
      .def_property_readonly("stacks",
                             [](const regret::HandState& h) {
                               std::vector<regret::Chips> v;
                               for (int i = 0; i < h.num_players(); ++i) v.push_back(h.stack(i));
                               return v;
                             })
      .def_property_readonly("bets",
                             [](const regret::HandState& h) {
                               std::vector<regret::Chips> v;
                               for (int i = 0; i < h.num_players(); ++i) v.push_back(h.bet(i));
                               return v;
                             })
      .def_property_readonly("contributed",
                             [](const regret::HandState& h) {
                               std::vector<regret::Chips> v;
                               for (int i = 0; i < h.num_players(); ++i)
                                 v.push_back(h.contributed(i));
                               return v;
                             })
      .def_property_readonly("statuses",
                             [](const regret::HandState& h) {
                               std::vector<regret::PlayerStatus> v;
                               for (int i = 0; i < h.num_players(); ++i) v.push_back(h.status(i));
                               return v;
                             })
      .def_property_readonly(
          "history",
          [](const regret::HandState& h) {
            py::list out;
            for (const auto& e : h.history())
              out.append(py::make_tuple(e.street, e.seat, e.action));
            return out;
          },
          "List of (street, seat, action).")
      .def("legal_actions", &regret::HandState::legal_actions)
      .def(
          "why_illegal",
          [](const regret::HandState& h, const regret::Action& a) -> py::object {
            const char* why = h.why_illegal(a);
            return why ? py::str(why) : py::object(py::none());
          },
          py::arg("action"), "None if legal, else the reason.")
      .def("apply", &regret::HandState::apply, py::arg("action"))
      .def("pots", &regret::HandState::pots)
      .def(
          "payoffs",
          [](const regret::HandState& h, const std::vector<std::vector<int>>& hole,
             const std::vector<int>& board) {
            std::vector<std::array<Card, 2>> hc;
            for (const auto& cards : hole) {
              if (cards.size() != 2) throw py::value_error("each seat needs 2 hole cards");
              hc.push_back({checked_card(cards[0]), checked_card(cards[1])});
            }
            std::array<Card, 5> b{};
            if (h.is_showdown()) {
              if (board.size() != 5) throw py::value_error("showdown needs a 5-card board");
              for (int i = 0; i < 5; ++i) b[i] = checked_card(board[i]);
            }
            return h.payoffs(hc, b);
          },
          py::arg("hole_cards"), py::arg("board"), "Net chips won per seat once the hand is over.")
      .def("copy", [](const regret::HandState& h) { return regret::HandState(h); });

  // ---- cfr
  py::class_<regret::NashConvResult>(m, "NashConvResult")
      .def_readonly("best_response", &regret::NashConvResult::best_response)
      .def_readonly("on_policy", &regret::NashConvResult::on_policy)
      .def_readonly("nash_conv", &regret::NashConvResult::nash_conv);

  py::class_<regret::CfrSolver, std::unique_ptr<regret::CfrSolver>>(
      m, "CfrSolver", "External-sampling MCCFR (Linear CFR + pruning) for a named game.")
      .def(py::init([](const std::string& game, std::uint64_t seed, int threads,
                       std::int64_t lcfr_until, std::int64_t discount_interval,
                       std::int64_t prune_after, double prune_probability, float prune_threshold,
                       float regret_floor, std::size_t max_infosets) {
             if (threads < 1) throw py::value_error("threads must be >= 1");
             if (discount_interval < 1) throw py::value_error("discount_interval must be >= 1");
             regret::CfrParams p;
             p.seed = seed;
             p.threads = threads;
             p.lcfr_until = lcfr_until;
             p.discount_interval = discount_interval;
             p.prune_after = prune_after;
             p.prune_probability = prune_probability;
             p.prune_threshold = prune_threshold;
             p.regret_floor = regret_floor;
             p.max_infosets = max_infosets;
             return regret::make_solver(game, p);
           }),
           py::arg("game"), py::arg("seed") = 0, py::arg("threads") = 1, py::arg("lcfr_until") = 0,
           py::arg("discount_interval") = 1000, py::arg("prune_after") = -1,
           py::arg("prune_probability") = 0.95, py::arg("prune_threshold") = -1e9f,
           py::arg("regret_floor") = -1e30f, py::arg("max_infosets") = std::size_t{1} << 20)
      .def(
          "run",
          [](regret::CfrSolver& s, std::int64_t max_iterations, double max_seconds) {
            py::gil_scoped_release release;
            return s.run(max_iterations, max_seconds);
          },
          py::arg("max_iterations"), py::arg("max_seconds") = 1e18,
          "Run up to max_iterations more iterations or until max_seconds pass; returns the "
          "number run.")
      .def_property_readonly("iteration", &regret::CfrSolver::iteration)
      .def_property_readonly("num_infosets", &regret::CfrSolver::num_infosets)
      .def_property_readonly("memory_bytes", &regret::CfrSolver::memory_bytes)
      .def_property_readonly("num_players", &regret::CfrSolver::num_players)
      .def(
          "nash_conv",
          [](const regret::CfrSolver& s) {
            py::gil_scoped_release release;
            return s.nash_conv();
          },
          "Exact NashConv of the average strategy (small games only).")
      .def("average_strategy", &regret::CfrSolver::average_strategy)
      .def("save", [](const regret::CfrSolver& s) { return py::bytes(s.save()); })
      .def(
          "load", [](regret::CfrSolver& s, const py::bytes& b) { s.load(std::string(b)); },
          py::arg("data"));

  // ---- card abstraction
  namespace ab = regret::abstraction;
  auto u8 = [](const std::vector<std::uint8_t>& v) {
    return py::array_t<std::uint8_t>(static_cast<py::ssize_t>(v.size()), v.data());
  };
  auto u16 = [](const std::vector<std::uint16_t>& v) {
    return py::array_t<std::uint16_t>(static_cast<py::ssize_t>(v.size()), v.data());
  };
  auto f32 = [](const std::vector<float>& v) {
    return py::array_t<float>(static_cast<py::ssize_t>(v.size()), v.data());
  };
  auto board_of = [](const std::vector<int>& ids, std::size_t n) {
    if (ids.size() != n) throw py::value_error("expected " + std::to_string(n) + " board cards");
    auto cards = checked_cards(ids);
    return cards;
  };
  using U8Array = py::array_t<std::uint8_t, py::array::c_style | py::array::forcecast>;
  using U16Array = py::array_t<std::uint16_t, py::array::c_style | py::array::forcecast>;
  using F32Array = py::array_t<float, py::array::c_style | py::array::forcecast>;
  auto vec_u8 = [](const U8Array& a) {
    return std::vector<std::uint8_t>(a.data(), a.data() + a.size());
  };
  auto vec_u16 = [](const U16Array& a) {
    return std::vector<std::uint16_t>(a.data(), a.data() + a.size());
  };
  auto vec_f32 = [](const F32Array& a) {
    return std::vector<float>(a.data(), a.data() + a.size());
  };

  m.def(
      "hole_index", [](int a, int b) { return ab::hole_index(checked_card(a), checked_card(b)); },
      py::arg("a"), py::arg("b"), "Index 0..1325 of an unordered hole pair.");
  m.def(
      "river_equities",
      [=](const std::vector<int>& board) {
        const auto b = board_of(board, 5);
        std::vector<float> eq(ab::kNumHoles);
        {
          py::gil_scoped_release release;
          ab::river_equities(b.data(), eq.data());
        }
        return f32(eq);
      },
      py::arg("board"),
      "Equity of every hole index vs a random hand on a 5-card board (-1 if blocked).");
  m.def(
      "river_ochs",
      [=](const std::vector<int>& board, const U8Array& cluster_of, int k) {
        const auto b = board_of(board, 5);
        if (cluster_of.size() != ab::kNumHoles)
          throw py::value_error("cluster_of needs 1326 entries");
        std::vector<float> out(static_cast<std::size_t>(ab::kNumHoles) * k);
        ab::river_ochs(b.data(), cluster_of.data(), k, out.data());
        return f32(out).reshape({ab::kNumHoles, k});
      },
      py::arg("board"), py::arg("cluster_of"), py::arg("k"));
  m.def(
      "river_ochs_one",
      [=](const std::vector<int>& hole, const std::vector<int>& board, const U8Array& cluster_of,
          int k) {
        const auto h = board_of(hole, 2);
        const auto b = board_of(board, 5);
        std::vector<float> out(k);
        ab::river_ochs_one(h[0], h[1], b.data(), cluster_of.data(), k, out.data());
        return f32(out);
      },
      py::arg("hole"), py::arg("board"), py::arg("cluster_of"), py::arg("k"));
  m.def(
      "turn_histograms",
      [=](const std::vector<int>& board, int bins) {
        const auto b = board_of(board, 4);
        std::vector<std::uint8_t> out(static_cast<std::size_t>(ab::kNumHoles) * bins);
        ab::turn_histograms(b.data(), bins, out.data());
        return u8(out).reshape({ab::kNumHoles, bins});
      },
      py::arg("board"), py::arg("bins"));
  m.def(
      "turn_histogram_one",
      [=](const std::vector<int>& hole, const std::vector<int>& board, int bins) {
        const auto h = board_of(hole, 2);
        const auto b = board_of(board, 4);
        std::vector<std::uint8_t> out(bins);
        ab::turn_histogram_one(h[0], h[1], b.data(), bins, out.data());
        return u8(out);
      },
      py::arg("hole"), py::arg("board"), py::arg("bins"));
  m.def("preflop_class_of_hole", [=] { return u8(ab::preflop_class_of_hole()); });
  m.def(
      "preflop_class_equity",
      [](int samples, std::uint64_t seed, int threads) {
        py::gil_scoped_release release;
        return ab::preflop_class_equity(samples, seed, threads);
      },
      py::arg("samples"), py::arg("seed"), py::arg("threads"));
  m.def(
      "sample_river_ochs",
      [=](int boards, std::uint64_t seed, const U8Array& cluster_of, int k, int threads) {
        const auto c = vec_u8(cluster_of);
        std::vector<float> out;
        {
          py::gil_scoped_release release;
          out = ab::sample_river_ochs(boards, seed, c, k, threads);
        }
        return f32(out).reshape(
            {static_cast<py::ssize_t>(out.size() / k), static_cast<py::ssize_t>(k)});
      },
      py::arg("boards"), py::arg("seed"), py::arg("cluster_of"), py::arg("k"), py::arg("threads"));
  m.def(
      "sample_turn_histograms",
      [=](int boards, std::uint64_t seed, int bins, int threads) {
        std::vector<std::uint8_t> out;
        {
          py::gil_scoped_release release;
          out = ab::sample_turn_histograms(boards, seed, bins, threads);
        }
        return u8(out).reshape(
            {static_cast<py::ssize_t>(out.size() / bins), static_cast<py::ssize_t>(bins)});
      },
      py::arg("boards"), py::arg("seed"), py::arg("bins"), py::arg("threads"));
  m.def(
      "flop_features",
      [=](const U16Array& turn_table, const U16Array& turn_rank, int threads) {
        const auto tt = vec_u16(turn_table);
        const auto tr = vec_u16(turn_rank);
        ab::FlopFeatures f;
        {
          py::gil_scoped_release release;
          f = ab::flop_features(tt, tr, threads);
        }
        const auto k = static_cast<py::ssize_t>(tr.size());
        return py::make_tuple(u8(f.counts).reshape({static_cast<py::ssize_t>(f.rows), k}),
                              f32(f.weights));
      },
      py::arg("turn_table"), py::arg("turn_rank"), py::arg("threads"));
  m.def(
      "build_river_table",
      [=](const U8Array& cluster_of, int k, const F32Array& centroids, int threads) {
        const auto c = vec_u8(cluster_of);
        const auto cent = vec_f32(centroids);
        std::vector<std::uint16_t> out;
        {
          py::gil_scoped_release release;
          out = ab::build_river_table(c, k, cent, threads);
        }
        return u16(out);
      },
      py::arg("cluster_of"), py::arg("k"), py::arg("centroids"), py::arg("threads"));
  m.def(
      "build_turn_table",
      [=](int bins, const F32Array& centroids, int threads) {
        const auto cent = vec_f32(centroids);
        std::vector<std::uint16_t> out;
        {
          py::gil_scoped_release release;
          out = ab::build_turn_table(bins, cent, threads);
        }
        return u16(out);
      },
      py::arg("bins"), py::arg("centroids"), py::arg("threads"));
  m.def(
      "build_flop_table",
      [=](const U16Array& turn_table, const U16Array& turn_rank, const F32Array& centroids,
          int threads) {
        const auto tt = vec_u16(turn_table);
        const auto tr = vec_u16(turn_rank);
        const auto cent = vec_f32(centroids);
        std::vector<std::uint16_t> out;
        {
          py::gil_scoped_release release;
          out = ab::build_flop_table(tt, tr, cent, threads);
        }
        return u16(out);
      },
      py::arg("turn_table"), py::arg("turn_rank"), py::arg("centroids"), py::arg("threads"));

  // ---- action abstraction
  py::class_<ab::PreflopSizing>(m, "PreflopSizing")
      .def(py::init<>())
      .def_readwrite("open_bb", &ab::PreflopSizing::open_bb)
      .def_readwrite("reraise_x_ip", &ab::PreflopSizing::reraise_x_ip)
      .def_readwrite("reraise_x_oop", &ab::PreflopSizing::reraise_x_oop)
      .def_readwrite("four_bet_x", &ab::PreflopSizing::four_bet_x)
      .def_readwrite("all_in", &ab::PreflopSizing::all_in)
      .def_readwrite("max_raises", &ab::PreflopSizing::max_raises);
  py::class_<ab::PostflopSizing>(m, "PostflopSizing")
      .def(py::init<>())
      .def_readwrite("bet_pot", &ab::PostflopSizing::bet_pot)
      .def_readwrite("raise_pot", &ab::PostflopSizing::raise_pot)
      .def_readwrite("all_in", &ab::PostflopSizing::all_in)
      .def_readwrite("max_raises", &ab::PostflopSizing::max_raises);
  py::class_<ab::ActionRules>(m, "ActionRules")
      .def(py::init<>())
      .def_readwrite("chips_per_bb", &ab::ActionRules::chips_per_bb)
      .def_readwrite("preflop", &ab::ActionRules::preflop)
      .def_readwrite("postflop", &ab::ActionRules::postflop);
  m.def("abstract_actions", &ab::abstract_actions, py::arg("state"), py::arg("rules"));
  m.def("pot_fraction", &ab::pot_fraction, py::arg("state"), py::arg("to"));
  m.def("pseudo_harmonic", &ab::pseudo_harmonic, py::arg("a"), py::arg("b"), py::arg("x"));
  m.def("translate", &ab::translate, py::arg("state"), py::arg("rules"), py::arg("action"));
}
