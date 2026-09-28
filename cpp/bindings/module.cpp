#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <string>
#include <vector>

#include "regret/cards.hpp"
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
}
