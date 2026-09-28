#pragma once

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

namespace regret {

// Adds `delta` to an atomic float (C++17 has no fetch_add for floats).
inline void atomic_add(std::atomic<float>& x, float delta) {
  float cur = x.load(std::memory_order_relaxed);
  while (!x.compare_exchange_weak(cur, cur + delta, std::memory_order_relaxed)) {
  }
}

// Regrets and strategy sums per information set, created on first visit.
//
// A fixed-capacity, open-addressing hash table with lock-free lookups (atomic loads only) and
// lock-free inserts (CAS on the key slot); only allocating a new infoset's floats takes a
// mutex. Each infoset owns 2 * num_actions floats: regrets [0, n) then strategy sums [n, 2n).
// Floats live in fixed-size blocks that never move, so pointers stay valid.
//
// Capacity is fixed up front (the memory planner sizes it); inserting past 90% load throws.
class InfosetStore {
 public:
  static constexpr int kMaxActions = 16;

  struct Slot {
    std::atomic<float>* data = nullptr;  // regrets, then strategy sums
    int num_actions = 0;
  };

  explicit InfosetStore(std::size_t min_capacity = std::size_t{1} << 16);

  // Returns the infoset's slot, creating a zeroed one if needed. Throws if an existing infoset
  // is looked up with a different number of actions (a key collision or a game bug).
  Slot get_or_create(std::uint64_t key, int num_actions);
  // Returns an empty slot if the infoset was never visited.
  Slot find(std::uint64_t key) const;

  std::size_t size() const { return size_.load(std::memory_order_relaxed); }
  std::size_t capacity() const { return capacity_; }
  std::size_t memory_bytes() const;

  // Multiplies all regrets by `regret_factor` and strategy sums by `strategy_factor`
  // (Linear CFR discounting). Must not run concurrently with traversals.
  void scale(float regret_factor, float strategy_factor);

  // Sorted by key: a deterministic dump regardless of insertion order.
  std::vector<std::uint64_t> sorted_keys() const;

  // Binary form: magic, count, then per infoset (sorted by key): key, num_actions, 2n floats.
  std::string serialize() const;
  void deserialize(const std::string& bytes);  // replaces the contents
  void clear();

 private:
  static constexpr std::uint64_t kEmpty = ~std::uint64_t{0};
  static constexpr std::size_t kBlockFloats = std::size_t{1} << 20;  // 4 MB blocks
  static constexpr std::size_t kMaxBlocks = std::size_t{1} << 16;    // up to 256 GB

  static std::uint64_t mix(std::uint64_t key) {
    key ^= key >> 33;
    key *= 0xFF51AFD7ED558CCDULL;
    key ^= key >> 33;
    return key;
  }
  std::atomic<float>* allocate(int floats);

  std::size_t capacity_;
  std::size_t mask_;
  std::unique_ptr<std::atomic<std::uint64_t>[]> keys_;
  std::unique_ptr<std::atomic<std::atomic<float>*>[]> data_;
  std::unique_ptr<std::atomic<std::uint8_t>[]> actions_;

  std::mutex alloc_mu_;
  std::vector<std::unique_ptr<std::atomic<float>[]>> blocks_;
  std::size_t block_used_ = kBlockFloats;  // forces a first block on first allocation
  std::atomic<std::size_t> size_{0};
};

}  // namespace regret
