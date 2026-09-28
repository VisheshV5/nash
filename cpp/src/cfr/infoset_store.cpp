#include "regret/cfr/infoset_store.hpp"

#include <algorithm>
#include <cstring>
#include <stdexcept>
#include <thread>

namespace regret {
namespace {

constexpr char kMagic[8] = {'R', 'G', 'T', 'S', 'T', 'O', 'R', '1'};

template <class T>
void put(std::string& out, const T& v) {
  out.append(reinterpret_cast<const char*>(&v), sizeof(T));
}

template <class T>
T get(const std::string& in, std::size_t& pos) {
  if (pos + sizeof(T) > in.size()) throw std::invalid_argument("InfosetStore: truncated data");
  T v;
  std::memcpy(&v, in.data() + pos, sizeof(T));
  pos += sizeof(T);
  return v;
}

}  // namespace

InfosetStore::InfosetStore(std::size_t min_capacity) {
  capacity_ = 16;
  while (capacity_ < min_capacity) capacity_ <<= 1;
  mask_ = capacity_ - 1;
  keys_.reset(new std::atomic<std::uint64_t>[capacity_]);
  data_.reset(new std::atomic<std::atomic<float>*>[capacity_]);
  actions_.reset(new std::atomic<std::uint8_t>[capacity_]);
  blocks_.reserve(kMaxBlocks);
  clear();
}

void InfosetStore::clear() {
  for (std::size_t i = 0; i < capacity_; ++i) {
    keys_[i].store(kEmpty, std::memory_order_relaxed);
    data_[i].store(nullptr, std::memory_order_relaxed);
    actions_[i].store(0, std::memory_order_relaxed);
  }
  std::lock_guard<std::mutex> lock(alloc_mu_);
  blocks_.clear();
  block_used_ = kBlockFloats;
  size_.store(0);
}

std::atomic<float>* InfosetStore::allocate(int floats) {
  std::lock_guard<std::mutex> lock(alloc_mu_);
  if (block_used_ + static_cast<std::size_t>(floats) > kBlockFloats) {
    if (blocks_.size() == kMaxBlocks) throw std::runtime_error("InfosetStore: out of blocks");
    blocks_.emplace_back(new std::atomic<float>[kBlockFloats]);
    for (std::size_t i = 0; i < kBlockFloats; ++i) {
      blocks_.back()[i].store(0.0f, std::memory_order_relaxed);
    }
    block_used_ = 0;
  }
  std::atomic<float>* p = blocks_.back().get() + block_used_;
  block_used_ += static_cast<std::size_t>(floats);
  return p;
}

InfosetStore::Slot InfosetStore::get_or_create(std::uint64_t key, int num_actions) {
  if (num_actions < 1 || num_actions > kMaxActions) {
    throw std::invalid_argument("InfosetStore: bad number of actions");
  }
  if (key == kEmpty) throw std::invalid_argument("InfosetStore: reserved key");
  for (std::size_t i = mix(key) & mask_;; i = (i + 1) & mask_) {
    std::uint64_t k = keys_[i].load(std::memory_order_acquire);
    if (k == kEmpty) {
      if (size() * 10 >= capacity_ * 9) {
        throw std::runtime_error("InfosetStore: table is full; raise the infoset capacity");
      }
      if (keys_[i].compare_exchange_strong(k, key, std::memory_order_acq_rel)) {
        actions_[i].store(static_cast<std::uint8_t>(num_actions), std::memory_order_relaxed);
        data_[i].store(allocate(2 * num_actions), std::memory_order_release);
        size_.fetch_add(1, std::memory_order_relaxed);
        return {data_[i].load(std::memory_order_relaxed), num_actions};
      }
      // Lost the race for this slot: `k` now holds the winner's key.
    }
    if (k == key) {
      std::atomic<float>* d;
      while ((d = data_[i].load(std::memory_order_acquire)) == nullptr) {
        std::this_thread::yield();  // another thread is still allocating it
      }
      if (actions_[i].load(std::memory_order_relaxed) != num_actions) {
        throw std::logic_error("InfosetStore: action count changed for an infoset");
      }
      return {d, num_actions};
    }
  }
}

InfosetStore::Slot InfosetStore::find(std::uint64_t key) const {
  for (std::size_t i = mix(key) & mask_;; i = (i + 1) & mask_) {
    const std::uint64_t k = keys_[i].load(std::memory_order_acquire);
    if (k == kEmpty) return {};
    if (k == key) {
      std::atomic<float>* d = data_[i].load(std::memory_order_acquire);
      if (d == nullptr) return {};  // being created right now
      return {d, actions_[i].load(std::memory_order_relaxed)};
    }
  }
}

std::size_t InfosetStore::memory_bytes() const {
  return blocks_.size() * kBlockFloats * sizeof(float) +
         capacity_ * (sizeof(std::uint64_t) + sizeof(void*) + 1);
}

void InfosetStore::scale(float regret_factor, float strategy_factor) {
  for (std::size_t i = 0; i < capacity_; ++i) {
    std::atomic<float>* d = data_[i].load(std::memory_order_relaxed);
    if (d == nullptr) continue;
    const int n = actions_[i].load(std::memory_order_relaxed);
    for (int a = 0; a < n; ++a) {
      d[a].store(d[a].load(std::memory_order_relaxed) * regret_factor, std::memory_order_relaxed);
      d[n + a].store(d[n + a].load(std::memory_order_relaxed) * strategy_factor,
                     std::memory_order_relaxed);
    }
  }
}

std::vector<std::uint64_t> InfosetStore::sorted_keys() const {
  std::vector<std::uint64_t> keys;
  keys.reserve(size());
  for (std::size_t i = 0; i < capacity_; ++i) {
    const std::uint64_t k = keys_[i].load(std::memory_order_relaxed);
    if (k != kEmpty) keys.push_back(k);
  }
  std::sort(keys.begin(), keys.end());
  return keys;
}

std::string InfosetStore::serialize() const {
  const std::vector<std::uint64_t> keys = sorted_keys();
  std::string out(kMagic, sizeof(kMagic));
  put<std::uint64_t>(out, keys.size());
  for (std::uint64_t key : keys) {
    const Slot slot = find(key);
    put<std::uint64_t>(out, key);
    put<std::uint8_t>(out, static_cast<std::uint8_t>(slot.num_actions));
    for (int i = 0; i < 2 * slot.num_actions; ++i) {
      put<float>(out, slot.data[i].load(std::memory_order_relaxed));
    }
  }
  return out;
}

void InfosetStore::deserialize(const std::string& bytes) {
  if (bytes.size() < sizeof(kMagic) || std::memcmp(bytes.data(), kMagic, sizeof(kMagic)) != 0) {
    throw std::invalid_argument("InfosetStore: not a regret store");
  }
  clear();
  std::size_t pos = sizeof(kMagic);
  const auto count = get<std::uint64_t>(bytes, pos);
  for (std::uint64_t i = 0; i < count; ++i) {
    const auto key = get<std::uint64_t>(bytes, pos);
    const int n = get<std::uint8_t>(bytes, pos);
    const Slot slot = get_or_create(key, n);
    for (int j = 0; j < 2 * n; ++j) {
      slot.data[j].store(get<float>(bytes, pos), std::memory_order_relaxed);
    }
  }
  if (pos != bytes.size()) throw std::invalid_argument("InfosetStore: trailing bytes");
}

}  // namespace regret
