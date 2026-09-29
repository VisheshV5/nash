#pragma once

#include <algorithm>
#include <atomic>
#include <cstdint>
#include <exception>
#include <mutex>
#include <thread>
#include <vector>

namespace regret {

// Runs fn(i) for i in [0, n) on `threads` workers (dynamic scheduling). Rethrows the first
// exception after all workers stop.
template <class Fn>
void parallel_for(std::int64_t n, int threads, Fn&& fn) {
  threads = std::max(1, std::min<int>(threads, static_cast<int>(std::max<std::int64_t>(n, 1))));
  std::atomic<std::int64_t> next{0};
  std::exception_ptr error;
  std::mutex error_mu;
  auto worker = [&] {
    try {
      for (std::int64_t i; (i = next.fetch_add(1)) < n;) fn(i);
    } catch (...) {
      std::lock_guard<std::mutex> lock(error_mu);
      if (!error) error = std::current_exception();
      next.store(n);
    }
  };
  if (threads == 1) {
    worker();
  } else {
    std::vector<std::thread> pool;
    for (int t = 0; t < threads; ++t) pool.emplace_back(worker);
    for (auto& th : pool) th.join();
  }
  if (error) std::rethrow_exception(error);
}

}  // namespace regret
