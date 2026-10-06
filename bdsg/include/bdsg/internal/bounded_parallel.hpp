//
// bounded_parallel.hpp
//
// Small internal helpers for bounded parallel work.
//

#ifndef BDSG_BOUNDED_PARALLEL_HPP_INCLUDED
#define BDSG_BOUNDED_PARALLEL_HPP_INCLUDED

#include <algorithm>
#include <atomic>
#include <cstddef>
#include <exception>
#include <mutex>
#include <thread>
#include <utility>
#include <vector>

namespace bdsg {
namespace internal {

/// Run numbered blocks with at most `workers` threads. The first worker error
/// stops new work, joins every thread, and is rethrown by the caller.
template<class Work>
void bounded_parallel_for(std::size_t blocks, std::size_t workers, Work&& work) {
    if (blocks == 0) {
        return;
    }
    workers = std::max<std::size_t>(1, std::min(workers, blocks));
    if (workers == 1) {
        for (std::size_t i = 0; i < blocks; ++i) {
            work(i);
        }
        return;
    }

    std::atomic<std::size_t> next{0};
    std::atomic<bool> cancelled{false};
    std::mutex error_mutex;
    std::exception_ptr error;
    std::vector<std::thread> threads;
    threads.reserve(workers);
    try {
        for (std::size_t worker = 0; worker < workers; ++worker) {
            threads.emplace_back([&]() {
                try {
                    while (!cancelled.load(std::memory_order_relaxed)) {
                        const std::size_t block = next.fetch_add(1, std::memory_order_relaxed);
                        if (block >= blocks) {
                            break;
                        }
                        work(block);
                    }
                } catch (...) {
                    std::lock_guard<std::mutex> lock(error_mutex);
                    if (!error) {
                        error = std::current_exception();
                    }
                    cancelled.store(true, std::memory_order_relaxed);
                }
            });
        }
    } catch (...) {
        cancelled.store(true, std::memory_order_relaxed);
        for (auto& thread : threads) {
            thread.join();
        }
        throw;
    }
    for (auto& thread : threads) {
        thread.join();
    }
    if (error) {
        std::rethrow_exception(error);
    }
}

} // namespace internal
} // namespace bdsg

#endif
