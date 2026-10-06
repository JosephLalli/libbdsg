//
// bounded_parallel.hpp
//
// Small bounded helpers for deterministic parallel producers.  Workers may
// finish out of order, but bytes are consumed in block order and at most one
// completed chunk per active block is retained.
//

#ifndef BDSG_BOUNDED_PARALLEL_HPP_INCLUDED
#define BDSG_BOUNDED_PARALLEL_HPP_INCLUDED

#include <algorithm>
#include <atomic>
#include <condition_variable>
#include <cstddef>
#include <exception>
#include <memory>
#include <mutex>
#include <ostream>
#include <stdexcept>
#include <streambuf>
#include <string>
#include <thread>
#include <utility>
#include <vector>

namespace bdsg {
namespace internal {

/// Run numbered blocks with a fixed worker bound. The first exception cancels
/// outstanding work, all workers are joined, and then the exception is
/// rethrown in the caller.
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
                    {
                        std::lock_guard<std::mutex> lock(error_mutex);
                        if (!error) {
                            error = std::current_exception();
                        }
                    }
                    cancelled.store(true, std::memory_order_relaxed);
                }
            });
        }
    } catch (...) {
        const std::exception_ptr creation_error = std::current_exception();
        cancelled.store(true, std::memory_order_relaxed);
        for (auto& thread : threads) {
            thread.join();
        }
        std::rethrow_exception(creation_error);
    }
    for (auto& thread : threads) {
        thread.join();
    }
    if (error) {
        std::rethrow_exception(error);
    }
}

namespace detail {

struct OrderedBlockState {
    std::mutex mutex;
    std::condition_variable changed;
    std::string chunk;
    bool chunk_ready = false;
    bool done = false;
    std::size_t owner = 0;
};

/// Stream buffer that hands fixed-size strings to one block queue. A queue has
/// capacity one, so later blocks cannot consume unbounded memory while the
/// writer waits for an earlier block.
class OrderedChunkBuffer : public std::streambuf {
public:
    OrderedChunkBuffer(OrderedBlockState& state,
                       std::atomic<bool>& cancelled,
                       std::size_t chunk_bytes) :
        state(state), cancelled(cancelled), chunk_bytes(std::max<std::size_t>(1, chunk_bytes)) {
        buffer.reserve(this->chunk_bytes);
    }

    void finish() {
        flush();
    }

protected:
    int_type overflow(int_type character) override {
        if (traits_type::eq_int_type(character, traits_type::eof())) {
            return traits_type::not_eof(character);
        }
        const char value = traits_type::to_char_type(character);
        append(&value, 1);
        return cancelled.load(std::memory_order_relaxed) ? traits_type::eof() : character;
    }

    std::streamsize xsputn(const char* data, std::streamsize count) override {
        if (count <= 0) {
            return 0;
        }
        const std::size_t requested = static_cast<std::size_t>(count);
        std::size_t written = 0;
        while (written < requested && !cancelled.load(std::memory_order_relaxed)) {
            const std::size_t available = chunk_bytes - buffer.size();
            const std::size_t take = std::min(available, requested - written);
            buffer.append(data + written, take);
            written += take;
            if (buffer.size() == chunk_bytes) {
                flush();
            }
        }
        return static_cast<std::streamsize>(written);
    }

    int sync() override {
        flush();
        return cancelled.load(std::memory_order_relaxed) ? -1 : 0;
    }

private:
    void append(const char* data, std::size_t count) {
        (void) xsputn(data, static_cast<std::streamsize>(count));
    }

    void flush() {
        if (buffer.empty()) {
            return;
        }
        std::unique_lock<std::mutex> lock(state.mutex);
        state.changed.wait(lock, [&]() {
            return cancelled.load(std::memory_order_relaxed) || !state.chunk_ready;
        });
        if (cancelled.load(std::memory_order_relaxed)) {
            throw std::runtime_error("Ordered output cancelled");
        }
        state.chunk = std::move(buffer);
        state.chunk_ready = true;
        lock.unlock();
        state.changed.notify_all();
        buffer.clear();
        buffer.reserve(chunk_bytes);
    }

    OrderedBlockState& state;
    std::atomic<bool>& cancelled;
    std::size_t chunk_bytes;
    std::string buffer;
};

} // namespace detail

/**
 * Run byte-producing blocks concurrently and copy their output to `out` in
 * ascending block order. `produce(block, ostream)` must write one complete
 * block. `before(block, out)` and `after(block, out)` run on the writer thread
 * and allow small boundary records to be interleaved without buffering them in
 * workers.
 *
 * Retained payload is bounded by one queued chunk per block, one filling chunk
 * per worker, and the writer's current chunk. Any producer or writer failure
 * cancels waiters and joins every worker before it is rethrown.
 */
template<class Produce, class Before, class After>
void bounded_ordered_output(std::ostream& out,
                            std::size_t blocks,
                            std::size_t workers,
                            std::size_t chunk_bytes,
                            Produce&& produce,
                            Before&& before,
                            After&& after) {
    if (blocks == 0) {
        return;
    }
    workers = std::max<std::size_t>(1, std::min(workers, blocks));
    chunk_bytes = std::max<std::size_t>(1, chunk_bytes);

    std::vector<std::unique_ptr<detail::OrderedBlockState>> states;
    // Reuse a small ring of slots. The owner is an absolute block number,
    // preventing a producer from overwriting an earlier use of the same slot.
    const std::size_t slots = std::min(blocks, workers <= blocks / 2 ? workers * 2 : blocks);
    states.reserve(slots);
    for (std::size_t i = 0; i < slots; ++i) {
        states.emplace_back(new detail::OrderedBlockState());
        states.back()->owner = i;
    }

    std::atomic<std::size_t> next{0};
    std::atomic<bool> cancelled{false};
    std::mutex error_mutex;
    std::exception_ptr error;
    auto cancel = [&](std::exception_ptr incoming) {
        {
            std::lock_guard<std::mutex> lock(error_mutex);
            if (!error) {
                error = incoming;
            }
        }
        cancelled.store(true, std::memory_order_relaxed);
        for (auto& state : states) {
            // Pair cancellation with the predicate's mutex. Otherwise a waiter
            // can observe false, miss an unlocked notification, and sleep
            // forever even though the atomic subsequently became true.
            std::lock_guard<std::mutex> lock(state->mutex);
            state->changed.notify_all();
        }
    };

    std::vector<std::thread> threads;
    threads.reserve(workers);
    try {
        for (std::size_t worker = 0; worker < workers; ++worker) {
            threads.emplace_back([&]() {
                while (!cancelled.load(std::memory_order_relaxed)) {
                    const std::size_t block = next.fetch_add(1, std::memory_order_relaxed);
                    if (block >= blocks) {
                        break;
                    }
                    auto& state = *states[block % slots];
                    try {
                        {
                            std::unique_lock<std::mutex> lock(state.mutex);
                            state.changed.wait(lock, [&]() {
                                return cancelled.load(std::memory_order_relaxed) || state.owner == block;
                            });
                            if (cancelled.load(std::memory_order_relaxed)) { break; }
                        }
                        detail::OrderedChunkBuffer buffer(state, cancelled, chunk_bytes);
                        std::ostream block_out(&buffer);
                        block_out.exceptions(std::ios::badbit | std::ios::failbit);
                        produce(block, block_out);
                        buffer.finish();
                        {
                            std::lock_guard<std::mutex> lock(state.mutex);
                            state.done = true;
                        }
                        state.changed.notify_all();
                    } catch (...) {
                        cancel(std::current_exception());
                        break;
                    }
                }
            });
        }
    } catch (...) {
        const std::exception_ptr creation_error = std::current_exception();
        cancel(creation_error);
        for (auto& thread : threads) {
            thread.join();
        }
        std::rethrow_exception(creation_error);
    }

    try {
        for (std::size_t block = 0; block < blocks; ++block) {
            before(block, out);
            auto& state = *states[block % slots];
            while (true) {
                std::string chunk;
                bool done = false;
                {
                    std::unique_lock<std::mutex> lock(state.mutex);
                    state.changed.wait(lock, [&]() {
                        return cancelled.load(std::memory_order_relaxed) ||
                               (state.owner == block && (state.chunk_ready || state.done));
                    });
                    if (state.chunk_ready) {
                        chunk = std::move(state.chunk);
                        state.chunk.clear();
                        state.chunk_ready = false;
                    }
                    done = state.done && !state.chunk_ready;
                }
                state.changed.notify_all();
                if (!chunk.empty()) {
                    out.write(chunk.data(), static_cast<std::streamsize>(chunk.size()));
                    if (!out) {
                        throw std::runtime_error("Error writing ordered output block");
                    }
                }
                if (done) {
                    break;
                }
                if (cancelled.load(std::memory_order_relaxed)) {
                    std::lock_guard<std::mutex> lock(error_mutex);
                    if (error) {
                        std::rethrow_exception(error);
                    }
                    throw std::runtime_error("Ordered output cancelled");
                }
            }
            after(block, out);
            if (!out) {
                throw std::runtime_error("Error writing ordered output boundary");
            }
            {
                std::lock_guard<std::mutex> lock(state.mutex);
                state.done = false;
                // There cannot be another producer for this slot until this
                // release. The final chunk has already been written above.
                state.owner = slots < blocks - block ? block + slots : blocks;
            }
            state.changed.notify_all();
        }
    } catch (...) {
        cancel(std::current_exception());
    }

    for (auto& thread : threads) {
        thread.join();
    }
    if (error) {
        std::rethrow_exception(error);
    }
}

template<class Produce>
void bounded_ordered_output(std::ostream& out,
                            std::size_t blocks,
                            std::size_t workers,
                            std::size_t chunk_bytes,
                            Produce&& produce) {
    bounded_ordered_output(out, blocks, workers, chunk_bytes,
                           std::forward<Produce>(produce),
                           [](std::size_t, std::ostream&) {},
                           [](std::size_t, std::ostream&) {});
}

} // namespace internal
} // namespace bdsg

#endif
