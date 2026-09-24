//
//  packed_graph.cpp
//

#include "bdsg/packed_graph.hpp"

#include <handlegraph/util.hpp>
#include <algorithm>
#include <atomic>
#include <exception>

namespace bdsg {

    using namespace handlegraph;
    
    BasePackedGraph<>* PackedGraph::get() {
        return &implementation;
    }
    
    const BasePackedGraph<>* PackedGraph::get() const {
        return &implementation;
    }
    
    BasePackedGraph<MappedBackend>* MappedPackedGraph::get() {
        return implementation.get();
    }
    
    const BasePackedGraph<MappedBackend>* MappedPackedGraph::get() const {
        return implementation.get();
    }
    
    MappedPackedGraph::MappedPackedGraph() {
        // Make sure our implementation pointer is never null.
        implementation.construct(get_prefix());
    }

    MappedPackedGraph::MappedPackedGraph(int backing_fd, size_t initial_arena_bytes) {
        if (backing_fd <= 0) {
            throw std::invalid_argument("A file-backed MappedPackedGraph requires a positive file descriptor");
        }
        implementation.construct_in_fd(backing_fd, get_prefix(), initial_arena_bytes);
    }
    
    // Delegate copy and move to the copy and move constructors for the things that exist in mapped memory.
    
    MappedPackedGraph::MappedPackedGraph(const MappedPackedGraph& other) {
        implementation.construct(get_prefix(), *other.get());
    }
    
    MappedPackedGraph& MappedPackedGraph::operator=(const MappedPackedGraph& other) {
        if (get() != other.get()) {
            implementation.construct(get_prefix(), *other.get());
        }
        return *this;
    }
    
    MappedPackedGraph::MappedPackedGraph(MappedPackedGraph&& other) {
        implementation.construct(get_prefix(), std::move(*other.get()));
    }
    
    MappedPackedGraph& MappedPackedGraph::operator=(MappedPackedGraph&& other) {
        if (get() != other.get()) {
            implementation.construct(get_prefix(), std::move(*other.get()));
        }
        return *this;
    }
    
    void MappedPackedGraph::dissociate() {
        implementation.dissociate();
    }

    void MappedPackedGraph::checkpoint_and_evict() const {
        implementation.checkpoint_and_evict();
    }

    void MappedPackedGraph::serialize_packed_graph(
        std::ostream& out,
        size_t checkpoint_interval_bytes) const {

        if (checkpoint_interval_bytes < sizeof(uint64_t)) {
            throw std::invalid_argument("PackedGraph checkpoint interval must be at least 8 bytes");
        }

        // Begin from a durable, nonresident arena. This also rejects an
        // anonymous graph before emitting a partial output stream.
        checkpoint_and_evict();

        constexpr size_t MAX_WRITE_CHUNK = 1024 * 1024;
        size_t max_chunk_bytes = std::min(checkpoint_interval_bytes, MAX_WRITE_CHUNK);
        max_chunk_bytes -= max_chunk_bytes % sizeof(uint64_t);

        size_t bytes_since_checkpoint = 0;
        auto progress = [&](size_t bytes_written) {
            if (bytes_written >= checkpoint_interval_bytes -
                                 std::min(bytes_since_checkpoint, checkpoint_interval_bytes)) {
                checkpoint_and_evict();
                bytes_since_checkpoint = 0;
            } else {
                bytes_since_checkpoint += bytes_written;
            }
        };

        get()->serialize_standard(out, max_chunk_bytes, progress);
        checkpoint_and_evict();
    }

    void MappedPackedGraph::deserialize_packed_graph(
        std::istream& in,
        size_t checkpoint_interval_bytes) {

        if (checkpoint_interval_bytes < sizeof(uint64_t)) {
            throw std::invalid_argument("PackedGraph checkpoint interval must be at least 8 bytes");
        }

        // Reject anonymous storage before reading any input. BasePackedGraph
        // performs the stronger member-level fresh-graph check.
        checkpoint_and_evict();

        constexpr size_t MAX_READ_CHUNK = 1024 * 1024;
        size_t max_chunk_bytes = std::min(checkpoint_interval_bytes, MAX_READ_CHUNK);
        max_chunk_bytes -= max_chunk_bytes % sizeof(uint64_t);

        size_t bytes_since_checkpoint = 0;
        auto progress = [&](size_t bytes_read) {
            const size_t remaining = checkpoint_interval_bytes - bytes_since_checkpoint;
            if (bytes_read >= remaining) {
                checkpoint_and_evict();
                bytes_since_checkpoint = 0;
            } else {
                bytes_since_checkpoint += bytes_read;
            }
        };

        try {
            get()->deserialize_standard(in, max_chunk_bytes, progress);
            checkpoint_and_evict();
        } catch (...) {
            // Keep the first parsing or checkpoint failure, but make a best
            // effort to leave every completed input chunk durable for diagnosis.
            std::exception_ptr original = std::current_exception();
            try {
                checkpoint_and_evict();
            } catch (...) {
                // The original failure remains the actionable one.
            }
            std::rethrow_exception(original);
        }
    }
    
    void MappedPackedGraph::serialize(const std::function<void(const void*, size_t)>& iteratee) const {
        // Pass the same iteratee back to the implementation pointer.
        implementation.save(iteratee);
    }
    
    void MappedPackedGraph::serialize(int fd) {
        implementation.save(fd);
    }
    
    void MappedPackedGraph::deserialize(int fd) {
        implementation.load(fd, get_prefix());
    }
    
    void MappedPackedGraph::serialize_members(std::ostream& out) const {
        // libhandlegraph already wrote our magic number prefix.
        implementation.save_after_prefix(out, get_prefix());
    }
    
    void MappedPackedGraph::deserialize_members(std::istream& in) {
        // libhandlegraph stole our magic number, and checked it.
        implementation.load_after_prefix(in, get_prefix());
    }
    
    uint32_t MappedPackedGraph::get_magic_number() const {
        // Chosen by fair dice roll, guaranteed to be magic.
        return 672226447;
    }
    
    std::string MappedPackedGraph::get_prefix() const {
        // Put into network byte order
        uint32_t magic_number = htonl(get_magic_number());
        // Then convert to a string, bounding length because it is not null terminated.
        return std::string((char*) &magic_number, sizeof(magic_number) / sizeof(char));
    }
    
   
}
