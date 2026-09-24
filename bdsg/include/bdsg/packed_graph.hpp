//
//  packed_graph.hpp
//  
//  Contains a implementation of a sequence graph based on bit-packed integer
//  vectors.
//

#ifndef BDSG_PACKED_GRAPH_HPP_INCLUDED
#define BDSG_PACKED_GRAPH_HPP_INCLUDED

#include <utility>

#include <handlegraph/trivially_serializable.hpp>

#include "bdsg/internal/base_packed_graph.hpp"
#include "bdsg/graph_proxy.hpp"


namespace bdsg {
    
using namespace std;
using namespace handlegraph;

/*
 * In-memory implementation of MutablePathDeletableHandleGraph
 */
class PackedGraph : public GraphProxy<BasePackedGraph<>> {
public:
    /// Stream a serial batch of existing, distinct edges to the packed backing
    /// implementation and defer its defragmentation check until the batch ends.
    template<typename EdgeProducer>
    void destroy_edges_bulk(EdgeProducer&& produce_edges) {
        get()->destroy_edges_bulk(std::forward<EdgeProducer>(produce_edges));
    }

    bool can_serialize_with_paths() const {
        return get()->can_serialize_with_paths();
    }

    /// Serialize replayable walks into ordinary PackedGraph path storage.
    template<class PathName, class PathSize, class PathSteps>
    void serialize_with_paths(std::ostream& out, size_t path_count,
                              const PathName& path_name, const PathSize& path_size,
                              const PathSteps& path_steps) const {
        get()->serialize_with_paths(out, path_count, path_name, path_size, path_steps);
    }

    /// Parallel bounded-memory form. The five-argument overload above remains
    /// the exact serial implementation.
    template<class PathName, class PathSize, class PathSteps>
    void serialize_with_paths(
        std::ostream& out, size_t path_count,
        const PathName& path_name, const PathSize& path_size,
        const PathSteps& path_steps, size_t workers,
        size_t extra_memory_budget,
        const PathSerializationStageCallback& stage = {}) const {
        get()->serialize_with_paths(out, path_count, path_name, path_size,
                                    path_steps, workers, extra_memory_budget, stage);
    }

protected:
    /**
     * Get the object that actually provides the graph methods.
     */
    BasePackedGraph<>* get();
    
    /**
     * Get the object that actually provides the graph methods.
     */
    const BasePackedGraph<>* get() const;
    
    /**
     * We just directly contain the BasePackedGraph using the default
     * in-normal-memory backend data structures.
     */
    BasePackedGraph<> implementation;
};

/*
 * Memory-mapped implementation of MutablePathDeletableHandleGraph
 */
class MappedPackedGraph : public GraphProxy<BasePackedGraph<MappedBackend>>, public TriviallySerializable {
public:

    /// Stream a serial batch of existing, distinct edges to the mapped packed
    /// implementation and defer its defragmentation check until the batch ends.
    template<typename EdgeProducer>
    void destroy_edges_bulk(EdgeProducer&& produce_edges) {
        get()->destroy_edges_bulk(std::forward<EdgeProducer>(produce_edges));
    }

    // We need constructors, destructors, copy, and move because we are keeping
    // the graph we are proxying for in mapped memory.

    MappedPackedGraph();

    /**
     * Construct an empty graph directly associated with a writable file
     * descriptor. The destination file is truncated and becomes the graph's
     * write-back arena.
     */
    explicit MappedPackedGraph(int backing_fd, size_t initial_arena_bytes = 0);

    ~MappedPackedGraph() = default;
    
    MappedPackedGraph(const MappedPackedGraph& other);
    MappedPackedGraph& operator=(const MappedPackedGraph& other);
    
    MappedPackedGraph(MappedPackedGraph&& other);
    MappedPackedGraph& operator=(MappedPackedGraph&& other);
    
    // We need to say that TriviallySerializable's serialize and deserialize
    // should still be available.
    using TriviallySerializable::serialize;
    using TriviallySerializable::deserialize;
   
    /**
     * Cut the memory mapping connection to any backing file.
     */
    void dissociate();

    /**
     * Durably synchronize the file-backed arena and advise the kernel to
     * release its resident pages. Throws for an anonymous graph.
     */
    void checkpoint_and_evict() const;

    /**
     * Emit byte-identical ordinary PackedGraph serialization while bounding
     * mapped input residency with periodic checkpoint-and-evict operations.
     * The graph must be file-backed and quiescent for mutation.
     */
    void serialize_packed_graph(std::ostream& out,
                                size_t checkpoint_interval_bytes = 64ull * 1024 * 1024) const;

    /**
     * Replace an empty, file-backed graph from the ordinary PackedGraph wire
     * format. Packed payloads are copied into the mapped arena in bounded
     * chunks, with durable page eviction at the requested interval.
     */
    void deserialize_packed_graph(
        std::istream& in,
        size_t checkpoint_interval_bytes = 256ull * 1024 * 1024);
    
    /**
     * Serialize us as a series of in-memory blocks shown to the given finction.
     * Backs const serialization to FDs, and serialization to streams.
     */
    void serialize(const std::function<void(const void*, size_t)>& iteratee) const;
    
    /**
     * Serialize us to the given file descriptor and establish a write-back
     * link.
     */
    void serialize(int fd);
    
    /**
     * Deserialize us from the given file descriptor.
     */
    void deserialize(int fd);
    
    // We aren't going to override serialize() and deserialize() for streams,
    // because TriviallySerializable has a nice implementation for them, so we
    // still need to implement serialization and reading of everything past the
    // magic number.
    
    /**
     * Serialize everything except the magic number to the given stream.
     */
    void serialize_members(std::ostream& out) const;
    
    /**
     * Deserialize everything except the magic number from the given stream.
     */
    void deserialize_members(std::istream& in);
    
   
protected:

    /**
     * Return the magic number to use at the start of files.
     *
     * Different than the magic number that BasePackedGraph<> uses, because we
     * are really storing a memory-mapped actual BasePackedGraph<> and not just
     * a packed-up description.
     */
    uint32_t get_magic_number() const;
    
    /**
     * Return the magic number as a string representing the bytes it will be
     * represented by on disk.
     */
    std::string get_prefix() const;
    
    /**
     * Get the object that actually provides the graph methods.
     */
    BasePackedGraph<MappedBackend>* get();
    
    /**
     * Get the object that actually provides the graph methods.
     */
    const BasePackedGraph<MappedBackend>* get() const;
    
    /**
     * We have a pointer to the memory-mapped graph that actually implements
     * everything.
     */
    yomo::UniqueMappedPointer<BasePackedGraph<MappedBackend>> implementation;
};

}

#endif
