//
// Merge distance indexes built on disjoint graphs into one index.
//
// An index is one packed integer vector: a root record, one pointer per
// connected component, a table of two entries per node ID, and then the
// records of the snarl tree. No record refers to a record of another
// connected component, so indexes of disjoint graphs can be concatenated as
// long as every stored record offset, node ID and root-level component number
// is rewritten. Distances are copied, never recomputed, so no graph is needed.
//

#include "bdsg/snarl_distance_index.hpp"

#include <algorithm>
#include <cerrno>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <fcntl.h>
#include <fstream>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

namespace bdsg {

using namespace std;
using namespace handlegraph;

namespace {

// Entries are read and written through the vector's packed word array.
// MappedIntVector::at() resolves the vector's yomo::Pointer on every access,
// and on a read-only mapping that takes the Manager's lock each time, while a
// merge touches every entry of every part.
struct PackedWords : public MappedIntVector {
    static const uint64_t* of(const MappedIntVector& vector) {
        return (vector.*(&PackedWords::data)).get_first();
    }
    static uint64_t* of(MappedIntVector& vector) {
        return (vector.*(&PackedWords::data)).get_first();
    }
};

size_t bits_needed(uint64_t value) {
    return value == 0 ? 0 : 64 - __builtin_clzll(value);
}

/// One bit for each position in a range.
class PositionBits {
public:
    explicit PositionBits(size_t size) : bits((size + 63) / 64, 0) {}

    bool get(size_t i) const {
        return (bits[i >> 6] >> (i & 63)) & 1;
    }

    /// Set the bit for i and say whether it was already set.
    bool test_and_set(size_t i) {
        uint64_t mask = uint64_t(1) << (i & 63);
        bool was_set = bits[i >> 6] & mask;
        bits[i >> 6] |= mask;
        return was_set;
    }

    /// Set the bits for [from, from + count) and say whether any was already set.
    bool test_and_set(size_t from, size_t count) {
        bool any_set = false;
        size_t end = from + count;
        while (from < end) {
            size_t low = from & 63;
            size_t n = std::min<size_t>(64 - low, end - from);
            uint64_t mask = (n == 64 ? ~uint64_t(0) : (uint64_t(1) << n) - 1) << low;
            any_set |= (bits[from >> 6] & mask) != 0;
            bits[from >> 6] |= mask;
            from += n;
        }
        return any_set;
    }

    /// The whole 64-bit word of bits holding position i.
    uint64_t word_at(size_t i) const {
        return bits[i >> 6];
    }

private:
    vector<uint64_t> bits;
};

// Record type names in record_t order; the class's own table is in another order.
const char* const RECORD_TYPE_NAMES[] = {"?", "ROOT", "NODE", "DISTANCED_NODE", "TRIVIAL_SNARL",
    "DISTANCED_TRIVIAL_SNARL", "SIMPLE_SNARL", "DISTANCED_SIMPLE_SNARL", "SNARL", "DISTANCED_SNARL",
    "OVERSIZED_SNARL", "ROOT_SNARL", "DISTANCED_ROOT_SNARL", "CHAIN", "DISTANCED_CHAIN",
    "MULTICOMPONENT_CHAIN", "CHILDREN"};

string duration(double seconds) {
    long whole = (long) seconds;
    char buffer[32];
    snprintf(buffer, sizeof(buffer), "%ld:%02ld:%02ld", whole / 3600, (whole / 60) % 60, whole % 60);
    return buffer;
}

/// Drop the pages of [start, start + bytes) from this process. The range is
/// widened to whole pages, which stay inside the mapping that holds it.
void release_pages(const void* start, size_t bytes, const string& what) {
    static const size_t page_size = sysconf(_SC_PAGESIZE);
    uintptr_t from = (uintptr_t) start & ~(uintptr_t) (page_size - 1);
    uintptr_t to = ((uintptr_t) start + bytes + page_size - 1) & ~(uintptr_t) (page_size - 1);
    if (madvise((void*) from, to - from, MADV_DONTNEED) != 0) {
        throw runtime_error(what + ": could not release mapped pages: " + strerror(errno));
    }
}

}

class SnarlDistanceIndex::IndexMerger {
public:
    IndexMerger(const string& output_path, const vector<pair<string, nid_t>>& inputs, ostream* log);
    void run();

private:

    /// Entries between releases of mapped pages, so that neither the parts
    /// nor the output stay resident as they are read and written.
    static constexpr size_t RELEASE_INTERVAL = size_t(1) << 27;
    /// Record visits between releases of a part's pages while it is walked.
    static constexpr size_t VISIT_RELEASE_INTERVAL = size_t(1) << 20;

    /// What the header pass learns about a part, and where it goes.
    struct Part {
        string path;
        nid_t id_offset = 0;
        uint64_t root_tag = 0;
        uint64_t version_entry = 0;
        size_t components = 0;
        size_t node_slots = 0;
        nid_t min_id = 0;
        nid_t max_id = 0;
        size_t max_depth = 0;
        size_t entries = 0;
        size_t width = 0;
        /// The first entry after the part's root record and tables.
        size_t records_start = 0;
        /// Added to every record offset of the part.
        size_t offset_shift = 0;
        /// Added to every connected component number of the part.
        size_t component_base = 0;
    };

    /// Files smaller than this are read into memory rather than mapped. The
    /// memory mapper only maps a file that can hold its first 1 KiB link and
    /// otherwise grows it, which a read-only descriptor refuses.
    static constexpr size_t MAP_MIN_BYTES = size_t(1) << 20;

    /// A part's index, opened read-only.
    class Source {
    public:
        explicit Source(const string& path);
        uint64_t at(size_t i) const {
            size_t bit = i * width;
            return sdsl::bits::read_int(words + (bit >> 6), bit & 63, width);
        }
        void release() const {
            // Only pages of a file mapping can be dropped and read back.
            if (mapped) {
                release_pages(words, ((size * width + 63) / 64) * sizeof(uint64_t), path);
            }
        }
        string path;
        SnarlDistanceIndex index;
        bool mapped = false;
        const uint64_t* words = nullptr;
        size_t width = 0;
        size_t size = 0;
    };

    /// Bytes of the output written back at a time. The kernel pauses a
    /// process whose dirty pages pass its share of the dirty limit, and under
    /// a memory cgroup that share can be a few megabytes, so the output never
    /// holds more than a few windows of unwritten pages.
    static constexpr size_t FLUSH_BYTES = size_t(1) << 20;

    /// The merged index, written mostly in order into a file-backed mapping
    /// that is extended and written back a window at a time.
    class Output {
    public:
        Output(int fd, const string& path, const string& prefix, size_t entries, size_t width);
        void put(size_t i, uint64_t value) {
            if (i >= ready) {
                extend_to(i + 1);
            }
            if (width < 64 && (value >> width) != 0) {
                throw runtime_error(path + ": value " + to_string(value) + " for entry " + to_string(i)
                                    + " does not fit in " + to_string(width) + " bits");
            }
            if (i < dirty_begin || i > dirty_end) {
                flush();
                dirty_begin = dirty_end = i;
            }
            size_t bit = i * width;
            sdsl::bits::write_int(words + (bit >> 6), value, bit & 63, width);
            dirty_end = std::max(dirty_end, i + 1);
            if (dirty_end - dirty_begin >= flush_entries) {
                flush();
            }
            if (++since_release == RELEASE_INTERVAL) {
                release();
            }
        }
        void extend_to(size_t count);
        void release();
        void finish();
        /// Also released along with the output.
        const Source* source = nullptr;
    private:
        void flush();
        void wait_for_writeback();
        string path;
        int fd;
        bdsg::yomo::UniqueMappedPointer<bdsg::MappedIntVector> records;
        uint64_t* words = nullptr;
        /// Where the packed entries start in the file.
        size_t words_offset = 0;
        size_t entries = 0;
        size_t width = 0;
        size_t flush_entries = 0;
        size_t ready = 0;
        /// Entries written since they were last sent to be written back.
        size_t dirty_begin = 0;
        size_t dirty_end = 0;
        /// File bytes sent to be written back and not yet waited for.
        size_t writing_offset = 0;
        size_t writing_bytes = 0;
        size_t since_release = 0;
    };

    /// What was found while walking one part.
    struct PartWalk {
        explicit PartWalk(size_t region) : record_starts(region), covered(region), child_list_entries(region) {}
        PositionBits record_starts;
        PositionBits covered;
        PositionBits child_list_entries;
        vector<size_t> pending;
        size_t visits = 0;
        size_t records_by_type[CHILDREN + 1] = {};
        size_t snarls_without_child_list = 0;
        size_t uncovered_zero_entries = 0;
    };

    static bool is_node(record_t type) {return type == NODE || type == DISTANCED_NODE;}
    static bool is_chain(record_t type) {return type == CHAIN || type == DISTANCED_CHAIN || type == MULTICOMPONENT_CHAIN;}
    static bool is_trivial_snarl(record_t type) {return type == TRIVIAL_SNARL || type == DISTANCED_TRIVIAL_SNARL;}
    static bool is_simple_snarl(record_t type) {return type == SIMPLE_SNARL || type == DISTANCED_SIMPLE_SNARL;}
    static bool is_chain_snarl(record_t type) {return type == SNARL || type == DISTANCED_SNARL || type == OVERSIZED_SNARL;}
    static bool is_root_snarl(record_t type) {return type == ROOT_SNARL || type == DISTANCED_ROOT_SNARL;}
    static size_t parent_field(record_t type);

    void read_headers();
    void lay_out();
    void check_output_path() const;
    void merge_part(Part& merging, Output& output);

    // Walking a part: every record reachable from the component table, the
    // snarls' child lists and the node table, checking the layout as it goes.
    record_t type_at(size_t i) const;
    uint64_t field(size_t record, size_t offset) const;
    size_t record_size(size_t record, record_t type) const;
    void check_offset(size_t offset, const string& what) const;
    void claim(size_t from, size_t count, size_t record);
    void walk_part();
    void descend(size_t record);
    void visit(size_t record);
    void walk_chain(size_t chain);
    void walk_child_list(size_t snarl);
    void climb(nid_t id, size_t record, size_t node_offset);
    void check_coverage();
    void count_visit(record_t type);

    // Writing a part: its tables, then its record region in order.
    void write_part(Output& output);
    size_t write_record(Output& output, size_t record);
    uint64_t relocate(uint64_t offset) const;
    uint64_t shift_id(uint64_t id) const;
    uint64_t shift_oriented_id(uint64_t oriented_id) const;

    [[noreturn]] void fail(const string& message) const;
    void note(const string& message) const;

    string output_path;
    ostream* log;
    vector<Part> parts;
    chrono::steady_clock::time_point start_time;

    uint64_t root_tag = 0;
    uint64_t version_entry = 0;
    size_t total_components = 0;
    nid_t min_id = 0;
    nid_t max_id = 0;
    size_t node_slots = 0;
    size_t max_depth = 0;
    size_t header_entries = 0;
    size_t total_entries = 0;
    size_t width = 0;

    // The part being merged.
    const Part* part = nullptr;
    const Source* source = nullptr;
    PartWalk* walk = nullptr;
};

SnarlDistanceIndex::IndexMerger::Source::Source(const string& path) : path(path) {
    int fd = ::open(path.c_str(), O_RDONLY);
    if (fd < 0) {
        throw runtime_error(path + ": could not open: " + strerror(errno));
    }
    struct stat info;
    if (fstat(fd, &info) != 0) {
        int error = errno;
        ::close(fd);
        throw runtime_error(path + ": could not stat: " + strerror(error));
    }
    // The stream loader only warns about a wrong magic number and reads on.
    string prefix = index.get_prefix();
    string found(prefix.size(), '\0');
    if (pread(fd, &found[0], found.size(), 0) != (ssize_t) found.size() || found != prefix) {
        ::close(fd);
        throw runtime_error(path + ": is not a distance index");
    }
    try {
        if ((size_t) info.st_size >= MAP_MIN_BYTES) {
            // A read-only descriptor gets a read-only mapping, which cannot write
            // back to the file the way a writable one sets a flag in it.
            index.deserialize(fd);
            mapped = true;
        } else {
            ifstream in(path, ios::binary);
            if (!in) {
                throw runtime_error(strerror(errno));
            }
            index.deserialize(in);
        }
    } catch (const exception& e) {
        ::close(fd);
        throw runtime_error(path + ": could not load as a distance index: " + e.what());
    }
    // A mapping holds its own duplicate of the descriptor.
    ::close(fd);
    width = index.snarl_tree_records->width();
    size = index.snarl_tree_records->size();
    words = PackedWords::of(*index.snarl_tree_records);
}

SnarlDistanceIndex::IndexMerger::Output::Output(int fd, const string& path, const string& prefix, size_t entries,
                                                size_t width) :
    path(path), fd(fd), entries(entries), width(width), flush_entries(std::max<size_t>(64, FLUSH_BYTES * 8 / width)) {
    size_t data_bytes = ((entries * width + 63) / 64) * sizeof(uint64_t);
    // The mapping starts large enough for the allocator's own records and the
    // whole vector, so it is one link and the vector never moves or doubles;
    // the unused tail is cut off the file when the mapping is torn down, which
    // leaves no spare capacity to be saved, as the fitted save path requires.
    records.construct_in_fd(fd, prefix, data_bytes + (size_t(1) << 16));
    records->width(width);
    records->reserve(entries);
    if (records->capacity() < entries) {
        throw runtime_error(path + ": could not reserve " + to_string(entries) + " entries");
    }
    words = PackedWords::of(*records);
    // A file-backed chain maps the file from its start, so chain positions are file offsets.
    words_offset = bdsg::yomo::Manager::get_chain_and_position(words).second;
}

void SnarlDistanceIndex::IndexMerger::Output::extend_to(size_t count) {
    if (count > entries) {
        throw runtime_error(path + ": entry " + to_string(count - 1) + " is past the planned "
                            + to_string(entries) + " entries");
    }
    // Growing the vector writes zeroes into the new entries, dirtying their
    // pages, so it runs only a window ahead of the writes, which dirty the
    // same pages again before they are written back.
    while (ready < count) {
        size_t zeroed = ready;
        ready = std::min(entries, ready + flush_entries);
        records->resize(ready);
        if (ready < count) {
            // Nothing is about to be written here, so write the zeroes back now.
            flush();
            dirty_begin = zeroed;
            dirty_end = ready;
            flush();
        }
    }
}

void SnarlDistanceIndex::IndexMerger::Output::flush() {
    if (dirty_end == dirty_begin) {
        return;
    }
    static const size_t page_size = sysconf(_SC_PAGESIZE);
    size_t begin = words_offset + ((dirty_begin * width) / 64) * sizeof(uint64_t);
    size_t end = words_offset + ((dirty_end * width + 63) / 64) * sizeof(uint64_t);
    begin -= begin % page_size;
    end = ((end + page_size - 1) / page_size) * page_size;
#ifdef __linux__
    // Start this window's writeback, then wait for the previous one, so the
    // disk works while the next window is filled.
    if (sync_file_range(fd, begin, end - begin, SYNC_FILE_RANGE_WRITE) != 0) {
        throw runtime_error(path + ": could not start writing back: " + strerror(errno));
    }
    wait_for_writeback();
    writing_offset = begin;
    writing_bytes = end - begin;
#else
    char* mapped = (char*) words - words_offset;
    if (msync(mapped + begin, end - begin, MS_SYNC) != 0) {
        throw runtime_error(path + ": could not write back: " + strerror(errno));
    }
#endif
    dirty_begin = dirty_end;
}

void SnarlDistanceIndex::IndexMerger::Output::wait_for_writeback() {
#ifdef __linux__
    if (writing_bytes != 0 && sync_file_range(fd, writing_offset, writing_bytes, SYNC_FILE_RANGE_WAIT_BEFORE
                                              | SYNC_FILE_RANGE_WRITE | SYNC_FILE_RANGE_WAIT_AFTER) != 0) {
        throw runtime_error(path + ": could not write back: " + strerror(errno));
    }
    writing_bytes = 0;
#endif
}

void SnarlDistanceIndex::IndexMerger::Output::release() {
    flush();
    wait_for_writeback();
    records.checkpoint_and_evict();
    if (source != nullptr) {
        source->release();
    }
    since_release = 0;
}

void SnarlDistanceIndex::IndexMerger::Output::finish() {
    if (ready != entries || records->size() != entries) {
        throw runtime_error(path + ": only " + to_string(ready) + " of " + to_string(entries) + " entries were written");
    }
    flush();
    wait_for_writeback();
    records.checkpoint_and_evict();
    // Tearing the mapping down truncates the free space after the vector.
    records.reset();
}

size_t SnarlDistanceIndex::IndexMerger::parent_field(record_t type) {
    if (is_node(type)) {
        return NODE_PARENT_OFFSET;
    } else if (is_chain(type)) {
        return CHAIN_PARENT_OFFSET;
    } else if (is_trivial_snarl(type)) {
        return TRIVIAL_SNARL_PARENT_OFFSET;
    } else if (is_simple_snarl(type)) {
        return SIMPLE_SNARL_PARENT_OFFSET;
    } else {
        return SNARL_PARENT_OFFSET;
    }
}

SnarlDistanceIndex::IndexMerger::IndexMerger(const string& output_path, const vector<pair<string, nid_t>>& inputs,
                                             ostream* log) :
    output_path(output_path), log(log), start_time(chrono::steady_clock::now()) {
    if (inputs.empty()) {
        throw runtime_error("no distance indexes to merge into " + output_path);
    }
    for (auto& input : inputs) {
        parts.emplace_back();
        parts.back().path = input.first;
        parts.back().id_offset = input.second;
    }
}

void SnarlDistanceIndex::IndexMerger::fail(const string& message) const {
    throw runtime_error((part != nullptr ? part->path + ": " : string()) + message);
}

void SnarlDistanceIndex::IndexMerger::note(const string& message) const {
    if (log != nullptr) {
        double elapsed = chrono::duration<double>(chrono::steady_clock::now() - start_time).count();
        *log << "[merge_indexes " << duration(elapsed) << "] " << message << endl;
    }
}

void SnarlDistanceIndex::IndexMerger::read_headers() {
    for (Part& p : parts) {
        part = &p;
        Source mapped(p.path);
        if (mapped.size < ROOT_RECORD_SIZE) {
            fail("too short to hold a root record");
        }
        p.root_tag = mapped.at(0);
        p.version_entry = mapped.at(VERSION_NUMBER_OFFSET);
        p.components = mapped.at(COMPONENT_COUNT_OFFSET);
        p.node_slots = mapped.at(NODE_COUNT_OFFSET);
        p.min_id = mapped.at(MIN_NODE_ID_OFFSET);
        p.max_depth = mapped.at(MAX_TREE_DEPTH_OFFSET);
        p.entries = mapped.size;
        p.width = mapped.width;
        if (get_record_type(p.root_tag) != ROOT) {
            fail("does not start with a root record");
        }
        // Loading accepts version 3 with a warning, but its records differ.
        if ((VERSION_NUMBER_SENTINEL ^ p.version_entry) != CURRENT_VERSION_NUMBER) {
            fail("is a version " + to_string(VERSION_NUMBER_SENTINEL ^ p.version_entry)
                 + " distance index; only version " + to_string(CURRENT_VERSION_NUMBER) + " can be merged");
        }
        if (p.root_tag != parts.front().root_tag || p.version_entry != parts.front().version_entry) {
            fail("root record tag or version differs from " + parts.front().path);
        }
        if (p.components == 0 || p.node_slots == 0 || p.min_id < 1) {
            fail("has no connected components or no node IDs");
        }
        p.records_start = ROOT_RECORD_SIZE + p.components + 2 * p.node_slots;
        if (p.records_start > p.entries) {
            fail("its tables run past its " + to_string(p.entries) + " entries");
        }
        p.max_id = p.min_id + (nid_t) p.node_slots - 1;
        if ((p.id_offset > 0 && p.max_id > numeric_limits<nid_t>::max() - p.id_offset)
            || p.min_id + p.id_offset < 1) {
            fail("node ID offset " + to_string(p.id_offset) + " moves node IDs " + to_string(p.min_id) + "-"
                 + to_string(p.max_id) + " out of range");
        }
        note("read " + p.path + ": " + to_string(p.components) + " components, node IDs " + to_string(p.min_id)
             + "-" + to_string(p.max_id) + ", " + to_string(p.entries) + " entries at " + to_string(p.width) + " bits");
    }
    part = nullptr;
}

void SnarlDistanceIndex::IndexMerger::lay_out() {
    vector<const Part*> by_id;
    for (const Part& p : parts) {
        by_id.push_back(&p);
    }
    sort(by_id.begin(), by_id.end(), [](const Part* a, const Part* b) {
        return a->min_id + a->id_offset < b->min_id + b->id_offset;
    });
    for (size_t i = 1; i < by_id.size(); i++) {
        if (by_id[i]->min_id + by_id[i]->id_offset <= by_id[i - 1]->max_id + by_id[i - 1]->id_offset) {
            throw runtime_error("node IDs of " + by_id[i - 1]->path + " and " + by_id[i]->path + " overlap after shifting");
        }
    }
    root_tag = parts.front().root_tag;
    version_entry = parts.front().version_entry;
    min_id = by_id.front()->min_id + by_id.front()->id_offset;
    max_id = by_id.back()->max_id + by_id.back()->id_offset;
    node_slots = max_id - min_id + 1;
    size_t max_width = 0;
    for (Part& p : parts) {
        p.component_base = total_components;
        total_components += p.components;
        max_depth = std::max(max_depth, p.max_depth);
        max_width = std::max(max_width, p.width);
    }
    header_entries = ROOT_RECORD_SIZE + total_components + 2 * node_slots;
    total_entries = header_entries;
    for (Part& p : parts) {
        p.offset_shift = total_entries - p.records_start;
        total_entries += p.entries - p.records_start;
    }
    // Net handles keep record offsets in their top 49 bits.
    if (total_entries >= (size_t(1) << (64 - BITS_FOR_TRIVIAL_NODE_OFFSET - 7))) {
        throw runtime_error("a merged index of " + to_string(total_entries) + " entries is too large to address");
    }
    // The parts' widths already hold their distances. Offsets and IDs grow, and
    // get_snarl_tree_records leaves 2 spare bits for the flags packed beside
    // them, so the same rule sizes the merged entries.
    width = std::max(max_width, bits_needed(std::max(total_entries, (size_t) max_id)) + 2);
    if (width > 64) {
        throw runtime_error("a merged index of " + to_string(total_entries) + " entries needs more than 64 bits per entry");
    }
    note("merged index: " + to_string(parts.size()) + " parts, " + to_string(total_components) + " components, node IDs "
         + to_string(min_id) + "-" + to_string(max_id) + ", " + to_string(total_entries) + " entries at " + to_string(width)
         + " bits (widest part " + to_string(max_width) + ")");
}

void SnarlDistanceIndex::IndexMerger::check_output_path() const {
    // The output file is truncated when it is mapped, so it must not be a part.
    for (const string& path : {output_path, output_path + ".incomplete"}) {
        struct stat out_stat;
        if (stat(path.c_str(), &out_stat) != 0) {
            continue;
        }
        for (const Part& p : parts) {
            struct stat in_stat;
            if (stat(p.path.c_str(), &in_stat) == 0 && in_stat.st_dev == out_stat.st_dev && in_stat.st_ino == out_stat.st_ino) {
                throw runtime_error("output " + path + " is the same file as input " + p.path);
            }
        }
    }
}

void SnarlDistanceIndex::IndexMerger::run() {
    read_headers();
    lay_out();
    check_output_path();

    string partial_path = output_path + ".incomplete";
    int fd = ::open(partial_path.c_str(), O_RDWR | O_CREAT | O_TRUNC, 0644);
    if (fd < 0) {
        throw runtime_error(partial_path + ": could not create: " + strerror(errno));
    }
    try {
        {
            Output output(fd, partial_path, SnarlDistanceIndex().get_prefix(), total_entries, width);
            output.extend_to(header_entries);
            output.put(0, root_tag);
            output.put(VERSION_NUMBER_OFFSET, version_entry);
            output.put(COMPONENT_COUNT_OFFSET, total_components);
            output.put(NODE_COUNT_OFFSET, node_slots);
            output.put(MIN_NODE_ID_OFFSET, min_id);
            output.put(MAX_TREE_DEPTH_OFFSET, max_depth);
            for (Part& p : parts) {
                merge_part(p, output);
            }
            output.finish();
        }
        if (fsync(fd) != 0) {
            throw runtime_error(partial_path + ": could not sync: " + strerror(errno));
        }
        if (::close(fd) != 0) {
            fd = -1;
            throw runtime_error(partial_path + ": could not close: " + strerror(errno));
        }
        fd = -1;

        // Reload the finished file the way a mapper would, before it takes the output's name.
        {
            Source merged(partial_path);
            if (merged.size != total_entries || merged.width != width || merged.at(0) != root_tag
                || merged.at(COMPONENT_COUNT_OFFSET) != total_components || merged.at(NODE_COUNT_OFFSET) != node_slots
                || merged.at(MIN_NODE_ID_OFFSET) != (uint64_t) min_id) {
                throw runtime_error(partial_path + ": does not reload with the entries that were written");
            }
        }
        if (rename(partial_path.c_str(), output_path.c_str()) != 0) {
            throw runtime_error(partial_path + ": could not rename to " + output_path + ": " + strerror(errno));
        }
    } catch (...) {
        if (fd >= 0) {
            ::close(fd);
        }
        unlink(partial_path.c_str());
        throw;
    }
    note("wrote " + output_path);
}

void SnarlDistanceIndex::IndexMerger::merge_part(Part& p, Output& output) {
    part = &p;
    Source mapped(p.path);
    source = &mapped;
    if (mapped.size != p.entries || mapped.width != p.width || mapped.at(0) != p.root_tag
        || mapped.at(COMPONENT_COUNT_OFFSET) != p.components || mapped.at(NODE_COUNT_OFFSET) != p.node_slots
        || mapped.at(MIN_NODE_ID_OFFSET) != (uint64_t) p.min_id) {
        fail("changed since its header was read");
    }

    PartWalk part_walk(p.entries - p.records_start);
    walk = &part_walk;
    walk_part();
    mapped.release();
    note("walked " + p.path + ": " + to_string(walk->visits) + " records, "
         + to_string(walk->snarls_without_child_list) + " snarls without child lists, "
         + to_string(walk->uncovered_zero_entries) + " zero entries outside records");
    string census;
    for (size_t type = ROOT; type < CHILDREN; type++) {
        if (walk->records_by_type[type] != 0) {
            census += string(" ") + RECORD_TYPE_NAMES[type] + "=" + to_string(walk->records_by_type[type]);
        }
    }
    note("record types in " + p.path + ":" + census);

    output.source = &mapped;
    write_part(output);
    output.release();
    output.source = nullptr;
    note("copied " + p.path + " with record offsets shifted by " + to_string(p.offset_shift) + ", node IDs by "
         + to_string(p.id_offset) + " and component numbers by " + to_string(p.component_base));

    walk = nullptr;
    source = nullptr;
    part = nullptr;
}

SnarlDistanceIndex::record_t SnarlDistanceIndex::IndexMerger::type_at(size_t i) const {
    return get_record_type(source->at(i));
}

uint64_t SnarlDistanceIndex::IndexMerger::field(size_t record, size_t offset) const {
    if (offset >= part->entries - record) {
        fail("record at " + to_string(record) + " is cut off by the end of the index");
    }
    return source->at(record + offset);
}

void SnarlDistanceIndex::IndexMerger::check_offset(size_t offset, const string& what) const {
    if (offset < part->records_start || offset >= part->entries) {
        fail(what + " " + to_string(offset) + " is outside the record region [" + to_string(part->records_start)
             + ", " + to_string(part->entries) + ")");
    }
}

size_t SnarlDistanceIndex::IndexMerger::record_size(size_t record, record_t type) const {
    size_t size;
    if (is_node(type)) {
        size = NODE_RECORD_SIZE;
    } else if (is_chain(type)) {
        size = CHAIN_RECORD_SIZE;
    } else if (is_trivial_snarl(type)) {
        size_t nodes = field(record, TRIVIAL_SNARL_NODE_COUNT_OFFSET);
        if (nodes == 0 || nodes > MAX_TRIVIAL_SNARL_NODE_COUNT) {
            fail("run of nodes at " + to_string(record) + " claims " + to_string(nodes) + " nodes");
        }
        size = type == TRIVIAL_SNARL ? DISTANCELESS_TRIVIAL_SNARL_RECORD_SIZE + nodes
                                     : DISTANCED_TRIVIAL_SNARL_RECORD_SIZE + 2 * nodes;
    } else if (is_simple_snarl(type)) {
        size = SIMPLE_SNARL_RECORD_SIZE + 2 * (field(record, SIMPLE_SNARL_NODE_COUNT_AND_LENGTHS_OFFSET) >> 22);
    } else if (is_chain_snarl(type) || is_root_snarl(type)) {
        size_t children = field(record, SNARL_NODE_COUNT_OFFSET);
        if (children > part->entries) {
            fail("snarl at " + to_string(record) + " claims " + to_string(children) + " children");
        }
        size = SnarlRecord::record_size(type, children);
    } else {
        fail("entry " + to_string(record) + " has record type " + to_string((size_t) type)
             + ", which is not a snarl tree record");
    }
    if (size > part->entries - record) {
        fail("record at " + to_string(record) + " runs past the end of the index");
    }
    return size;
}

void SnarlDistanceIndex::IndexMerger::claim(size_t from, size_t count, size_t record) {
    if (from < part->records_start || count > part->entries - from) {
        fail("record at " + to_string(record) + " extends outside the record region");
    }
    if (walk->covered.test_and_set(from - part->records_start, count)) {
        fail("entries [" + to_string(from) + ", " + to_string(from + count) + ") of the record at " + to_string(record)
             + " also belong to another record");
    }
}

void SnarlDistanceIndex::IndexMerger::count_visit(record_t type) {
    walk->records_by_type[type]++;
    if (++walk->visits % VISIT_RELEASE_INTERVAL == 0) {
        source->release();
    }
}

void SnarlDistanceIndex::IndexMerger::walk_part() {
    for (size_t i = 0; i < part->components; i++) {
        size_t record = source->at(ROOT_RECORD_SIZE + i);
        check_offset(record, "component " + to_string(i) + " record");
        record_t type = type_at(record);
        size_t rank;
        if (is_node(type)) {
            rank = field(record, NODE_RANK_OFFSET);
        } else if (is_chain(type)) {
            rank = field(record, CHAIN_RANK_OFFSET) >> 1;
        } else if (is_root_snarl(type)) {
            rank = field(record, SNARL_MIN_LENGTH_OFFSET);
        } else {
            fail("component " + to_string(i) + " is a record of type " + to_string((size_t) type));
        }
        if (field(record, parent_field(type)) != 0) {
            fail("component " + to_string(i) + " record at " + to_string(record) + " has a parent");
        }
        // Component numbers are rewritten by adding a base, which is only
        // right if each root-level record's rank is its own slot.
        if (rank != i) {
            fail("component " + to_string(i) + " record at " + to_string(record) + " gives its rank as " + to_string(rank));
        }
        descend(record);
    }

    const size_t table = ROOT_RECORD_SIZE + part->components;
    for (size_t j = 0; j < part->node_slots; j++) {
        size_t record = source->at(table + 2 * j);
        if (record != 0) {
            climb(part->min_id + (nid_t) j, record, source->at(table + 2 * j + 1));
        }
        if ((j + 1) % VISIT_RELEASE_INTERVAL == 0) {
            source->release();
        }
    }
    check_coverage();
}

void SnarlDistanceIndex::IndexMerger::descend(size_t record) {
    walk->pending.push_back(record);
    while (!walk->pending.empty()) {
        size_t next = walk->pending.back();
        walk->pending.pop_back();
        visit(next);
    }
}

void SnarlDistanceIndex::IndexMerger::visit(size_t record) {
    check_offset(record, "record");
    if (walk->record_starts.test_and_set(record - part->records_start)) {
        fail("record at " + to_string(record) + " is reached twice");
    }
    record_t type = type_at(record);
    if (!is_node(type) && !is_chain(type) && !is_root_snarl(type)) {
        fail("record at " + to_string(record) + " has type " + to_string((size_t) type)
             + ", but only nodes, chains and root snarls are children of snarls or of the root");
    }
    count_visit(type);
    claim(record, record_size(record, type), record);
    if (is_chain(type)) {
        walk_chain(record);
    } else if (is_root_snarl(type)) {
        walk_child_list(record);
    }
}

void SnarlDistanceIndex::IndexMerger::walk_chain(size_t chain) {
    // Children are stored inline after the chain record, each bracketed by its
    // size before and after, except the last run of nodes, which is never
    // closed: its leading size stays 0 and nothing follows it.
    uint64_t last_entry = field(chain, CHAIN_LAST_CHILD_OFFSET);
    size_t last = last_entry >> 2;
    bool last_is_snarl = (last_entry >> 1) & 1;
    check_offset(last, "last child of chain at " + to_string(chain));
    size_t nodes = 0;
    size_t child = chain + CHAIN_RECORD_SIZE + 1;
    while (true) {
        if (child > last) {
            fail("chain at " + to_string(chain) + " runs past its last child at " + to_string(last));
        }
        record_t type = type_at(child);
        if (!is_trivial_snarl(type) && !is_simple_snarl(type) && !is_chain_snarl(type)) {
            fail("chain at " + to_string(chain) + " has a child at " + to_string(child) + " of type " + to_string((size_t) type));
        }
        size_t size = record_size(child, type);
        if (field(child, parent_field(type)) != chain) {
            fail("child at " + to_string(child) + " of chain at " + to_string(chain) + " gives its parent as "
                 + to_string(field(child, parent_field(type))));
        }
        if (walk->record_starts.test_and_set(child - part->records_start)) {
            fail("record at " + to_string(child) + " is reached twice");
        }
        count_visit(type);
        if (is_trivial_snarl(type)) {
            nodes += field(child, TRIVIAL_SNARL_NODE_COUNT_OFFSET);
        }
        uint64_t leading = source->at(child - 1);
        if (child == last && is_trivial_snarl(type)) {
            if (last_is_snarl || leading != 0) {
                fail("last run of nodes at " + to_string(child) + " in chain at " + to_string(chain) + " is marked as closed");
            }
            claim(child - 1, size + 1, child);
        } else {
            if (size + 1 > part->entries - child || leading != size || source->at(child + size) != size) {
                fail("child at " + to_string(child) + " of chain at " + to_string(chain)
                     + " is not bracketed by its size " + to_string(size));
            }
            claim(child - 1, size + 2, child);
            if (is_chain_snarl(type)) {
                walk_child_list(child);
            }
            if (child == last && !last_is_snarl) {
                fail("last child at " + to_string(child) + " of chain at " + to_string(chain) + " is a snarl marked as a node");
            }
        }
        if (child == last) {
            break;
        }
        child += size + 2;
    }
    // A looping chain lists its first node again at the end without storing it twice.
    size_t count = field(chain, CHAIN_NODE_COUNT_OFFSET);
    bool looping = (field(chain, CHAIN_START_NODE_OFFSET) >> 1) == (field(chain, CHAIN_END_NODE_OFFSET) >> 1);
    if (nodes != count && !(looping && nodes + 1 == count)) {
        fail("chain at " + to_string(chain) + " holds " + to_string(nodes) + " nodes but counts " + to_string(count));
    }
}

void SnarlDistanceIndex::IndexMerger::walk_child_list(size_t snarl) {
    size_t children = field(snarl, SNARL_NODE_COUNT_OFFSET);
    size_t list = field(snarl, SNARL_CHILD_RECORD_OFFSET);
    if (list == 0) {
        // The builder skips the list of a root snarl whose distances it did
        // not store; its children are reached from the node table instead.
        walk->snarls_without_child_list++;
        return;
    }
    check_offset(list, "child list of snarl at " + to_string(snarl));
    claim(list, children, snarl);
    walk->child_list_entries.test_and_set(list - part->records_start, children);
    for (size_t i = 0; i < children; i++) {
        size_t child = field(list, i);
        check_offset(child, "child " + to_string(i) + " of snarl at " + to_string(snarl));
        record_t type = type_at(child);
        if (!is_node(type) && !is_chain(type)) {
            fail("child at " + to_string(child) + " of snarl at " + to_string(snarl) + " has type " + to_string((size_t) type));
        }
        if (field(child, parent_field(type)) != snarl) {
            fail("child at " + to_string(child) + " of snarl at " + to_string(snarl) + " gives its parent as "
                 + to_string(field(child, parent_field(type))));
        }
        walk->pending.push_back(child);
    }
}

void SnarlDistanceIndex::IndexMerger::climb(nid_t id, size_t record, size_t node_offset) {
    // Find the highest ancestor not yet reached; its parent must be a root
    // snarl without a child list, and walking down from it reaches the node.
    check_offset(record, "record of node " + to_string(id));
    size_t ancestor = record;
    size_t below = 0;
    size_t steps = 0;
    while (ancestor != 0 && !walk->record_starts.get(ancestor - part->records_start)) {
        record_t type = type_at(ancestor);
        if (is_root_snarl(type)) {
            fail("root snarl at " + to_string(ancestor) + " is not in the component table");
        }
        if (!is_node(type) && !is_chain(type) && !is_trivial_snarl(type) && !is_simple_snarl(type) && !is_chain_snarl(type)) {
            fail("record of node " + to_string(id) + " leads to entry " + to_string(ancestor) + " of type " + to_string((size_t) type));
        }
        below = ancestor;
        ancestor = field(ancestor, parent_field(type));
        if (ancestor != 0) {
            check_offset(ancestor, "parent of record at " + to_string(below));
        }
        // Snarl trees are nowhere near this deep; a longer walk is a cycle.
        if (++steps > (size_t(1) << 24)) {
            fail("parents of the record of node " + to_string(id) + " do not reach the root");
        }
    }
    if (below != 0) {
        if (ancestor == 0) {
            fail("record at " + to_string(below) + " has no parent but is not in the component table");
        }
        record_t below_type = type_at(below);
        if (!is_node(below_type) && !is_chain(below_type)) {
            fail("record at " + to_string(below) + " is not among the children of its chain at " + to_string(ancestor));
        }
        if (!is_root_snarl(type_at(ancestor)) || field(ancestor, SNARL_CHILD_RECORD_OFFSET) != 0) {
            fail("record at " + to_string(below) + " gives its parent as " + to_string(ancestor)
                 + ", which does not list it as a child");
        }
        descend(below);
    }
    if (!walk->record_starts.get(record - part->records_start)) {
        fail("record of node " + to_string(id) + " at " + to_string(record) + " is not a record");
    }

    // The node table entry must lead back to this node.
    record_t type = type_at(record);
    bool found;
    if (is_node(type)) {
        found = node_offset == 0 && field(record, NODE_ID_OFFSET) == (uint64_t) id;
    } else if (is_trivial_snarl(type)) {
        size_t nodes = field(record, TRIVIAL_SNARL_NODE_COUNT_OFFSET);
        size_t entry = type == TRIVIAL_SNARL ? record + DISTANCELESS_TRIVIAL_SNARL_RECORD_SIZE + node_offset
                                             : record + DISTANCED_TRIVIAL_SNARL_RECORD_SIZE + 2 * node_offset;
        found = node_offset < nodes && (source->at(entry) >> 1) == (uint64_t) id;
    } else if (is_simple_snarl(type)) {
        size_t nodes = field(record, SIMPLE_SNARL_NODE_COUNT_AND_LENGTHS_OFFSET) >> 22;
        found = node_offset >= 2 && node_offset - 2 < nodes
                && source->at(record + SIMPLE_SNARL_RECORD_SIZE + 2 * (node_offset - 2)) == (uint64_t) id;
    } else {
        found = false;
    }
    if (!found) {
        fail("node table entry for node " + to_string(id) + " (record " + to_string(record) + ", offset "
             + to_string(node_offset) + ") does not lead to that node");
    }
}

void SnarlDistanceIndex::IndexMerger::check_coverage() {
    const size_t start = part->records_start;
    const size_t region = part->entries - start;
    size_t uncovered_nonzero = 0;
    size_t first_nonzero = 0;
    for (size_t i = 0; i < region; i++) {
        if ((i & 63) == 0 && region - i >= 64 && walk->covered.word_at(i) == ~uint64_t(0)) {
            i += 63;
            continue;
        }
        if (!walk->covered.get(i)) {
            if (source->at(start + i) != 0) {
                if (uncovered_nonzero++ == 0) {
                    first_nonzero = start + i;
                }
            } else {
                walk->uncovered_zero_entries++;
            }
        }
        if ((i + 1) % RELEASE_INTERVAL == 0) {
            source->release();
        }
    }
    if (uncovered_nonzero != 0) {
        fail(to_string(uncovered_nonzero) + " nonzero entries belong to no record reached from the root or the node"
             " table, the first at " + to_string(first_nonzero));
    }
}

uint64_t SnarlDistanceIndex::IndexMerger::relocate(uint64_t offset) const {
    if (offset == 0) {
        return 0;
    }
    if (offset < part->records_start || offset >= part->entries) {
        fail("record offset " + to_string(offset) + " is outside the record region");
    }
    return offset + part->offset_shift;
}

uint64_t SnarlDistanceIndex::IndexMerger::shift_id(uint64_t id) const {
    if (id < (uint64_t) part->min_id || id > (uint64_t) part->max_id) {
        fail("stored node ID " + to_string(id) + " is outside the index's node IDs");
    }
    return id + part->id_offset;
}

uint64_t SnarlDistanceIndex::IndexMerger::shift_oriented_id(uint64_t oriented_id) const {
    return (shift_id(oriented_id >> 1) << 1) | (oriented_id & 1);
}

void SnarlDistanceIndex::IndexMerger::write_part(Output& output) {
    const Part& p = *part;
    for (size_t i = 0; i < p.components; i++) {
        output.put(ROOT_RECORD_SIZE + p.component_base + i, relocate(source->at(ROOT_RECORD_SIZE + i)));
    }
    const size_t table = ROOT_RECORD_SIZE + p.components;
    const size_t merged_table = ROOT_RECORD_SIZE + total_components + 2 * (p.min_id + p.id_offset - min_id);
    for (size_t j = 0; j < p.node_slots; j++) {
        size_t record = source->at(table + 2 * j);
        if (record != 0) {
            output.put(merged_table + 2 * j, relocate(record));
            // The second entry is a position inside the record, not an offset.
            output.put(merged_table + 2 * j + 1, source->at(table + 2 * j + 1));
        }
    }
    size_t i = p.records_start;
    while (i < p.entries) {
        size_t bit = i - p.records_start;
        if (walk->record_starts.get(bit)) {
            i += write_record(output, i);
        } else if (walk->child_list_entries.get(bit)) {
            output.put(i + p.offset_shift, relocate(source->at(i)));
            i++;
        } else {
            // Chain child size brackets and unused zeros.
            output.put(i + p.offset_shift, source->at(i));
            i++;
        }
    }
}

size_t SnarlDistanceIndex::IndexMerger::write_record(Output& output, size_t record) {
    const Part& p = *part;
    record_t type = type_at(record);
    size_t size = record_size(record, type);
    // Numbers of connected components are stored only by root-level records,
    // as their rank: raw for nodes, beside an orientation bit for chains, and
    // in the length slot for root snarls.
    bool at_root = is_root_snarl(type) || ((is_node(type) || is_chain(type)) && source->at(record + parent_field(type)) == 0);
    // Everything after a snarl's first fields is distances.
    size_t fields = is_chain_snarl(type) || is_root_snarl(type) ? SNARL_RECORD_SIZE : size;
    for (size_t f = 0; f < fields; f++) {
        uint64_t value = source->at(record + f);
        if (f == 0) {
            // The tag.
        } else if (is_node(type)) {
            if (f == NODE_ID_OFFSET) {
                value = shift_id(value);
            } else if (f == NODE_PARENT_OFFSET) {
                value = relocate(value);
            } else if (f == NODE_RANK_OFFSET && at_root) {
                value += p.component_base;
            }
        } else if (is_chain(type)) {
            if (f == CHAIN_PARENT_OFFSET) {
                value = relocate(value);
            } else if (f == CHAIN_RANK_OFFSET && at_root) {
                value = (((value >> 1) + p.component_base) << 1) | (value & 1);
            } else if (f == CHAIN_START_NODE_OFFSET || f == CHAIN_END_NODE_OFFSET) {
                value = shift_oriented_id(value);
            } else if (f == CHAIN_LAST_CHILD_OFFSET) {
                // (offset << 2) | (is_snarl << 1) | loopable
                value = (relocate(value >> 2) << 2) | (value & 3);
            }
        } else if (is_trivial_snarl(type)) {
            size_t header = type == TRIVIAL_SNARL ? DISTANCELESS_TRIVIAL_SNARL_RECORD_SIZE : DISTANCED_TRIVIAL_SNARL_RECORD_SIZE;
            size_t stride = type == TRIVIAL_SNARL ? 1 : 2;
            if (f == TRIVIAL_SNARL_PARENT_OFFSET) {
                value = relocate(value);
            } else if (f >= header && (f - header) % stride == 0) {
                // (node ID << 1) | reversed; a distanced run also stores a prefix sum after each.
                value = shift_oriented_id(value);
            }
        } else if (is_simple_snarl(type)) {
            if (f == SIMPLE_SNARL_PARENT_OFFSET) {
                value = relocate(value);
            } else if (f >= SIMPLE_SNARL_RECORD_SIZE && (f - SIMPLE_SNARL_RECORD_SIZE) % 2 == 0) {
                // Node ID, then (length << 1) | reversed.
                value = shift_id(value);
            }
        } else {
            // Chain snarls and root snarls.
            if (f == SNARL_PARENT_OFFSET || f == SNARL_CHILD_RECORD_OFFSET) {
                value = relocate(value);
            } else if (f == SNARL_MIN_LENGTH_OFFSET && is_root_snarl(type)) {
                value += p.component_base;
            }
        }
        output.put(record + f + p.offset_shift, value);
    }
    for (size_t f = fields; f < size; f++) {
        output.put(record + f + p.offset_shift, source->at(record + f));
    }
    return size;
}

void SnarlDistanceIndex::merge_indexes(const string& output_path, const vector<pair<string, nid_t>>& parts, ostream* log) {
    IndexMerger(output_path, parts, log).run();
}

}
