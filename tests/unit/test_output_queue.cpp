#include "arch/output_queue.hpp"
#include "common/entry.hpp"
#include <iostream>
#include <vector>

/* All comments are in English */

using namespace sf;

namespace {
int g_failures = 0;
void CHECK(bool cond, const char* msg) {
    if (!cond) { ++g_failures; std::cerr << "[FAIL] " << msg << "\n"; }
}

void TEST_BasicEnqueueDequeue() {
    std::cout << "[RUN ] BasicEnqueueDequeue\n";
    OutputQueue q;

    CHECK(q.empty(), "queue should start empty");
    q.enqueue({1, 42});
    q.enqueue({3, 99});
    CHECK(q.size() == 2, "queue size should be 2 after two enqueues");

    auto head1 = q.peek_head();
    CHECK(head1 && head1->ts == 1 && head1->neuron_id == 42, "head should be <1,42>");

    q.dequeue();
    auto head2 = q.peek_head();
    CHECK(head2 && head2->ts == 3 && head2->neuron_id == 99, "head should be <3,99>");
    q.dequeue();
    CHECK(q.empty(), "queue should be empty after dequeuing all");
    std::cout << "[DONE] BasicEnqueueDequeue\n";
}

void TEST_EmptyVsFlush() {
    std::cout << "[RUN ] EmptyVsFlush\n";
    OutputQueue q;
    q.enqueue({5, 11});
    q.enqueue({6, 12});
    CHECK(!q.empty(), "queue should not be empty before flush");

    q.flush();
    CHECK(q.empty(), "queue should be empty after flush");
    CHECK(q.size() == 0, "size should be 0 after flush");

    // empty() only checks state, flush() actually clears content
    q.enqueue({7, 77});
    CHECK(q.size() == 1, "queue should have 1 entry after enqueue");
    CHECK(!q.empty(), "empty() should return false when queue has entries");
    std::cout << "[DONE] EmptyVsFlush\n";
}

void TEST_StoreToDRAM() {
    std::cout << "[RUN ] StoreToDRAM\n";
    // Fake DRAM initialized with dummy entries
    std::vector<Entry> dram(10, {0, 0});

    OutputQueue q;
    q.enqueue({10, 100});
    q.enqueue({12, 120});

    // Valid store at index 4 (byte address = 4*sizeof(Entry))
    bool ok = q.store_to_dram(4 * sizeof(Entry), dram);
    CHECK(ok, "store_to_dram should succeed for aligned, in-bounds request");
    CHECK(dram[4].ts == 10 && dram[4].neuron_id == 100, "dram[4] should match first entry");
    CHECK(dram[5].ts == 12 && dram[5].neuron_id == 120, "dram[5] should match second entry");

    // Misaligned base address
    bool bad_align = q.store_to_dram(sizeof(Entry)/2, dram);
    CHECK(!bad_align, "store should fail if base address not aligned");

    // Out-of-bounds (not enough space to store both entries)
    bool bad_oob = q.store_to_dram(9 * sizeof(Entry), dram);
    CHECK(!bad_oob, "store should fail if not enough space in DRAM");
    std::cout << "[DONE] StoreToDRAM\n";
}

} // namespace

int main() {
    std::cout << "=== OutputQueue Unit Tests ===\n";
    TEST_BasicEnqueueDequeue();
    TEST_EmptyVsFlush();
    TEST_StoreToDRAM();
    if (g_failures == 0) {
        std::cout << "[PASS] All tests passed.\n";
        return 0;
    } else {
        std::cout << "[FAIL] " << g_failures << " test(s) failed.\n";
        return 1;
    }
}
