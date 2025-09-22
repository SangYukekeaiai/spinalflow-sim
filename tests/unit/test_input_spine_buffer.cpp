#include "arch/input_spine_buffer.hpp"
#include "common/entry.hpp"
#include <iostream>
#include <vector>
#include <cstdint>

/* All comments are in English */

using namespace sf;

namespace {
int g_failures = 0;

void CHECK(bool cond, const char* msg) {
    if (!cond) {
        ++g_failures;
        std::cerr << "[FAIL] " << msg << "\n";
    }
}

void TEST_LoadSuccessAndPeekDequeue() {
    std::cout << "[RUN ] LoadSuccessAndPeekDequeue\n";
    // Fake DRAM as flat vector of Entry
    std::vector<Entry> dram = { {5,10}, {2,11}, {4,12} };

    InputSpineBuffer buf;
    // Load dram[1] only
    bool ok = buf.load_from_dram(1 * sizeof(Entry), 1 * sizeof(Entry), dram);
    CHECK(ok, "load_from_dram should succeed for aligned, in-bounds request");
    CHECK(!buf.empty(), "buffer should not be empty after load");
    CHECK(buf.size() == 1, "buffer size should be 1 after single entry load");

    auto head = buf.peek_head();
    CHECK(head.has_value(), "peek_head should return a value");
    CHECK(head->ts == 2 && head->neuron_id == 11, "head should be <ts=2, neuron=11>");

    buf.dequeue();
    CHECK(buf.empty(), "buffer should be empty after dequeue");
    std::cout << "[DONE] LoadSuccessAndPeekDequeue\n";
}

void TEST_LoadAlignmentAndBounds() {
    std::cout << "[RUN ] LoadAlignmentAndBounds\n";
    std::vector<Entry> dram = { {1,1}, {2,2} };
    InputSpineBuffer buf;

    // Misaligned size
    bool ok1 = buf.load_from_dram(0, /*bytes*/ 1, dram);
    CHECK(!ok1, "load should fail when size_bytes is not multiple of sizeof(Entry)");

    // Misaligned base addr
    bool ok2 = buf.load_from_dram(/*addr*/ sizeof(Entry)/2, /*bytes*/ sizeof(Entry), dram);
    CHECK(!ok2, "load should fail when base_addr is not Entry-aligned");

    // Out-of-bounds (start beyond dram size)
    bool ok3 = buf.load_from_dram(10 * sizeof(Entry), sizeof(Entry), dram);
    CHECK(!ok3, "load should fail when base index is beyond dram size");

    // Out-of-bounds (count too large)
    bool ok4 = buf.load_from_dram(1 * sizeof(Entry), 2 * sizeof(Entry), dram);
    CHECK(!ok4, "load should fail when count exceeds dram end");

    CHECK(buf.empty(), "buffer should remain empty after failed loads");
    std::cout << "[DONE] LoadAlignmentAndBounds\n";
}

void TEST_SpineHeadView() {
    std::cout << "[RUN ] SpineHeadView\n";
    std::vector<Entry> dram = { {9,7} };
    InputSpineBuffer buf;
    SpineHeadView view(&buf);

    // Empty case
    HeadInfo h0 = view.get_head();
    CHECK(!h0.valid, "head view should be invalid on empty buffer");

    // Load one
    bool ok = buf.load_from_dram(0, sizeof(Entry), dram);
    CHECK(ok, "aligned in-bounds load should succeed");

    HeadInfo h1 = view.get_head();
    CHECK(h1.valid, "head view should be valid after load");
    CHECK(h1.ts == 9 && h1.neuron_id == 7, "head view should match loaded entry");
    std::cout << "[DONE] SpineHeadView\n";
}

} // namespace

int main() {
    std::cout << "=== InputSpineBuffer Unit Tests ===\n";
    TEST_LoadSuccessAndPeekDequeue();
    TEST_LoadAlignmentAndBounds();
    TEST_SpineHeadView();
    if (g_failures == 0) {
        std::cout << "[PASS] All tests passed.\n";
        return 0;
    } else {
        std::cout << "[FAIL] " << g_failures << " test(s) failed.\n";
        return 1;
    }
}
