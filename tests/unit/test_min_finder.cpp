#include "arch/input_spine_buffer.hpp"
#include "arch/min_finder.hpp"
#include "common/entry.hpp"
#include <iostream>
#include <array>
#include <vector>

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

void TEST_EmptyAll() {
    std::cout << "[RUN ] EmptyAll\n";
    InputSpineBuffer b0,b1,b2,b3;
    SpineHeadView v0(&b0), v1(&b1), v2(&b2), v3(&b3);
    std::array<SpineHeadView*,4> views{&v0,&v1,&v2,&v3};
    MinFinder<4> mf(views);

    MinPick p = mf.step();
    CHECK(!p.valid, "min pick should be invalid when all inputs are empty");
    std::cout << "[DONE] EmptyAll\n";
}

void TEST_MinSelectAndDequeue() {
    std::cout << "[RUN ] MinSelectAndDequeue\n";
    std::vector<Entry> dram = { {5,10}, {2,11}, {2,9}, {4,13} };

    InputSpineBuffer b0,b1,b2,b3;
    SpineHeadView v0(&b0), v1(&b1), v2(&b2), v3(&b3);
    std::array<SpineHeadView*,4> views{&v0,&v1,&v2,&v3};
    MinFinder<4> mf(views);

    // Load single entries: b0<-0, b1<-1, b2<-2, b3<-3
    bool ok0 = b0.load_from_dram(0*sizeof(Entry), 1*sizeof(Entry), dram);
    bool ok1 = b1.load_from_dram(1*sizeof(Entry), 1*sizeof(Entry), dram);
    bool ok2 = b2.load_from_dram(2*sizeof(Entry), 1*sizeof(Entry), dram);
    bool ok3 = b3.load_from_dram(3*sizeof(Entry), 1*sizeof(Entry), dram);
    CHECK(ok0 && ok1 && ok2 && ok3, "all loads should succeed");

    // First min: tie on ts=2; choose smaller neuron_id=9
    MinPick p1 = mf.step();
    CHECK(p1.valid, "first pick should be valid");
    CHECK(p1.ts == 2 && p1.neuron_id == 9, "first pick should be <ts=2, neuron=9>");

    // Simulate consumer removing the chosen head (we know it's in b2)
    b2.dequeue();

    // Second min: now <ts=2, neuron=11>
    MinPick p2 = mf.step();
    CHECK(p2.valid, "second pick should be valid");
    CHECK(p2.ts == 2 && p2.neuron_id == 11, "second pick should be <ts=2, neuron=11>");
    std::cout << "[DONE] MinSelectAndDequeue\n";
}

void TEST_TieBreakByNeuronThenStable() {
    std::cout << "[RUN ] TieBreakByNeuronThenStable\n";
    InputSpineBuffer b0,b1,b2,b3;
    SpineHeadView v0(&b0), v1(&b1), v2(&b2), v3(&b3);
    std::array<SpineHeadView*,4> views{&v0,&v1,&v2,&v3};
    MinFinder<4> mf(views);

    // Two heads with same ts, different neuron_id
    std::vector<Entry> dram = { {3,5}, {3,2} };
    bool okA = b0.load_from_dram(0*sizeof(Entry), 1*sizeof(Entry), dram); // <3,5>
    bool okB = b1.load_from_dram(1*sizeof(Entry), 1*sizeof(Entry), dram); // <3,2>
    CHECK(okA && okB, "loads should succeed");

    MinPick p = mf.step();
    CHECK(p.valid, "pick should be valid");
    CHECK(p.ts == 3 && p.neuron_id == 2, "smaller neuron_id (2) wins tie on ts");
    std::cout << "[DONE] TieBreakByNeuronThenStable\n";
}

} // namespace

int main() {
    std::cout << "=== MinFinder Unit Tests ===\n";
    TEST_EmptyAll();
    TEST_MinSelectAndDequeue();
    TEST_TieBreakByNeuronThenStable();
    if (g_failures == 0) {
        std::cout << "[PASS] All tests passed.\n";
        return 0;
    } else {
        std::cout << "[FAIL] " << g_failures << " test(s) failed.\n";
        return 1;
    }
}
