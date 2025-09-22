#include "arch/pe.hpp"
#include <iostream>
#include <string>

/* All comments are in English.
 * Minimal test harness:
 *  - Use CHECK(cond, msg) to record failures without stopping the run.
 *  - Each test is a function; main() aggregates results.
 */

using namespace sf;

namespace {
int g_failures = 0;

void CHECK(bool cond, const std::string& msg) {
    if (!cond) {
        ++g_failures;
        std::cerr << "[FAIL] " << msg << "\n";
    }
}

void TEST_DefaultCtorAndNoSpike() {
    std::cout << "[RUN ] DefaultCtorAndNoSpike\n";
    PE pe;
    CHECK(pe.vmem() == 0, "vmem should be 0 after default ctor");
    CHECK(!pe.spiked(), "spiked should be false after default ctor");
    CHECK(pe.reset_vmem() == 0, "reset_vmem should be 0 after default ctor");

    // Below-threshold update should not spike
    int8_t out = pe.Process(/*timestamp*/ 7, /*filter*/ 3, /*threshold*/ 5);
    CHECK(out == PE::kNoSpike, "no spike expected below threshold");
    CHECK(pe.vmem() == 3, "vmem should become 3");
    CHECK(!pe.spiked(), "spiked should remain false");
    std::cout << "[DONE] DefaultCtorAndNoSpike\n";
}

void TEST_SpikeAndReset() {
    std::cout << "[RUN ] SpikeAndReset\n";
    PE pe;
    pe.set_reset_vmem(-2);

    int8_t out1 = pe.Process(/*ts*/ 1, /*filter*/ 4, /*thr*/ 6);
    CHECK(out1 == PE::kNoSpike, "first call should not spike");
    CHECK(pe.vmem() == 4, "vmem should be 4 after first call");
    CHECK(!pe.spiked(), "spiked should be false after first call");

    int8_t out2 = pe.Process(/*ts*/ 2, /*filter*/ 3, /*thr*/ 6);
    CHECK(out2 == 2, "should spike and output timestamp 2");
    CHECK(pe.spiked(), "spiked should be true after crossing threshold");
    CHECK(pe.vmem() == -2, "vmem should reset to -2 after spike");
    std::cout << "[DONE] SpikeAndReset\n";
}

void TEST_SaturatingAdd() {
    std::cout << "[RUN ] SaturatingAdd\n";
    PE pe;

    // Push vmem up near int8_t max using repeated additions
    for (int i = 0; i < 20; ++i) {
        (void)pe.Process(/*ts*/ i, /*filter*/ 10, /*thr*/ 120);
    }
    int8_t before = pe.vmem();
    std::cout << " vmem before large positive filter: " << static_cast<int>(before) << "\n";
    // Large positive filter -> must not overflow; clamp to +127
    (void)pe.Process(/*ts*/ 21, /*filter*/ 100, /*thr*/ 127);
    std::cout << " vmem after large positive filter: " << static_cast<int>(pe.vmem()) << "\n";
    CHECK(pe.vmem() <= 127, "vmem must be clamped to <= 127");
    CHECK(pe.vmem() >= before, "vmem should be monotonic up to saturation");
    std::cout << "[DONE] SaturatingAdd\n";
}

} // namespace

int main() {
    std::cout << "=== PE Unit Tests (no external deps) ===\n";
    TEST_DefaultCtorAndNoSpike();
    TEST_SpikeAndReset();
    TEST_SaturatingAdd();
    if (g_failures == 0) {
        std::cout << "[PASS] All tests passed.\n";
        return 0;
    } else {
        std::cout << "[FAIL] " << g_failures << " test(s) failed.\n";
        return 1;
    }
}
