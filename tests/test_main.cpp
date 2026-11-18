#include "test_harness.h"

#include <cstdio>
#include <cstring>

namespace tctest {

namespace {
int g_failures_in_current_test = 0;
}

std::vector<TestCase>& registry() {
    static std::vector<TestCase> tests;
    return tests;
}

void report_failure(const char* file, int line, const std::string& message) {
    ++g_failures_in_current_test;
    // Cap the noise: a broken codec can fail thousands of checks in one loop.
    if (g_failures_in_current_test <= 10) {
        std::printf("    FAIL %s:%d: %s\n", file, line, message.c_str());
    }
}

}  // namespace tctest

int main(int argc, char** argv) {
    const char* group_filter = argc > 1 ? argv[1] : nullptr;

    int ran = 0;
    int failed = 0;
    for (const tctest::TestCase& test : tctest::registry()) {
        if (group_filter != nullptr && std::strcmp(group_filter, test.group) != 0) continue;
        ++ran;
        tctest::g_failures_in_current_test = 0;
        try {
            test.fn();
        } catch (const tctest::RequireFailed&) {
            // already reported
        } catch (const std::exception& e) {
            tctest::report_failure(__FILE__, __LINE__, std::string("uncaught exception: ") + e.what());
        } catch (...) {
            tctest::report_failure(__FILE__, __LINE__, "uncaught non-std exception");
        }
        const bool ok = tctest::g_failures_in_current_test == 0;
        std::printf("[%s] %s.%s\n", ok ? " ok " : "FAIL", test.group, test.name);
        if (!ok) ++failed;
    }

    if (ran == 0) {
        std::printf("no tests matched group '%s'\n", group_filter ? group_filter : "");
        return 2;   // an empty selection must not look like a pass
    }
    std::printf("%d test(s), %d failed\n", ran, failed);
    return failed == 0 ? 0 : 1;
}
