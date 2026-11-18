#pragma once

// Minimal in-repo test harness: no external framework, so the suite builds
// anywhere the library does.
//
//   TC_TEST(group, name) { TC_CHECK(1 + 1 == 2); }
//
// `tc_tests` runs everything; `tc_tests <group>` runs one group (this is how
// CTest registers them). A failed TC_CHECK records the failure and keeps going;
// TC_REQUIRE aborts the current test.

#include <cstdint>
#include <exception>
#include <stdexcept>
#include <string>
#include <vector>

namespace tctest {

struct TestCase {
    const char* group;
    const char* name;
    void (*fn)();
};

std::vector<TestCase>& registry();
void report_failure(const char* file, int line, const std::string& message);

struct Registrar {
    Registrar(const char* group, const char* name, void (*fn)()) {
        registry().push_back({group, name, fn});
    }
};

struct RequireFailed : std::exception {};

}  // namespace tctest

#define TC_TEST(group, name)                                                        \
    static void tc_test_##group##_##name();                                         \
    static tctest::Registrar tc_registrar_##group##_##name(#group, #name,           \
                                                           &tc_test_##group##_##name); \
    static void tc_test_##group##_##name()

#define TC_CHECK(cond)                                                              \
    do {                                                                            \
        if (!(cond)) tctest::report_failure(__FILE__, __LINE__, "TC_CHECK(" #cond ")"); \
    } while (0)

#define TC_CHECK_MSG(cond, msg)                                                     \
    do {                                                                            \
        if (!(cond)) tctest::report_failure(__FILE__, __LINE__,                     \
                                            std::string("TC_CHECK(" #cond "): ") + (msg)); \
    } while (0)

#define TC_REQUIRE(cond)                                                            \
    do {                                                                            \
        if (!(cond)) {                                                              \
            tctest::report_failure(__FILE__, __LINE__, "TC_REQUIRE(" #cond ")");    \
            throw tctest::RequireFailed{};                                          \
        }                                                                           \
    } while (0)

// Passes only if `expr` throws exactly ExceptionType (or a type derived from it).
#define TC_CHECK_THROWS(expr, ExceptionType)                                        \
    do {                                                                            \
        bool tc_threw = false;                                                      \
        try {                                                                       \
            (void)(expr);                                                           \
        } catch (const ExceptionType&) {                                            \
            tc_threw = true;                                                        \
        } catch (...) {                                                             \
            tctest::report_failure(__FILE__, __LINE__,                              \
                                   "wrong exception type from: " #expr);            \
            tc_threw = true;                                                        \
        }                                                                           \
        if (!tc_threw) tctest::report_failure(__FILE__, __LINE__,                   \
                                              "expected " #ExceptionType " from: " #expr); \
    } while (0)
