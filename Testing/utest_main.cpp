
/** \file utest_main.cpp
 *  Entry point for unit tests
 */

#include "gtest/gtest.h"
#include "test_utils/AmrexTestEnv.H"

//! Global instance of the environment (for access in tests)
GEMPIC_tests::AmrexTestEnv* utest_env = nullptr;

int main(int argc, char* argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
  // gtest takes ownership of the TestEnvironment ptr - we don't delete it.
  auto utest_env = new GEMPIC_tests::AmrexTestEnv(argc, argv);
  ::testing::AddGlobalTestEnvironment(utest_env);
  return RUN_ALL_TESTS();
}
