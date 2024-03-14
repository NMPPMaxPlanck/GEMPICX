
/** \file GEMPIC_UnitTests.cpp
 *  Entry point for unit tests
 */

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include "TestUtils/GEMPIC_AmrexTestEnv.H"

//! Global instance of the environment (for access in tests)
GEMPIC_Tests::AmrexTestEnv* utestEnv = nullptr;

int main (int argc, char* argv[])
{
    ::testing::InitGoogleTest(&argc, argv);
    ::testing::InitGoogleMock(&argc, argv);
    // gtest takes ownership of the TestEnvironment ptr - we don't delete it.
    auto* utestEnv = new GEMPIC_Tests::AmrexTestEnv(argc, argv);
    ::testing::AddGlobalTestEnvironment(utestEnv);
    return RUN_ALL_TESTS();
}
