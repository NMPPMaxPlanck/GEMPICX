/**************************************************************************************************
 * Copyright (c) 2021 GEMPICX                                                                     *
 * SPDX-License-Identifier: BSD-3-Clause                                                          *
 **************************************************************************************************/

/** \file GEMPIC_UnitTests.cpp
 *  Entry point for unit tests
 */

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include "TestUtils/GEMPIC_AmrexTestEnv.H"

//! Global instance of the environment (for access in tests)
GempicTests::AmrexTestEnv* utestEnv = nullptr;

int main (int argc, char* argv[])
{
    ::testing::InitGoogleTest(&argc, argv);
    ::testing::InitGoogleMock(&argc, argv);

    // gtest takes ownership of the TestEnvironment ptr - we don't delete it.
    auto* utestEnv = new GempicTests::AmrexTestEnv(argc, argv);
    ::testing::AddGlobalTestEnvironment(utestEnv);

    testing::TestEventListeners& listeners = testing::UnitTest::GetInstance()->listeners();
    // Add a listener to the end. gtest takes ownership of the listener, too.
    // This listener ensures that the parmparse object is destroyed after each test --
    // unless keepParameters is set.
    listeners.Append(new GempicTests::AmrexTestIsolation);
    return RUN_ALL_TESTS();
}
