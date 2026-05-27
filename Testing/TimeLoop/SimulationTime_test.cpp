/**************************************************************************************************
 * Copyright (c) 2021 GEMPICX                                                                     *
 * SPDX-License-Identifier: BSD-3-Clause                                                          *
 **************************************************************************************************/
#include <filesystem>

#include <gtest/gtest.h>

#include "GEMPIC_HDF5Interface.H"
#include "GEMPIC_TimeStepper.H"

using namespace Gempic;

TEST(SimulationTimeTest, MemberProperties)
{
    DiscreteTime time{0.1, 10};
    EXPECT_EQ(time.current_step(), time.initial_step());
    EXPECT_LE(time.t(), time.initial());
    while (time.continue_simulation())
    {
        EXPECT_LE(time.current_step(), time.final_step());
        EXPECT_LE(time.t(), time.final());
        time.step();
    };
}

TEST(SimulationTimeTest, SerializeDeserializeUnity)
{
    double const dt = 0.1;
    size_t const N = 10;
    size_t const n = 3;

    DiscreteTime original{dt, N, n};

    std::string const filename = "simulationTimeFile";
    std::string const groupname = "simulationTime";

    {
        H5FileHandle file{filename, H5FileHandle::Mode::CreateExclusive};
        H5GroupHandle group{file.h5id(), groupname, H5GroupHandle::Mode::CreateExclusive};

        serialize(groupname, original, group);
    }

    H5FileHandle file{filename, H5FileHandle::Mode::ReadOnly};
    H5GroupHandle group{file.h5id(), groupname, H5GroupHandle::Mode::ReadWrite};

    DiscreteTime restored{0.0, 0, 0};
    deserialize(groupname, restored, group);

    EXPECT_EQ(original, restored);
    EXPECT_TRUE(std::filesystem::remove(filename + H5FileHandle::s_extension));
}
