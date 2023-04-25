#include <AMReX.H>
#include <GEMPIC_particle_mesh_coupling.H>
#include "gtest/gtest.h"
#include "test_utils/GEMPIC_test_utils.H"

namespace {
    class PushVTest : public testing::Test {
        protected:

        static const int vDim{3};
        
        amrex::Real dt{1};
        amrex::Real chargemass{1};
        
        amrex::GpuArray<amrex::Real, 3> vel{1, 1, 1};
        amrex::GpuArray<amrex::Real, 3> Ep{1, 1, 1};

    };

    TEST_F(PushVTest, NullTest) {

        amrex::GpuArray<amrex::Real, vDim> newPos = Gempic::Particles::push_v_efield<vDim>(vel, dt, chargemass, Ep);

        EXPECT_EQ(newPos[0], 2);
        EXPECT_EQ(newPos[1], 2);
        EXPECT_EQ(newPos[2], 2);
    }

    TEST_F(PushVTest, VelTest) {

        chargemass = 0;

        amrex::GpuArray<amrex::Real, vDim> newPos = Gempic::Particles::push_v_efield<vDim>(vel, dt, chargemass, Ep);

        EXPECT_EQ(newPos[0], vel[0]);
        EXPECT_EQ(newPos[1], vel[1]);
        EXPECT_EQ(newPos[2], vel[2]);
    }

    TEST_F(PushVTest, EpTest) {

        vel = {0, 0, 0};
        Ep = {1, 2, 3};

        amrex::GpuArray<amrex::Real, vDim> newPos = Gempic::Particles::push_v_efield<vDim>(vel, dt, chargemass, Ep);

        EXPECT_EQ(newPos[0], 1);
        EXPECT_EQ(newPos[1], 2);
        EXPECT_EQ(newPos[2], 3);
    }
}