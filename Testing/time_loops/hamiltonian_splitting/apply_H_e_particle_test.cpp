#include <AMReX.H>
#include <GEMPIC_particle_groups.H>
#include <GEMPIC_FDDeRhamComplex.H>
#include <GEMPIC_Fields.H>
#include <GEMPIC_particle_groups.H>
#include <GEMPIC_particle_mesh_coupling_C2.H>
#include "gtest/gtest.h"
#include "test_utils/GEMPIC_test_utils.H"
#include "gmock/gmock.h"

#include "GEMPIC_hs_zigzag.H"

using namespace Particles;
using namespace GEMPIC_FDDeRhamComplex;
using namespace GEMPIC_Fields;
using namespace Time_Loop;

using ::testing::Mock;
using ::testing::Exactly;
using ::testing::_;

namespace {
    template<int degx, int degy, int degz>
    class MockSpline: public splines_at_particles<degx, degy, degz> {
        public:

        MOCK_METHOD(void, init_particles, (
            (amrex::GpuArray<amrex::Real, GEMPIC_SPACEDIM> &position),
            (amrex::GpuArray<amrex::Real, GEMPIC_SPACEDIM> const &plo),
            (amrex::GpuArray<amrex::Real, GEMPIC_SPACEDIM + 1> const &dxi)));
    };

    class HEParticleTest : public testing::Test {
            protected:

            static const int degX{1};
            static const int degY{1};
            static const int degZ{1};
            static const int degmw{2};

            static const int numSpec{1};
            static const int vDim{3};
            static const int nData{1};

            HSZigZagC2<vDim, numSpec, degX, degY, degZ, degmw, nData> hsZigZag();
    };

    TEST_F(HEParticleTest, NullTest) {
        MockSpline<degX, degY, degZ> mockSpline;

        // EXPECT_CALL(mockSpline, init_particles).Times(Exactly(1));

        // hsZigZag.apply_H_e_particle();

    }
}