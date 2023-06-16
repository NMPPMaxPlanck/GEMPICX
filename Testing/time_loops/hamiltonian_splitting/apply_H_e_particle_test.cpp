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

    template <int vdim, int degx, int degy, int degz>
    AMREX_GPU_HOST_DEVICE amrex::GpuArray<amrex::Real, vdim> mock_evaluate_efield(
        splines_at_particles<degx, degy, degz> &spline,
        amrex::GpuArray<amrex::Array4<amrex::Real>, vdim> const &eArray)
    {
        amrex::GpuArray<amrex::Real, vdim> fields;
        for (int comp = 0; comp < vdim; comp++)
        {
            fields[comp] = 0.;
        }
        return fields;
    };

    template <int vdim>
    AMREX_GPU_HOST_DEVICE amrex::GpuArray<amrex::Real, vdim> mock_push_v_efield(
        amrex::GpuArray<amrex::Real, vdim> vel, amrex::Real dt,
        amrex::Real chargemass,  // charge/mass
        amrex::GpuArray<amrex::Real, vdim> &Ep)
    {
        amrex::GpuArray<amrex::Real, vdim> newPos;
        return (newPos);
    };

    class HEParticleTest : public testing::Test {
        protected:

        static const int degX{1};
        static const int degY{1};
        static const int degZ{1};
        static const int degmw{2};

        static const int numSpec{1};
        static const int vDim{3};
        static const int spec{0};
        static const int nData{1};
        const int Nghost{GEMPIC_TestUtils::initNGhost(degX, degY, degZ)};

        double charge{1};
        double mass{1};

        computational_domain infra;
        amrex::GpuArray<std::unique_ptr<particle_groups<vDim>>, numSpec> particleGroup;

        HSZigZagC2<vDim, numSpec, degX, degY, degZ, degmw, nData> hsZigZag();

        // virtual void SetUp() will be called before each test is run.
        void SetUp() override {
            /* Initialize the infrastructure */
            const amrex::RealBox realBox({AMREX_D_DECL(0.0, 0.0, 0.0)},
                                            {AMREX_D_DECL(10.0, 10.0, 10.0)});
            const amrex::IntVect nCell{AMREX_D_DECL(10, 10, 10)};
            const amrex::IntVect maxGridSize{AMREX_D_DECL(10, 10, 10)};

            // This class does the same as GEMPIC_parameters.H and needs to be urgently redesigned.
            // {1, 1, 1} represents periodicity, has different types than Params and gempic_parameters.
            infra.initialize_computational_domain(nCell, maxGridSize, {1, 1, 1}, realBox);
            
            // particles
            for (int spec{0}; spec < numSpec; spec++)
            {
                particleGroup[spec] =
                    std::make_unique<particle_groups<vDim>>(charge, mass, infra);
            }
        }
    };

    TEST_F(HEParticleTest, NullTest) {
        // Adding particle to one cell
        const int numParticles{1};
        amrex::Array<amrex::GpuArray<amrex::Real, GEMPIC_SPACEDIM>, numParticles> positions{*infra.geom.ProbLo()};
        amrex::Array<amrex::Real, numParticles> weights{1};
        GEMPIC_TestUtils::addSingleParticles<vDim, numSpec, numParticles>(particleGroup, infra, weights, positions);

        // (default) charge correctly transferred from addSingleParticles
        EXPECT_EQ(1, particleGroup[0]->getCharge()); 

        MockSpline<degX, degY, degZ> mockSpline;
        // auto original_evaluate_efield = evaluate_efield<vDim, degX, degY, degZ>;
        // evaluate_efield<3, 1, 1, 1> = mock_evaluate_efield<3, 1, 1, 1>;

        // EXPECT_CALL(mockSpline, init_particles).Times(Exactly(1));

        bool particleLoopRun{false};
        for (amrex::ParIter<0, 0, vDim + 1, 0> pti(*particleGroup[spec], 0); pti.isValid(); ++pti)
        {
            particleLoopRun = true;

            const long np{pti.numParticles()};
            EXPECT_EQ(1, np); // Only one particle added by addSingleParticles

            // hsZigZag.apply_H_e_particle();
        }
        ASSERT_TRUE(particleLoopRun);

    }
}