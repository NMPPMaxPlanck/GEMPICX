#include <AMReX.H>
#include <GEMPIC_particle_groups.H>
#include <GEMPIC_FDDeRhamComplex.H>
#include <GEMPIC_Fields.H>
#include <GEMPIC_particle_groups.H>
#include "gtest/gtest.h"
#include "test_utils/GEMPIC_test_utils.H"
#include "gmock/gmock.h"

#include "GEMPIC_hs_zigzag.H"

using namespace Particles;
using namespace GEMPIC_FDDeRhamComplex;
using namespace GEMPIC_Fields;
using namespace Time_Loop;

using ::testing::Mock;

namespace {

    template<int vdim, int numspec, int degx, int degy, int degz, int degmw,
         int ndata, bool electromagnetic, bool profiling>
    class MockSplitting : public Time_Loop::HSZigZagC2<vdim, numspec, degx, degy, degz, degmw, ndata, electromagnetic, profiling> {
        public:

        MOCK_METHOD(void, time_step, ((std::shared_ptr<FDDeRhamComplex> deRham),
                           (DeRhamField<Grid::dual, Space::cell>& rho),
                           (DeRhamField<Grid::primal, Space::edge>& E),
                           (DeRhamField<Grid::primal, Space::face>& B),
                           (DeRhamField<Grid::dual, Space::face>& D),
                           (DeRhamField<Grid::dual, Space::edge>& H),
                           (DeRhamField<Grid::dual, Space::face>& J),
                           (DeRhamField<Grid::primal, Space::face>& auxPrimalF2),
                           (DeRhamField<Grid::primal, Space::face>& auxPrimalF2_2),
                           (DeRhamField<Grid::dual, Space::face>& auxDualF2),
                           (DeRhamField<Grid::dual, Space::face>& auxDualF2_2),
                           (computational_domain infra),
                           (const amrex::Real dt),
                           (amrex::GpuArray<std::unique_ptr<particle_groups<vdim, ndata>>, numspec>& partGr)));
    };

    class SplittingTest : public testing::Test {
        protected:

        // Degree of splines in each direction
        static const int degX{1};
        static const int degY{1};
        static const int degZ{1};
        static const int degmw{2};

        // Number of species (second species only used for DoubleParticleMultipleSpecies)
        static const int numSpec{2};
        // Number of velocity dimensions.
        static const int vDim{3};

        static const int nData{1};
        const int nSteps{1};
        const int strangOrder{2};
        const amrex::Real dt{1};

        Parameters params;

        static const bool electromagnetic{true};
        static const bool profiling{false};
        computational_domain infra;
        amrex::GpuArray<std::unique_ptr<particle_groups<vDim>>, numSpec> particleGroup;

        void SetUp() override {
            /* Initialize the infrastructure */
            const amrex::RealBox realBox({AMREX_D_DECL(0.0, 0.0, 0.0)},
                                            {AMREX_D_DECL(10.0, 10.0, 10.0)});
            const amrex::IntVect nCell{AMREX_D_DECL(10, 10, 10)};
            const amrex::IntVect maxGridSize{AMREX_D_DECL(10, 10, 10)};
            const amrex::Array<int, GEMPIC_SPACEDIM> isPeriodic{1, 1, 1};
            const int hodgeDegree{2};

            infra.initialize_computational_domain(nCell, maxGridSize, {1, 1, 1}, realBox);

            params = Parameters(realBox, nCell, maxGridSize, isPeriodic, hodgeDegree);

        }
    };

    TEST_F(SplittingTest, NullTest) {
        MockSplitting<vDim, numSpec, degX, degY, degZ, degmw, nData, electromagnetic, profiling> mockSplitting;

        // Initialize the De Rham Complex
        std::shared_ptr<FDDeRhamComplex> deRham = std::make_shared<FDDeRhamComplex>(params);

        // Declare the fields 
        DeRhamField<Grid::dual, Space::face> D(deRham);
        DeRhamField<Grid::dual, Space::face> J(deRham);
        DeRhamField<Grid::primal, Space::face> B(deRham);
        DeRhamField<Grid::primal, Space::edge> E(deRham);
        DeRhamField<Grid::dual, Space::edge> H(deRham);
        DeRhamField<Grid::dual, Space::cell> rho(deRham);
        DeRhamField<Grid::primal, Space::node> phi(deRham);

        HSZigZagC2<vDim, numSpec, degX, degY, degZ, degmw, nData> time_looper;
        time_looper.time_loop(deRham, rho, E, B, D, H, J, infra, dt, particleGroup, nSteps, strangOrder);
    }
}