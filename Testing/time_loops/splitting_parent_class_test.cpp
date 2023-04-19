#include <AMReX.H>
#include <GEMPIC_particle_groups.H>
#include <GEMPIC_FDDeRhamComplex.H>
#include <GEMPIC_Fields.H>
#include <GEMPIC_particle_groups.H>
#include "gtest/gtest.h"
#include "test_utils/GEMPIC_test_utils.H"
#include "gmock/gmock.h"

#include "GEMPIC_Splitting.H"

using namespace Particles;
using namespace GEMPIC_FDDeRhamComplex;
using namespace GEMPIC_Fields;
using namespace Time_Loop;

using ::testing::Mock;
using ::testing::Exactly;
using ::testing::_;

namespace {

    template<int vdim, int numspec, int degx, int degy, int degz, int degmw,
         int ndata, bool electromagnetic, bool profiling>
    class MockSplitting : public Time_Loop::Splitting<vdim, numspec, degx, degy, degz, degmw, ndata, electromagnetic, profiling> {
        public:

        // This mock of the treat_partiles method is not used in the tests. It is needed 
        // because treat_particles is virtual and if it is not implemented we can not 
        // instantiate the Splitting mock.
        MOCK_METHOD(void, treat_particles, (
            (std::shared_ptr<FDDeRhamComplex> deRham),
            (DeRhamField<Grid::primal, Space::edge>& E),
            (DeRhamField<Grid::primal, Space::face>& B),
            (DeRhamField<Grid::dual, Space::face>& D),
            (DeRhamField<Grid::dual, Space::face>& J),
            computational_domain infra,
            const amrex::Real dt,
            (amrex::GpuArray<std::unique_ptr<particle_groups<vdim, ndata>>, numspec>& partGr)),
            (override));

        MOCK_METHOD(void, time_step, (
            (std::shared_ptr<FDDeRhamComplex> deRham),
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
            (amrex::GpuArray<std::unique_ptr<particle_groups<vdim, ndata>>, numspec>& partGr)),
            (override));
    };

    class SplittingTest : public testing::Test {
        protected:

        // Degree of splines in each direction
        static const int degX{1};
        static const int degY{1};
        static const int degZ{1};
        static const int degmw{2};

        static const int numSpec{1};
        // Number of velocity dimensions.
        static const int vDim{3};

        static const int nData{1};
        const int strangOrder{2};
        const amrex::Real dt{1};

        Parameters params;
        std::shared_ptr<FDDeRhamComplex> deRham;

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
            deRham = std::make_shared<FDDeRhamComplex>(params);
        }
    };

    TEST_F(SplittingTest, NullTest) {
        const int nSteps{1};
        MockSplitting<vDim, numSpec, degX, degY, degZ, degmw, nData, electromagnetic, profiling> mockSplitting;

        EXPECT_CALL(mockSplitting, time_step).Times(Exactly(1));

        // Declare the fields 
        DeRhamField<Grid::primal, Space::edge> E(deRham);
        DeRhamField<Grid::primal, Space::face> B(deRham);
        DeRhamField<Grid::dual, Space::face> D(deRham);
        DeRhamField<Grid::dual, Space::edge> H(deRham);
        DeRhamField<Grid::dual, Space::face> J(deRham);
        DeRhamField<Grid::dual, Space::cell> rho(deRham);

        mockSplitting.time_loop(deRham, rho, E, B, D, H, J, infra, dt, particleGroup, nSteps, strangOrder);
    }

    TEST_F(SplittingTest, FiveSteps) {
        const int nSteps{5};
        MockSplitting<vDim, numSpec, degX, degY, degZ, degmw, nData, electromagnetic, profiling> mockSplitting;

        // Declare the fields 
        DeRhamField<Grid::primal, Space::edge> E(deRham);
        DeRhamField<Grid::primal, Space::face> B(deRham);
        DeRhamField<Grid::dual, Space::face> D(deRham);
        DeRhamField<Grid::dual, Space::edge> H(deRham);
        DeRhamField<Grid::dual, Space::face> J(deRham);
        DeRhamField<Grid::dual, Space::cell> rho(deRham);

        EXPECT_CALL(mockSplitting, time_step(deRham, _, _, _, _, _, _, _, _, _, _, _, 1, _)).Times(Exactly(5));

        mockSplitting.time_loop(deRham, rho, E, B, D, H, J, infra, dt, particleGroup, nSteps, strangOrder);
    }
}