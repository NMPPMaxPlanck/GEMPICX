#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <AMReX.H>
#include <AMReX_ParmParse.H>

#include "GEMPIC_FDDeRhamComplex.H"
#include "GEMPIC_Fields.H"
#include "GEMPIC_Parameters.H"
#include "GEMPIC_ParticleGroups.H"
#include "GEMPIC_ParticleMeshCoupling.H"
#include "GEMPIC_SplineClass.H"
#include "GEMPIC_Splitting.H"
#include "TestUtils/GEMPIC_TestUtils.H"

using namespace Gempic;
using namespace Forms;
using namespace Particle;
using ::testing::Mock;

#define check_field(...) Gempic::Test::Utils::check_field(__FILE__, __LINE__, __VA_ARGS__)
#define compare_fields(...) Gempic::Test::Utils::compare_fields(__FILE__, __LINE__, __VA_ARGS__)

namespace
{
/**
 * @brief Test fixture. Sets up clean environment before each test of the SplineWithPrimitive class
 */
class HamiltonianSplittingOrderTest : public testing::Test
{
public:
    static constexpr int s_degX{3};
    static constexpr int s_degY{3};
    static constexpr int s_degZ{3};
    static constexpr int s_vDim{3};

    inline static const int s_maxSplineDegree{
        AMREX_D_PICK(s_degX, std::max(s_degX, s_degY), std::max(std::max(s_degX, s_degY), s_degZ))};
    inline static const int s_hodgeDegree{2};

    static const int s_nVar = GEMPIC_SPACEDIM + 1;  // x, y, z, t
    amrex::Parser m_parserRho, m_parserPhi;
    amrex::ParserExecutor<s_nVar> m_funcRho, m_funcPhi;

    static void SetUpTestSuite ()
    {
        const amrex::Vector<amrex::Real> domainLo{AMREX_D_DECL(0.0, 0.0, 0.0)};
        const amrex::Vector<amrex::Real> domainHi{AMREX_D_DECL(2 * M_PI, 2 * M_PI, 2 * M_PI)};
        const amrex::Vector<int> maxGridSize{AMREX_D_DECL(16, 16, 16)};
        const amrex::Vector<int> isPeriodic{AMREX_D_DECL(1, 1, 1)};

        /* Initialize the infrastructure */
        amrex::ParmParse pp;
        pp.addarr("domainLo", domainLo);
        pp.addarr("domainHi", domainHi);

        pp.addarr("maxGridSizeVector", maxGridSize);
        pp.addarr("isPeriodicVector", isPeriodic);
    }

    void maxwellstrang_error (int n, double &bError, double &dError)
    {
        // Analytical solutions in every direction (assuming k=1 in all directions)
#if (GEMPIC_SPACEDIM == 1)
        const amrex::Array<std::string, 3> analyticalD = {
            "0.",
            "cos(x-t)",
            "cos(x-t)",
        };

        const amrex::Array<std::string, 3> analyticalB = {
            "0.",
            "-cos(x-t)",
            "cos(x-t)",
        };
#endif
#if (GEMPIC_SPACEDIM == 2)
        const amrex::Array<std::string, 3> analyticalD = {
            "cos(x+y-sqrt(2.0)*t)",
            "-cos(x+y-sqrt(2.0)*t)",
            "-sqrt(2)*cos(x+y-sqrt(2.0)*t)",
        };

        const amrex::Array<std::string, 3> analyticalB = {
            "-cos(x+y-sqrt(2.0)*t)",
            "cos(x+y-sqrt(2.0)*t)",
            "-sqrt(2)*cos(x+y-sqrt(2.0)*t)",
        };
#endif
#if (GEMPIC_SPACEDIM == 3)
        const amrex::Array<std::string, 3> analyticalD = {
            "cos(x+y+z-sqrt(3.0)*t)",
            "-2*cos(x+y+z-sqrt(3.0)*t)",
            "cos(x+y+z-sqrt(3.0)*t)",
        };

        const amrex::Array<std::string, 3> analyticalB = {
            "sqrt(3)*cos(x+y+z-sqrt(3.0)*t)",
            "0.0",
            "-sqrt(3)*cos(x+y+z-sqrt(3.0)*t)",
        };
#endif
        // Initialize computational_domain
        const amrex::Vector<int> nCell{AMREX_D_DECL(n, n, n)};

        Gempic::Io::Parameters parameters{};
        parameters.set("nCellVector", nCell);
        ComputationalDomain infra;

        // Project B and D to a primal and dual two form respectively
        amrex::Array<amrex::ParserExecutor<s_nVar>, 3> funcD;
        amrex::Array<amrex::ParserExecutor<s_nVar>, 3> funcB;
        amrex::Array<amrex::Parser, 3> parserD;
        amrex::Array<amrex::Parser, 3> parserB;
        for (int i = 0; i < 3; ++i)
        {
            parserD[i].define(analyticalD[i]);
            parserD[i].registerVariables({AMREX_D_DECL("x", "y", "z"), "t"});
            funcD[i] = parserD[i].compile<s_nVar>();
        }

        for (int i = 0; i < 3; ++i)
        {
            parserB[i].define(analyticalB[i]);
            parserB[i].registerVariables({AMREX_D_DECL("x", "y", "z"), "t"});
            funcB[i] = parserB[i].compile<s_nVar>();
        }
        // Define the fields
        auto deRham = std::make_shared<FDDeRhamComplex>(infra, s_hodgeDegree, s_maxSplineDegree,
                                                        HodgeScheme::FDHodge);
        DeRhamField<Grid::dual, Space::face> D(deRham, funcD);
        DeRhamField<Grid::primal, Space::face> B(deRham, funcB);
        DeRhamField<Grid::primal, Space::edge> E(deRham);
        DeRhamField<Grid::dual, Space::edge> H(deRham);

        // Advance Maxwell's equations using second-order Hamiltonian Strang splitting
        Gempic::TimeLoop::OperatorHamilton<s_vDim, s_degX, s_degY, s_degZ, s_hodgeDegree>
            operatorHamilton;
        int nt = 2;  // number of time steps
        amrex::Real dt = 0.001;
        for (int i = 0; i < nt; ++i)
        {
            // solve Faraday equation for a half step
            operatorHamilton.apply_h_e_field(B, deRham, E, D, dt / 2);

            // solve Ampère equation for a full step
            operatorHamilton.apply_h_b(D, deRham, B, H, dt);

            // solve Faraday equation for a half step
            operatorHamilton.apply_h_e_field(B, deRham, E, D, dt / 2);
        }

        for (int comp = 0; comp < 3; ++comp)
        {
            dError += max_error_midpoint<s_hodgeDegree>(
                infra.m_geom, funcD[comp], D.m_data[comp],
                amrex::RealVect{AMREX_D_DECL(infra.m_dx[xDir], infra.m_dx[yDir], infra.m_dx[zDir])},
                2, true, comp, nt * dt);
        }

        for (int comp = 0; comp < 3; ++comp)
        {
            bError += max_error_midpoint<s_hodgeDegree>(
                infra.m_geom, funcB[comp], B.m_data[comp],
                amrex::RealVect{AMREX_D_DECL(infra.m_dx[xDir], infra.m_dx[yDir], infra.m_dx[zDir])},
                2, false, comp, nt * dt);
        }
    }
};

TEST_F(HamiltonianSplittingOrderTest, MaxwellTest)
{
    // Solves the sourceless Maxwell equations on a few time steps
    // Enables to test apply_h_b and apply_h_e_field
    const int coarse = 30, fine = 60;
    amrex::Real bErrorCoarse, bErrorFine, dErrorCoarse, dErrorFine;
    amrex::Real tol = 0.01;

    this->maxwellstrang_error(coarse, bErrorCoarse, dErrorCoarse);
    this->maxwellstrang_error(fine, bErrorFine, dErrorFine);

    amrex::Real rateOfConvergenceB = std::log2(bErrorCoarse / bErrorFine);
    amrex::Real rateOfConvergenceD = std::log2(dErrorCoarse / dErrorFine);

    // test: strang splitting really results in order 2
    EXPECT_NEAR(rateOfConvergenceB, 2, tol);
    EXPECT_NEAR(rateOfConvergenceD, 2, tol);
}

}  // namespace