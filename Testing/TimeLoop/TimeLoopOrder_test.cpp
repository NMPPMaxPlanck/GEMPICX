#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <AMReX.H>

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

namespace
{
/**
 * @brief Test fixture. Sets up clean environment before each test of the SplineWithPrimitive class
 */
class HamiltonianSplittingOrderTest : public testing::Test
{
public:
    static constexpr double s_sV{4.0};
    static constexpr double s_sOmega{3.0};

    static constexpr int s_degX{3};
    static constexpr int s_degY{3};
    static constexpr int s_degZ{3};
    static constexpr int s_vDim{3};

    inline static const int s_maxSplineDegree{
        AMREX_D_PICK(s_degX, std::max(s_degX, s_degY), std::max(std::max(s_degX, s_degY), s_degZ))};
    inline static const int s_hodgeDegree{2};

    static const int s_nVar = AMREX_SPACEDIM + 1; // x, y, z, t
    amrex::Parser m_parserRho, m_parserPhi;
    amrex::ParserExecutor<s_nVar> m_funcRho, m_funcPhi;

    ComputationalDomain set_params (Gempic::Io::Parameters& parameters, const amrex::IntVect& nCell)
    {
        parameters.set("sV", s_sV);
        parameters.set("sOmega", s_sOmega);

        /* Initialize the infrastructure */
        const amrex::Array<amrex::Real, AMREX_SPACEDIM> domainLo{AMREX_D_DECL(0.0, 0.0, 0.0)};
        const amrex::Array<amrex::Real, AMREX_SPACEDIM> domainHi{
            AMREX_D_DECL(2 * M_PI, 2 * M_PI, 2 * M_PI)};
        const amrex::IntVect maxGridSize{AMREX_D_DECL(16, 16, 16)};
        const amrex::Array<int, AMREX_SPACEDIM> isPeriodic{AMREX_D_DECL(1, 1, 1)};

        return ComputationalDomain(domainLo, domainHi, nCell, maxGridSize, isPeriodic,
                                   amrex::CoordSys::cartesian);
    }

    void maxwellstrang_error (double& bError,
                              double& dError,
                              int n,
                              amrex::Real const sOmegaSquared,
                              amrex::Real const sV)
    {
        // Analytical solutions in every direction (assuming k=1 in all directions)
#if (AMREX_SPACEDIM == 1)
        const amrex::Array<std::string, 3> analyticalD = {
            "1 / sOmegaSquared * 0.",
            "1 / sOmegaSquared * cos(x - sV * t)",
            "1 / sOmegaSquared * cos(x - sV * t)",
        };

        const amrex::Array<std::string, 3> analyticalB = {
            "1 / sV * 0.",
            "-1 / sV * cos(x - sV * t)",
            "1 / sV * cos(x - sV * t)",
        };
#endif
#if (AMREX_SPACEDIM == 2)
        const amrex::Array<std::string, 3> analyticalD = {
            "1 / sOmegaSquared * cos(x + y -sqrt(2.0) * sV * t)",
            "-1 / sOmegaSquared * cos(x + y -sqrt(2.0) * sV * t)",
            "-1 / sOmegaSquared * sqrt(2) * cos(x + y -sqrt(2.0) * sV * t)",
        };

        const amrex::Array<std::string, 3> analyticalB = {
            "-1 / sV * cos(x + y -sqrt(2.0) * sV * t)",
            "1 / sV * cos(x + y -sqrt(2.0) * sV * t)",
            "-1 / sV * sqrt(2) * cos(x + y -sqrt(2.0) * sV * t)",
        };
#endif
#if (AMREX_SPACEDIM == 3)
        const amrex::Array<std::string, 3> analyticalD = {
            "1 / sOmegaSquared * cos(x + y + z -sqrt(3.0) * sV * t)",
            "-2 / sOmegaSquared * cos(x + y + z -sqrt(3.0) * sV * t)",
            "1 / sOmegaSquared * cos(x + y + z -sqrt(3.0) * sV * t)",
        };

        const amrex::Array<std::string, 3> analyticalB = {
            "1 / sV * sqrt(3) * cos(x + y + z -sqrt(3.0) * sV * t)",
            "1 / sV * 0.0",
            "-1 / sV * sqrt(3) * cos(x + y + z -sqrt(3.0) * sV * t)",
        };
#endif
        // Initialize computational_domain
        const amrex::IntVect nCell{AMREX_D_DECL(n, n, n)};
        Gempic::Io::Parameters parameters;
        ComputationalDomain infra = set_params(parameters, nCell);

        // Project B and D to a primal and dual two form respectively
        amrex::Array<amrex::ParserExecutor<s_nVar>, 3> funcD;
        amrex::Array<amrex::ParserExecutor<s_nVar>, 3> funcB;
        amrex::Array<amrex::Parser, 3> parserD;
        amrex::Array<amrex::Parser, 3> parserB;
        for (int i = 0; i < 3; ++i)
        {
            parserD[i].define(analyticalD[i]);
            parserD[i].setConstant("sOmegaSquared", sOmegaSquared);
            parserD[i].setConstant("sV", sV);
            parserD[i].registerVariables({AMREX_D_DECL("x", "y", "z"), "t"});
            funcD[i] = parserD[i].compile<s_nVar>();
        }

        for (int i = 0; i < 3; ++i)
        {
            parserB[i].define(analyticalB[i]);
            parserB[i].setConstant("sOmegaSquared", sOmegaSquared);
            parserB[i].setConstant("sV", sV);
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
        Gempic::TimeLoop::OperatorHamilton<s_vDim, s_degX, s_degY, s_degZ> operatorHamilton;
        int nt = 2; // number of time steps
        amrex::Real dt = 0.001;
        for (int i = 0; i < nt; ++i)
        {
            // solve Faraday equation for a half step
            operatorHamilton.apply_h_e_field(B, E, deRham, D, dt / 2);

            // solve Ampère equation for a full step
            operatorHamilton.apply_h_b(D, deRham, B, H, dt);

            // solve Faraday equation for a half step
            operatorHamilton.apply_h_e_field(B, E, deRham, D, dt / 2);
        }

        dError = 0;
        for (int comp = 0; comp < 3; ++comp)
        {
            dError += max_error_midpoint<s_hodgeDegree>(infra.m_geom, funcD[comp], D.m_data[comp],
                                                        infra.geometry().CellSizeArray(), 2, true,
                                                        comp, nt * dt);
        }

        bError = 0;
        for (int comp = 0; comp < 3; ++comp)
        {
            bError += max_error_midpoint<s_hodgeDegree>(infra.m_geom, funcB[comp], B.m_data[comp],
                                                        infra.geometry().CellSizeArray(), 2, false,
                                                        comp, nt * dt);
        }
    }
};

TEST_F(HamiltonianSplittingOrderTest, MaxwellTest)
{
    // Solves the sourceless Maxwell equations on a few time steps
    // Enables to test apply_h_b and apply_h_e_field
    // including scaling parameters (i.e., epsilon_0 and mu_0 != 1)
    // D = 1/s_omega^2 * E
    // s_omega^2 = 3
    // s_v = 4
    // 1D
    // E = [0, cos(x-s_v*t), cos(x-s_v*t)]
    // B = 1/s_v * [0, -cos(x-s_v*t), cos(x-s_v*t)]
    // 2D
    // E = [cos(x+y-sqrt(2)*s_v*t), -cos(x+y-sqrt(2)*s_v*t), -sqrt(2)*cos(x+y-sqrt(2)*s_v*t)]
    // B = 1/s_v * [cos(x+y-sqrt(2)*s_v*t), cos(x+y-sqrt(2)*s_v*t), -sqrt(2)*cos(x+y-sqrt(2)*s_v*t)]
    // 3D
    // E = [cos(x+y+z-sqrt(3)*s_v*t), -2*cos(x+y+z-sqrt(3)*s_v*t), cos(x+y+z-sqrt(3)*s_v*t)]
    // B = 1/s_V * [sqrt(3)*cos(x+y+z-sqrt(3)*s_v*t), 0, -sqrt(3)*cos(x+y+z-sqrt(3)*s_v*t)]
    const int coarse = 30, fine = 60;
    amrex::Real bErrorCoarse, bErrorFine, dErrorCoarse, dErrorFine;
    amrex::Real tol = 0.01;
    const amrex::Real sOmegaSquared = s_sOmega * s_sOmega;
    const amrex::Real sV = s_sV;

    this->maxwellstrang_error(bErrorCoarse, dErrorCoarse, coarse, sOmegaSquared, sV);
    this->maxwellstrang_error(bErrorFine, dErrorFine, fine, sOmegaSquared, sV);

    amrex::Real rateOfConvergenceB = std::log2(bErrorCoarse / bErrorFine);
    amrex::Real rateOfConvergenceD = std::log2(dErrorCoarse / dErrorFine);

    // test: strang splitting really results in order 2
    EXPECT_NEAR(rateOfConvergenceB, 2, tol);
    EXPECT_NEAR(rateOfConvergenceD, 2, tol);
}

} // namespace
