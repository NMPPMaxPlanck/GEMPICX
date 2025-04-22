/*------------------------------------------------------------------------------
 Test Maxwell solver convergence rates with Finite Difference Hodges.

  Performs 10 time steps with dt = 0.0001 so that spatial discretization error dominates.
  Analytical solutions are
  in 1D:                           (     0    )
                          D(x,t) = ( cos(x-t) )
                                   ( cos(x-t) )

                                   (     0     )
                          B(x,t) = ( -cos(x-t) )
                                   (  cos(x-t) )

  in 2D:                           (  cos(x+y-sqrt(2)*t)         )
                        D(x,y,t) = ( -cos(x+y-sqrt(2)*t)         )
                                   ( -sqrt(2)*cos(x+y-sqrt(2)*t) )

                                   ( -cos(x+y-sqrt(2)*t)         )
                        B(x,y,t) = (  cos(x+y-sqrt(2)*t)         )
                                   ( -sqrt(2)*cos(x+y-sqrt(2)*t) )

  in 3D:                           (    cos(x+y+z-sqrt(3)*t)   )
                      D(x,y,z,t) = ( -2*cos(x+y+z-sqrt(3)*t)   )
                                   (    cos(x+y+z-sqrt(3)*t)   )

                                   (  sqrt(3)*cos(x+y+z-sqrt(3)*t) )
                      B(x,y,z,t) = (                0              )
                                   ( -sqrt(3)*cos(x+y+z-sqrt(3)*t) ).

  And epsilon = mu = 1.
  They are computed for 16 and 32 nodes in each direction. The convergence rate is estimated by
log_2 (error_16 / error_32)
------------------------------------------------------------------------------*/

#include <gtest/gtest.h>

#include <AMReX.H>

#include "GEMPIC_FDDeRhamComplex.H"
#include "TestUtils/GEMPIC_TestUtils.H"

using namespace Gempic;
using namespace Forms;

namespace
{
/**
 * @brief Test fixture. Sets up clean environment before each test of the SplineWithPrimitive class
 */
template <typename Degree>
class MaxwellFDSolverTest : public testing::Test
{
public:
    static constexpr int s_degX{3};
    static constexpr int s_degY{3};
    static constexpr int s_degZ{3};
    static constexpr int s_hodgeDegree{Degree()};

    inline static const int s_maxSplineDegree{
        AMREX_D_PICK(s_degX, std::max(s_degX, s_degY), std::max(std::max(s_degX, s_degY), s_degZ))};
};

template <int hodgeDegree, int maxSplineDegree>
void maxwellstrang_error (double& bError, double& dError, const int n)
{
    // Analytical solutions in every direction
#if (AMREX_SPACEDIM == 1)
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
#if (AMREX_SPACEDIM == 2)
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
#if (AMREX_SPACEDIM == 3)
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
    Gempic::Io::Parameters parameters{};
    const amrex::IntVect nCell{AMREX_D_DECL(n, n, n)};
    auto infra = Gempic::Test::Utils::get_compdom(nCell);

    // Project B and D to a primal and dual two form respectively
    constexpr int nVar = AMREX_SPACEDIM + 1; // x, y, z, t
    amrex::Array<amrex::ParserExecutor<nVar>, 3> funcD;
    amrex::Array<amrex::ParserExecutor<nVar>, 3> funcB;
    amrex::Array<amrex::Parser, 3> parserD;
    amrex::Array<amrex::Parser, 3> parserB;
    for (int i = 0; i < 3; ++i)
    {
        parserD[i].define(analyticalD[i]);
        parserD[i].registerVariables({AMREX_D_DECL("x", "y", "z"), "t"});
        funcD[i] = parserD[i].compile<nVar>();
    }

    for (int i = 0; i < 3; ++i)
    {
        parserB[i].define(analyticalB[i]);
        parserB[i].registerVariables({AMREX_D_DECL("x", "y", "z"), "t"});
        funcB[i] = parserB[i].compile<nVar>();
    }

    // Define the fields
    auto deRham = std::make_shared<FDDeRhamComplex>(infra, hodgeDegree, maxSplineDegree,
                                                    HodgeScheme::FDHodge);
    DeRhamField<Grid::dual, Space::face> D(deRham, funcD);
    DeRhamField<Grid::primal, Space::face> B(deRham, funcB);
    DeRhamField<Grid::primal, Space::edge> E(deRham);
    DeRhamField<Grid::dual, Space::edge> H(deRham);

    DeRhamField<Grid::primal, Space::face> curlE(deRham);
    DeRhamField<Grid::dual, Space::face> curlH(deRham);

    // Advance Maxwell's equations using second-order Hamiltonian Strang splitting
    const int nt = 10; // number of time steps
    const amrex::Real dt = 0.0001;
    for (int i = 0; i < nt; ++i)
    {
        // solve Faraday equation for a half step
        deRham->hodge(E, D);
        deRham->curl(curlE, E);
        curlE *= dt / 2;
        B -= curlE;

        // solve Ampère equation for a full step
        deRham->hodge(H, B);
        deRham->curl(curlH, H);
        curlH *= dt;
        D += curlH;

        // solve Faraday's equation again for a half step
        deRham->hodge(E, D);
        deRham->curl(curlE, E);
        curlE *= dt / 2;
        B -= curlE;
    }

    auto dr{infra.cell_size_array()};
    dError = 0;
    bError = 0;
    // Calculate max error of D and B
    for (int comp = 0; comp < 3; ++comp)
    {
        dError += max_error_midpoint<hodgeDegree>(infra.m_geom, funcD[comp], D.m_data[comp], dr, 2,
                                                  true, comp, nt * dt);
        bError += max_error_midpoint<hodgeDegree>(infra.m_geom, funcB[comp], B.m_data[comp], dr, 2,
                                                  false, comp, nt * dt);
    }
}

using MyTypes = ::testing::Types<std::integral_constant<int, 2>,
                                 std::integral_constant<int, 4>,
                                 std::integral_constant<int, 6>>;
TYPED_TEST_SUITE(MaxwellFDSolverTest, MyTypes);

TYPED_TEST(MaxwellFDSolverTest, ConvergenceTest)
{
    constexpr int hodgeDegree{TestFixture::s_hodgeDegree};
    constexpr int maxSplineDegree{TestFixture::s_maxSplineDegree};
    const int coarse = 16, fine = 32;
    amrex::Real bErrorCoarse, bErrorFine, dErrorCoarse, dErrorFine;
    amrex::Real tol = 0.05;

    maxwellstrang_error<hodgeDegree, maxSplineDegree>(bErrorCoarse, dErrorCoarse, coarse);
    maxwellstrang_error<hodgeDegree, maxSplineDegree>(bErrorFine, dErrorFine, fine);

    amrex::Real rateOfConvergenceB = std::log2(bErrorCoarse / bErrorFine);
    amrex::Real rateOfConvergenceD = std::log2(dErrorCoarse / dErrorFine);

    // test: strang splitting really results in order hodgeDegree
    EXPECT_NEAR(rateOfConvergenceB, hodgeDegree, tol);
    EXPECT_NEAR(rateOfConvergenceD, hodgeDegree, tol);
}
} // namespace