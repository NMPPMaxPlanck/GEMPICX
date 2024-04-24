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

#include "GEMPIC_ComputationalDomain.H"
#include "GEMPIC_FDDeRhamComplex.H"
#include "GEMPIC_Fields.H"
#include "GEMPIC_Interpolation.H"
#include "GEMPIC_Parameters.H"

using namespace Gempic::Forms;

const int hodgeDegree = 2;
const int maxSplineDegree = 1;

std::tuple<amrex::Real, amrex::Real> maxwell (const int n)
{
    BL_PROFILE("Gempic::Forms::std::tuple<amrex::Real, amrex::Real> maxwell()");
    /* Initialize the infrastructure */
    // const amrex::RealBox realBox({AMREX_D_DECL(-M_PI + 0.3, -M_PI + 0.6, -M_PI +
    // 0.4)},{AMREX_D_DECL(M_PI + 0.3, M_PI + 0.6, M_PI + 0.4)});
    const amrex::Vector<amrex::Real> domainLo{AMREX_D_DECL(-M_PI + 0.3, -M_PI + 0.6, -M_PI + 0.4)};
    const amrex::Vector<amrex::Real> k{AMREX_D_DECL(1.0, 1.0, 1.0)};
    const amrex::Vector<int> nCell{AMREX_D_DECL(n, n, n)};
    const amrex::Vector<int> maxGridSize{AMREX_D_DECL(8, 9, 7)};
    const amrex::Array<int, GEMPIC_SPACEDIM> isPeriodic{AMREX_D_DECL(1, 1, 1)};

    const amrex::Real dt = 0.0001;
    const int nt = 10;

    Gempic::Io::Parameters parameters{};
    parameters.set("domainLo", domainLo);
    parameters.set("k", k);
    parameters.set("nCellVector", nCell);
    parameters.set("maxGridSizeVector", maxGridSize);
    parameters.set("isPeriodicVector", isPeriodic);

    // Initialize computational_domain
    Gempic::ComputationalDomain infra;

    // Initialize the De Rham Complex
    auto deRham = std::make_shared<FDDeRhamComplex>(infra, hodgeDegree, maxSplineDegree,
                                                    HodgeScheme::FDHodge);

    const amrex::Geometry geom = infra.m_geom;

    // Declare the fields
    DeRhamField<Grid::dual, Space::face> D(deRham);
    DeRhamField<Grid::primal, Space::face> B(deRham);
    DeRhamField<Grid::primal, Space::edge> E(deRham);
    DeRhamField<Grid::dual, Space::edge> H(deRham);

    DeRhamField<Grid::dual, Space::cell> divD(deRham);
    DeRhamField<Grid::primal, Space::cell> divB(deRham);

    DeRhamField<Grid::primal, Space::face> curlE(deRham);
    DeRhamField<Grid::dual, Space::face> curlH(deRham);

    // Analytical solutions in every direction
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

    // Project B and D to a primal and dual two form respectively
    const int nVar = GEMPIC_SPACEDIM + 1;  // x, y, z, t
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

    deRham->projection(funcD, 0.0, D);
    deRham->projection(funcB, 0.0, B);

    // Advance Maxwell's equations using second-order Hamiltonian Strang splitting
    for (int i = 0; i < nt; ++i)
    {
        // solve Faraday equation for a half step
        deRham->hodge(D, E);
        deRham->curl(E, curlE);
        curlE *= dt / 2;
        B -= curlE;

        // solve Ampère equation for a full step
        deRham->hodge(B, H);
        deRham->curl(H, curlH);
        curlH *= dt;
        D += curlH;

        // solve Faraday's equation again for a half step
        deRham->hodge(D, E);
        deRham->curl(E, curlE);
        curlE *= dt / 2;
        B -= curlE;
    }

    deRham->div(D, divD);
    deRham->div(B, divB);
    amrex::Print() << "Gauss errors: max(div D) = " << divD.m_data.norm0()
                   << ", max(div B) = " << divB.m_data.norm0() << std::endl;

    // Calculate max error of D and B
    amrex::Real dError = 0;
    amrex::Real bError = 0;
    for (int comp = 0; comp < 3; ++comp)
    {
        dError += max_error_midpoint<hodgeDegree>(
            geom, funcD[comp], D.m_data[comp],
            amrex::RealVect{AMREX_D_DECL(infra.m_dx[xDir], infra.m_dx[yDir], infra.m_dx[zDir])}, 2,
            true, comp, nt * dt);
    }
    for (int comp = 0; comp < 3; ++comp)
    {
        bError += max_error_midpoint<hodgeDegree>(
            geom, funcB[comp], B.m_data[comp],
            amrex::RealVect{AMREX_D_DECL(infra.m_dx[xDir], infra.m_dx[yDir], infra.m_dx[zDir])}, 2,
            false, comp, nt * dt);
    }

    return std::make_tuple(dError, bError);
}

int main (int argc, char *argv[])
{
    amrex::Initialize(argc, argv);
    {
        BL_PROFILE("main()");

        const int coarse = 16;
        const int fine = 32;
        amrex::Real errorCoarseD;
        amrex::Real errorCoarseB;
        amrex::Real errorFineD;
        amrex::Real errorFineB;

        amrex::GpuArray<amrex::Real, 2> rate;

        std::tie(errorCoarseD, errorCoarseB) = maxwell(coarse);
        std::tie(errorFineD, errorFineB) = maxwell(fine);
        rate[0] = std::log2(errorCoarseD / errorFineD);
        rate[1] = std::log2(errorCoarseB / errorFineB);

        amrex::PrintToFile("test_maxwell_FDsolver_degree_2.output") << std::endl;
        amrex::PrintToFile("test_maxwell_FDsolver_degree_2.output")
            << GEMPIC_SPACEDIM << "D Maxwell degree 2 convergence test:" << std::endl;
        amrex::PrintToFile("test_maxwell_FDsolver_degree_2.output") << std::endl;
        amrex::PrintToFile("test_maxwell_FDsolver_degree_2.output").SetPrecision(3)
            << "D field rate: " << rate[0] << std::endl;
        amrex::PrintToFile("test_maxwell_FDsolver_degree_2.output").SetPrecision(3)
            << "B field rate: " << rate[1] << std::endl;

        if (amrex::ParallelDescriptor::MyProc() == 0)
        {
            std::rename("test_maxwell_FDsolver_degree_2.output.0",
                        "test_maxwell_FDsolver_degree_2.output");
        }
    }
    amrex::Finalize();
}
