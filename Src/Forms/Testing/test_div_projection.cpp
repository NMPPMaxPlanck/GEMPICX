/*------------------------------------------------------------------------------
 Test the discrete div D on both primal and dual grid for
 periodic boundary conditions.
    The following errors are computed: max |R_3 div f - D R_2 f|, i.e. the
    analytical f is projected to a discrete 2-form followed by the discrete
    divergence. The result is compared with the restriction of the analytical
    divergence of f to a discrete 3-form.
    Test passes if all projected DOFs are within 1e-15 of the analytical value.
------------------------------------------------------------------------------*/
#include "GEMPIC_FDDeRhamComplex.H"
#include "GEMPIC_Fields.H"
#include "GEMPIC_Parameters.H"

using namespace Gempic::Forms;

int main (int argc, char *argv[])
{
    amrex::Initialize(argc, argv);
    Gempic::Io::Parameters parameters{};
    {
        // error tolerance
        const amrex::Real tol = 1e-15;

        // number of quadrature points
        int gaussNodes = 6;

        // const amrex::RealBox realBox({AMREX_D_DECL(-M_PI + 0.3, -M_PI + 0.6, -M_PI +
        // 0.4)},{AMREX_D_DECL(M_PI + 0.3, M_PI + 0.6, M_PI + 0.4)});
        const amrex::Vector<amrex::Real> domainLo{
            AMREX_D_DECL(-M_PI + 0.3, -M_PI + 0.6, -M_PI + 0.4)};
        const amrex::Vector<amrex::Real> k{AMREX_D_DECL(1.0, 1.0, 1.0)};
        const amrex::Vector<int> nCell{AMREX_D_DECL(9, 11, 7)};
        const amrex::Vector<int> maxGridSize{AMREX_D_DECL(3, 4, 5)};
        const amrex::Array<int, GEMPIC_SPACEDIM> isPeriodic{AMREX_D_DECL(1, 1, 1)};
        const int hodgeDegree = 2;
        const int maxSplineDegree = 1;

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

        // Declare the fields
        DeRhamField<Grid::primal, Space::face> B(deRham);
        DeRhamField<Grid::primal, Space::cell> divB(deRham);

        // Parse analytical fields and and initialize func
#if (GEMPIC_SPACEDIM == 1)
        const amrex::Array<std::string, 3> analyticalB = {
            "-cos(x)",
            "sin(x)",
            "-sin(x)",
        };
#endif
#if (GEMPIC_SPACEDIM == 2)
        const amrex::Array<std::string, 3> analyticalB = {
            "-cos(x)*cos(y)",
            "sin(x)*cos(y)",
            "-sin(x)*sin(y)",
        };
#endif

#if (GEMPIC_SPACEDIM == 3)
        const amrex::Array<std::string, 3> analyticalB = {
            "-cos(x)*cos(y)*sin(z)",
            "sin(x)*cos(y)*sin(z)",
            "-sin(x)*sin(y)*cos(z)",
        };
#endif
        const int nVar = GEMPIC_SPACEDIM + 1;  // x, y, z, t
        amrex::Array<amrex::ParserExecutor<nVar>, 3> funcB;
        amrex::Array<amrex::Parser, 3> parserB;
        for (int i = 0; i < 3; ++i)
        {
            parserB[i].define(analyticalB[i]);
            parserB[i].registerVariables({AMREX_D_DECL("x", "y", "z"), "t"});
            funcB[i] = parserB[i].compile<nVar>();
        }

        // Compute the projection of the field
        deRham->projection(funcB, 0.0, B, gaussNodes);

        // Calculate divB from B
        deRham->div(B, divB);

        // Analytical divB
#if (GEMPIC_SPACEDIM == 1)
        const std::string analyticalDivB = "sin(x)";
#endif
#if (GEMPIC_SPACEDIM == 2)
        const std::string analyticalDivB = "sin(x)*cos(y) - sin(x)*sin(y)";
#endif
#if (GEMPIC_SPACEDIM == 3)
        const std::string analyticalDivB =
            "sin(x)*cos(y)*sin(z) - sin(x)*sin(y)*sin(z) + sin(x)*sin(y)*sin(z)";
#endif

        amrex::ParserExecutor<nVar> funcDivB;
        amrex::Parser parserDivB;

        parserDivB.define(analyticalDivB);
        parserDivB.registerVariables({AMREX_D_DECL("x", "y", "z"), "t"});
        funcDivB = parserDivB.compile<nVar>();

        // Declare rho
        DeRhamField<Grid::primal, Space::cell> rho(deRham);
        deRham->projection(funcDivB, 0.0, rho, gaussNodes);

        // Calculate errorE
        bool passRho{false};
        DeRhamField<Grid::primal, Space::cell> errorRho(deRham);

        for (amrex::MFIter mfi(rho.m_data); mfi.isValid(); ++mfi)
        {
            const amrex::Box &bx = mfi.validbox();
            amrex::Array4<amrex::Real> const &analyticalMF = (rho.m_data)[mfi].array();
            amrex::Array4<amrex::Real> const &projectionMF = (divB.m_data)[mfi].array();
            amrex::Array4<amrex::Real> const &errorRhoMF = (errorRho.m_data)[mfi].array();

            ParallelFor(
                bx, [=] AMREX_GPU_DEVICE(int i, int j, int k)
                { errorRhoMF(i, j, k) = std::abs(projectionMF(i, j, k) - analyticalMF(i, j, k)); });
        }

        // Test div projection for Dual
        // Declare the fields
        DeRhamField<Grid::dual, Space::face> D(deRham);
        DeRhamField<Grid::dual, Space::cell> divD(deRham);

        // Parse analytical fields and and initialize func

#if (GEMPIC_SPACEDIM == 1)
        const amrex::Array<std::string, 3> analyticalD = {
            "-cos(x)",
            "sin(x)",
            "-sin(x)",
        };
#endif
#if (GEMPIC_SPACEDIM == 2)
        const amrex::Array<std::string, 3> analyticalD = {
            "-cos(x)*cos(y)",
            "sin(x)*cos(y)",
            "-sin(x)*sin(y)",
        };
#endif

#if (GEMPIC_SPACEDIM == 3)
        const amrex::Array<std::string, 3> analyticalD = {
            "-cos(x)*cos(y)*sin(z)",
            "sin(x)*cos(y)*sin(z)",
            "-sin(x)*sin(y)*cos(z)",
        };
#endif
        amrex::Array<amrex::ParserExecutor<nVar>, 3> funcD;
        amrex::Array<amrex::Parser, 3> parserD;
        for (int i = 0; i < 3; ++i)
        {
            parserD[i].define(analyticalD[i]);
            parserD[i].registerVariables({AMREX_D_DECL("x", "y", "z"), "t"});
            funcD[i] = parserD[i].compile<nVar>();
        }

        // Compute the projection of the field
        deRham->projection(funcD, 0.0, D, gaussNodes);

        // Calculate divB from B
        deRham->div(D, divD);

        // Analytical divB
#if (GEMPIC_SPACEDIM == 1)
        const std::string analyticalDivD = "sin(x)";
#endif
#if (GEMPIC_SPACEDIM == 2)
        const std::string analyticalDivD = "sin(x)*cos(y) - sin(x)*sin(y)";
#endif
#if (GEMPIC_SPACEDIM == 3)
        const std::string analyticalDivD =
            "sin(x)*cos(y)*sin(z) - sin(x)*sin(y)*sin(z) + sin(x)*sin(y)*sin(z)";
#endif

        amrex::ParserExecutor<nVar> funcDivD;
        amrex::Parser parserDivD;

        parserDivD.define(analyticalDivD);
        parserDivD.registerVariables({AMREX_D_DECL("x", "y", "z"), "t"});
        funcDivD = parserDivD.compile<nVar>();

        // Declare rho
        DeRhamField<Grid::dual, Space::cell> rhoDual(deRham);
        deRham->projection(funcDivD, 0.0, rhoDual, gaussNodes);

        // Calculate errorE
        bool passRhoDual{false};
        DeRhamField<Grid::dual, Space::cell> errorRhoDual(deRham);

        for (amrex::MFIter mfi(errorRhoDual.m_data); mfi.isValid(); ++mfi)
        {
            const amrex::Box &bx = mfi.validbox();
            amrex::Array4<amrex::Real> const &analyticalMF = (rhoDual.m_data)[mfi].array();
            amrex::Array4<amrex::Real> const &projectionMF = (divD.m_data)[mfi].array();
            amrex::Array4<amrex::Real> const &errorRhoMF = (errorRhoDual.m_data)[mfi].array();

            ParallelFor(
                bx, [=] AMREX_GPU_DEVICE(int i, int j, int k)
                { errorRhoMF(i, j, k) = std::abs(projectionMF(i, j, k) - analyticalMF(i, j, k)); });
        }

        amrex::Real errorRhoNorm0 = errorRho.m_data.norm0();
        amrex::Real errorRhoDualNorm0 = errorRhoDual.m_data.norm0();

        /*
        amrex::Print() << "errorRho_norm0 = " << errorRho_norm0 << std::endl;
        amrex::Print() << "errorRhoDual_norm0 = " << errorRhoDual_norm0 << std::endl;
        */

        passRho = (errorRhoNorm0 < tol);
        passRhoDual = (errorRhoDualNorm0 < tol);

        if (passRho == true && passRhoDual == true)
        {
            amrex::PrintToFile("test_div_projection.output") << std::endl;
            amrex::PrintToFile("test_div_projection.output")
                << GEMPIC_SPACEDIM << "D test passed" << std::endl;
        }
        else
        {
            amrex::PrintToFile("test_div_projection.output") << std::endl;
            amrex::PrintToFile("test_div_projection.output")
                << GEMPIC_SPACEDIM << "D test failed" << std::endl;
        }

        amrex::PrintToFile("test_div_projection.output")
            << "max Error divB = " << errorRhoNorm0 << std::endl;
        amrex::PrintToFile("test_div_projection.output")
            << "max Error divD = " << errorRhoDualNorm0 << std::endl;

        if (amrex::ParallelDescriptor::MyProc() == 0)
        {
            std::rename("test_div_projection.output.0", "test_div_projection.output");
        }
        amrex::Print() << "IOProcessorNumber " << amrex::ParallelDescriptor::IOProcessorNumber()
                       << std::endl;
    }
    amrex::Finalize();
}
