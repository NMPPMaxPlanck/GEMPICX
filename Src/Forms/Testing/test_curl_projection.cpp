/*------------------------------------------------------------------------------
 Test the discrete curl C on both primal and dual grid for
 periodic boundary conditions.
    The following errors are computed: max |R_2 curl f - C R_1 f|, i.e. the
    analytical f is projected to a discrete 1-form followed by the discrete
    curl. The result is compared with the restriction of the analytical
    curl of f to a discrete 2-form.
    Test passes if all projected DOFs are within 1e-15 of the analytical value.
------------------------------------------------------------------------------*/
#include <AMReX_ParmParse.H>

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

        /* Initialize the infrastructure */
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
        DeRhamField<Grid::primal, Space::edge> E(deRham);
        DeRhamField<Grid::primal, Space::face> curlE(deRham);

        // Parse analytical fields and and initialize func
#if (GEMPIC_SPACEDIM == 1)
        const amrex::Array<std::string, 3> analyticalE = {
            "-cos(x)",
            "sin(x)",
            "-sin(x)",
        };
#endif
#if (GEMPIC_SPACEDIM == 2)
        const amrex::Array<std::string, 3> analyticalE = {
            "-cos(x)*cos(y)",
            "sin(x)*cos(y)",
            "-sin(x)*sin(y)",
        };
#endif

#if (GEMPIC_SPACEDIM == 3)
        const amrex::Array<std::string, 3> analyticalE = {
            "-cos(x)*cos(y)*sin(z)",
            "sin(x)*cos(y)*sin(z)",
            "-sin(x)*sin(y)*cos(z)",
        };
#endif

        const int nVar = GEMPIC_SPACEDIM + 1;  // x, y, z, t
        amrex::Array<amrex::ParserExecutor<nVar>, 3> func;
        amrex::Array<amrex::Parser, 3> parser;
        for (int i = 0; i < 3; ++i)
        {
            parser[i].define(analyticalE[i]);
            parser[i].registerVariables({AMREX_D_DECL("x", "y", "z"), "t"});
            func[i] = parser[i].compile<nVar>();
        }

        // Compute the projection of the field
        deRham->projection(func, 0.0, E, gaussNodes);

        // Calculate curlE from E
        deRham->curl(E, curlE);

        // Analytical curlE
#if (GEMPIC_SPACEDIM == 1)
        const amrex::Array<std::string, 3> analyticalCurlE = {
            "0.",
            "cos(x)",
            "cos(x)",
        };
#endif
#if (GEMPIC_SPACEDIM == 2)
        const amrex::Array<std::string, 3> analyticalCurlE = {
            "-sin(x)*cos(y)",
            "cos(x)*sin(y)",
            "cos(x)*cos(y) - cos(x)*sin(y)",
        };
#endif
#if (GEMPIC_SPACEDIM == 3)
        const amrex::Array<std::string, 3> analyticalCurlE = {
            "-2*sin(x)*cos(y)*cos(z)",
            "-cos(x)*cos(y)*cos(z) + cos(x)*sin(y)*cos(z)",
            "cos(x)*cos(y)*sin(z) - cos(x)*sin(y)*sin(z)",
        };
#endif

        for (int i = 0; i < 3; ++i)
        {
            parser[i].define(analyticalCurlE[i]);
            parser[i].registerVariables({AMREX_D_DECL("x", "y", "z"), "t"});
            func[i] = parser[i].compile<nVar>();
        }

        // Declare B field
        DeRhamField<Grid::primal, Space::face> B(deRham);
        deRham->projection(func, 0.0, B, gaussNodes);

        // Calculate errorE
        bool passE{false};
        DeRhamField<Grid::primal, Space::face> errorE(deRham);

        for (int comp = 0; comp < 3; ++comp)
        {
            for (amrex::MFIter mfi(B.m_data[comp]); mfi.isValid(); ++mfi)
            {
                const amrex::Box &bx = mfi.validbox();
                amrex::Array4<amrex::Real> const &bmf = (B.m_data[comp])[mfi].array();
                amrex::Array4<amrex::Real> const &curlEMF = (curlE.m_data[comp])[mfi].array();
                amrex::Array4<amrex::Real> const &errorEMF = (errorE.m_data[comp])[mfi].array();

                ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k)
                            { errorEMF(i, j, k) = std::abs(bmf(i, j, k) - curlEMF(i, j, k)); });
            }
        }

        // Test curl projection = projection curl for dual
        // Declare the fields
        DeRhamField<Grid::dual, Space::edge> H(deRham);
        DeRhamField<Grid::dual, Space::face> curlH(deRham);

        // Parse analytical fields and and initialize parserEval
#if (GEMPIC_SPACEDIM == 1)
        const amrex::Array<std::string, 3> analyticalH = {
            "-cos(x)",
            "sin(x)",
            "-sin(x)",
        };
#endif
#if (GEMPIC_SPACEDIM == 2)
        const amrex::Array<std::string, 3> analyticalH = {
            "-cos(x)*cos(y)",
            "sin(x)*cos(y)",
            "-sin(x)*sin(y)",
        };
#endif

#if (GEMPIC_SPACEDIM == 3)
        const amrex::Array<std::string, 3> analyticalH = {
            "-cos(x)*cos(y)*sin(z)",
            "sin(x)*cos(y)*sin(z)",
            "-sin(x)*sin(y)*cos(z)",
        };
#endif
        for (int i = 0; i < 3; ++i)
        {
            parser[i].define(analyticalH[i]);
            parser[i].registerVariables({AMREX_D_DECL("x", "y", "z"), "t"});
            func[i] = parser[i].compile<nVar>();
        }

        // Compute the projection of the field
        deRham->projection(func, 0.0, H, gaussNodes);

        // Calculate curlH from H
        deRham->curl(H, curlH);

        // Analytical curlH

#if (GEMPIC_SPACEDIM == 1)
        const amrex::Array<std::string, 3> analyticalCurlH = {
            "0.",
            "cos(x)",
            "cos(x)",
        };
#endif
#if (GEMPIC_SPACEDIM == 2)
        const amrex::Array<std::string, 3> analyticalCurlH = {
            "-sin(x)*cos(y)",
            "cos(x)*sin(y)",
            "cos(x)*cos(y) - cos(x)*sin(y)",
        };
#endif
#if (GEMPIC_SPACEDIM == 3)
        const amrex::Array<std::string, 3> analyticalCurlH = {
            "-2*sin(x)*cos(y)*cos(z)",
            "-cos(x)*cos(y)*cos(z) + cos(x)*sin(y)*cos(z)",
            "cos(x)*cos(y)*sin(z) - cos(x)*sin(y)*sin(z)",
        };
#endif

        for (int i = 0; i < 3; ++i)
        {
            parser[i].define(analyticalCurlH[i]);
            parser[i].registerVariables({AMREX_D_DECL("x", "y", "z"), "t"});
            func[i] = parser[i].compile<nVar>();
        }

        // Declare D field
        DeRhamField<Grid::dual, Space::face> D(deRham);
        deRham->projection(func, 0.0, D, gaussNodes);

        // Calculate errorH
        bool passH{false};
        DeRhamField<Grid::dual, Space::face> errorH(deRham);

        for (int comp = 0; comp < 3; ++comp)
        {
            for (amrex::MFIter mfi(D.m_data[comp]); mfi.isValid(); ++mfi)
            {
                const amrex::Box &bx = mfi.validbox();
                amrex::Array4<amrex::Real> const &dmf = (D.m_data[comp])[mfi].array();
                amrex::Array4<amrex::Real> const &curlHMF = (curlH.m_data[comp])[mfi].array();
                amrex::Array4<amrex::Real> const &errorHMF = (errorH.m_data[comp])[mfi].array();

                ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k)
                            { errorHMF(i, j, k) = std::abs(dmf(i, j, k) - curlHMF(i, j, k)); });
            }
        }

        amrex::Real errorCurlExNorm0 = errorE.m_data[xDir].norm0();
        amrex::Real errorCurlEyNorm0 = errorE.m_data[yDir].norm0();
        amrex::Real errorCurlEzNorm0 = errorE.m_data[zDir].norm0();

        amrex::Real errorCurlHxNorm0 = errorH.m_data[xDir].norm0();
        amrex::Real errorCurlHyNorm0 = errorH.m_data[yDir].norm0();
        amrex::Real errorCurlHzNorm0 = errorH.m_data[zDir].norm0();

        /*
        amrex::Print() << "errorCurlEx_norm0: " << errorCurlEx_norm0 << std::endl;
        amrex::Print() << "errorCurlEy_norm0: " << errorCurlEy_norm0 << std::endl;
        amrex::Print() << "errorCurlEz_norm0: " << errorCurlEz_norm0 << std::endl;
        amrex::Print() << "errorCurlHx_norm0: " << errorCurlHx_norm0 << std::endl;
        amrex::Print() << "errorCurlHy_norm0: " << errorCurlHy_norm0 << std::endl;
        amrex::Print() << "errorCurlHz_norm0: " << errorCurlHz_norm0 << std::endl;
        */

        if (std::max({errorCurlExNorm0, errorCurlEyNorm0, errorCurlEzNorm0}) < tol)
        {
            passE = true;
        }
        if (std::max({errorCurlHxNorm0, errorCurlHyNorm0, errorCurlHzNorm0}) < tol)
        {
            passH = true;
        }

        if (passE == true && passH == true)
        {
            amrex::PrintToFile("test_curl_projection.output") << std::endl;
            amrex::PrintToFile("test_curl_projection.output")
                << GEMPIC_SPACEDIM << "D test passed" << std::endl;
        }
        else
        {
            amrex::PrintToFile("test_curl_projection.output") << std::endl;
            amrex::PrintToFile("test_curl_projection.output")
                << GEMPIC_SPACEDIM << "D test failed" << std::endl;
        }

        amrex::PrintToFile("test_curl_projection.output")
            << "max Error curlE[xDir] = " << errorCurlExNorm0 << std::endl;
        amrex::PrintToFile("test_curl_projection.output")
            << "max Error curlE[yDir] = " << errorCurlEyNorm0 << std::endl;
        amrex::PrintToFile("test_curl_projection.output")
            << "max Error curlE[zDir] = " << errorCurlEzNorm0 << std::endl;
        amrex::PrintToFile("test_curl_projection.output")
            << "max Error curlH[xDir] = " << errorCurlHxNorm0 << std::endl;
        amrex::PrintToFile("test_curl_projection.output")
            << "max Error curlH[yDir] = " << errorCurlHyNorm0 << std::endl;
        amrex::PrintToFile("test_curl_projection.output")
            << "max Error curlH[zDir] = " << errorCurlHzNorm0 << std::endl;

        if (amrex::ParallelDescriptor::MyProc() == 0)
        {
            std::rename("test_curl_projection.output.0", "test_curl_projection.output");
        }
        amrex::Print() << "IOProcessorNumber " << amrex::ParallelDescriptor::IOProcessorNumber()
                       << std::endl;
    }
    amrex::Finalize();
}
