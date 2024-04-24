/*------------------------------------------------------------------------------
 Test the discrete gradient G on both primal and dual grid for
 periodic boundary conditions.
    The following errors are computed: max |R_1 grad f - G R_0 f|, i.e. the
    analytical f is projected to a discrete 0-form followed by the discrete
    gradient. The result is compared with the restriction of the analytical
    gradient of f to a discrete 1-form.
    Test passes if all projected DOFs are within 1e-14 of the analytical value.
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
        BL_PROFILE("main()");
        // error tolerance
        const amrex::Real tol = 1e-14;

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
        DeRhamField<Grid::primal, Space::node> Q(deRham);
        DeRhamField<Grid::primal, Space::edge> gradQ(deRham);

        // Parse analytical fields and and initialize func
#if (GEMPIC_SPACEDIM == 1)
        const std::string analyticalQ = "sin(x)";
#endif
#if (GEMPIC_SPACEDIM == 2)
        const std::string analyticalQ = "sin(x + 2*y)";
#endif
#if (GEMPIC_SPACEDIM == 3)
        const std::string analyticalQ = "sin(x + 2*y - z)";
#endif

        const int nVar = GEMPIC_SPACEDIM + 1;  // x, y, z, t
        amrex::ParserExecutor<nVar> funcQ;
        amrex::Parser parserQ;

        parserQ.define(analyticalQ);
        parserQ.registerVariables({AMREX_D_DECL("x", "y", "z"), "t"});
        funcQ = parserQ.compile<nVar>();

        // Compute the projection of the field
        deRham->projection(funcQ, 0.0, Q);

        // Calculate gradQ from Q
        deRham->grad(Q, gradQ);

        // Analytical gradQ
#if (GEMPIC_SPACEDIM == 1)
        const amrex::Array<std::string, 3> analyticalGradQ = {
            "cos(x)",
            "0.",
            "0.",
        };
#endif
#if (GEMPIC_SPACEDIM == 2)
        const amrex::Array<std::string, 3> analyticalGradQ = {
            "cos(x + 2*y)",
            "2*cos(x + 2*y)",
            "0.",
        };
#endif
#if (GEMPIC_SPACEDIM == 3)
        const amrex::Array<std::string, 3> analyticalGradQ = {
            "cos(x + 2*y - z)",
            "2*cos(x + 2*y - z)",
            "-cos(x + 2*y - z)",
        };
#endif

        amrex::Array<amrex::ParserExecutor<nVar>, 3> funcGradQ;
        amrex::Array<amrex::Parser, 3> parserGradQ;
        for (int i = 0; i < 3; ++i)
        {
            parserGradQ[i].define(analyticalGradQ[i]);
            parserGradQ[i].registerVariables({AMREX_D_DECL("x", "y", "z"), "t"});
            funcGradQ[i] = parserGradQ[i].compile<nVar>();
        }

        // Declare E field
        DeRhamField<Grid::primal, Space::edge> E(deRham);
        deRham->projection(funcGradQ, 0.0, E, gaussNodes);

        // Calculate errorQ
        bool passQ{false};
        DeRhamField<Grid::primal, Space::edge> errorQ(deRham);

        for (int comp = 0; comp < 3; ++comp)
        {
            for (amrex::MFIter mfi(errorQ.m_data[comp]); mfi.isValid(); ++mfi)
            {
                const amrex::Box &bx = mfi.validbox();

                amrex::Array4<amrex::Real> const &projectionMF = (E.m_data[comp])[mfi].array();
                amrex::Array4<amrex::Real> const &analyticalMF = (gradQ.m_data[comp])[mfi].array();
                amrex::Array4<amrex::Real> const &errorMF = (errorQ.m_data[comp])[mfi].array();

                ParallelFor(bx,
                            [=] AMREX_GPU_DEVICE(int i, int j, int k)
                            {
                                errorMF(i, j, k) =
                                    std::abs(projectionMF(i, j, k) - analyticalMF(i, j, k));
                            });
            }
        }

        // Test grad for dual
        // Declare the fields
        DeRhamField<Grid::dual, Space::node> qDual(deRham);
        DeRhamField<Grid::dual, Space::edge> gradDualQ(deRham);

        // Parse analytical fields and and initialize func
#if (GEMPIC_SPACEDIM == 1)
        const std::string analyticalDualQ = "sin(x)";
#endif
#if (GEMPIC_SPACEDIM == 2)
        const std::string analyticalDualQ = "sin(x + 2*y)";
#endif
#if (GEMPIC_SPACEDIM == 3)
        const std::string analyticalDualQ = "sin(x + 2*y - z)";
#endif

        amrex::ParserExecutor<nVar> funcDualQ;
        amrex::Parser parserDualQ;

        parserDualQ.define(analyticalDualQ);
        parserDualQ.registerVariables({AMREX_D_DECL("x", "y", "z"), "t"});
        funcDualQ = parserDualQ.compile<nVar>();

        // Compute the projection of the field
        deRham->projection(funcDualQ, 0.0, qDual);

        // Calculate gradQ from Q
        deRham->grad(qDual, gradDualQ);

        // Analytical gradE
#if (GEMPIC_SPACEDIM == 1)
        const amrex::Array<std::string, 3> analyticalGradDualQ = {
            "cos(x)",
            "0.",
            "0.",
        };
#endif
#if (GEMPIC_SPACEDIM == 2)
        const amrex::Array<std::string, 3> analyticalGradDualQ = {
            "cos(x + 2*y)",
            "2*cos(x + 2*y)",
            "0.",
        };
#endif
#if (GEMPIC_SPACEDIM == 3)
        const amrex::Array<std::string, 3> analyticalGradDualQ = {
            "cos(x + 2*y - z)",
            "2*cos(x + 2*y - z)",
            "-cos(x + 2*y - z)",
        };
#endif

        amrex::Array<amrex::ParserExecutor<nVar>, 3> funcGradDualQ;
        amrex::Array<amrex::Parser, 3> parserGradDualQ;
        for (int i = 0; i < 3; ++i)
        {
            parserGradDualQ[i].define(analyticalGradDualQ[i]);
            parserGradDualQ[i].registerVariables({AMREX_D_DECL("x", "y", "z"), "t"});
            funcGradDualQ[i] = parserGradDualQ[i].compile<nVar>();
        }

        // Declare H field
        DeRhamField<Grid::dual, Space::edge> H(deRham);
        deRham->projection(funcGradDualQ, 0.0, H, gaussNodes);

        // Calculate errorQDual
        bool passDualQ{false};
        DeRhamField<Grid::dual, Space::edge> errorDualQ(deRham);

        for (int comp = 0; comp < 3; ++comp)
        {
            for (amrex::MFIter mfi(errorDualQ.m_data[comp]); mfi.isValid(); ++mfi)
            {
                const amrex::Box &bx = mfi.validbox();

                amrex::Array4<amrex::Real> const &projectionMF = (H.m_data[comp])[mfi].array();
                amrex::Array4<amrex::Real> const &analyticalMF =
                    (gradDualQ.m_data[comp])[mfi].array();
                amrex::Array4<amrex::Real> const &errorMF = (errorDualQ.m_data[comp])[mfi].array();

                ParallelFor(bx,
                            [=] AMREX_GPU_DEVICE(int i, int j, int k)
                            {
                                errorMF(i, j, k) =
                                    std::abs(projectionMF(i, j, k) - analyticalMF(i, j, k));
                            });
            }
        }

        amrex::Real errorGradQxNorm0 = errorQ.m_data[xDir].norm0();
        amrex::Real errorGradQyNorm0 = errorQ.m_data[yDir].norm0();
        amrex::Real errorGradQzNorm0 = errorQ.m_data[zDir].norm0();

        amrex::Real errorGradQDualxNorm0 = errorDualQ.m_data[xDir].norm0();
        amrex::Real errorGradQDualyNorm0 = errorDualQ.m_data[yDir].norm0();
        amrex::Real errorGradQDualzNorm0 = errorDualQ.m_data[zDir].norm0();

        /*
        amrex::Print() << "errorGradQx_norm0 = " << errorGradQx_norm0 << std::endl;
        amrex::Print() << "errorGradQy_norm0 = " << errorGradQy_norm0 << std::endl;
        amrex::Print() << "errorGradQz_norm0 = " << errorGradQz_norm0 << std::endl;
        amrex::Print() << "errorGradQDualx_norm0 = " << errorGradQDualx_norm0 << std::endl;
        amrex::Print() << "errorGradQDualy_norm0 = " << errorGradQDualy_norm0 << std::endl;
        amrex::Print() << "errorGradQDualz_norm0 = " << errorGradQDualz_norm0 << std::endl;
        */

        if (std::max({errorGradQxNorm0, errorGradQyNorm0, errorGradQzNorm0}) < tol)
        {
            passQ = true;
        }
        if (std::max({errorGradQDualxNorm0, errorGradQDualyNorm0, errorGradQDualzNorm0}) < tol)
        {
            passDualQ = true;
        }

        if (passQ == true && passDualQ == true)
        {
            amrex::PrintToFile("test_grad_projection.output") << std::endl;
            amrex::PrintToFile("test_grad_projection.output")
                << GEMPIC_SPACEDIM << "D test passed" << std::endl;
        }
        else
        {
            amrex::PrintToFile("test_grad_projection.output") << std::endl;
            amrex::PrintToFile("test_grad_projection.output")
                << GEMPIC_SPACEDIM << "D test failed" << std::endl;
        }

        amrex::PrintToFile("test_grad_projection.output")
            << "max Error gradQ[xDir] = " << errorGradQxNorm0 << std::endl;
        amrex::PrintToFile("test_grad_projection.output")
            << "max Error gradQ[yDir] = " << errorGradQyNorm0 << std::endl;
        amrex::PrintToFile("test_grad_projection.output")
            << "max Error gradQ[zDir] = " << errorGradQzNorm0 << std::endl;
        amrex::PrintToFile("test_grad_projection.output")
            << "max Error gradQDual[xDir] = " << errorGradQDualxNorm0 << std::endl;
        amrex::PrintToFile("test_grad_projection.output")
            << "max Error gradQDual[yDir] = " << errorGradQDualyNorm0 << std::endl;
        amrex::PrintToFile("test_grad_projection.output")
            << "max Error gradQDual[zDir] = " << errorGradQDualzNorm0 << std::endl;

        if (amrex::ParallelDescriptor::MyProc() == 0)
        {
            std::rename("test_grad_projection.output.0", "test_grad_projection.output");
        }
        amrex::Print() << "IOProcessorNumber " << amrex::ParallelDescriptor::IOProcessorNumber()
                       << std::endl;
    }
    amrex::Finalize();
}
