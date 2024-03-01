/*------------------------------------------------------------------------------
 Test the restriction operators R_1, R_2 on both primal and dual grid for
 periodic boundary conditions.
    Integration uses enough Gauss nodes so that the quadrature error can be
    smaller thant 1e-15.
    Test passes if all projected DOFs are within 1e-15 of the analytical value.
------------------------------------------------------------------------------*/
#include "GEMPIC_FDDeRhamComplex.H"
#include "GEMPIC_Fields.H"
#include "GEMPIC_parameters.H"

using namespace GEMPIC_Fields;
using namespace GEMPIC_FDDeRhamComplex;

int main (int argc, char *argv[])
{
    amrex::Initialize(argc, argv);
    Parameters parameters{};
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
        const amrex::Vector<int> nCell{AMREX_D_DECL(9, 8, 7)};
        const amrex::Vector<int> maxGridSize{AMREX_D_DECL(3, 4, 5)};
        const amrex::Array<int, GEMPIC_SPACEDIM> isPeriodic{AMREX_D_DECL(1, 1, 1)};
        const int hodgeDegree = 2;
        const int maxSplineDegree = 1;

        parameters.set("domain_lo", domainLo);
        parameters.set("k", k);
        parameters.set("n_cell_vector", nCell);
        parameters.set("max_grid_size_vector", maxGridSize);
        parameters.set("is_periodic_vector", isPeriodic);

        // Initialize computational_domain
        Gempic::CompDom::ComputationalDomain infra;

        // Initialize the De Rham Complex
        auto deRham = std::make_shared<FDDeRhamComplex>(infra, hodgeDegree, maxSplineDegree,
                                                        HodgeScheme::FDHodge);

        // Declare the fields
        DeRhamField<Grid::primal, Space::edge> E(deRham);

        // Parse analytical fields and and initialize parserEval
#if (GEMPIC_SPACEDIM == 1)
        const amrex::Array<std::string, 3> analyticalE = {
            "cos(x)",
            "cos(x)",
            "cos(x)",
        };
#endif

#if (GEMPIC_SPACEDIM == 2)
        const amrex::Array<std::string, 3> analyticalE = {
            "cos(x) * cos(y)",
            "cos(x) * cos(y)",
            "cos(x) * cos(y)",
        };
#endif

#if (GEMPIC_SPACEDIM == 3)
        const amrex::Array<std::string, 3> analyticalE = {
            "cos(x) * cos(y) * cos(z)",
            "cos(x) * cos(y) * cos(z)",
            "cos(x) * cos(y) * cos(z)",
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

        // Compute the analytical result of the integral
        DeRhamField<Grid::primal, Space::edge> lineIntegral(deRham);

        for (int comp = 0; comp < 3; ++comp)
        {
            for (amrex::MFIter mfi(lineIntegral.m_data[comp]); mfi.isValid(); ++mfi)
            {
                const amrex::Box &bx = mfi.validbox();
                amrex::Array4<amrex::Real> const &oneForm =
                    (lineIntegral.m_data[comp])[mfi].array();

                const amrex::RealVect dr = amrex::RealVect{
                    AMREX_D_DECL(infra.m_dx[xDir], infra.m_dx[yDir], infra.m_dx[zDir])};
                const amrex::GpuArray<amrex::Real, GEMPIC_SPACEDIM> r0 = infra.m_geom.ProbLoArray();

                ParallelFor(
                    bx,
                    [=] AMREX_GPU_DEVICE(int i, int j, int k)
                    {
                        amrex::GpuArray<amrex::Real, GEMPIC_SPACEDIM> r = {
                            AMREX_D_DECL(r0[xDir] + i * dr[xDir], r0[yDir] + j * dr[yDir],
                                         r0[zDir] + k * dr[zDir])};

                        if (comp == xDir)
                        {
                            oneForm(i, j, k) =
                                GEMPIC_D_MULT(std::sin(r[xDir] + dr[xDir]) - std::sin(r[xDir]),
                                              std::cos(r[yDir]), std::cos(r[zDir]));
                        }

                        if (comp == yDir)
                        {
                            oneForm(i, j, k) = GEMPIC_D_MULT(
                                std::cos(r[xDir]), std::sin(r[yDir] + dr[yDir]) - std::sin(r[yDir]),
                                std::cos(r[zDir]));
                        }

                        if (comp == zDir)
                        {
                            oneForm(i, j, k) =
                                GEMPIC_D_MULT(std::cos(r[xDir]), std::cos(r[yDir]),
                                              std::sin(r[zDir] + dr[zDir]) - std::sin(r[zDir]));
                        }
                    });
            }
        }

        lineIntegral.average_sync();
        lineIntegral.fill_boundary();

        // Compute the projection of the field
        deRham->projection(func, 0.0, E, gaussNodes);

        // Test passes if error < GEMPIC_CTEST_TOL
        bool passE = false;

        DeRhamField<Grid::primal, Space::edge> errorE(deRham);
        for (int comp = 0; comp < 3; ++comp)
        {
            for (amrex::MFIter mfi(E.m_data[comp]); mfi.isValid(); ++mfi)
            {
                const amrex::Box &bx = mfi.validbox();
                amrex::Array4<amrex::Real> const &projectionMF = (E.m_data[comp])[mfi].array();
                amrex::Array4<amrex::Real> const &analyticalMF =
                    (lineIntegral.m_data[comp])[mfi].array();
                amrex::Array4<amrex::Real> const &errorMF = (errorE.m_data[comp])[mfi].array();

                ParallelFor(bx,
                            [=] AMREX_GPU_DEVICE(int i, int j, int k)
                            {
                                errorMF(i, j, k) =
                                    std::abs(projectionMF(i, j, k) - analyticalMF(i, j, k));
                            });
            }
        }

        amrex::Real errorExNorm0 = errorE.m_data[xDir].norm0();
        amrex::Real errorEyNorm0 = errorE.m_data[yDir].norm0();
        amrex::Real errorEzNorm0 = errorE.m_data[zDir].norm0();

        if (std::max({errorExNorm0, errorEyNorm0, errorEzNorm0}) < tol)
        {
            passE = true;
        }

        // Run the same test for face integrals
        // Declare the fields
        DeRhamField<Grid::primal, Space::face> B(deRham);
        DeRhamField<Grid::primal, Space::face> faceIntegral(deRham);

        // Parse analytical fields and and initialize parserEval
#if (GEMPIC_SPACEDIM == 1)
        const amrex::Array<std::string, 3> analyticalB = {
            "cos(x)",
            "cos(x)",
            "cos(x)",
        };
#endif
#if (GEMPIC_SPACEDIM == 2)
        const amrex::Array<std::string, 3> analyticalB = {
            "cos(x) * cos(y)",
            "cos(x) * cos(y)",
            "cos(x) * cos(y)",
        };
#endif
#if (GEMPIC_SPACEDIM == 3)
        const amrex::Array<std::string, 3> analyticalB = {
            "cos(x) * cos(y) * cos(z)",
            "cos(x) * cos(y) * cos(z)",
            "cos(x) * cos(y) * cos(z)",
        };
#endif

        for (int i = 0; i < 3; ++i)
        {
            parser[i].define(analyticalB[i]);
            parser[i].registerVariables({AMREX_D_DECL("x", "y", "z"), "t"});
            func[i] = parser[i].compile<nVar>();
        }

        // Analytical face integral representing the projection
        for (int comp = 0; comp < 3; ++comp)
        {
            for (amrex::MFIter mfi(faceIntegral.m_data[comp]); mfi.isValid(); ++mfi)
            {
                const amrex::Box &bx = mfi.validbox();
                amrex::Array4<amrex::Real> const &oneForm =
                    (faceIntegral.m_data[comp])[mfi].array();

                const amrex::RealVect dr = amrex::RealVect{
                    AMREX_D_DECL(infra.m_dx[xDir], infra.m_dx[yDir], infra.m_dx[zDir])};
                const amrex::GpuArray<amrex::Real, GEMPIC_SPACEDIM> r0 = infra.m_geom.ProbLoArray();

                ParallelFor(
                    bx,
                    [=] AMREX_GPU_DEVICE(int i, int j, int k)
                    {
                        amrex::GpuArray<amrex::Real, GEMPIC_SPACEDIM> r = {
                            AMREX_D_DECL(r0[xDir] + i * dr[xDir], r0[yDir] + j * dr[yDir],
                                         r0[zDir] + k * dr[zDir])};

                        if (comp == xDir)
                        {
                            oneForm(i, j, k) = GEMPIC_D_MULT(
                                std::cos(r[xDir]), std::sin(r[yDir] + dr[yDir]) - std::sin(r[yDir]),
                                std::sin(r[zDir] + dr[zDir]) - std::sin(r[zDir]));
                        }

                        if (comp == yDir)
                        {
                            oneForm(i, j, k) = GEMPIC_D_MULT(
                                std::sin(r[xDir] + dr[xDir]) - std::sin(r[xDir]), std::cos(r[yDir]),
                                std::sin(r[zDir] + dr[zDir]) - std::sin(r[zDir]));
                        }

                        if (comp == zDir)
                        {
                            oneForm(i, j, k) =
                                GEMPIC_D_MULT(std::sin(r[xDir] + dr[xDir]) - std::sin(r[xDir]),
                                              std::sin(r[yDir] + dr[yDir]) - std::sin(r[yDir]),
                                              std::cos(r[zDir]));
                        }
                    });
            }
        }

        faceIntegral.average_sync();
        faceIntegral.fill_boundary();

        // Compute the projection of B
        deRham->projection(func, 0.0, B, gaussNodes);

        // Test passes if error < GEMPIC_CTEST_TOL
        bool passB = false;

        DeRhamField<Grid::primal, Space::face> errorB(deRham);
        for (int comp = 0; comp < 3; ++comp)
        {
            for (amrex::MFIter mfi(B.m_data[comp]); mfi.isValid(); ++mfi)
            {
                const amrex::Box &bx = mfi.validbox();
                amrex::Array4<amrex::Real> const &projectionMF = (B.m_data[comp])[mfi].array();
                amrex::Array4<amrex::Real> const &analyticalMF =
                    (faceIntegral.m_data[comp])[mfi].array();
                amrex::Array4<amrex::Real> const &errorMF = (errorB.m_data[comp])[mfi].array();

                ParallelFor(bx,
                            [=] AMREX_GPU_DEVICE(int i, int j, int k)
                            {
                                errorMF(i, j, k) =
                                    std::abs(projectionMF(i, j, k) - analyticalMF(i, j, k));
                            });
            }
        }

        amrex::Real errorBxNorm0 = errorB.m_data[xDir].norm0();
        amrex::Real errorByNorm0 = errorB.m_data[yDir].norm0();
        amrex::Real errorBzNorm0 = errorB.m_data[zDir].norm0();

        if (std::max({errorBxNorm0, errorByNorm0, errorBzNorm0}) < tol)
        {
            passB = true;
        }

        // same test on dual grid
        DeRhamField<Grid::dual, Space::edge> H(deRham);

#if (GEMPIC_SPACEDIM == 1)
        const amrex::Array<std::string, 3> analyticalH = {
            "cos(x)",
            "cos(x)",
            "cos(x)",
        };
#endif
#if (GEMPIC_SPACEDIM == 2)
        const amrex::Array<std::string, 3> analyticalH = {
            "cos(x) * cos(y)",
            "cos(x) * cos(y)",
            "cos(x) * cos(y)",
        };
#endif
#if (GEMPIC_SPACEDIM == 3)
        const amrex::Array<std::string, 3> analyticalH = {
            "cos(x) * cos(y) * cos(z)",
            "cos(x) * cos(y) * cos(z)",
            "cos(x) * cos(y) * cos(z)",
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

        // Compute the analytical solution
        DeRhamField<Grid::dual, Space::edge> lineIntegralDual(deRham);

        for (int comp = 0; comp < 3; ++comp)
        {
            for (amrex::MFIter mfi(lineIntegralDual.m_data[comp]); mfi.isValid(); ++mfi)
            {
                const amrex::Box &bx = mfi.validbox();
                amrex::Array4<amrex::Real> const &oneForm =
                    (lineIntegralDual.m_data[comp])[mfi].array();

                const amrex::RealVect dr = amrex::RealVect{
                    AMREX_D_DECL(infra.m_dx[xDir], infra.m_dx[yDir], infra.m_dx[zDir])};
                const amrex::GpuArray<amrex::Real, GEMPIC_SPACEDIM> r0 = infra.m_geom.ProbLoArray();

                ParallelFor(
                    bx,
                    [=] AMREX_GPU_DEVICE(int i, int j, int k)
                    {
                        amrex::GpuArray<amrex::Real, GEMPIC_SPACEDIM> r = {
                            AMREX_D_DECL(r0[xDir] + 0.5 * dr[xDir] + i * dr[xDir],
                                         r0[yDir] + 0.5 * dr[yDir] + j * dr[yDir],
                                         r0[zDir] + 0.5 * dr[zDir] + k * dr[zDir])};

                        if (comp == xDir)
                        {
                            oneForm(i, j, k) =
                                GEMPIC_D_MULT(std::sin(r[xDir]) - std::sin(r[xDir] - dr[xDir]),
                                              std::cos(r[yDir]), std::cos(r[zDir]));
                        }

                        if (comp == yDir)
                        {
                            oneForm(i, j, k) = GEMPIC_D_MULT(
                                std::cos(r[xDir]), std::sin(r[yDir]) - std::sin(r[yDir] - dr[yDir]),
                                std::cos(r[zDir]));
                        }

                        if (comp == zDir)
                        {
                            oneForm(i, j, k) =
                                GEMPIC_D_MULT(std::cos(r[xDir]), std::cos(r[yDir]),
                                              std::sin(r[zDir]) - std::sin(r[zDir] - dr[zDir]));
                        }
                    });
            }
        }

        lineIntegralDual.average_sync();
        lineIntegralDual.fill_boundary();

        // Calculate error
        DeRhamField<Grid::dual, Space::edge> errorH(deRham);
        for (int comp = 0; comp < 3; ++comp)
        {
            for (amrex::MFIter mfi(H.m_data[comp]); mfi.isValid(); ++mfi)
            {
                const amrex::Box &bx = mfi.validbox();

                amrex::Array4<amrex::Real> const &projectionMF = (H.m_data[comp])[mfi].array();
                amrex::Array4<amrex::Real> const &analyticalMF =
                    (lineIntegralDual.m_data[comp])[mfi].array();
                amrex::Array4<amrex::Real> const &errorMF = (errorH.m_data[comp])[mfi].array();

                ParallelFor(bx,
                            [=] AMREX_GPU_DEVICE(int i, int j, int k)
                            {
                                errorMF(i, j, k) =
                                    std::abs(projectionMF(i, j, k) - analyticalMF(i, j, k));
                            });
            }
        }

        bool passH = false;

        amrex::Real errorHxNorm0 = errorH.m_data[xDir].norm0();
        amrex::Real errorHyNorm0 = errorH.m_data[yDir].norm0();
        amrex::Real errorHzNorm0 = errorH.m_data[zDir].norm0();

        if (std::max({errorHxNorm0, errorHyNorm0, errorHzNorm0}) < tol)
        {
            passH = true;
        }

        // Declare the fields
        DeRhamField<Grid::dual, Space::face> D(deRham);

#if (GEMPIC_SPACEDIM == 1)
        const amrex::Array<std::string, 3> analyticalD = {
            "cos(x)",
            "cos(x)",
            "cos(x)",
        };
#endif
#if (GEMPIC_SPACEDIM == 2)
        const amrex::Array<std::string, 3> analyticalD = {
            "cos(x) * cos(y)",
            "cos(x) * cos(y)",
            "cos(x) * cos(y)",
        };
#endif
#if (GEMPIC_SPACEDIM == 3)
        const amrex::Array<std::string, 3> analyticalD = {
            "cos(x) * cos(y) * cos(z)",
            "cos(x) * cos(y) * cos(z)",
            "cos(x) * cos(y) * cos(z)",
        };
#endif

        for (int i = 0; i < 3; ++i)
        {
            parser[i].define(analyticalD[i]);
            parser[i].registerVariables({AMREX_D_DECL("x", "y", "z"), "t"});
            func[i] = parser[i].compile<nVar>();
        }

        // Compute the projection of the field
        deRham->projection(func, 0.0, D, gaussNodes);

        // Compute the analytical solution
        DeRhamField<Grid::dual, Space::face> faceIntegralDual(deRham);

        for (int comp = 0; comp < 3; ++comp)
        {
            for (amrex::MFIter mfi(faceIntegralDual.m_data[comp]); mfi.isValid(); ++mfi)
            {
                const amrex::Box &bx = mfi.validbox();
                amrex::Array4<amrex::Real> const &oneForm =
                    (faceIntegralDual.m_data[comp])[mfi].array();

                const amrex::RealVect dr = amrex::RealVect{
                    AMREX_D_DECL(infra.m_dx[xDir], infra.m_dx[yDir], infra.m_dx[zDir])};
                const amrex::GpuArray<amrex::Real, GEMPIC_SPACEDIM> r0 = infra.m_geom.ProbLoArray();

                ParallelFor(
                    bx,
                    [=] AMREX_GPU_DEVICE(int i, int j, int k)
                    {
                        amrex::GpuArray<amrex::Real, GEMPIC_SPACEDIM> r = {
                            AMREX_D_DECL(r0[xDir] + 0.5 * dr[xDir] + i * dr[xDir],
                                         r0[yDir] + 0.5 * dr[yDir] + j * dr[yDir],
                                         r0[zDir] + 0.5 * dr[zDir] + k * dr[zDir])};

                        if (comp == xDir)
                        {
                            oneForm(i, j, k) = GEMPIC_D_MULT(
                                std::cos(r[xDir]), std::sin(r[yDir]) - std::sin(r[yDir] - dr[yDir]),
                                std::sin(r[zDir]) - std::sin(r[zDir] - dr[zDir]));
                        }

                        if (comp == yDir)
                        {
                            oneForm(i, j, k) = GEMPIC_D_MULT(
                                std::sin(r[xDir]) - std::sin(r[xDir] - dr[xDir]), std::cos(r[yDir]),
                                std::sin(r[zDir]) - std::sin(r[zDir] - dr[zDir]));
                        }

                        if (comp == zDir)
                        {
                            oneForm(i, j, k) =
                                GEMPIC_D_MULT(std::sin(r[xDir]) - std::sin(r[xDir] - dr[xDir]),
                                              std::sin(r[yDir]) - std::sin(r[yDir] - dr[yDir]),
                                              std::cos(r[zDir]));
                        }
                    });
            }
        }

        faceIntegralDual.average_sync();
        faceIntegralDual.fill_boundary();

        // Calculate error
        DeRhamField<Grid::dual, Space::face> errorD(deRham);
        for (int comp = 0; comp < 3; ++comp)
        {
            for (amrex::MFIter mfi(D.m_data[comp]); mfi.isValid(); ++mfi)
            {
                const amrex::Box &bx = mfi.validbox();

                amrex::Array4<amrex::Real> const &projectionMF = (D.m_data[comp])[mfi].array();
                amrex::Array4<amrex::Real> const &analyticalMF =
                    (faceIntegralDual.m_data[comp])[mfi].array();
                amrex::Array4<amrex::Real> const &errorMF = (errorD.m_data[comp])[mfi].array();

                ParallelFor(bx,
                            [=] AMREX_GPU_DEVICE(int i, int j, int k)
                            {
                                errorMF(i, j, k) =
                                    std::abs(projectionMF(i, j, k) - analyticalMF(i, j, k));
                            });
            }
        }

        bool passD = false;

        amrex::Real errorDxNorm0 = errorD.m_data[xDir].norm0();
        amrex::Real errorDyNorm0 = errorD.m_data[yDir].norm0();
        amrex::Real errorDzNorm0 = errorD.m_data[zDir].norm0();

        /*
        amrex::Print() << "errorEx_norm0 = " << errorEx_norm0 << ", errorEy_norm0 = " <<
        errorEy_norm0 << ", errorEz_norm0 = " << errorEz_norm0 << std::endl; amrex::Print() <<
        "errorBx_norm0 = " << errorBx_norm0 << ", errorBy_norm0 = " << errorBy_norm0 << ",
        errorBz_norm0 = " << errorBz_norm0 << std::endl; amrex::Print() << "errorHx_norm0 = " <<
        errorHx_norm0 << ", errorHy_norm0 = " << errorHy_norm0 << ", errorHz_norm0 = " <<
        errorHz_norm0 << std::endl; amrex::Print() << "errorDx_norm0 = " << errorDx_norm0 << ",
        errorDy_norm0 = " << errorDy_norm0 << ", errorDz_norm0 = " << errorDz_norm0 << std::endl;
        */

        if (std::max({errorDxNorm0, errorDyNorm0, errorDzNorm0}) < tol)
        {
            passD = true;
        }

        if (passE == true && passB == true && passD == true && passH == true)
        {
            amrex::PrintToFile("test_projection_one_two_forms.output") << std::endl;
            amrex::PrintToFile("test_projection_one_two_forms.output")
                << GEMPIC_SPACEDIM << "D test passed" << std::endl;
        }
        else
        {
            amrex::PrintToFile("test_projection_one_two_forms.output") << std::endl;
            amrex::PrintToFile("test_projection_one_two_forms.output")
                << GEMPIC_SPACEDIM << "D test failed" << std::endl;
        }

        amrex::PrintToFile("test_projection_one_two_forms.output")
            << "max Error E[xDir] = " << errorExNorm0 << std::endl;
        amrex::PrintToFile("test_projection_one_two_forms.output")
            << "max Error E[yDir] = " << errorEyNorm0 << std::endl;
        amrex::PrintToFile("test_projection_one_two_forms.output")
            << "max Error E[zDir] = " << errorEzNorm0 << std::endl;
        amrex::PrintToFile("test_projection_one_two_forms.output")
            << "max Error B[xDir] = " << errorBxNorm0 << std::endl;
        amrex::PrintToFile("test_projection_one_two_forms.output")
            << "max Error B[yDir] = " << errorByNorm0 << std::endl;
        amrex::PrintToFile("test_projection_one_two_forms.output")
            << "max Error B[zDir] = " << errorBzNorm0 << std::endl;
        amrex::PrintToFile("test_projection_one_two_forms.output")
            << "max Error H[xDir] = " << errorHxNorm0 << std::endl;
        amrex::PrintToFile("test_projection_one_two_forms.output")
            << "max Error H[yDir] = " << errorHyNorm0 << std::endl;
        amrex::PrintToFile("test_projection_one_two_forms.output")
            << "max Error H[zDir] = " << errorHzNorm0 << std::endl;
        amrex::PrintToFile("test_projection_one_two_forms.output")
            << "max Error D[xDir] = " << errorDxNorm0 << std::endl;
        amrex::PrintToFile("test_projection_one_two_forms.output")
            << "max Error D[yDir] = " << errorDyNorm0 << std::endl;
        amrex::PrintToFile("test_projection_one_two_forms.output")
            << "max Error D[zDir] = " << errorDzNorm0 << std::endl;

        if (amrex::ParallelDescriptor::MyProc() == 0)
        {
            std::rename("test_projection_one_two_forms.output.0",
                        "test_projection_one_two_forms.output");
        }
        amrex::Print() << "IOProcessorNumber " << amrex::ParallelDescriptor::IOProcessorNumber()
                       << std::endl;
    }
    amrex::Finalize();
}
