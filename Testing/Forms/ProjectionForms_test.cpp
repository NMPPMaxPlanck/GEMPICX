/*------------------------------------------------------------------------------
 Test the restriction operators R_0, R_1, R_2, R_3 on both primal and dual grid for
 periodic boundary conditions.
    Integration uses enough Gauss nodes so that the quadrature error can be
    smaller thant 1e-15.
    Test passes if all projected DOFs are within 1e-15 of the analytical value.
------------------------------------------------------------------------------*/

#include <gtest/gtest.h>

#include "GEMPIC_FDDeRhamComplex.H"
#include "GEMPIC_Fields.H"
#include "GEMPIC_Parameters.H"

using namespace Gempic;
using namespace Forms;

namespace
{
/**
 * @brief Test fixture. Sets up clean environment before each test of the SplineWithPrimitive class
 */
class ProjectionFormsTest : public testing::Test
{
public:
    static constexpr int s_degX{3};
    static constexpr int s_degY{3};
    static constexpr int s_degZ{3};

    inline static const int s_maxSplineDegree{
        AMREX_D_PICK(s_degX, std::max(s_degX, s_degY), std::max(std::max(s_degX, s_degY), s_degZ))};

    static const int s_nVar = GEMPIC_SPACEDIM + 1;  // x, y, z, t

    static void SetUpTestSuite ()
    {
        const amrex::Vector<amrex::Real> domainLo{
            AMREX_D_DECL(-M_PI + 0.3, -M_PI + 0.6, -M_PI + 0.4)};
        const amrex::Vector<amrex::Real> k{AMREX_D_DECL(1.0, 1.0, 1.0)};
        const amrex::Vector<int> nCell{AMREX_D_DECL(9, 8, 7)};
        const amrex::Vector<int> maxGridSize{AMREX_D_DECL(3, 4, 5)};
        const amrex::Vector<int> isPeriodic{AMREX_D_DECL(1, 1, 1)};

        /* Initialize the infrastructure */
        amrex::ParmParse pp;
        pp.addarr("domainLo", domainLo);
        pp.addarr("k", k);
        pp.addarr("nCellVector", nCell);
        pp.addarr("maxGridSizeVector", maxGridSize);
        pp.addarr("isPeriodicVector", isPeriodic);
    }

    template <int hodgeDegree>
    void projection_r1_r2_error (const int gaussNodes, const int field, const double tol)
    {
        // Initialize computational_domain
        Gempic::Io::Parameters parameters{};
        ComputationalDomain infra;

        auto deRham = std::make_shared<FDDeRhamComplex>(infra, hodgeDegree, s_maxSplineDegree,
                                                        HodgeScheme::FDHodge);

        // Parse analytical fields and and initialize parserEval
#if (GEMPIC_SPACEDIM == 1)
        const amrex::Array<std::string, 3> analyticalF = {
            "cos(x)",
            "cos(x)",
            "cos(x)",
        };
#endif

#if (GEMPIC_SPACEDIM == 2)
        const amrex::Array<std::string, 3> analyticalF = {
            "cos(x) * cos(y)",
            "cos(x) * cos(y)",
            "cos(x) * cos(y)",
        };
#endif

#if (GEMPIC_SPACEDIM == 3)
        const amrex::Array<std::string, 3> analyticalF = {
            "cos(x) * cos(y) * cos(z)",
            "cos(x) * cos(y) * cos(z)",
            "cos(x) * cos(y) * cos(z)",
        };
#endif

        amrex::Array<amrex::ParserExecutor<s_nVar>, 3> func;
        amrex::Array<amrex::Parser, 3> parser;
        for (int i = 0; i < 3; ++i)
        {
            parser[i].define(analyticalF[i]);
            parser[i].registerVariables({AMREX_D_DECL("x", "y", "z"), "t"});
            func[i] = parser[i].compile<s_nVar>();
        }

        // test the Grid:: primal, Space::edge Field E
        if (field == 0)
        {
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
                    const amrex::GpuArray<amrex::Real, GEMPIC_SPACEDIM> r0 =
                        infra.m_geom.ProbLoArray();

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
                                oneForm(i, j, k) =
                                    GEMPIC_D_MULT(std::cos(r[xDir]),
                                                  std::sin(r[yDir] + dr[yDir]) - std::sin(r[yDir]),
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
            DeRhamField<Grid::primal, Space::edge> E(deRham);
            deRham->projection(func, 0.0, E, gaussNodes);

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

            // check, if error is smaller than tolerance
            EXPECT_LE(std::max({errorExNorm0, errorEyNorm0, errorEzNorm0}), tol);
        }

        // test the Grid:: primal, Space::face Field B
        else if (field == 1)
        {
            // Compute the analytical solution
            DeRhamField<Grid::primal, Space::face> faceIntegral(deRham);
            for (int comp = 0; comp < 3; ++comp)
            {
                for (amrex::MFIter mfi(faceIntegral.m_data[comp]); mfi.isValid(); ++mfi)
                {
                    const amrex::Box &bx = mfi.validbox();
                    amrex::Array4<amrex::Real> const &oneForm =
                        (faceIntegral.m_data[comp])[mfi].array();

                    const amrex::RealVect dr = amrex::RealVect{
                        AMREX_D_DECL(infra.m_dx[xDir], infra.m_dx[yDir], infra.m_dx[zDir])};
                    const amrex::GpuArray<amrex::Real, GEMPIC_SPACEDIM> r0 =
                        infra.m_geom.ProbLoArray();

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
                                    GEMPIC_D_MULT(std::cos(r[xDir]),
                                                  std::sin(r[yDir] + dr[yDir]) - std::sin(r[yDir]),
                                                  std::sin(r[zDir] + dr[zDir]) - std::sin(r[zDir]));
                            }

                            if (comp == yDir)
                            {
                                oneForm(i, j, k) =
                                    GEMPIC_D_MULT(std::sin(r[xDir] + dr[xDir]) - std::sin(r[xDir]),
                                                  std::cos(r[yDir]),
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
            DeRhamField<Grid::primal, Space::face> B(deRham);
            deRham->projection(func, 0.0, B, gaussNodes);

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

            // check, if error is smaller than tolerance
            EXPECT_LE(std::max({errorBxNorm0, errorByNorm0, errorBzNorm0}), tol);
        }

        // test the Grid::dual, Space::edge Field H
        else if (field == 2)
        {
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
                    const amrex::GpuArray<amrex::Real, GEMPIC_SPACEDIM> r0 =
                        infra.m_geom.ProbLoArray();

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
                                oneForm(i, j, k) =
                                    GEMPIC_D_MULT(std::cos(r[xDir]),
                                                  std::sin(r[yDir]) - std::sin(r[yDir] - dr[yDir]),
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

            DeRhamField<Grid::dual, Space::edge> H(deRham);
            deRham->projection(func, 0.0, H, gaussNodes);

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
            amrex::Real errorHxNorm0 = errorH.m_data[xDir].norm0();
            amrex::Real errorHyNorm0 = errorH.m_data[yDir].norm0();
            amrex::Real errorHzNorm0 = errorH.m_data[zDir].norm0();

            // check, if error is smaller than tolerance
            EXPECT_LE(std::max({errorHxNorm0, errorHyNorm0, errorHzNorm0}), tol);
        }

        // test the Grid::dual, Space::face Field D
        else if (field == 3)
        {
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
                    const amrex::GpuArray<amrex::Real, GEMPIC_SPACEDIM> r0 =
                        infra.m_geom.ProbLoArray();

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
                                    GEMPIC_D_MULT(std::cos(r[xDir]),
                                                  std::sin(r[yDir]) - std::sin(r[yDir] - dr[yDir]),
                                                  std::sin(r[zDir]) - std::sin(r[zDir] - dr[zDir]));
                            }

                            if (comp == yDir)
                            {
                                oneForm(i, j, k) =
                                    GEMPIC_D_MULT(std::sin(r[xDir]) - std::sin(r[xDir] - dr[xDir]),
                                                  std::cos(r[yDir]),
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

            DeRhamField<Grid::dual, Space::face> D(deRham);
            deRham->projection(func, 0.0, D, gaussNodes);
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
            amrex::Real errorDxNorm0 = errorD.m_data[xDir].norm0();
            amrex::Real errorDyNorm0 = errorD.m_data[yDir].norm0();
            amrex::Real errorDzNorm0 = errorD.m_data[zDir].norm0();

            // check, if error is smaller than tolerance
            EXPECT_LE(std::max({errorDxNorm0, errorDyNorm0, errorDzNorm0}), tol);
        }
    }

    template <int hodgeDegree>
    void projection_r0_r3_error (const int gaussNodes, const int field, const double tol)
    {
        // Initialize computational_domain
        Gempic::Io::Parameters parameters{};
        ComputationalDomain infra;

        auto deRham = std::make_shared<FDDeRhamComplex>(infra, hodgeDegree, s_maxSplineDegree,
                                                        HodgeScheme::FDHodge);

        // Parse analytical fields and and initialize parserEval
#if (GEMPIC_SPACEDIM == 1)
        const std::string analyticalF = "cos(x)";
#endif

#if (GEMPIC_SPACEDIM == 2)
        const std::string analyticalF = "cos(x) * cos(y)";
#endif

#if (GEMPIC_SPACEDIM == 3)
        const std::string analyticalF = "cos(x) * cos(y) * cos(z)";
#endif

        amrex::ParserExecutor<s_nVar> func;
        amrex::Parser parser;

        parser.define(analyticalF);
        parser.registerVariables({AMREX_D_DECL("x", "y", "z"), "t"});
        func = parser.compile<s_nVar>();

        // test the Grid::primal, Space::node Field Q
        if (field == 0)
        {
            // Compute the analytical result of the field
            DeRhamField<Grid::primal, Space::node> pointVals(deRham);

            for (amrex::MFIter mfi(pointVals.m_data); mfi.isValid(); ++mfi)
            {
                const amrex::Box &bx = mfi.validbox();
                amrex::Array4<amrex::Real> const &zeroForm = (pointVals.m_data)[mfi].array();

                const amrex::RealVect dr = amrex::RealVect{
                    AMREX_D_DECL(infra.m_dx[xDir], infra.m_dx[yDir], infra.m_dx[zDir])};
                const amrex::GpuArray<amrex::Real, GEMPIC_SPACEDIM> r0 = infra.m_geom.ProbLoArray();

                ParallelFor(bx,
                            [=] AMREX_GPU_DEVICE(int i, int j, int k)
                            {
                                amrex::GpuArray<amrex::Real, GEMPIC_SPACEDIM> r = {
                                    AMREX_D_DECL(r0[xDir] + i * dr[xDir], r0[yDir] + j * dr[yDir],
                                                 r0[zDir] + k * dr[zDir])};

                                zeroForm(i, j, k) = GEMPIC_D_MULT(
                                    std::cos(r[xDir]), std::cos(r[yDir]), std::cos(r[zDir]));
                            });
            }

            pointVals.average_sync();
            pointVals.fill_boundary();

            DeRhamField<Grid::primal, Space::node> Q(deRham);
            deRham->projection(func, 0.0, Q);
            // Calculate error
            DeRhamField<Grid::primal, Space::node> errorQ(deRham);
            for (amrex::MFIter mfi(Q.m_data); mfi.isValid(); ++mfi)
            {
                const amrex::Box &bx = mfi.validbox();

                amrex::Array4<amrex::Real> const &projectionMF = (Q.m_data)[mfi].array();
                amrex::Array4<amrex::Real> const &analyticalMF = (pointVals.m_data)[mfi].array();
                amrex::Array4<amrex::Real> const &errorMF = (errorQ.m_data)[mfi].array();

                ParallelFor(bx,
                            [=] AMREX_GPU_DEVICE(int i, int j, int k)
                            {
                                errorMF(i, j, k) =
                                    std::abs(projectionMF(i, j, k) - analyticalMF(i, j, k));
                            });
            }

            amrex::Real errorQNorm0 = errorQ.m_data.norm0();

            // check, if error is smaller than tolerance
            EXPECT_LE(errorQNorm0, tol);
        }

        // test the Grid::dual, Space::node Field qDual
        else if (field == 1)
        {
            // Compute the analytical solution
            DeRhamField<Grid::dual, Space::node> pointValsDual(deRham);

            for (amrex::MFIter mfi(pointValsDual.m_data); mfi.isValid(); ++mfi)
            {
                const amrex::Box &bx = mfi.validbox();
                amrex::Array4<amrex::Real> const &zeroForm = (pointValsDual.m_data)[mfi].array();

                const amrex::RealVect dr = amrex::RealVect{
                    AMREX_D_DECL(infra.m_dx[xDir], infra.m_dx[yDir], infra.m_dx[zDir])};
                const amrex::GpuArray<amrex::Real, GEMPIC_SPACEDIM> r0 = infra.m_geom.ProbLoArray();

                ParallelFor(bx,
                            [=] AMREX_GPU_DEVICE(int i, int j, int k)
                            {
                                amrex::GpuArray<amrex::Real, GEMPIC_SPACEDIM> r = {
                                    AMREX_D_DECL(r0[xDir] + 0.5 * dr[xDir] + i * dr[xDir],
                                                 r0[yDir] + 0.5 * dr[yDir] + j * dr[yDir],
                                                 r0[zDir] + 0.5 * dr[zDir] + k * dr[zDir])};

                                zeroForm(i, j, k) = GEMPIC_D_MULT(
                                    std::cos(r[xDir]), std::cos(r[yDir]), std::cos(r[zDir]));
                            });
            }

            pointValsDual.average_sync();
            pointValsDual.fill_boundary();

            // Compute the projection of qDual
            DeRhamField<Grid::dual, Space::node> qDual(deRham);
            deRham->projection(func, 0.0, qDual);

            // Calculate error
            DeRhamField<Grid::dual, Space::node> errorQDual(deRham);
            for (amrex::MFIter mfi(qDual.m_data); mfi.isValid(); ++mfi)
            {
                const amrex::Box &bx = mfi.validbox();

                amrex::Array4<amrex::Real> const &projectionMF = (qDual.m_data)[mfi].array();
                amrex::Array4<amrex::Real> const &analyticalMF =
                    (pointValsDual.m_data)[mfi].array();
                amrex::Array4<amrex::Real> const &errorMF = (errorQDual.m_data)[mfi].array();

                ParallelFor(bx,
                            [=] AMREX_GPU_DEVICE(int i, int j, int k)
                            {
                                errorMF(i, j, k) =
                                    std::abs(projectionMF(i, j, k) - analyticalMF(i, j, k));
                            });
            }
            amrex::Real errorQDualNorm0 = errorQDual.m_data.norm0();

            // check, if error is smaller than tolerance
            EXPECT_LE(errorQDualNorm0, tol);
        }

        // test the Grid::primal, Space::cell Field rho
        else if (field == 2)
        {
            // Compute the analytical solution
            DeRhamField<Grid::primal, Space::cell> rhoAn(deRham);
            for (amrex::MFIter mfi(rhoAn.m_data); mfi.isValid(); ++mfi)
            {
                const amrex::Box &bx = mfi.validbox();
                amrex::Array4<amrex::Real> const &threeForm = (rhoAn.m_data)[mfi].array();

                const amrex::RealVect dr = amrex::RealVect{
                    AMREX_D_DECL(infra.m_dx[xDir], infra.m_dx[yDir], infra.m_dx[zDir])};
                const amrex::GpuArray<amrex::Real, GEMPIC_SPACEDIM> r0 = infra.m_geom.ProbLoArray();

                ParallelFor(bx,
                            [=] AMREX_GPU_DEVICE(int i, int j, int k)
                            {
                                amrex::GpuArray<amrex::Real, GEMPIC_SPACEDIM> r = {
                                    AMREX_D_DECL(r0[xDir] + i * dr[xDir], r0[yDir] + j * dr[yDir],
                                                 r0[zDir] + k * dr[zDir])};

                                threeForm(i, j, k) = GEMPIC_D_MULT(
                                    (std::sin(r[xDir] + dr[xDir]) - std::sin(r[xDir])),
                                    (std::sin(r[yDir] + dr[yDir]) - std::sin(r[yDir])),
                                    (std::sin(r[zDir] + dr[zDir]) - std::sin(r[zDir])));
                            });
            }

            rhoAn.fill_boundary();

            DeRhamField<Grid::primal, Space::cell> rho(deRham);
            deRham->projection(func, 0.0, rho, gaussNodes);
            // Calculate error
            DeRhamField<Grid::primal, Space::cell> errorRho(deRham);
            for (amrex::MFIter mfi(rho.m_data); mfi.isValid(); ++mfi)
            {
                const amrex::Box &bx = mfi.validbox();

                amrex::Array4<amrex::Real> const &projectionMF = (rho.m_data)[mfi].array();
                amrex::Array4<amrex::Real> const &analyticalMF = (rhoAn.m_data)[mfi].array();
                amrex::Array4<amrex::Real> const &errorMF = (errorRho.m_data)[mfi].array();

                ParallelFor(bx,
                            [=] AMREX_GPU_DEVICE(int i, int j, int k)
                            {
                                errorMF(i, j, k) =
                                    std::abs(projectionMF(i, j, k) - analyticalMF(i, j, k));
                            });
            }
            amrex::Real errorRhoNorm0 = errorRho.m_data.norm0();

            // check, if error is smaller than tolerance
            EXPECT_LE(errorRhoNorm0, tol);
        }

        // test the Grid::dual, Space::cell Field rhoDual
        else if (field == 3)
        {
            // Compute the analytical result of the field
            DeRhamField<Grid::dual, Space::cell> rhoAnDual(deRham);

            for (amrex::MFIter mfi(rhoAnDual.m_data); mfi.isValid(); ++mfi)
            {
                const amrex::Box &bx = mfi.validbox();
                amrex::Array4<amrex::Real> const &threeForm = (rhoAnDual.m_data)[mfi].array();

                const amrex::RealVect dr = amrex::RealVect{
                    AMREX_D_DECL(infra.m_dx[xDir], infra.m_dx[yDir], infra.m_dx[zDir])};
                const amrex::GpuArray<amrex::Real, GEMPIC_SPACEDIM> r0 = infra.m_geom.ProbLoArray();

                ParallelFor(bx,
                            [=] AMREX_GPU_DEVICE(int i, int j, int k)
                            {
                                amrex::GpuArray<amrex::Real, GEMPIC_SPACEDIM> r = {
                                    AMREX_D_DECL(r0[xDir] + 0.5 * dr[xDir] + i * dr[xDir],
                                                 r0[yDir] + 0.5 * dr[yDir] + j * dr[yDir],
                                                 r0[zDir] + 0.5 * dr[zDir] + k * dr[zDir])};

                                threeForm(i, j, k) = GEMPIC_D_MULT(
                                    (std::sin(r[xDir]) - std::sin(r[xDir] - dr[xDir])),
                                    (std::sin(r[yDir]) - std::sin(r[yDir] - dr[yDir])),
                                    (std::sin(r[zDir]) - std::sin(r[zDir] - dr[zDir])));
                            });
            }

            rhoAnDual.average_sync();
            rhoAnDual.fill_boundary();

            // Calculate error
            DeRhamField<Grid::dual, Space::cell> rhoDual(deRham);
            deRham->projection(func, 0.0, rhoDual, gaussNodes);

            DeRhamField<Grid::dual, Space::cell> errorRhoDual(deRham);
            for (amrex::MFIter mfi(rhoDual.m_data); mfi.isValid(); ++mfi)
            {
                const amrex::Box &bx = mfi.validbox();

                amrex::Array4<amrex::Real> const &projectionMF = (rhoDual.m_data)[mfi].array();
                amrex::Array4<amrex::Real> const &analyticalMF = (rhoAnDual.m_data)[mfi].array();
                amrex::Array4<amrex::Real> const &errorMF = (errorRhoDual.m_data)[mfi].array();

                ParallelFor(bx,
                            [=] AMREX_GPU_DEVICE(int i, int j, int k)
                            {
                                errorMF(i, j, k) =
                                    std::abs(projectionMF(i, j, k) - analyticalMF(i, j, k));
                            });
            }

            amrex::Real errorRhoDualNorm0 = errorRhoDual.m_data.norm0();

            // check, if error is smaller than tolerance
            EXPECT_LE(errorRhoDualNorm0, tol);
        }
    }
};

TEST_F(ProjectionFormsTest, TestPrimalEdge)
{
    int gaussNodes = 6;
    const amrex::Real tol = 1e-15;
    int testFieldE = 0;

    this->projection_r1_r2_error<2>(gaussNodes, testFieldE, tol);
}

TEST_F(ProjectionFormsTest, TestPrimalFace)
{
    int gaussNodes = 6;
    const amrex::Real tol = 1e-15;
    int testFieldB = 1;

    this->projection_r1_r2_error<2>(gaussNodes, testFieldB, tol);
}

TEST_F(ProjectionFormsTest, TestDualEdge)
{
    int gaussNodes = 6;
    const amrex::Real tol = 1e-15;
    int testFieldH = 2;

    this->projection_r1_r2_error<2>(gaussNodes, testFieldH, tol);
}

TEST_F(ProjectionFormsTest, TestDualFace)
{
    int gaussNodes = 6;
    const amrex::Real tol = 1e-15;
    int testFieldD = 3;

    this->projection_r1_r2_error<2>(gaussNodes, testFieldD, tol);
}

TEST_F(ProjectionFormsTest, TestPrimalNode)
{
    int gaussNodes = 6;
    const amrex::Real tol = 1e-15;
    int testFieldQ = 0;

    this->projection_r0_r3_error<2>(gaussNodes, testFieldQ, tol);
}

TEST_F(ProjectionFormsTest, TestDualNode)
{
    int gaussNodes = 6;
    const amrex::Real tol = 1e-15;
    int testFieldQDual = 1;

    this->projection_r0_r3_error<2>(gaussNodes, testFieldQDual, tol);
}

TEST_F(ProjectionFormsTest, TestPrimalCell)
{
    int gaussNodes = 6;
    const amrex::Real tol = 1e-15;
    int testFieldRho = 2;

    this->projection_r0_r3_error<2>(gaussNodes, testFieldRho, tol);
}

TEST_F(ProjectionFormsTest, TestDualCell)
{
    int gaussNodes = 6;
    const amrex::Real tol = 1e-15;
    int testFieldRhoDual = 2;

    this->projection_r0_r3_error<2>(gaussNodes, testFieldRhoDual, tol);
}

}  // namespace