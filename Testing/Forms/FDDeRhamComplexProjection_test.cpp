#include <gtest/gtest.h>

#include <AMReX.H>
#include <AMReX_ParmParse.H>

#include "GEMPIC_ComputationalDomain.H"
#include "GEMPIC_FDDeRhamComplex.H"
#include "GEMPIC_Fields.H"
#include "GEMPIC_GempicNorm.H"
#include "GEMPIC_Parameters.H"
#include "TestUtils/GEMPIC_TestUtils.H"

// E = one form
// rho = three form
// phi = zero form

using namespace Gempic;
using namespace Forms;

namespace
{
// When using amrex::ParallelFor you have to create a standalone helper function that does the
// execution on GPU and call that function from the unit test because of how GTest creates tests
// within a TEST_F fixture.
void update_one_form_primal_parallel_for (amrex::MFIter& mfi,
                                          DeRhamField<Grid::primal, Space::edge>& lineIntegral,
                                          ComputationalDomain& mInfra,
                                          int comp)
{
    const amrex::Box& bx = mfi.validbox();
    amrex::Array4<amrex::Real> const& oneForm = (lineIntegral.m_data[comp])[mfi].array();

    const amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> dr = mInfra.geometry().CellSizeArray();
    const amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> r0 = mInfra.m_geom.ProbLoArray();

    amrex::ParallelFor(
        bx,
        [=] AMREX_GPU_DEVICE(int i, int j, int k)
        {
            amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> r = {AMREX_D_DECL(
                r0[xDir] + i * dr[xDir], r0[yDir] + j * dr[yDir], r0[zDir] + k * dr[zDir])};

            if (comp == xDir)
            {
                oneForm(i, j, k) = GEMPIC_D_MULT(std::sin(r[xDir] + dr[xDir]) - std::sin(r[xDir]),
                                                 std::cos(r[yDir]), std::cos(r[zDir]));
            }

            if (comp == yDir)
            {
                oneForm(i, j, k) = GEMPIC_D_MULT(std::cos(r[xDir]),
                                                 std::sin(r[yDir] + dr[yDir]) - std::sin(r[yDir]),
                                                 std::cos(r[zDir]));
            }

            if (comp == zDir)
            {
                oneForm(i, j, k) = GEMPIC_D_MULT(std::cos(r[xDir]), std::cos(r[yDir]),
                                                 std::sin(r[zDir] + dr[zDir]) - std::sin(r[zDir]));
            }
        });
}

void update_one_form_primal_error_parallel_for (
    amrex::MFIter& mfi,
    DeRhamField<Grid::primal, Space::edge>& lineIntegral,
    DeRhamField<Grid::primal, Space::edge>& E,
    DeRhamField<Grid::primal, Space::edge>& errorE,
    int comp)
{
    const amrex::Box& bx = mfi.validbox();
    amrex::Array4<amrex::Real> const& projectionMF = (E.m_data[comp])[mfi].array();
    amrex::Array4<amrex::Real> const& analyticalMF = (lineIntegral.m_data[comp])[mfi].array();
    amrex::Array4<amrex::Real> const& errorMF = (errorE.m_data[comp])[mfi].array();

    amrex::ParallelFor(
        bx, [=] AMREX_GPU_DEVICE(int i, int j, int k)
        { errorMF(i, j, k) = std::abs(projectionMF(i, j, k) - analyticalMF(i, j, k)); });
}

void update_two_form_primal_parallel_for (amrex::MFIter& mfi,
                                          DeRhamField<Grid::primal, Space::face>& faceIntegral,
                                          ComputationalDomain& mInfra,
                                          int comp)
{
    const amrex::Box& bx = mfi.validbox();
    amrex::Array4<amrex::Real> const& oneForm = (faceIntegral.m_data[comp])[mfi].array();

    const amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> dr = mInfra.geometry().CellSizeArray();
    const amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> r0 = mInfra.m_geom.ProbLoArray();

    amrex::ParallelFor(
        bx,
        [=] AMREX_GPU_DEVICE(int i, int j, int k)
        {
            amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> r = {AMREX_D_DECL(
                r0[xDir] + i * dr[xDir], r0[yDir] + j * dr[yDir], r0[zDir] + k * dr[zDir])};

            if (comp == xDir)
            {
                oneForm(i, j, k) = GEMPIC_D_MULT(std::cos(r[xDir]),
                                                 std::sin(r[yDir] + dr[yDir]) - std::sin(r[yDir]),
                                                 std::sin(r[zDir] + dr[zDir]) - std::sin(r[zDir]));
            }

            if (comp == yDir)
            {
                oneForm(i, j, k) = GEMPIC_D_MULT(std::sin(r[xDir] + dr[xDir]) - std::sin(r[xDir]),
                                                 std::cos(r[yDir]),
                                                 std::sin(r[zDir] + dr[zDir]) - std::sin(r[zDir]));
            }

            if (comp == zDir)
            {
                oneForm(i, j, k) = GEMPIC_D_MULT(std::sin(r[xDir] + dr[xDir]) - std::sin(r[xDir]),
                                                 std::sin(r[yDir] + dr[yDir]) - std::sin(r[yDir]),
                                                 std::cos(r[zDir]));
            }
        });
}

void update_two_form_primal_error_parallel_for (
    amrex::MFIter& mfi,
    DeRhamField<Grid::primal, Space::face>& faceIntegral,
    DeRhamField<Grid::primal, Space::face>& B,
    DeRhamField<Grid::primal, Space::face>& errorB,
    int comp)
{
    const amrex::Box& bx = mfi.validbox();
    amrex::Array4<amrex::Real> const& projectionMF = (B.m_data[comp])[mfi].array();
    amrex::Array4<amrex::Real> const& analyticalMF = (faceIntegral.m_data[comp])[mfi].array();
    amrex::Array4<amrex::Real> const& errorMF = (errorB.m_data[comp])[mfi].array();

    amrex::ParallelFor(
        bx, [=] AMREX_GPU_DEVICE(int i, int j, int k)
        { errorMF(i, j, k) = std::abs(projectionMF(i, j, k) - analyticalMF(i, j, k)); });
}

void update_one_form_dual_parallel_for (amrex::MFIter& mfi,
                                        DeRhamField<Grid::dual, Space::edge>& lineIntegralDual,
                                        ComputationalDomain& mInfra,
                                        int comp)
{
    const amrex::Box& bx = mfi.validbox();
    amrex::Array4<amrex::Real> const& oneForm = (lineIntegralDual.m_data[comp])[mfi].array();

    const amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> dr = mInfra.geometry().CellSizeArray();
    const amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> r0 = mInfra.m_geom.ProbLoArray();

    amrex::ParallelFor(
        bx,
        [=] AMREX_GPU_DEVICE(int i, int j, int k)
        {
            amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> r = {AMREX_D_DECL(
                r0[xDir] + 0.5 * dr[xDir] + i * dr[xDir], r0[yDir] + 0.5 * dr[yDir] + j * dr[yDir],
                r0[zDir] + 0.5 * dr[zDir] + k * dr[zDir])};

            if (comp == xDir)
            {
                oneForm(i, j, k) = GEMPIC_D_MULT(std::sin(r[xDir]) - std::sin(r[xDir] - dr[xDir]),
                                                 std::cos(r[yDir]), std::cos(r[zDir]));
            }

            if (comp == yDir)
            {
                oneForm(i, j, k) = GEMPIC_D_MULT(std::cos(r[xDir]),
                                                 std::sin(r[yDir]) - std::sin(r[yDir] - dr[yDir]),
                                                 std::cos(r[zDir]));
            }

            if (comp == zDir)
            {
                oneForm(i, j, k) = GEMPIC_D_MULT(std::cos(r[xDir]), std::cos(r[yDir]),
                                                 std::sin(r[zDir]) - std::sin(r[zDir] - dr[zDir]));
            }
        });
}

void update_one_form_dual_error_parallel_for (
    amrex::MFIter& mfi,
    DeRhamField<Grid::dual, Space::edge>& lineIntegralDual,
    DeRhamField<Grid::dual, Space::edge>& H,
    DeRhamField<Grid::dual, Space::edge>& errorH,
    int comp)
{
    const amrex::Box& bx = mfi.validbox();

    amrex::Array4<amrex::Real> const& projectionMF = (H.m_data[comp])[mfi].array();
    amrex::Array4<amrex::Real> const& analyticalMF = (lineIntegralDual.m_data[comp])[mfi].array();
    amrex::Array4<amrex::Real> const& errorMF = (errorH.m_data[comp])[mfi].array();

    amrex::ParallelFor(
        bx, [=] AMREX_GPU_DEVICE(int i, int j, int k)
        { errorMF(i, j, k) = std::abs(projectionMF(i, j, k) - analyticalMF(i, j, k)); });
}

void update_two_form_dual_parallel_for (amrex::MFIter& mfi,
                                        DeRhamField<Grid::dual, Space::face>& faceIntegralDual,
                                        ComputationalDomain& mInfra,
                                        int comp)
{
    const amrex::Box& bx = mfi.validbox();
    amrex::Array4<amrex::Real> const& oneForm = (faceIntegralDual.m_data[comp])[mfi].array();

    const amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> dr = mInfra.geometry().CellSizeArray();
    const amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> r0 = mInfra.m_geom.ProbLoArray();

    amrex::ParallelFor(
        bx,
        [=] AMREX_GPU_DEVICE(int i, int j, int k)
        {
            amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> r = {AMREX_D_DECL(
                r0[xDir] + 0.5 * dr[xDir] + i * dr[xDir], r0[yDir] + 0.5 * dr[yDir] + j * dr[yDir],
                r0[zDir] + 0.5 * dr[zDir] + k * dr[zDir])};

            if (comp == xDir)
            {
                oneForm(i, j, k) = GEMPIC_D_MULT(std::cos(r[xDir]),
                                                 std::sin(r[yDir]) - std::sin(r[yDir] - dr[yDir]),
                                                 std::sin(r[zDir]) - std::sin(r[zDir] - dr[zDir]));
            }

            if (comp == yDir)
            {
                oneForm(i, j, k) = GEMPIC_D_MULT(std::sin(r[xDir]) - std::sin(r[xDir] - dr[xDir]),
                                                 std::cos(r[yDir]),
                                                 std::sin(r[zDir]) - std::sin(r[zDir] - dr[zDir]));
            }

            if (comp == zDir)
            {
                oneForm(i, j, k) = GEMPIC_D_MULT(std::sin(r[xDir]) - std::sin(r[xDir] - dr[xDir]),
                                                 std::sin(r[yDir]) - std::sin(r[yDir] - dr[yDir]),
                                                 std::cos(r[zDir]));
            }
        });
}

void update_two_form_dual_error_parallel_for (
    amrex::MFIter& mfi,
    DeRhamField<Grid::dual, Space::face>& faceIntegralDual,
    DeRhamField<Grid::dual, Space::face>& D,
    DeRhamField<Grid::dual, Space::face>& errorD,
    int comp)
{
    const amrex::Box& bx = mfi.validbox();

    amrex::Array4<amrex::Real> const& projectionMF = (D.m_data[comp])[mfi].array();
    amrex::Array4<amrex::Real> const& analyticalMF = (faceIntegralDual.m_data[comp])[mfi].array();
    amrex::Array4<amrex::Real> const& errorMF = (errorD.m_data[comp])[mfi].array();

    amrex::ParallelFor(
        bx, [=] AMREX_GPU_DEVICE(int i, int j, int k)
        { errorMF(i, j, k) = std::abs(projectionMF(i, j, k) - analyticalMF(i, j, k)); });
}

void update_zero_form_primal_parallel_for (amrex::MFIter& mfi,
                                           DeRhamField<Grid::primal, Space::node>& pointVals,
                                           ComputationalDomain& mInfra)
{
    const amrex::Box& bx = mfi.validbox();
    amrex::Array4<amrex::Real> const& zeroForm = (pointVals.m_data)[mfi].array();

    const amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> dr = mInfra.geometry().CellSizeArray();
    const amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> r0 = mInfra.m_geom.ProbLoArray();

    amrex::ParallelFor(
        bx,
        [=] AMREX_GPU_DEVICE(int i, int j, int k)
        {
            amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> r = {AMREX_D_DECL(
                r0[xDir] + i * dr[xDir], r0[yDir] + j * dr[yDir], r0[zDir] + k * dr[zDir])};

            zeroForm(i, j, k) =
                GEMPIC_D_MULT(std::cos(r[xDir]), std::cos(r[yDir]), std::cos(r[zDir]));
        });
}

void update_zero_form_primal_error_parallel_for (amrex::MFIter& mfi,
                                                 DeRhamField<Grid::primal, Space::node>& pointVals,
                                                 DeRhamField<Grid::primal, Space::node>& Q,
                                                 DeRhamField<Grid::primal, Space::node>& errorQ)
{
    const amrex::Box& bx = mfi.validbox();

    amrex::Array4<amrex::Real> const& projectionMF = (Q.m_data)[mfi].array();
    amrex::Array4<amrex::Real> const& analyticalMF = (pointVals.m_data)[mfi].array();
    amrex::Array4<amrex::Real> const& errorMF = (errorQ.m_data)[mfi].array();

    amrex::ParallelFor(
        bx, [=] AMREX_GPU_DEVICE(int i, int j, int k)
        { errorMF(i, j, k) = std::abs(projectionMF(i, j, k) - analyticalMF(i, j, k)); });
}

void update_zero_form_dual_parallel_for (amrex::MFIter& mfi,
                                         DeRhamField<Grid::dual, Space::node>& pointValsDual,
                                         ComputationalDomain& mInfra)
{
    const amrex::Box& bx = mfi.validbox();
    amrex::Array4<amrex::Real> const& zeroForm = (pointValsDual.m_data)[mfi].array();

    const amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> dr = mInfra.geometry().CellSizeArray();
    const amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> r0 = mInfra.m_geom.ProbLoArray();

    amrex::ParallelFor(
        bx,
        [=] AMREX_GPU_DEVICE(int i, int j, int k)
        {
            amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> r = {AMREX_D_DECL(
                r0[xDir] + 0.5 * dr[xDir] + i * dr[xDir], r0[yDir] + 0.5 * dr[yDir] + j * dr[yDir],
                r0[zDir] + 0.5 * dr[zDir] + k * dr[zDir])};

            zeroForm(i, j, k) =
                GEMPIC_D_MULT(std::cos(r[xDir]), std::cos(r[yDir]), std::cos(r[zDir]));
        });
}

void update_zero_form_dual_error_parallel_for (amrex::MFIter& mfi,
                                               DeRhamField<Grid::dual, Space::node>& pointValsDual,
                                               DeRhamField<Grid::dual, Space::node>& qDual,
                                               DeRhamField<Grid::dual, Space::node>& errorQDual)
{
    const amrex::Box& bx = mfi.validbox();

    amrex::Array4<amrex::Real> const& projectionMF = (qDual.m_data)[mfi].array();
    amrex::Array4<amrex::Real> const& analyticalMF = (pointValsDual.m_data)[mfi].array();
    amrex::Array4<amrex::Real> const& errorMF = (errorQDual.m_data)[mfi].array();

    amrex::ParallelFor(
        bx, [=] AMREX_GPU_DEVICE(int i, int j, int k)
        { errorMF(i, j, k) = std::abs(projectionMF(i, j, k) - analyticalMF(i, j, k)); });
}

void update_three_form_primal_parallel_for (amrex::MFIter& mfi,
                                            DeRhamField<Grid::primal, Space::cell>& rhoAn,
                                            ComputationalDomain& mInfra)
{
    const amrex::Box& bx = mfi.validbox();
    amrex::Array4<amrex::Real> const& threeForm = (rhoAn.m_data)[mfi].array();

    const amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> dr = mInfra.geometry().CellSizeArray();
    const amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> r0 = mInfra.m_geom.ProbLoArray();

    amrex::ParallelFor(
        bx,
        [=] AMREX_GPU_DEVICE(int i, int j, int k)
        {
            amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> r = {AMREX_D_DECL(
                r0[xDir] + i * dr[xDir], r0[yDir] + j * dr[yDir], r0[zDir] + k * dr[zDir])};

            threeForm(i, j, k) = GEMPIC_D_MULT((std::sin(r[xDir] + dr[xDir]) - std::sin(r[xDir])),
                                               (std::sin(r[yDir] + dr[yDir]) - std::sin(r[yDir])),
                                               (std::sin(r[zDir] + dr[zDir]) - std::sin(r[zDir])));
        });
}

void update_three_form_primal_error_parallel_for (amrex::MFIter& mfi,
                                                  DeRhamField<Grid::primal, Space::cell>& rhoAn,
                                                  DeRhamField<Grid::primal, Space::cell>& rho,
                                                  DeRhamField<Grid::primal, Space::cell>& errorRho)
{
    const amrex::Box& bx = mfi.validbox();

    amrex::Array4<amrex::Real> const& projectionMF = (rho.m_data)[mfi].array();
    amrex::Array4<amrex::Real> const& analyticalMF = (rhoAn.m_data)[mfi].array();
    amrex::Array4<amrex::Real> const& errorMF = (errorRho.m_data)[mfi].array();

    amrex::ParallelFor(
        bx, [=] AMREX_GPU_DEVICE(int i, int j, int k)
        { errorMF(i, j, k) = std::abs(projectionMF(i, j, k) - analyticalMF(i, j, k)); });
}

void update_three_form_dual_parallel_for (amrex::MFIter& mfi,
                                          DeRhamField<Grid::dual, Space::cell>& rhoAnDual,
                                          ComputationalDomain& mInfra)
{
    const amrex::Box& bx = mfi.validbox();
    amrex::Array4<amrex::Real> const& threeForm = (rhoAnDual.m_data)[mfi].array();

    const amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> dr = mInfra.geometry().CellSizeArray();
    const amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> r0 = mInfra.m_geom.ProbLoArray();

    amrex::ParallelFor(
        bx,
        [=] AMREX_GPU_DEVICE(int i, int j, int k)
        {
            amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> r = {AMREX_D_DECL(
                r0[xDir] + 0.5 * dr[xDir] + i * dr[xDir], r0[yDir] + 0.5 * dr[yDir] + j * dr[yDir],
                r0[zDir] + 0.5 * dr[zDir] + k * dr[zDir])};

            threeForm(i, j, k) = GEMPIC_D_MULT((std::sin(r[xDir]) - std::sin(r[xDir] - dr[xDir])),
                                               (std::sin(r[yDir]) - std::sin(r[yDir] - dr[yDir])),
                                               (std::sin(r[zDir]) - std::sin(r[zDir] - dr[zDir])));
        });
}

void update_three_form_dual_error_parallel_for (amrex::MFIter& mfi,
                                                DeRhamField<Grid::dual, Space::cell>& rhoAnDual,
                                                DeRhamField<Grid::dual, Space::cell>& rhoDual,
                                                DeRhamField<Grid::dual, Space::cell>& errorRhoDual)
{
    const amrex::Box& bx = mfi.validbox();

    amrex::Array4<amrex::Real> const& projectionMF = (rhoDual.m_data)[mfi].array();
    amrex::Array4<amrex::Real> const& analyticalMF = (rhoAnDual.m_data)[mfi].array();
    amrex::Array4<amrex::Real> const& errorMF = (errorRhoDual.m_data)[mfi].array();

    amrex::ParallelFor(
        bx, [=] AMREX_GPU_DEVICE(int i, int j, int k)
        { errorMF(i, j, k) = std::abs(projectionMF(i, j, k) - analyticalMF(i, j, k)); });
}

class FDDeRhamComplexProjectionTest : public testing::Test
{
protected:
    // Linear splines is ok, and lower dimension Hodge is good enough
    // Spline degreesx
    static const int s_degX{1};
    static const int s_degY{1};
    static const int s_degZ{1};
    inline static const int s_maxSplineDegree{std::max(std::max(s_degX, s_degY), s_degZ)};
    inline static const int s_hodgeDegree{2};

    Gempic::Io::Parameters m_parameters{};
    Gempic::ComputationalDomain m_infra{false}; // "uninitialized" computational domain

    int m_gaussNodes = 6;
    const amrex::Real m_tol = 1e-15;

    static void SetUpTestSuite ()
    {
        /* Initialize the infrastructure */

        const amrex::Vector<amrex::Real> domainLo{
            AMREX_D_DECL(-M_PI + 0.3, -M_PI + 0.6, -M_PI + 0.4)};
        const amrex::Vector<amrex::Real> k{AMREX_D_DECL(1.0, 1.0, 1.0)};
        const amrex::Vector<int> nCell{AMREX_D_DECL(9, 8, 7)};
        const amrex::Vector<int> maxGridSize{AMREX_D_DECL(3, 4, 5)};
        const amrex::Vector<int> isPeriodic{AMREX_D_DECL(1, 1, 1)};

        amrex::ParmParse pp;
        pp.addarr("ComputationalDomain.domainLo", domainLo);
        pp.addarr("k", k);
        pp.addarr("ComputationalDomain.nCell", nCell);
        pp.addarr("ComputationalDomain.maxGridSize", maxGridSize);
        pp.addarr("ComputationalDomain.isPeriodic", isPeriodic);
    }

    void SetUp () override { m_infra = Gempic::ComputationalDomain{}; }
};

TEST_F(FDDeRhamComplexProjectionTest, TestProjectionOneTwoForms)
{
    // Parse analytical fields and and initialize parserEval
#if (AMREX_SPACEDIM == 1)
    const amrex::Array<std::string, 3> analyticalE = {
        "cos(x)",
        "cos(x)",
        "cos(x)",
    };
#endif

#if (AMREX_SPACEDIM == 2)
    const amrex::Array<std::string, 3> analyticalE = {
        "cos(x) * cos(y)",
        "cos(x) * cos(y)",
        "cos(x) * cos(y)",
    };
#endif

#if (AMREX_SPACEDIM == 3)
    const amrex::Array<std::string, 3> analyticalE = {
        "cos(x) * cos(y) * cos(z)",
        "cos(x) * cos(y) * cos(z)",
        "cos(x) * cos(y) * cos(z)",
    };
#endif

    const int nVar = AMREX_SPACEDIM + 1; // x, y, z, t
    amrex::Array<amrex::ParserExecutor<nVar>, 3> func;
    amrex::Array<amrex::Parser, 3> parser;
    for (int i = 0; i < 3; ++i)
    {
        parser[i].define(analyticalE[i]);
        parser[i].registerVariables({AMREX_D_DECL("x", "y", "z"), "t"});
        func[i] = parser[i].compile<nVar>();
    }

    // Initialize the De Rham Complex
    auto deRham = std::make_shared<FDDeRhamComplex>(m_infra, s_hodgeDegree, s_maxSplineDegree,
                                                    HodgeScheme::FDHodge);
    // Declare the fields
    DeRhamField<Grid::primal, Space::edge> E(deRham);

    // Compute the analytical result of the integral
    DeRhamField<Grid::primal, Space::edge> lineIntegral(deRham);

    for (int comp = 0; comp < 3; ++comp)
    {
        for (amrex::MFIter mfi(lineIntegral.m_data[comp]); mfi.isValid(); ++mfi)
        {
            update_one_form_primal_parallel_for(mfi, lineIntegral, m_infra, comp);
        }
    }

    lineIntegral.fill_boundary();

    // Compute the projection of the field
    deRham->projection(func, 0.0, E, m_gaussNodes);

    DeRhamField<Grid::primal, Space::edge> errorE(deRham);
    for (int comp = 0; comp < 3; ++comp)
    {
        for (amrex::MFIter mfi(E.m_data[comp]); mfi.isValid(); ++mfi)
        {
            update_one_form_primal_error_parallel_for(mfi, lineIntegral, E, errorE, comp);
        }
    }

    amrex::Real errorExNorm0 = errorE.m_data[xDir].norm0();
    amrex::Real errorEyNorm0 = errorE.m_data[yDir].norm0();
    amrex::Real errorEzNorm0 = errorE.m_data[zDir].norm0();

    EXPECT_NEAR(0, errorExNorm0, m_tol);
    EXPECT_NEAR(0, errorEyNorm0, m_tol);
    EXPECT_NEAR(0, errorEzNorm0, m_tol);

    // Run the same test for face integrals
    // Declare the fields
    DeRhamField<Grid::primal, Space::face> B(deRham);
    DeRhamField<Grid::primal, Space::face> faceIntegral(deRham);

    // Parse analytical fields and and initialize parserEval
#if (AMREX_SPACEDIM == 1)
    const amrex::Array<std::string, 3> analyticalB = {
        "cos(x)",
        "cos(x)",
        "cos(x)",
    };
#endif
#if (AMREX_SPACEDIM == 2)
    const amrex::Array<std::string, 3> analyticalB = {
        "cos(x) * cos(y)",
        "cos(x) * cos(y)",
        "cos(x) * cos(y)",
    };
#endif
#if (AMREX_SPACEDIM == 3)
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
            update_two_form_primal_parallel_for(mfi, faceIntegral, m_infra, comp);
        }
    }

    faceIntegral.fill_boundary();

    // Compute the projection of B
    deRham->projection(func, 0.0, B, m_gaussNodes);

    DeRhamField<Grid::primal, Space::face> errorB(deRham);
    for (int comp = 0; comp < 3; ++comp)
    {
        for (amrex::MFIter mfi(B.m_data[comp]); mfi.isValid(); ++mfi)
        {
            update_two_form_primal_error_parallel_for(mfi, faceIntegral, B, errorB, comp);
        }
    }

    amrex::Real errorBxNorm0 = errorB.m_data[xDir].norm0();
    amrex::Real errorByNorm0 = errorB.m_data[yDir].norm0();
    amrex::Real errorBzNorm0 = errorB.m_data[zDir].norm0();

    EXPECT_NEAR(0, errorBxNorm0, m_tol);
    EXPECT_NEAR(0, errorByNorm0, m_tol);
    EXPECT_NEAR(0, errorBzNorm0, m_tol);

    // same test on dual grid
    DeRhamField<Grid::dual, Space::edge> H(deRham);

#if (AMREX_SPACEDIM == 1)
    const amrex::Array<std::string, 3> analyticalH = {
        "cos(x)",
        "cos(x)",
        "cos(x)",
    };
#endif
#if (AMREX_SPACEDIM == 2)
    const amrex::Array<std::string, 3> analyticalH = {
        "cos(x) * cos(y)",
        "cos(x) * cos(y)",
        "cos(x) * cos(y)",
    };
#endif
#if (AMREX_SPACEDIM == 3)
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
    deRham->projection(func, 0.0, H, m_gaussNodes);

    // Compute the analytical solution
    DeRhamField<Grid::dual, Space::edge> lineIntegralDual(deRham);

    for (int comp = 0; comp < 3; ++comp)
    {
        for (amrex::MFIter mfi(lineIntegralDual.m_data[comp]); mfi.isValid(); ++mfi)
        {
            update_one_form_dual_parallel_for(mfi, lineIntegralDual, m_infra, comp);
        }
    }

    lineIntegralDual.fill_boundary();

    DeRhamField<Grid::dual, Space::edge> errorH(deRham);
    for (int comp = 0; comp < 3; ++comp)
    {
        for (amrex::MFIter mfi(H.m_data[comp]); mfi.isValid(); ++mfi)
        {
            update_one_form_dual_error_parallel_for(mfi, lineIntegralDual, H, errorH, comp);
        }
    }

    amrex::Real errorHxNorm0 = errorH.m_data[xDir].norm0();
    amrex::Real errorHyNorm0 = errorH.m_data[yDir].norm0();
    amrex::Real errorHzNorm0 = errorH.m_data[zDir].norm0();

    EXPECT_NEAR(0, std::max({errorHxNorm0, errorHyNorm0, errorHzNorm0}), m_tol);

    // Declare the fields
    DeRhamField<Grid::dual, Space::face> D(deRham);

#if (AMREX_SPACEDIM == 1)
    const amrex::Array<std::string, 3> analyticalD = {
        "cos(x)",
        "cos(x)",
        "cos(x)",
    };
#endif
#if (AMREX_SPACEDIM == 2)
    const amrex::Array<std::string, 3> analyticalD = {
        "cos(x) * cos(y)",
        "cos(x) * cos(y)",
        "cos(x) * cos(y)",
    };
#endif
#if (AMREX_SPACEDIM == 3)
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
    deRham->projection(func, 0.0, D, m_gaussNodes);

    // Compute the analytical solution
    DeRhamField<Grid::dual, Space::face> faceIntegralDual(deRham);

    for (int comp = 0; comp < 3; ++comp)
    {
        for (amrex::MFIter mfi(faceIntegralDual.m_data[comp]); mfi.isValid(); ++mfi)
        {
            update_two_form_dual_parallel_for(mfi, faceIntegralDual, m_infra, comp);
        }
    }

    faceIntegralDual.fill_boundary();

    DeRhamField<Grid::dual, Space::face> errorD(deRham);
    for (int comp = 0; comp < 3; ++comp)
    {
        for (amrex::MFIter mfi(D.m_data[comp]); mfi.isValid(); ++mfi)
        {
            update_two_form_dual_error_parallel_for(mfi, faceIntegralDual, D, errorD, comp);
        }
    }

    amrex::Real errorDxNorm0 = errorD.m_data[xDir].norm0();
    amrex::Real errorDyNorm0 = errorD.m_data[yDir].norm0();
    amrex::Real errorDzNorm0 = errorD.m_data[zDir].norm0();

    EXPECT_NEAR(0, errorDxNorm0, m_tol);
    EXPECT_NEAR(0, errorDyNorm0, m_tol);
    EXPECT_NEAR(0, errorDzNorm0, m_tol);
}

TEST_F(FDDeRhamComplexProjectionTest, TestProjectionZeroThreeForms)
{
    // Initialize the De Rham Complex
    auto deRham = std::make_shared<FDDeRhamComplex>(m_infra, s_hodgeDegree, s_maxSplineDegree,
                                                    HodgeScheme::FDHodge);

    // Declare the fields
    DeRhamField<Grid::primal, Space::node> Q(deRham);

#if (AMREX_SPACEDIM == 1)
    const std::string analyticalQ = "cos(x)";
#endif
#if (AMREX_SPACEDIM == 2)
    const std::string analyticalQ = "cos(x) * cos(y)";
#endif
#if (AMREX_SPACEDIM == 3)
    const std::string analyticalQ = "cos(x) * cos(y) * cos(z)";
#endif

    const int nVar = AMREX_SPACEDIM + 1; // x, y, z, t
    amrex::ParserExecutor<nVar> func;
    amrex::Parser parser;

    parser.define(analyticalQ);
    parser.registerVariables({AMREX_D_DECL("x", "y", "z"), "t"});
    func = parser.compile<nVar>();

    // Compute the projection of the field
    deRham->projection(func, 0.0, Q);

    // Compute the analytical result of the field
    DeRhamField<Grid::primal, Space::node> pointVals(deRham);

    for (amrex::MFIter mfi(pointVals.m_data); mfi.isValid(); ++mfi)
    {
        update_zero_form_primal_parallel_for(mfi, pointVals, m_infra);
    }

    pointVals.fill_boundary();

    // Calculate error
    DeRhamField<Grid::primal, Space::node> errorQ(deRham);
    for (amrex::MFIter mfi(Q.m_data); mfi.isValid(); ++mfi)
    {
        update_zero_form_primal_error_parallel_for(mfi, pointVals, Q, errorQ);
    }

    amrex::Real errorQNorm0 = errorQ.m_data.norm0();

    EXPECT_NEAR(0, errorQNorm0, m_tol);

    // Same test for dual, node
    DeRhamField<Grid::dual, Space::node> qDual(deRham);

#if (AMREX_SPACEDIM == 1)
    const std::string analyticalQDual = "cos(x)";
#endif
#if (AMREX_SPACEDIM == 2)
    const std::string analyticalQDual = "cos(x) * cos(y)";
#endif
#if (AMREX_SPACEDIM == 3)
    const std::string analyticalQDual = "cos(x) * cos(y) * cos(z)";
#endif

    parser.define(analyticalQDual);
    parser.registerVariables({AMREX_D_DECL("x", "y", "z"), "t"});
    func = parser.compile<nVar>();

    // Compute the projection of the field
    deRham->projection(func, 0.0, qDual);

    // Compute the analytical result of the field
    DeRhamField<Grid::dual, Space::node> pointValsDual(deRham);

    for (amrex::MFIter mfi(pointValsDual.m_data); mfi.isValid(); ++mfi)
    {
        update_zero_form_dual_parallel_for(mfi, pointValsDual, m_infra);
    }

    pointValsDual.fill_boundary();

    // Calculate error
    DeRhamField<Grid::dual, Space::node> errorQDual(deRham);
    for (amrex::MFIter mfi(qDual.m_data); mfi.isValid(); ++mfi)
    {
        update_zero_form_dual_error_parallel_for(mfi, pointValsDual, qDual, errorQDual);
    }

    amrex::Real errorQDualNorm0 = errorQDual.m_data.norm0();

    EXPECT_NEAR(0, errorQDualNorm0, m_tol);

    // Test projection for primal three form
    DeRhamField<Grid::primal, Space::cell> rho(deRham);

#if (AMREX_SPACEDIM == 1)
    const std::string analyticalRho = "cos(x)";
#endif
#if (AMREX_SPACEDIM == 2)
    const std::string analyticalRho = "cos(x) * cos(y)";
#endif
#if (AMREX_SPACEDIM == 3)
    const std::string analyticalRho = "cos(x) * cos(y) * cos(z)";
#endif

    parser.define(analyticalRho);
    parser.registerVariables({AMREX_D_DECL("x", "y", "z"), "t"});
    func = parser.compile<nVar>();

    // Compute the projection of the field
    deRham->projection(func, 0.0, rho, m_gaussNodes);

    // Compute the analytical result of the field
    DeRhamField<Grid::primal, Space::cell> rhoAn(deRham);

    for (amrex::MFIter mfi(rhoAn.m_data); mfi.isValid(); ++mfi)
    {
        update_three_form_primal_parallel_for(mfi, rhoAn, m_infra);
    }

    rhoAn.fill_boundary();

    // Calculate error
    DeRhamField<Grid::primal, Space::cell> errorRho(deRham);
    for (amrex::MFIter mfi(rho.m_data); mfi.isValid(); ++mfi)
    {
        update_three_form_primal_error_parallel_for(mfi, rhoAn, rho, errorRho);
    }

    amrex::Real errorRhoNorm0 = errorRho.m_data.norm0();

    EXPECT_NEAR(0, errorRhoNorm0, m_tol);

    // Test three form projection dual
    DeRhamField<Grid::dual, Space::cell> rhoDual(deRham);

#if (AMREX_SPACEDIM == 1)
    const std::string analyticalRhoDual = "cos(x)";
#endif
#if (AMREX_SPACEDIM == 2)
    const std::string analyticalRhoDual = "cos(x) * cos(y)";
#endif
#if (AMREX_SPACEDIM == 3)
    const std::string analyticalRhoDual = "cos(x) * cos(y) * cos(z)";
#endif

    parser.define(analyticalRhoDual);
    parser.registerVariables({AMREX_D_DECL("x", "y", "z"), "t"});
    func = parser.compile<nVar>();

    // Compute the projection of the field
    deRham->projection(func, 0.0, rhoDual, m_gaussNodes);

    // Compute the analytical result of the field
    DeRhamField<Grid::dual, Space::cell> rhoAnDual(deRham);

    for (amrex::MFIter mfi(rhoAnDual.m_data); mfi.isValid(); ++mfi)
    {
        update_three_form_dual_parallel_for(mfi, rhoAnDual, m_infra);
    }

    rhoAnDual.fill_boundary();

    // Calculate error
    DeRhamField<Grid::dual, Space::cell> errorRhoDual(deRham);
    for (amrex::MFIter mfi(rhoDual.m_data); mfi.isValid(); ++mfi)
    {
        update_three_form_dual_error_parallel_for(mfi, rhoAnDual, rhoDual, errorRhoDual);
    }

    amrex::Real errorRhoDualNorm0 = errorRhoDual.m_data.norm0();

    EXPECT_NEAR(0, errorRhoDualNorm0, m_tol);
}
} // namespace