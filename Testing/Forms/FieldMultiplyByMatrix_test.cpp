#include <gtest/gtest.h>

#include <AMReX.H>

#include "GEMPIC_ComputationalDomain.H"
#include "GEMPIC_FDDeRhamComplex.H"
#include "GEMPIC_Fields.H"
#include "GEMPIC_GempicNorm.H"
#include "GEMPIC_Parameters.H"
#include "TestUtils/GEMPIC_TestUtils.H"

using namespace Gempic;
using namespace Forms;

namespace
{

void initialize_tensor (amrex::MFIter& mfi,
                        amrex::MultiFab& mf,
                        amrex::GpuArray<amrex::Real, 3> val)
{
    amrex::Box const& bx = mfi.tilebox();
    amrex::Array4<amrex::Real> tensorArray = mf[mfi].array();
    ParallelFor(bx,
                [=] AMREX_GPU_DEVICE(int i, int j, int k)
                {
                    tensorArray(i, j, k, xDir) = val[0];
                    tensorArray(i, j, k, yDir) = val[1];
                    tensorArray(i, j, k, zDir) = val[2];
                });
}

ComputationalDomain get_compdom ()
{
    std::array<amrex::Real, AMREX_SPACEDIM> const domainLo{AMREX_D_DECL(0.0, 0.0, 0.0)};
    std::array<amrex::Real, AMREX_SPACEDIM> const domainHi{AMREX_D_DECL(1.0, 1.0, 1.0)};
    amrex::IntVect const nCell{AMREX_D_DECL(2, 2, 2)};
    amrex::IntVect const maxGridSize{AMREX_D_DECL(2, 2, 2)};
    std::array<int, AMREX_SPACEDIM> const isPeriodic{AMREX_D_DECL(1, 1, 1)};

    return ComputationalDomain(domainLo, domainHi, nCell, maxGridSize, isPeriodic);
}

class FieldMultiplyByMatrixTest : public testing::Test
{
protected:
    // Linear splines is ok, and lower dimension Hodge is good enough
    // Spline degreesx
    static int const s_degX{1};
    static int const s_degY{1};
    static int const s_degZ{1};
    inline static int const s_maxSplineDegree{std::max(std::max(s_degX, s_degY), s_degZ)};

    Io::Parameters m_parameters{};
    ComputationalDomain m_infra;
    amrex::Real m_tol{1e-11};

    FieldMultiplyByMatrixTest() : m_infra{get_compdom()}
    {
        // Not checking particles
        int const nGhostExtra{1};
        m_parameters.set("nGhostExtra", nGhostExtra);
    }
};
//Test Eout = M*D
//input D: constant field
//tensor M=[1 1 1; 1 1 1; 1 1 1]
TEST_F(FieldMultiplyByMatrixTest, ConstField)
{
#if AMREX_SPACEDIM != 3
    GTEST_SKIP();
#endif
    constexpr int hodgeDegree{2};

    // Initialize the De Rham Complex with deg 2
    auto deRham = std::make_shared<FDDeRhamComplex>(m_infra, hodgeDegree, s_maxSplineDegree,
                                                    HodgeScheme::FDHodge);

    DeRhamField<Grid::dual, Space::face> D(deRham);
    DeRhamField<Grid::primal, Space::edge> eOut(deRham);
    DeRhamField<Grid::primal, Space::edge> eResult(deRham);

    // define tensor to multiply with by a fields with three components and three variables each
    DeRhamField<Grid::primal, Space::edge> tensor(deRham, 3);

    // initialize fields and tensor
    auto dx = m_infra.cell_size_array();
    amrex::Real scalingX = GEMPIC_D_MULT(dx[xDir], 1 / dx[yDir], 1 / dx[zDir]);
    amrex::Real scalingY = GEMPIC_D_MULT(1 / dx[xDir], dx[yDir], 1 / dx[zDir]);
    amrex::Real scalingZ = GEMPIC_D_MULT(1 / dx[xDir], 1 / dx[yDir], dx[zDir]);
    // works only for equal number of cells in each direction
    ASSERT_TRUE((scalingX == scalingY) && (scalingX == scalingZ));
    for (int comp = 0; comp < 3; comp++)
    {
        D.m_data[comp].setVal(1.0 / scalingX);
        eResult.m_data[comp].setVal(3.0);
        tensor.m_data[comp].setVal(1.0);
    }

    deRham->hodge_dk(eOut, D, tensor);
    bool loopRun{false};

    for (int comp = 0; comp < 3; ++comp)
    {
        loopRun = false;
        for (amrex::MFIter mfi(D.m_data[comp]); mfi.isValid(); ++mfi)
        {
            loopRun = true;
            amrex::Box const& bx = mfi.validbox();
            COMPARE_FIELDS((eOut.m_data[comp])[mfi].array(), (eResult.m_data[comp])[mfi].array(),
                           bx, m_tol);
        }
        ASSERT_TRUE(loopRun);
    }
}

// Test cOut = M * C
// Input: C (linear field)
// Tensor M is an identity matrix: [1 0 0; 0 1 0; 0 0 1]
TEST_F(FieldMultiplyByMatrixTest, LinearFieldDiagTensor)
{
#if AMREX_SPACEDIM != 3
    GTEST_SKIP();
#endif

    constexpr int hodgeDegree{2};

#if (AMREX_SPACEDIM == 1)
    amrex::Array<std::string, 3> const analyticalWeightFunc = {
        "x",
        "x",
        "x",
    };
#endif

#if (AMREX_SPACEDIM == 2)
    amrex::Array<std::string, 3> const analyticalWeightFunc = {
        "x * y",
        "x * y",
        "x * y",
    };
#endif

#if (AMREX_SPACEDIM == 3)
    amrex::Array<std::string, 3> const analyticalWeightFunc = {
        "x + y + z",
        "x + y + z",
        "x + y + z",
    };
#endif

    int const nVar = AMREX_SPACEDIM + 1; // x, y, z, t

    amrex::Array<amrex::ParserExecutor<nVar>, 3> weightFunc;
    amrex::Array<amrex::Parser, 3> weightParser;
    for (int i = 0; i < 3; ++i)
    {
        weightParser[i].define(analyticalWeightFunc[i]);
        weightParser[i].registerVariables({AMREX_D_DECL("x", "y", "z"), "t"});
        weightFunc[i] = weightParser[i].compile<nVar>();
    }

    amrex::Vector<amrex::Array<amrex::ParserExecutor<nVar>, 3>> weightFuncs{weightFunc, weightFunc,
                                                                            weightFunc};

    // Initialize the De Rham Complex with deg 2
    auto deRham = std::make_shared<FDDeRhamComplex>(m_infra, hodgeDegree, s_maxSplineDegree,
                                                    HodgeScheme::FDHodge);

    DeRhamField<Grid::dual, Space::face> C(deRham, weightFunc);
    DeRhamField<Grid::primal, Space::edge> cOut(deRham);
    DeRhamField<Grid::primal, Space::edge> cResult(deRham, weightFunc);

    // define tensor to multiply with by a field with three components and three variables each
    DeRhamField<Grid::primal, Space::edge> tensor(deRham, 3);
    amrex::GpuArray<amrex::Real, 3> val;

    // initialize tensor
    for (amrex::MFIter mfi(tensor.m_data[xDir], true); mfi.isValid(); ++mfi)
    {
        val = {1, 0, 0};
        initialize_tensor(mfi, tensor.m_data[xDir], val);
    }
    for (amrex::MFIter mfi(tensor.m_data[yDir], true); mfi.isValid(); ++mfi)
    {
        val = {0, 1, 0};
        initialize_tensor(mfi, tensor.m_data[yDir], val);
    }
    for (amrex::MFIter mfi(tensor.m_data[zDir], true); mfi.isValid(); ++mfi)
    {
        val = {0, 0, 1};
        initialize_tensor(mfi, tensor.m_data[zDir], val);
    }

    deRham->hodge_dk(cOut, C, tensor);

    bool loopRun{false};

    for (int comp = 0; comp < 3; ++comp)
    {
        loopRun = false;
        for (amrex::MFIter mfi(C.m_data[comp]); mfi.isValid(); ++mfi)
        {
            loopRun = true;
            amrex::Box const& bx = mfi.validbox();
            COMPARE_FIELDS((cOut.m_data[comp])[mfi].array(), (cResult.m_data[comp])[mfi].array(),
                           bx, m_tol);
        }
        ASSERT_TRUE(loopRun);
    }
}
} // namespace
