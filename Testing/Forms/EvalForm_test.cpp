/**************************************************************************************************
 * Copyright (c) 2021 GEMPICX                                                                     *
 * SPDX-License-Identifier: BSD-3-Clause                                                          *
 **************************************************************************************************/
#include <gtest/gtest.h>

#include <AMReX.H>

#include "GEMPIC_ComputationalDomain.H"
#include "GEMPIC_FDDeRhamComplex.H"
#include "GEMPIC_FieldMethods.H"
#include "GEMPIC_Fields.H"
#include "GEMPIC_GempicNorm.H"
#include "GEMPIC_Parameters.H"
#include "TestUtils/GEMPIC_TestUtils.H"

// E = one form
// rho = three form
// phi = zero form

namespace
{
using namespace Gempic;
using namespace Forms;

template <Grid grid, Space space>
struct GridSpaceTypes
{
    static constexpr Grid s_grid{grid};
    static constexpr Space s_space{space};
};

// Supplements the gtest naming of typed tests
void testname_addition (Grid grid, Space space)
{
    switch (grid)
    {
        case Grid::primal:
            amrex::Print() << "Primal ";
            break;
        case Grid::dual:
            amrex::Print() << "Dual ";
            break;
        default:
            amrex::Print() << "Unknown grid, ";
            break;
    }
    switch (space)
    {
        case Space::node:
            amrex::Print() << "zero-form (node)\n";
            break;
        case Space::edge:
            amrex::Print() << "one-form (edge)\n";
            break;
        case Space::face:
            amrex::Print() << "two-form (face)\n";
            break;
        case Space::cell:
            amrex::Print() << "three-form (cell)\n";
            break;
        default:
            amrex::Print() << "unknown form\n";
            break;
    }
}

namespace
{
// Calculate the analytical point values of rho: func(x,y,z)
void compute_analytical_scalar_function_parallel_for (
    amrex::MFIter& mfi,
    ComputationalDomain& mInfra,
    amrex::ParserExecutor<AMREX_SPACEDIM + 1>& func,
    amrex::MultiFab& analyticalPointValues,
    amrex::GpuArray<amrex::Real, AMREX_SPACEDIM>* const evalShiftGpuPtr,
    size_t nValues)
{
    amrex::Box const& bx = mfi.validbox();
    amrex::Array4<amrex::Real> const& analyticalPointValuesMF = analyticalPointValues[mfi].array();

    amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const dr = mInfra.geometry().CellSizeArray();
    amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const r0 = mInfra.m_geom.ProbLoArray();

    ParallelFor(bx, nValues,
                [=] AMREX_GPU_DEVICE(int i, int j, int k, int n)
                {
                    // Compute the position of the point i, j, k
                    amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> rr = {AMREX_D_DECL(
                        r0[xDir] + i * dr[xDir], r0[yDir] + j * dr[yDir], r0[zDir] + k * dr[zDir])};

                    amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> r = {
                        AMREX_D_DECL(rr[xDir] + evalShiftGpuPtr[n][xDir] * dr[xDir],
                                     rr[yDir] + evalShiftGpuPtr[n][yDir] * dr[yDir],
                                     rr[zDir] + evalShiftGpuPtr[n][zDir] * dr[zDir])};

                    analyticalPointValuesMF(i, j, k, n) =
                        func({AMREX_D_DECL(r[xDir], r[yDir], r[zDir]), 0.0});
                });
}

// Calculate the analytical point values of a vector valued function: func(x,y,z)
void compute_analytical_vector_function_parallel_for (
    amrex::MFIter& mfi,
    ComputationalDomain& mInfra,
    amrex::ParserExecutor<AMREX_SPACEDIM + 1>* const funcPtr,
    amrex::MultiFab& analyticalPointValues,
    amrex::GpuArray<amrex::Real, AMREX_SPACEDIM>* const evalShiftGpuPtr,
    size_t nEvals,
    int nComp)
{
    amrex::Box const& bx = mfi.validbox();
    amrex::Array4<amrex::Real> const& analyticalPointValuesMF = analyticalPointValues[mfi].array();

    amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const dr = mInfra.geometry().CellSizeArray();
    amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const r0 = mInfra.m_geom.ProbLoArray();

    ParallelFor(
        bx, nComp,
        [=] AMREX_GPU_DEVICE(int i, int j, int k, int n)
        {
            for (int evalInd = 0; evalInd < nEvals; ++evalInd)
            {
                // Compute the position of the point i, j, k
                amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> rr = {AMREX_D_DECL(
                    r0[xDir] + i * dr[xDir], r0[yDir] + j * dr[yDir], r0[zDir] + k * dr[zDir])};

                amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> r = {
                    AMREX_D_DECL(rr[xDir] + evalShiftGpuPtr[evalInd][xDir] * dr[xDir],
                                 rr[yDir] + evalShiftGpuPtr[evalInd][yDir] * dr[yDir],
                                 rr[zDir] + evalShiftGpuPtr[evalInd][zDir] * dr[zDir])};

                amrex::Real fval = funcPtr[n]({AMREX_D_DECL(r[xDir], r[yDir], r[zDir]), 0.0});

                analyticalPointValuesMF(i, j, k, n * nEvals + evalInd) = fval;
            }
        });
}

//@todo: These tests should work with the default ComputationDomain, but don't. Why?
ComputationalDomain get_compdom ()
{
    std::array<amrex::Real, AMREX_SPACEDIM> const domainLo{AMREX_D_DECL(0.0, 0.0, 0.0)};
    std::array<amrex::Real, AMREX_SPACEDIM> const domainHi{AMREX_D_DECL(1.0, 1.0, 1.0)};
    amrex::IntVect const nCell{AMREX_D_DECL(10, 10, 10)};
    amrex::IntVect const maxGridSize{AMREX_D_DECL(10, 10, 10)};
    std::array<int, AMREX_SPACEDIM> const isPeriodic{AMREX_D_DECL(0, 0, 0)};

    return ComputationalDomain(domainLo, domainHi, nCell, maxGridSize, isPeriodic);
}

template <typename Form>
class FDDeRhamComplexEvalFormTest : public testing::Test
{
protected:
    static constexpr Grid s_grid{Form::s_grid};
    static constexpr Space s_space{Form::s_space};

    // Linear splines is ok, and lower dimension Hodge is good enough
    // Spline degreesx
    static int const s_degX{1};
    static int const s_degY{1};
    static int const s_degZ{1};
    inline static int const s_maxSplineDegree{std::max(std::max(s_degX, s_degY), s_degZ)};

    Io::Parameters m_parameters{};
    ComputationalDomain m_infra;

    int m_gaussNodes = 6;
    amrex::Real const m_tol = 1e-13;

    FDDeRhamComplexEvalFormTest() : m_infra{get_compdom()}
    {
        // Not checking particles
        int const nGhostExtra{1}; //{-s_maxSplineDegree};

        m_parameters.set("nGhostExtra", nGhostExtra);
    }
};

using ZeroOrThreeFormTypes = ::testing::Types<GridSpaceTypes<Grid::primal, Space::node>,
                                              GridSpaceTypes<Grid::primal, Space::cell>,
                                              GridSpaceTypes<Grid::dual, Space::node>,
                                              GridSpaceTypes<Grid::dual, Space::cell>>;

TYPED_TEST_SUITE(FDDeRhamComplexEvalFormTest, ZeroOrThreeFormTypes);

TYPED_TEST(FDDeRhamComplexEvalFormTest, EvalFormZeroThreeForm)
{
    testname_addition(TestFixture::s_grid, TestFixture::s_space);
    constexpr int hodgeDegree{4};

    auto deRham = std::make_shared<FDDeRhamComplex>(this->m_infra, hodgeDegree,
                                                    this->s_maxSplineDegree, HodgeScheme::FDHodge);

    DeRhamField<TestFixture::s_grid, TestFixture::s_space> form(deRham);

#if (AMREX_SPACEDIM == 1)
    std::string const analyticalForm = "x";
#elif (AMREX_SPACEDIM == 2)
    std::string const analyticalForm = "x + y";
#elif (AMREX_SPACEDIM == 3)
    std::string const analyticalForm = "x + y + z";
#endif

    int const nVar = AMREX_SPACEDIM + 1; // x, y, z, t
    amrex::ParserExecutor<nVar> func;
    amrex::Parser parser;

    parser.define(analyticalForm);
    parser.registerVariables({AMREX_D_DECL("x", "y", "z"), "t"});
    func = parser.compile<nVar>();

    // Compute the projection of the field
    if constexpr (TestFixture::s_space == Space::node)
    {
        projection(func, 0.0, form); // Projection of func to the discrete 0-form
    }
    else
    {
        projection(func, 0.0, form,
                   this->m_gaussNodes); // Projection of func to the discrete 3-form
    }

    amrex::BoxArray const& ba = form.m_data.boxArray();
    amrex::DistributionMapping const& dm = form.m_data.DistributionMap();
    int nghost = form.m_data.nGrow();

    // test with 2 interpolation points
    size_t nValues = 2;
    amrex::MultiFab pointValues(ba, dm, nValues, nghost);
    amrex::MultiFab analyticalPointValues(ba, dm, nValues, nghost);

    // Compute the interpolations
    amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> evalShiftArray1{{AMREX_D_DECL(0.25, 0.25, 0.25)}};
    amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> evalShiftArray2{{AMREX_D_DECL(0.75, 0.75, 0.75)}};
    amrex::Vector<amrex::GpuArray<amrex::Real, AMREX_SPACEDIM>> evalShift = {evalShiftArray1,
                                                                             evalShiftArray2};

    eval_form(form, evalShift, pointValues);

    // Copy evalShift to GPU array
    amrex::Gpu::DeviceVector<amrex::GpuArray<amrex::Real, AMREX_SPACEDIM>> evalShiftGpu{nValues};
    amrex::Gpu::copyAsync(amrex::Gpu::hostToDevice, evalShift.begin(), evalShift.end(),
                          evalShiftGpu.begin());
    auto* const evalShiftGpuPtr = evalShiftGpu.dataPtr();

    // Calculate func in evalShift
    for (amrex::MFIter mfi(form.m_data); mfi.isValid(); ++mfi)
    {
        compute_analytical_scalar_function_parallel_for(
            mfi, this->m_infra, func, analyticalPointValues, evalShiftGpuPtr, nValues);
    }

    bool loopRun{false};

    for (amrex::MFIter mfi(form.m_data); mfi.isValid(); ++mfi)
    {
        loopRun = true;

        amrex::Box const& bx = mfi.tilebox();
        COMPARE_FIELDS(pointValues[mfi].array(), analyticalPointValues[mfi].array(), bx);
    }
    ASSERT_TRUE(loopRun);
}

TYPED_TEST(FDDeRhamComplexEvalFormTest, EvalFormMultivaluedZeroThreeForm)
{
    testname_addition(TestFixture::s_grid, TestFixture::s_space);
    constexpr int hodgeDegree{4};

    auto deRham = std::make_shared<FDDeRhamComplex>(this->m_infra, hodgeDegree,
                                                    this->s_maxSplineDegree, HodgeScheme::FDHodge);

    // test with 2 values
    int nComp{2};
    DeRhamField<TestFixture::s_grid, TestFixture::s_space> form(deRham, nComp);

#if (AMREX_SPACEDIM == 1)
    amrex::Vector<std::string> const analyticalform = {
        "x",
        "-x",
    };
#elif (AMREX_SPACEDIM == 2)
    amrex::Vector<std::string> const analyticalform = {
        "x + y",
        "-x - y",
    };
#elif (AMREX_SPACEDIM == 3)
    amrex::Vector<std::string> const analyticalform = {
        "x + y + z",
        "-x - y - z",
    };
#endif

    int const nVar = AMREX_SPACEDIM + 1; // x, y, z, t
    amrex::Vector<amrex::ParserExecutor<nVar>> funcs(nComp);
    amrex::Vector<amrex::Parser> parser(nComp);

    for (int i = 0; i < nComp; ++i)
    {
        parser[i].define(analyticalform[i]);
        parser[i].registerVariables({AMREX_D_DECL("x", "y", "z"), "t"});
        funcs[i] = parser[i].compile<nVar>();
    }

    // Compute the projection of the field
    if constexpr (TestFixture::s_space == Space::node)
    {
        projection(funcs, 0.0, form); // Projection of func to the discrete 0-form
    }
    else
    {
        projection(funcs, 0.0, form,
                   this->m_gaussNodes); // Projection of func to the discrete 3-form
    }

    amrex::BoxArray const& ba = form.m_data.boxArray();
    amrex::DistributionMapping const& dm = form.m_data.DistributionMap();
    int nghost = form.m_data.nGrow();

    // test with 2 interpolation points
    size_t nValues = 2;
    amrex::MultiFab pointValues(ba, dm, nValues * nComp, nghost);
    amrex::MultiFab analyticalPointValues(ba, dm, nValues * nComp, nghost);

    // Compute the interpolations
    amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> evalShiftArray1{{AMREX_D_DECL(0.25, 0.25, 0.25)}};
    amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> evalShiftArray2{{AMREX_D_DECL(0.75, 0.75, 0.75)}};
    amrex::Vector<amrex::GpuArray<amrex::Real, AMREX_SPACEDIM>> evalShift = {evalShiftArray1,
                                                                             evalShiftArray2};

    eval_form(form, evalShift, pointValues);

    // Copy evalShift to GPU array
    amrex::Gpu::DeviceVector<amrex::GpuArray<amrex::Real, AMREX_SPACEDIM>> evalShiftGpu{nValues};
    amrex::Gpu::copyAsync(amrex::Gpu::hostToDevice, evalShift.begin(), evalShift.end(),
                          evalShiftGpu.begin());
    auto* const evalShiftGpuPtr = evalShiftGpu.dataPtr();
    amrex::Gpu::DeviceVector<amrex::ParserExecutor<nVar>> funcsGpu{static_cast<size_t>(nComp)};
    amrex::Gpu::copyAsync(amrex::Gpu::hostToDevice, funcs.begin(), funcs.end(), funcsGpu.begin());
    auto* const funcsGpuPtr = funcsGpu.dataPtr();

    // Calculate func in evalShift
    for (amrex::MFIter mfi(form.m_data); mfi.isValid(); ++mfi)
    {
        compute_analytical_vector_function_parallel_for(mfi, this->m_infra, funcsGpuPtr,
                                                        analyticalPointValues, evalShiftGpuPtr,
                                                        nValues, nComp);
    }

    bool loopRun{false};

    for (amrex::MFIter mfi(form.m_data); mfi.isValid(); ++mfi)
    {
        loopRun = true;

        amrex::Box const& bx = mfi.tilebox();
        COMPARE_FIELDS(pointValues[mfi].array(), analyticalPointValues[mfi].array(), bx);
    }
    ASSERT_TRUE(loopRun);
}

template <typename Form>
class FDDeRhamComplexEvalFormTest2 : public FDDeRhamComplexEvalFormTest<Form>
{
};

using OneOrTwoFormTypes = ::testing::Types<GridSpaceTypes<Grid::primal, Space::edge>,
                                           GridSpaceTypes<Grid::primal, Space::face>,
                                           GridSpaceTypes<Grid::dual, Space::edge>,
                                           GridSpaceTypes<Grid::dual, Space::face>>;

TYPED_TEST_SUITE(FDDeRhamComplexEvalFormTest2, OneOrTwoFormTypes);
TYPED_TEST(FDDeRhamComplexEvalFormTest2, EvalFormOneTwoForm)
{
    testname_addition(TestFixture::s_grid, TestFixture::s_space);
    constexpr int hodgeDegree{4};

    auto deRham = std::make_shared<FDDeRhamComplex>(this->m_infra, hodgeDegree,
                                                    this->s_maxSplineDegree, HodgeScheme::FDHodge);

    DeRhamField<TestFixture::s_grid, TestFixture::s_space> form(deRham);

#if (AMREX_SPACEDIM == 1)
    amrex::Array<std::string, 3> const analyticalform = {
        "x",
        "x",
        "x",
    };
#elif (AMREX_SPACEDIM == 2)
    amrex::Array<std::string, 3> const analyticalform = {
        "x + y",
        "x + y",
        "x + y",
    };
#elif (AMREX_SPACEDIM == 3)
    amrex::Array<std::string, 3> const analyticalform = {
        "x + y + z",
        "x + y + z",
        "x + y + z",
    };
#endif

    int const nVar = AMREX_SPACEDIM + 1; // x, y, z, t
    amrex::Array<amrex::ParserExecutor<nVar>, 3> func;
    amrex::Array<amrex::Parser, 3> parser;

    for (int i = 0; i < 3; ++i)
    {
        parser[i].define(analyticalform[i]);
        parser[i].registerVariables({AMREX_D_DECL("x", "y", "z"), "t"});
        func[i] = parser[i].compile<nVar>();
    }

    // Compute the projection of the field
    projection(func, 0.0, form,
               this->m_gaussNodes); // Projection of func to the discrete form

    amrex::BoxArray const& ba =
        amrex::convert(form.m_data[xDir].boxArray(), amrex::IntVect(AMREX_D_DECL(0, 0, 0)));
    amrex::DistributionMapping const& dm = form.m_data[xDir].DistributionMap();
    int nghost = form.m_data[xDir].nGrow();
    // test with 2 interpolation points
    size_t nValues = 2;
    amrex::MultiFab pointValuesX(ba, dm, nValues, nghost);
    amrex::MultiFab analyticalPointValuesX(ba, dm, nValues, nghost);
    amrex::MultiFab pointValuesY(ba, dm, nValues, nghost);
    amrex::MultiFab analyticalPointValuesY(ba, dm, nValues, nghost);
    amrex::MultiFab pointValuesZ(ba, dm, nValues, nghost);
    amrex::MultiFab analyticalPointValuesZ(ba, dm, nValues, nghost);
    amrex::Array<amrex::MultiFab*, 3> pointValues{&pointValuesX, &pointValuesY, &pointValuesZ};
    amrex::Array<amrex::MultiFab*, 3> analyticalPointValues{
        &analyticalPointValuesX,
        &analyticalPointValuesY,
        &analyticalPointValuesZ,
    };

    // Compute the interpolations
    amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> evalShiftArray1{{AMREX_D_DECL(0.25, 0.25, 0.25)}};
    amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> evalShiftArray2{{AMREX_D_DECL(0.75, 0.75, 0.75)}};
    amrex::Vector<amrex::GpuArray<amrex::Real, AMREX_SPACEDIM>> evalShift = {evalShiftArray1,
                                                                             evalShiftArray2};

    // Copy evalShift to GPU array
    amrex::Gpu::DeviceVector<amrex::GpuArray<amrex::Real, AMREX_SPACEDIM>> evalShiftGpu{nValues};
    amrex::Gpu::copyAsync(amrex::Gpu::hostToDevice, evalShift.begin(), evalShift.end(),
                          evalShiftGpu.begin());
    auto* const evalShiftGpuPtr = evalShiftGpu.dataPtr();

    for (int comp{0}; comp < 3; ++comp)
    {
        // Calculate func in evalShift
        for (amrex::MFIter mfi(*analyticalPointValues[comp]); mfi.isValid(); ++mfi)
        {
            compute_analytical_scalar_function_parallel_for(mfi, this->m_infra, func[comp],
                                                            *analyticalPointValues[comp],
                                                            evalShiftGpuPtr, nValues);
        }
    }

    eval_form(form, evalShift, *pointValues[xDir], xDir);
    eval_form(form, evalShift, *pointValues[yDir], yDir);
    eval_form(form, evalShift, *pointValues[zDir], zDir);

    bool loopRun{false};

    for (int comp{0}; comp < 3; ++comp)
    {
        for (amrex::MFIter mfi(*analyticalPointValues[comp]); mfi.isValid(); ++mfi)
        {
            loopRun = true;

            amrex::Box const& bx = mfi.tilebox();
            COMPARE_FIELDS((*pointValues[comp])[mfi].array(),
                           (*analyticalPointValues[comp])[mfi].array(), bx);
        }
        ASSERT_TRUE(loopRun);

        // Preparation for the composite version eval_form test
        pointValues[comp]->setVal(0.0);
    }

    eval_form(form, evalShift, pointValues); // Check that this version conforms as well

    for (int comp{0}; comp < 3; ++comp)
    {
        for (amrex::MFIter mfi(*analyticalPointValues[comp]); mfi.isValid(); ++mfi)
        {
            amrex::Box const& bx = mfi.tilebox();
            COMPARE_FIELDS((*pointValues[comp])[mfi].array(),
                           (*analyticalPointValues[comp])[mfi].array(), bx);
        }
    }
}

TYPED_TEST(FDDeRhamComplexEvalFormTest2, EvalFormMultiValuedOneTwoForm)
{
    testname_addition(TestFixture::s_grid, TestFixture::s_space);
    constexpr int hodgeDegree{4};

    auto deRham = std::make_shared<FDDeRhamComplex>(this->m_infra, hodgeDegree,
                                                    this->s_maxSplineDegree, HodgeScheme::FDHodge);

    // test with 2 values
    int nComp{2};
    DeRhamField<TestFixture::s_grid, TestFixture::s_space> form(deRham, nComp);

#if (AMREX_SPACEDIM == 1)
    amrex::Vector<amrex::Array<std::string, 3>> const analyticalform{{
        {"x", "x", "x"},
        {"-x", "-x", "-x"},
    }};
#elif (AMREX_SPACEDIM == 2)
    amrex::Vector<amrex::Array<std::string, 3>> const analyticalform{{
        {"x + y", "x + y", "x + y"},
        {"-x  - y", "-x  - y", "-x  - y"},
    }};
#elif (AMREX_SPACEDIM == 3)
    amrex::Vector<amrex::Array<std::string, 3>> const analyticalform{{
        {"x + y + z", "x + y + z", "x + y + z"},
        {"-x  - y - z", "-x  - y - z", "-x  - y - z"},
    }};
#endif

    int const nVar = AMREX_SPACEDIM + 1; // x, y, z, t
    amrex::Vector<amrex::Array<amrex::ParserExecutor<nVar>, 3>> funcs(nComp);
    amrex::Vector<amrex::Array<amrex::Parser, 3>> parser(nComp);

    for (int i = 0; i < nComp; ++i)
    {
        for (int j = 0; j < 3; ++j)
        {
            parser[i][j].define(analyticalform[i][j]);
            parser[i][j].registerVariables({AMREX_D_DECL("x", "y", "z"), "t"});
            funcs[i][j] = parser[i][j].compile<nVar>();
        }
    }

    // Compute the projection of the field
    projection(funcs, 0.0, form,
               this->m_gaussNodes); // Projection of func to the discrete form

    amrex::BoxArray const& ba =
        amrex::convert(form.m_data[xDir].boxArray(), amrex::IntVect(AMREX_D_DECL(0, 0, 0)));
    amrex::DistributionMapping const& dm = form.m_data[xDir].DistributionMap();
    int nghost = form.m_data[xDir].nGrow();
    // test with 2 interpolation points
    size_t nValues = 2;
    amrex::MultiFab pointValuesX(ba, dm, nValues * nComp, nghost);
    amrex::MultiFab analyticalPointValuesX(ba, dm, nValues * nComp, nghost);
    amrex::MultiFab pointValuesY(ba, dm, nValues * nComp, nghost);
    amrex::MultiFab analyticalPointValuesY(ba, dm, nValues * nComp, nghost);
    amrex::MultiFab pointValuesZ(ba, dm, nValues * nComp, nghost);
    amrex::MultiFab analyticalPointValuesZ(ba, dm, nValues * nComp, nghost);
    amrex::Array<amrex::MultiFab*, 3> pointValues{&pointValuesX, &pointValuesY, &pointValuesZ};
    amrex::Array<amrex::MultiFab*, 3> analyticalPointValues{
        &analyticalPointValuesX,
        &analyticalPointValuesY,
        &analyticalPointValuesZ,
    };

    // Compute the interpolations
    amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> evalShiftArray1{{AMREX_D_DECL(0.25, 0.25, 0.25)}};
    amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> evalShiftArray2{{AMREX_D_DECL(0.75, 0.75, 0.75)}};
    amrex::Vector<amrex::GpuArray<amrex::Real, AMREX_SPACEDIM>> evalShift = {evalShiftArray1,
                                                                             evalShiftArray2};

    // Copy evalShift to GPU array
    amrex::Gpu::DeviceVector<amrex::GpuArray<amrex::Real, AMREX_SPACEDIM>> evalShiftGpu{nValues};
    amrex::Gpu::copyAsync(amrex::Gpu::hostToDevice, evalShift.begin(), evalShift.end(),
                          evalShiftGpu.begin());
    auto* const evalShiftGpuPtr = evalShiftGpu.dataPtr();

    for (int comp{0}; comp < 3; ++comp)
    {
        // transpose funcs for convenience
        amrex::Vector<amrex::ParserExecutor<nVar>> compFuncs(nComp);
        for (int i{0}; i < nComp; ++i)
        {
            compFuncs[i] = parser[i][comp].compile<nVar>();
        }
        // copy transposed funcs to GPU compatible vector
        amrex::Gpu::DeviceVector<amrex::ParserExecutor<nVar>> funcsGpu{static_cast<size_t>(nComp)};
        amrex::Gpu::copyAsync(amrex::Gpu::hostToDevice, compFuncs.begin(), compFuncs.end(),
                              funcsGpu.begin());
        auto* const funcsGpuPtr = funcsGpu.dataPtr();
        // Calculate func in evalShift
        for (amrex::MFIter mfi(*analyticalPointValues[comp]); mfi.isValid(); ++mfi)
        {
            compute_analytical_vector_function_parallel_for(mfi, this->m_infra, funcsGpuPtr,
                                                            *analyticalPointValues[comp],
                                                            evalShiftGpuPtr, nValues, nComp);
        }
    }

    eval_form(form, evalShift, *pointValues[xDir], xDir);
    eval_form(form, evalShift, *pointValues[yDir], yDir);
    eval_form(form, evalShift, *pointValues[zDir], zDir);

    bool loopRun{false};

    for (int comp{0}; comp < 3; ++comp)
    {
        for (amrex::MFIter mfi(*analyticalPointValues[comp]); mfi.isValid(); ++mfi)
        {
            loopRun = true;

            amrex::Box const& bx = mfi.tilebox();
            COMPARE_FIELDS((*pointValues[comp])[mfi].array(),
                           (*analyticalPointValues[comp])[mfi].array(), bx);
        }
        ASSERT_TRUE(loopRun);

        // Preparation for the composite version eval_form test
        pointValues[comp]->setVal(0.0);
    }

    eval_form(form, evalShift, pointValues); // Check that this version conforms as well

    for (int comp{0}; comp < 3; ++comp)
    {
        for (amrex::MFIter mfi(*analyticalPointValues[comp]); mfi.isValid(); ++mfi)
        {
            amrex::Box const& bx = mfi.tilebox();
            COMPARE_FIELDS((*pointValues[comp])[mfi].array(),
                           (*analyticalPointValues[comp])[mfi].array(), bx);
        }
    }
}
/// @todo: Add multiple value multifab fields tests
///        (using compute_analytical_vector_function_parallel_for)
} // namespace

} //namespace