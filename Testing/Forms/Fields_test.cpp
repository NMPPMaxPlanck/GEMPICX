#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include "GEMPIC_ComputationalDomain.H"
#include "GEMPIC_FDDeRhamComplex.H"
#include "GEMPIC_Fields.H"
#include "GEMPIC_Parameters.H"
#include "TestUtils/GEMPIC_TestUtils.H"

namespace
{
using namespace Gempic;
using namespace Forms;

class DiscreteFieldsTest : public ::testing::Test
{
public:
    DiscreteFieldsTest()
    {
        m_parameters.set("ComputationalDomain.domainLo", m_domainLo);
        m_parameters.set("k", m_k);
        m_parameters.set("ComputationalDomain.nCell", m_nCell);
        m_parameters.set("ComputationalDomain.maxGridSize", m_maxGridSize);
        m_parameters.set("ComputationalDomain.isPeriodic", m_isPeriodic);
    }
    Gempic::Io::Parameters m_parameters;
    amrex::Vector<amrex::Real> const m_domainLo{
        AMREX_D_DECL(-M_PI + 0.3, -M_PI + 0.6, -M_PI + 0.4)};
    amrex::Vector<amrex::Real> const m_k{AMREX_D_DECL(1.0, 1.0, 1.0)};
    amrex::Vector<int> const m_nCell{AMREX_D_DECL(9, 8, 7)};
    amrex::Vector<int> const m_maxGridSize{AMREX_D_DECL(3, 4, 5)};
    amrex::Vector<int> const m_isPeriodic{AMREX_D_DECL(1, 1, 1)};
};

void fill_scalar_field_with_one (DiscreteField& sf)
{
    auto fillFunc = [=] AMREX_GPU_HOST_DEVICE(
                        AMREX_D_DECL(amrex::Real x, amrex::Real y, amrex::Real z)) { return 1.0; };
    Gempic::fill(sf, fillFunc);
}
void fill_scalar_field_with_sin (DiscreteField& f)
{
    //! [DiscreteFieldExample.Fill]
    std::array<amrex::Real, AMREX_SPACEDIM> k0{};
    for (auto dir : {AMREX_D_DECL(Direction::xDir, Direction::yDir, Direction::zDir)})
    {
        k0[dir] = 2 * M_PI / f.discrete_grid().length(dir);
    }
    auto fillFunc = [=] AMREX_GPU_HOST_DEVICE(AMREX_D_DECL(double x, double y, double z))
    {
        return GEMPIC_D_MULT(std::sin(k0[Direction::xDir] * x), std::sin(k0[Direction::yDir] * y),
                             std::sin(k0[Direction::zDir] * z));
    };
    Gempic::fill(f, fillFunc);
    //! [DiscreteFieldExample.Fill]
}
void fill_vector_field_with_one_two_three (DiscreteVectorField& vf)
{
    auto fillFunc = [=] AMREX_GPU_HOST_DEVICE(
                        Direction dir, AMREX_D_DECL(amrex::Real x, amrex::Real y, amrex::Real z))
    {
        switch (dir)
        {
            case Direction::xDir:
                return 1.0;
            case Direction::yDir:
                return 2.0;
            case Direction::zDir:
                return 3.0;
        };
        AMREX_ALWAYS_ASSERT(false);
        return 0.0;
    };
    Gempic::fill(vf, fillFunc);
}

void fill_vector_field_with_sin (DiscreteVectorField& f)
{
    //! [DiscreteVectorFieldExample.Fill]
    std::array<amrex::Real, AMREX_SPACEDIM> k0{};
    for (auto gridDir : {AMREX_D_DECL(Direction::xDir, Direction::yDir, Direction::zDir)})
    {
        k0[gridDir] = 2 * M_PI / f.discrete_grid(Direction::xDir).length(gridDir);
    }
    auto fillFunc =
        [=] AMREX_GPU_HOST_DEVICE(Direction dir, AMREX_D_DECL(double x, double y, double z))
    {
        switch (dir)
        {
            case Direction::xDir:
            {
                return GEMPIC_D_MULT(std::sin(k0[Direction::xDir] * x),
                                     std::sin(k0[Direction::yDir] * y),
                                     std::sin(k0[Direction::zDir] * z));
            }
            case Direction::yDir:
            {
                return GEMPIC_D_MULT(std::cos(k0[Direction::xDir] * x),
                                     std::cos(k0[Direction::yDir] * y),
                                     std::sin(k0[Direction::zDir] * z));
            }
            case Direction::zDir:
            {
                return GEMPIC_D_MULT(std::sin(k0[Direction::xDir] * x),
                                     std::cos(k0[Direction::yDir] * y),
                                     std::sin(k0[Direction::zDir] * z));
            }
        }
        AMREX_ALWAYS_ASSERT(false);
        return 0.0;
    };
    Gempic::fill(f, fillFunc);
    //! [DiscreteVectorFieldExample.Fill]
}

TEST_F(DiscreteFieldsTest, fillScalarField)
{
    Gempic::Io::Parameters parameters;
    DiscreteField df{
        "df", parameters,
        DiscreteGrid{parameters,
                     {AMREX_D_DECL(DiscreteGrid::Cell, DiscreteGrid::Cell, DiscreteGrid::Cell)}}};

    df.multi_fab().setVal(0.0);
    fill_scalar_field_with_one(df);
    EXPECT_EQ(df.multi_fab().norm0(), 1.0);
    EXPECT_EQ(df.multi_fab().norm1(),
              GEMPIC_D_MULT(df.discrete_grid().size(xDir), df.discrete_grid().size(yDir),
                            df.discrete_grid().size(zDir)));
}

TEST_F(DiscreteFieldsTest, fillVectorField)
{
    std::array<DiscreteGrid::Position, AMREX_SPACEDIM> position{
        {AMREX_D_DECL(DiscreteGrid::Cell, DiscreteGrid::Cell, DiscreteGrid::Cell)}};
    std::array<DiscreteGrid, 3> grids{};
    for (Direction dir : {Direction::xDir, Direction::yDir, Direction::zDir})
    {
        grids[dir] = DiscreteGrid{m_parameters, position};
    }
    DiscreteVectorField df{"df", m_parameters, grids};

    for (Direction dir : {Direction::xDir, Direction::yDir, Direction::zDir})
    {
        df.multi_fab(dir).setVal(0.0);
    }

    fill_vector_field_with_one_two_three(df);

    for (Direction dir : {Direction::xDir, Direction::yDir, Direction::zDir})
    {
        EXPECT_EQ(df.multi_fab(dir).norm0(), dir + 1);
        EXPECT_EQ(df.multi_fab(dir).norm1(),
                  (dir + 1) * GEMPIC_D_MULT(df.discrete_grid(dir).size(xDir),
                                            df.discrete_grid(dir).size(yDir),
                                            df.discrete_grid(dir).size(zDir)));
    }
}

TEST_F(DiscreteFieldsTest, setGhostCellsScalarField)
{
    DiscreteField df{
        "df", m_parameters,
        DiscreteGrid{m_parameters,
                     {AMREX_D_DECL(DiscreteGrid::Cell, DiscreteGrid::Node, DiscreteGrid::Cell)}}};
    DiscreteField res{
        "df", m_parameters,
        DiscreteGrid{m_parameters,
                     {AMREX_D_DECL(DiscreteGrid::Cell, DiscreteGrid::Node, DiscreteGrid::Cell)}}};

    fill_scalar_field_with_sin(df);
    fill_scalar_field_with_sin(res);
    for (Direction dir : {AMREX_D_DECL(Direction::xDir, Direction::yDir, Direction::zDir)})
    {
        EXPECT_EQ(df.multi_fab().nGrow(dir), 0);
    }
    df.apply_boundary_conditions({AMREX_D_DECL(1, 2, 3)});
    for (Direction dir : {AMREX_D_DECL(Direction::xDir, Direction::yDir, Direction::zDir)})
    {
        EXPECT_EQ(df.multi_fab().nGrow(dir), dir + 1);
    }
    df.apply_boundary_conditions({AMREX_D_DECL(0, 1, 2)});
    for (Direction dir : {AMREX_D_DECL(Direction::xDir, Direction::yDir, Direction::zDir)})
    {
        EXPECT_EQ(df.multi_fab().nGrow(dir), dir + 1);
    }
    EXPECT_EQ(l_inf_error(df, res), 0.0);
}

TEST_F(DiscreteFieldsTest, setGhostCellsVectorField)
{
    std::array<DiscreteGrid::Position, AMREX_SPACEDIM> position{
        {AMREX_D_DECL(DiscreteGrid::Cell, DiscreteGrid::Node, DiscreteGrid::Cell)}};
    std::array<DiscreteGrid, 3> grids{};
    for (Direction dir : {Direction::xDir, Direction::yDir, Direction::zDir})
    {
        grids[dir] = DiscreteGrid{m_parameters, position};
    }
    DiscreteVectorField df{"df", m_parameters, grids};
    DiscreteVectorField res{"df", m_parameters, grids};

    fill_vector_field_with_sin(df);
    fill_vector_field_with_sin(res);
    for (Direction gridDir : {AMREX_D_DECL(Direction::xDir, Direction::yDir, Direction::zDir)})
    {
        for (Direction fieldDir : {AMREX_D_DECL(Direction::xDir, Direction::yDir, Direction::zDir)})
        {
            EXPECT_EQ(df.multi_fab(fieldDir).nGrow(gridDir), 0);
        }
    }
    df.apply_boundary_conditions({AMREX_D_DECL(1, 2, 3)});
    for (Direction gridDir : {AMREX_D_DECL(Direction::xDir, Direction::yDir, Direction::zDir)})
    {
        for (Direction fieldDir : {Direction::xDir, Direction::yDir, Direction::zDir})
        {
            EXPECT_EQ(df.multi_fab(fieldDir).nGrow(gridDir), gridDir + 1);
        }
    }
    df.apply_boundary_conditions({AMREX_D_DECL(0, 1, 2)});
    for (Direction gridDir : {AMREX_D_DECL(Direction::xDir, Direction::yDir, Direction::zDir)})
    {
        for (Direction fieldDir : {Direction::xDir, Direction::yDir, Direction::zDir})
        {
            EXPECT_EQ(df.multi_fab(fieldDir).nGrow(gridDir), gridDir + 1);
        }
    }
    std::array<amrex::Real, 3> lInfError{l_inf_error(df, res)};
    EXPECT_EQ(lInfError[xDir], 0.0);
    EXPECT_EQ(lInfError[yDir], 0.0);
    EXPECT_EQ(lInfError[zDir], 0.0);
}

void discrete_field_example_kernel (DiscreteField& f, DiscreteField& g)
{
    //! [DiscreteFieldExample.DiscreteDerivative]
    g.apply_boundary_conditions({AMREX_D_DECL(1, 0, 0)});
    for (amrex::MFIter mfi{g.multi_fab()}; mfi.isValid(); ++mfi)
    {
        f.select_box(mfi);
        g.select_box(mfi);
        amrex::ParallelFor(mfi.validbox(), [=] AMREX_GPU_HOST_DEVICE(int ix, int iy, int iz)
                           { f(ix, iy, iz) = g(ix, iy, iz) - g(ix - 1, iy, iz); });
    }
    //! [DiscreteFieldExample.DiscreteDerivative]
}

TEST_F(DiscreteFieldsTest, discreteFieldKernelExample)
{
    DiscreteField f{
        "f", m_parameters,
        DiscreteGrid{m_parameters,
                     {AMREX_D_DECL(DiscreteGrid::Cell, DiscreteGrid::Cell, DiscreteGrid::Cell)}}};
    DiscreteField g{
        "g", m_parameters,
        DiscreteGrid{m_parameters,
                     {AMREX_D_DECL(DiscreteGrid::Cell, DiscreteGrid::Cell, DiscreteGrid::Cell)}}};
    g.multi_fab().setVal(1);
    discrete_field_example_kernel(f, g);
    EXPECT_EQ(f.multi_fab().norm0(), 0);
}

void discrete_vector_field_example_kernel (DiscreteField& f, DiscreteVectorField& g)
{
    //! [DiscreteVectorFieldExample.DiscreteDivergence]
    g.apply_boundary_conditions({AMREX_D_DECL(1, 1, 1)});
    for (amrex::MFIter mfi{g.multi_fab(Direction::xDir)}; mfi.isValid(); ++mfi)
    {
        f.select_box(mfi);
        g.select_box(mfi);
        amrex::ParallelFor(
            mfi.validbox(),
            [=] AMREX_GPU_HOST_DEVICE(int ix, int iy, int iz)
            {
                f(ix, iy, iz) = GEMPIC_D_ADD(
                    g(Direction::xDir, ix, iy, iz) - g(Direction::xDir, ix - 1, iy, iz),
                    g(Direction::yDir, ix, iy, iz) - g(Direction::yDir, ix, iy - 1, iz),
                    g(Direction::zDir, ix, iy, iz) - g(Direction::zDir, ix, iy, iz - 1));
            });
    }
    //! [DiscreteVectorFieldExample.DiscreteDivergence]
}

TEST_F(DiscreteFieldsTest, DiscreteVectorFieldKernelExample)
{
    std::array<DiscreteGrid::Position, AMREX_SPACEDIM> position{
        {AMREX_D_DECL(DiscreteGrid::Cell, DiscreteGrid::Cell, DiscreteGrid::Cell)}};
    std::array<DiscreteGrid, 3> grids{};
    for (Direction dir : {Direction::xDir, Direction::yDir, Direction::zDir})
    {
        grids[dir] = DiscreteGrid{m_parameters, position};
    }
    DiscreteField f{"df", m_parameters, grids[Direction::xDir]};
    DiscreteVectorField g{"df", m_parameters, grids};
    fill_vector_field_with_one_two_three(g);
    g.multi_fab(Direction::xDir).setVal(1.0);
    g.multi_fab(Direction::yDir).setVal(2.0);
    g.multi_fab(Direction::zDir).setVal(3.0);
    discrete_vector_field_example_kernel(f, g);
    EXPECT_EQ(f.multi_fab().norm0(), 0);
}

void fill_reference_scalar_field (DiscreteField& sf, double t)
{
    fill(
        sf,
        [=] AMREX_GPU_HOST_DEVICE(AMREX_D_DECL(amrex::Real x, amrex::Real y, amrex::Real z),
                                  amrex::Real t)
        { return t * AMREX_D_PICK(x, (x - y), (x - y - z)); },
        t);
}
void fill_scalar_field_with_parse (DiscreteField& f,
                                   DiscreteFieldsFunctionParser const& parser,
                                   double t)
{
    fill(f, parser, t);
}
TEST(FunctionParser, fillDiscreteFieldWithParsedFunction)
{
    std::array<amrex::Real, AMREX_SPACEDIM> const domainLo{AMREX_D_DECL(-1.0, -0.6, -2.4)};
    std::array<amrex::Real, AMREX_SPACEDIM> const domainHi{AMREX_D_DECL(4.0, 4.5, 5.0)};
    std::array<int, AMREX_SPACEDIM> const nCell{AMREX_D_DECL(9, 8, 7)};
    std::array<int, AMREX_SPACEDIM> const isPeriodic{AMREX_D_DECL(1, 1, 1)};
    DiscreteGrid grid{domainLo,
                      domainHi,
                      nCell,
                      {AMREX_D_DECL(DiscreteGrid::Position::Cell, DiscreteGrid::Position::Cell,
                                    DiscreteGrid::Node)}};
    Gempic::Io::Parameters parameters;
    double t{0.5};
    // AMReX `ParamParse::add()` requires a reference for a string
    // Trying to directly pass the string to `Parameters::set()` on clang converts implicitly and
    // calls ParamParse::add(bool);
    std::string sfFunction{AMREX_D_PICK("t*x", "t*(x-y)", "t*(x-y-z)")};
    parameters.set("FunctionParser.sf", sfFunction);
    parameters.set("ComputationalDomain.maxGridSize", amrex::Vector<int>{9, 8, 7});

    DiscreteField sf{"sf", parameters, grid};
    DiscreteField sfRef{"sf", parameters, grid};

    DiscreteFieldsFunctionParser parseSf{"sf", parameters};
    fill_scalar_field_with_parse(sf, parseSf, t);
    fill_reference_scalar_field(sfRef, t);
    EXPECT_EQ(l_inf_error(sf, sfRef), 0.0);
}

void fill_reference_vector_field (DiscreteVectorField& vf, double t)
{
    fill(
        vf,
        [=] AMREX_GPU_HOST_DEVICE(
            Direction dir, AMREX_D_DECL(amrex::Real x, amrex::Real y, amrex::Real z), amrex::Real t)
        {
            switch (dir)
            {
                case Direction::xDir:
                    return t * AMREX_D_PICK(x, (x + y), (x + y + z));
                case Direction::yDir:
                    return t * AMREX_D_PICK(x, (x - y), (x - y + z));
                case Direction::zDir:
                    return t * AMREX_D_PICK(x, (x + y), (x + y - z));
            }
            AMREX_ALWAYS_ASSERT(false);
            return 0.0;
        },
        t);
}
void fill_vector_field_with_parse (DiscreteVectorField& f,
                                   DiscreteFieldsFunctionParser const& parser,
                                   double t)
{
    fill(f, parser, t);
}
TEST(FunctionParser, fillDiscreteVectorFieldWithParsedFunction)
{
    std::array<amrex::Real, AMREX_SPACEDIM> const domainLo{AMREX_D_DECL(-1.0, -0.6, -2.4)};
    std::array<amrex::Real, AMREX_SPACEDIM> const domainHi{AMREX_D_DECL(4.0, 4.5, 5.0)};
    std::array<int, AMREX_SPACEDIM> const nCell{AMREX_D_DECL(9, 8, 7)};
    std::array<int, AMREX_SPACEDIM> const isPeriodic{AMREX_D_DECL(1, 1, 1)};
    DiscreteGrid grid{domainLo,
                      domainHi,
                      nCell,
                      {AMREX_D_DECL(DiscreteGrid::Position::Cell, DiscreteGrid::Position::Cell,
                                    DiscreteGrid::Node)}};
    Gempic::Io::Parameters parameters;
    double t{0.5};
    // AMReX `ParamParse::add()` requires a reference for a string
    // Trying to directly pass the string to `Parameters::set()` on clang converts implicitly and
    // calls ParamParse::add(bool);
    std::string vfxFunction{AMREX_D_PICK("t*x", "t*(x+y)", "t*(x+y+z)")};
    std::string vfyFunction{AMREX_D_PICK("t*x", "t*(x-y)", "t*(x-y+z)")};
    std::string vfzFunction{AMREX_D_PICK("t*x", "t*(x+y)", "t*(x+y-z)")};
    parameters.set("FunctionParser.vfx", vfxFunction);
    parameters.set("FunctionParser.vfy", vfyFunction);
    parameters.set("FunctionParser.vfz", vfzFunction);
    parameters.set("ComputationalDomain.maxGridSize", amrex::Vector<int>{9, 8, 7});

    DiscreteVectorField vf{"vf", parameters, {grid, grid, grid}};
    DiscreteVectorField vfRef{"vf", parameters, {grid, grid, grid}};

    DiscreteFieldsFunctionParser parseVf{{"vfx", "vfy", "vfz"}, parameters};
    fill_vector_field_with_parse(vf, parseVf, t);
    fill_reference_vector_field(vfRef, t);
    EXPECT_EQ(l_inf_error(vf, vfRef)[Direction::xDir], 0.0);
    EXPECT_EQ(l_inf_error(vf, vfRef)[Direction::yDir], 0.0);
    EXPECT_EQ(l_inf_error(vf, vfRef)[Direction::zDir], 0.0);
}

void fill_scalar_field_with_nan (DiscreteField f)
{
    auto nan = [] AMREX_GPU_HOST_DEVICE(
                   AMREX_D_DECL(amrex::Real x, amrex::Real y, amrex::Real z)) -> amrex::Real
    {
        if (x < 1.0)
        {
            return std::numeric_limits<amrex::Real>::quiet_NaN();
        }
        else
        {
            return 1.0;
        }
    };
    Gempic::fill(f, nan);
}

TEST_F(DiscreteFieldsTest, DiscreteFieldNan)
{
    std::array<DiscreteGrid::Position, AMREX_SPACEDIM> position{
        {AMREX_D_DECL(DiscreteGrid::Cell, DiscreteGrid::Cell, DiscreteGrid::Cell)}};
    DiscreteGrid grid{m_parameters, position};
    DiscreteField f{"f", m_parameters, grid};
    fill_scalar_field_with_nan(f);
    EXPECT_TRUE(Gempic::is_nan(f));
    fill_scalar_field_with_one(f);
    EXPECT_FALSE(Gempic::is_nan(f));
}

TEST_F(DiscreteFieldsTest, DiscreteFieldLInfError)
{
    std::array<DiscreteGrid::Position, AMREX_SPACEDIM> position{
        {AMREX_D_DECL(DiscreteGrid::Cell, DiscreteGrid::Cell, DiscreteGrid::Cell)}};
    DiscreteGrid grid{m_parameters, position};
    DiscreteField f{"f", m_parameters, grid};
    DiscreteField g{"g", m_parameters, grid};
    fill_zero(f);
    fill_scalar_field_with_one(g);
    EXPECT_EQ(l_inf_error(f, g), 1.0);
    fill_scalar_field_with_one(f);
    fill_scalar_field_with_one(g);
    EXPECT_EQ(l_inf_error(f, g), 0.0);
    fill_scalar_field_with_one(f);
    fill_scalar_field_with_nan(g);
    EXPECT_TRUE(std::isnan(l_inf_error(f, g)));
}

void fill_vector_field_with_nan (DiscreteVectorField& f)
{
    auto nan = [] AMREX_GPU_HOST_DEVICE(Direction dir, AMREX_D_DECL(amrex::Real x, amrex::Real y,
                                                                    amrex::Real z)) -> amrex::Real
    {
        if (x < 1.0)
        {
            return std::numeric_limits<amrex::Real>::quiet_NaN();
        }
        else
        {
            return 0.0;
        }
    };
    Gempic::fill(f, nan);
}

TEST_F(DiscreteFieldsTest, DiscreteVectorFieldIsNan)
{
    std::array<DiscreteGrid::Position, AMREX_SPACEDIM> position{
        {AMREX_D_DECL(DiscreteGrid::Cell, DiscreteGrid::Cell, DiscreteGrid::Cell)}};
    DiscreteGrid grid{m_parameters, position};
    DiscreteVectorField f{"f", m_parameters, {grid, grid, grid}};
    fill_vector_field_with_nan(f);
    EXPECT_TRUE(Gempic::is_nan(f)[Direction::xDir]);
    EXPECT_TRUE(Gempic::is_nan(f)[Direction::yDir]);
    EXPECT_TRUE(Gempic::is_nan(f)[Direction::zDir]);
    fill_vector_field_with_one_two_three(f);
    EXPECT_FALSE(Gempic::is_nan(f)[Direction::xDir]);
    EXPECT_FALSE(Gempic::is_nan(f)[Direction::yDir]);
    EXPECT_FALSE(Gempic::is_nan(f)[Direction::zDir]);
}

TEST_F(DiscreteFieldsTest, DiscreteVectorFieldLInfError)
{
    std::array<DiscreteGrid::Position, AMREX_SPACEDIM> position{
        {AMREX_D_DECL(DiscreteGrid::Cell, DiscreteGrid::Cell, DiscreteGrid::Cell)}};
    DiscreteGrid grid{m_parameters, position};
    DiscreteVectorField f{"f", m_parameters, {grid, grid, grid}};
    DiscreteVectorField g{"g", m_parameters, {grid, grid, grid}};
    fill_zero(f);
    fill_vector_field_with_one_two_three(g);
    EXPECT_EQ(l_inf_error(f, g)[Direction::xDir], 1.0);
    EXPECT_EQ(l_inf_error(f, g)[Direction::yDir], 2.0);
    EXPECT_EQ(l_inf_error(f, g)[Direction::zDir], 3.0);
    fill_vector_field_with_one_two_three(f);
    fill_vector_field_with_one_two_three(g);
    EXPECT_EQ(l_inf_error(f, g)[Direction::xDir], 0.0);
    EXPECT_EQ(l_inf_error(f, g)[Direction::yDir], 0.0);
    EXPECT_EQ(l_inf_error(f, g)[Direction::zDir], 0.0);
    fill_vector_field_with_one_two_three(f);
    fill_vector_field_with_nan(g);
    EXPECT_TRUE(std::isnan(l_inf_error(f, g)[Direction::xDir]));
    EXPECT_TRUE(std::isnan(l_inf_error(f, g)[Direction::yDir]));
    EXPECT_TRUE(std::isnan(l_inf_error(f, g)[Direction::zDir]));
}

void compare_discrete_field_periodic_boundary (DiscreteField const& df)
{
    amrex::Real l2Err{};
    amrex::Real* l2ErrPtr{&l2Err};
    amrex::LoopOnCpu(
        Gempic::Impl::selected_ghost_box(df, {AMREX_D_DECL(1, 0, 0)}, Direction::xDir,
                                         Gempic::Impl::GhostRegion::low),
        [=] AMREX_GPU_HOST(int ix, int iy, int iz)
        {
            amrex::Real diff{df(ix, iy, iz) -
                             df(ix + df.discrete_grid().size(Direction::xDir), iy, iz)};
            amrex::Gpu::Atomic::Add(l2ErrPtr, diff * diff);
        });
    EXPECT_NEAR(l2Err, 0.0, 1.0e-15);
    l2Err = 0.0;
    amrex::LoopOnCpu(
        Gempic::Impl::selected_ghost_box(df, {AMREX_D_DECL(1, 0, 0)}, Direction::xDir,
                                         Gempic::Impl::GhostRegion::up),
        [=] AMREX_GPU_HOST(int ix, int iy, int iz)
        {
            amrex::Real diff{df(ix, iy, iz) -
                             df(ix - df.discrete_grid().size(Direction::xDir), iy, iz)};
            amrex::Gpu::Atomic::Add(l2ErrPtr, diff * diff);
        });
    EXPECT_NEAR(l2Err, 0.0, 1.0e-15);
#if AMREX_SPACEDIM > 1
    l2Err = 0.0;
    // Direction::yDir is has DOF positioned at Nodes which is why size needs to be reduces by 1 to
    // compare the boundary value
    amrex::LoopOnCpu(
        Gempic::Impl::selected_ghost_box(df, {AMREX_D_DECL(0, 2, 0)}, Direction::yDir,
                                         Gempic::Impl::GhostRegion::low),
        [=] AMREX_GPU_HOST(int ix, int iy, int iz)
        {
            amrex::Real diff{df(ix, iy, iz) -
                             df(ix, iy - 1 + df.discrete_grid().size(Direction::yDir), iz)};
            amrex::Gpu::Atomic::Add(l2ErrPtr, diff * diff);
        });
    EXPECT_NEAR(l2Err, 0.0, 1.0e-15);
    l2Err = 0.0;
    // Direction::yDir is has DOF positioned at Nodes which is why size needs to be increased by 1
    // to compare the boundary value
    amrex::LoopOnCpu(
        Gempic::Impl::selected_ghost_box(df, {AMREX_D_DECL(0, 2, 0)}, Direction::yDir,
                                         Gempic::Impl::GhostRegion::up),
        [=] AMREX_GPU_HOST(int ix, int iy, int iz)
        {
            amrex::Real diff{df(ix, iy, iz) -
                             df(ix, iy + 1 - df.discrete_grid().size(Direction::yDir), iz)};
            amrex::Gpu::Atomic::Add(l2ErrPtr, diff * diff);
        });
    EXPECT_NEAR(l2Err, 0.0, 1.0e-15);
#endif
#if AMREX_SPACEDIM > 2
    l2Err = 0.0;
    amrex::LoopOnCpu(
        Gempic::Impl::selected_ghost_box(df, {AMREX_D_DECL(0, 0, 3)}, Direction::zDir,
                                         Gempic::Impl::GhostRegion::low),
        [=] AMREX_GPU_HOST(int ix, int iy, int iz)
        {
            amrex::Real diff{df(ix, iy, iz) -
                             df(ix, iy, iz + df.discrete_grid().size(Direction::zDir))};
            amrex::Gpu::Atomic::Add(l2ErrPtr, diff * diff);
        });
    EXPECT_NEAR(l2Err, 0.0, 1.0e-15);
    l2Err = 0.0;
    amrex::LoopOnCpu(
        Gempic::Impl::selected_ghost_box(df, {AMREX_D_DECL(0, 0, 3)}, Direction::zDir,
                                         Gempic::Impl::GhostRegion::up),
        [=] AMREX_GPU_HOST(int ix, int iy, int iz)
        {
            amrex::Real diff{df(ix, iy, iz) -
                             df(ix, iy, iz - df.discrete_grid().size(Direction::zDir))};
            amrex::Gpu::Atomic::Add(l2ErrPtr, diff * diff);
        });
    EXPECT_NEAR(l2Err, 0.0, 1.0e-15);
    l2Err = 0.0;
#endif
}

TEST(DiscreteFieldBoundaryConditionsTest, periodic)
{
    Gempic::Io::Parameters parameters;
    amrex::Vector<amrex::Real> const domainLo{AMREX_D_DECL(-M_PI + 0.3, -M_PI + 0.6, -M_PI + 0.4)};
    amrex::Vector<amrex::Real> const k{AMREX_D_DECL(1.0, 1.0, 1.0)};
    amrex::Vector<int> const nCell{AMREX_D_DECL(9, 8, 7)};
    amrex::Vector<int> const isPeriodic{AMREX_D_DECL(1, 1, 1)};
    parameters.set("ComputationalDomain.domainLo", domainLo);
    parameters.set("k", k);
    parameters.set("ComputationalDomain.nCell", nCell);
    parameters.set("ComputationalDomain.isPeriodic", isPeriodic);

    DiscreteField df{
        "df", parameters,
        DiscreteGrid{parameters,
                     {AMREX_D_DECL(DiscreteGrid::Cell, DiscreteGrid::Node, DiscreteGrid::Cell)}}};
    ASSERT_EQ(df.multi_fab().boxArray().size(), 1);
    fill_scalar_field_with_sin(df);
    df.apply_boundary_conditions({AMREX_D_DECL(1, 2, 3)});
    for (amrex::MFIter mfi{df.multi_fab()}; mfi.isValid(); ++mfi) df.select_box(mfi);
    compare_discrete_field_periodic_boundary(df);
    compare_discrete_field_periodic_boundary(df);
    compare_discrete_field_periodic_boundary(df);
}

TEST(DiscreteFieldBoundaryConditionsTest, extrapolation)
{
    // ToDo implement tests for remaining boundary conditions.
    // Extrapolation and Dirichlet
}

class LinearAlgebraTest : public ::testing::Test
{
public:
    LinearAlgebraTest()
    {
        m_parameters.set("ComputationalDomain.domainLo", m_domainLo);
        m_parameters.set("k", m_k);
        m_parameters.set("ComputationalDomain.nCell", m_nCell);
        m_parameters.set("ComputationalDomain.maxGridSize", m_maxGridSize);
        m_parameters.set("ComputationalDomain.isPeriodic", m_isPeriodic);
    }
    Gempic::Io::Parameters m_parameters;
    amrex::Vector<amrex::Real> const m_domainLo{
        AMREX_D_DECL(-M_PI + 0.3, -M_PI + 0.6, -M_PI + 0.4)};
    amrex::Vector<amrex::Real> const m_k{AMREX_D_DECL(1.0, 1.0, 1.0)};
    amrex::Vector<int> const m_nCell{AMREX_D_DECL(9, 8, 7)};
    amrex::Vector<int> const m_maxGridSize{AMREX_D_DECL(9, 8, 7)};
    amrex::Vector<int> const m_isPeriodic{AMREX_D_DECL(1, 1, 1)};
};

TEST_F(LinearAlgebraTest, AddAssignDiscreteVectorFieldVectorField)
{
    std::array<DiscreteGrid::Position, AMREX_SPACEDIM> position{
        {AMREX_D_DECL(DiscreteGrid::Cell, DiscreteGrid::Node, DiscreteGrid::Cell)}};
    std::array<DiscreteGrid, 3> grids{};
    for (Direction dir : {Direction::xDir, Direction::yDir, Direction::zDir})
    {
        grids[dir] = DiscreteGrid{m_parameters, position};
    }
    DiscreteVectorField df{"df", m_parameters, grids};
    DiscreteVectorField res{"df", m_parameters, grids};
    for (Direction dir : {Direction::xDir, Direction::yDir, Direction::zDir})
    {
        df.multi_fab(dir).setVal(1);
        res.multi_fab(dir).setVal(0);
    }
    res += df;

    for (Direction dir : {Direction::xDir, Direction::yDir, Direction::zDir})
    {
        EXPECT_EQ(res.multi_fab(dir).norm0(), 1);
        EXPECT_EQ(res.multi_fab(dir).norm1(),
                  GEMPIC_D_MULT(res.discrete_grid(dir).size(Direction::xDir),
                                res.discrete_grid(dir).size(Direction::yDir),
                                res.discrete_grid(dir).size(Direction::zDir)));
    }
}

TEST_F(LinearAlgebraTest, MultiplyAssignScalarVectorField)
{
    std::array<DiscreteGrid::Position, AMREX_SPACEDIM> position{
        {AMREX_D_DECL(DiscreteGrid::Cell, DiscreteGrid::Node, DiscreteGrid::Cell)}};
    std::array<DiscreteGrid, 3> grids{};
    for (Direction dir : {Direction::xDir, Direction::yDir, Direction::zDir})
    {
        grids[dir] = DiscreteGrid{m_parameters, position};
    }
    DiscreteVectorField df{"df", m_parameters, grids};
    DiscreteVectorField res{"df", m_parameters, grids};
    for (Direction dir : {Direction::xDir, Direction::yDir, Direction::zDir})
    {
        res.multi_fab(dir).setVal(1);
    }
    res *= 2.0;

    for (Direction dir : {Direction::xDir, Direction::yDir, Direction::zDir})
    {
        EXPECT_EQ(res.multi_fab(dir).norm0(), 2);
        EXPECT_EQ(res.multi_fab(dir).norm1(),
                  2 * GEMPIC_D_MULT(res.discrete_grid(dir).size(Direction::xDir),
                                    res.discrete_grid(dir).size(Direction::yDir),
                                    res.discrete_grid(dir).size(Direction::zDir)));
    }
}

// Test fixture. Sets up clean environment before each test.
class FieldsTest : public testing::Test
{
protected:
    static int const s_degX{1};
    static int const s_degY{1};
    static int const s_degZ{1};
    inline static int const s_maxSplineDegree{std::max(std::max(s_degX, s_degY), s_degZ)};

    Io::Parameters m_parameters{};

    // Initialize computational_domain
    ComputationalDomain m_infra;

    int m_nComps{1};
    static int const s_gSize{5};

    FieldsTest() : m_infra{Gempic::Test::Utils::get_compdom(s_gSize)} {}
};

TEST_F(FieldsTest, OperatorsHodge2)
{
    constexpr int hodgeDegree{2};

    auto deRham = std::make_shared<FDDeRhamComplex>(m_infra, hodgeDegree, s_maxSplineDegree,
                                                    HodgeScheme::FDHodge);

    DeRhamField<Grid::primal, Space::node> primalNode(deRham);
    DeRhamField<Grid::primal, Space::edge> primalEdge(deRham);

    DeRhamField<Grid::dual, Space::face> dualFaceOther(deRham);
    DeRhamField<Grid::dual, Space::cell> dualCellOther(deRham);

    primalNode.m_data.setVal(1);
    primalNode *= 2; // result = 2
    dualCellOther.m_data.setVal(3);

    amrex::Real dotProd = dot_product(primalNode, dualCellOther);
    EXPECT_NEAR(dotProd, 2 * 3 * m_infra.box().numPts(), 1e-14);

    for (int comp = 0; comp < 3; comp++)
    {
        primalEdge.m_data[comp].setVal(1);
        dualFaceOther.m_data[comp].setVal(3);
    }
    dotProd = dot_product(primalEdge, dualFaceOther);
    EXPECT_NEAR(dotProd, 3 * 3 * m_infra.box().numPts(), 1e-14);
}

TEST_F(FieldsTest, LinearCombination)
{
    constexpr int hodgeDegree{2};

    auto deRham = std::make_shared<FDDeRhamComplex>(m_infra, hodgeDegree, s_maxSplineDegree,
                                                    HodgeScheme::FDHodge);

    DeRhamField<Grid::primal, Space::node> primalNode1(deRham);
    DeRhamField<Grid::primal, Space::edge> primalEdge1(deRham);
    DeRhamField<Grid::primal, Space::face> primalFace1(deRham);

    DeRhamField<Grid::primal, Space::node> primalNode2(deRham);
    DeRhamField<Grid::primal, Space::edge> primalEdge2(deRham);
    DeRhamField<Grid::dual, Space::edge> dualEdge2(deRham);
    DeRhamField<Grid::dual, Space::cell> dualCell2(deRham);

    primalNode1.m_data.setVal(1.0);
    primalNode2.m_data.setVal(2.0);
    Gempic::Forms::linear_combination(primalNode1, 2.0, primalNode1, 3.0, primalNode2);
    EXPECT_EQ(primalNode1.m_data.min(0), 2.0 * 1.0 + 3.0 * 2.0);

    primalNode1.m_data.setVal(1.0);
    dualCell2.m_data.setVal(2.0);
    Gempic::Forms::linear_combination(primalNode1, 2.0, primalNode1, 3.0, dualCell2);
    EXPECT_EQ(primalNode1.m_data.min(0), 2.0 * 1.0 + 3.0 * 2.0);

    for (int comp = 0; comp < 3; comp++)
    {
        primalEdge1.m_data[comp].setVal(1);
        primalEdge2.m_data[comp].setVal(2);
        primalFace1.m_data[comp].setVal(1);
        dualEdge2.m_data[comp].setVal(2);
    }

    linear_combination(primalEdge1, 2.0, primalEdge1, 3.0, primalEdge2);
    linear_combination(primalFace1, 2.0, primalFace1, 3.0, dualEdge2);

    for (int comp = 0; comp < 3; comp++)
    {
        EXPECT_EQ(primalEdge1.m_data[comp].min(0), 2.0 * 1.0 + 3.0 * 2.0);
        EXPECT_EQ(primalFace1.m_data[comp].min(0), 2.0 * 1.0 + 3.0 * 2.0);
    }
}

} // namespace
