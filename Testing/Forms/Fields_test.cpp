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
    DiscreteFieldsTest ()
    {
        amrex::ParmParse pp;
        pp.addarr("ComputationalDomain.domainLo", domainLo);
        pp.addarr("ComputationalDomain.k", k);
        pp.addarr("ComputationalDomain.nCell", nCell);
        pp.addarr("ComputationalDomain.maxGridSize", maxGridSize);
        pp.addarr("ComputationalDomain.isPeriodic", isPeriodic);
    }
    amrex::Vector<amrex::Real> const domainLo{AMREX_D_DECL(-M_PI + 0.3, -M_PI + 0.6, -M_PI + 0.4)};
    amrex::Vector<amrex::Real> const k{AMREX_D_DECL(1.0, 1.0, 1.0)};
    amrex::Vector<int> const nCell{AMREX_D_DECL(9, 8, 7)};
    amrex::Vector<int> const maxGridSize{AMREX_D_DECL(3, 4, 5)};
    amrex::Vector<int> const isPeriodic{AMREX_D_DECL(1, 1, 1)};
};

void fill_scalar_field_with_one (DiscreteField & sf)
{
    auto fillFunc = [=]AMREX_GPU_HOST_DEVICE(AMREX_D_DECL(amrex::Real x, amrex::Real y, amrex::Real z), int n)
    {
        return 1.0;
    };
    Gempic::fill(sf,fillFunc);
}
void fill_scalar_field_with_sin (DiscreteField & sf)
{
    auto fillFunc = [=] AMREX_GPU_HOST_DEVICE(AMREX_D_DECL(double x, double y, double z), int n)
         { return GEMPIC_D_MULT(std::sin(x) , std::sin(y) , std::sin(z)); };
    Gempic::fill(sf, fillFunc);
}
void fill_vector_field_with_one_two_three (DiscreteVectorField & vf)
{
    auto fillFunc =
        [=]AMREX_GPU_HOST_DEVICE(AMREX_D_DECL( amrex::Real x, amrex::Real y, amrex::Real z), Direction dir, int n)
    {
        switch (dir)
        {
            case Direction::xDir:
                return 1.0;
            case Direction::yDir:
                return 2.0;
            case Direction::zDir:
                return 3.0;
            default:
                return 0.0;
        };
    };
    Gempic::fill(vf, fillFunc);
}
void fill_vector_field_with_sin (DiscreteVectorField & vf)
{
    auto fillFunc = [=] AMREX_GPU_HOST_DEVICE(AMREX_D_DECL(double x, double y, double z), Direction dir, int n)
         { return GEMPIC_D_MULT( std::sin(x), std::sin(y), std::sin(z)) * (dir + 1); };
    Gempic::fill(vf, fillFunc);
}

TEST_F(DiscreteFieldsTest, fillScalarField)
{
    Gempic::Io::Parameters parameters;
    DiscreteField df{
        "df", parameters,
        DiscreteGrid{parameters, {AMREX_D_DECL(DiscreteGrid::Cell, DiscreteGrid::Cell, DiscreteGrid::Cell)}},
        Grid::dual, 3};

    df.multiFab().setVal(0.0);
    fill_scalar_field_with_one(df);
    EXPECT_EQ(df.multiFab().norm0(), 1.0);
    EXPECT_EQ(df.multiFab().norm1(), GEMPIC_D_MULT(df.discrete_grid().size(xDir) , df.discrete_grid().size(yDir) ,
                                         df.discrete_grid().size(zDir)));
}

TEST_F(DiscreteFieldsTest, fillVectorField)
{
    Gempic::Io::Parameters parameters;

    std::array<DiscreteGrid::Position, AMREX_SPACEDIM> position{
        {AMREX_D_DECL(DiscreteGrid::Cell, DiscreteGrid::Cell, DiscreteGrid::Cell)}};
    std::array<DiscreteGrid, AMREX_SPACEDIM> grids{};
    for (Direction dir : {AMREX_D_DECL(Direction::xDir, Direction::yDir, Direction::zDir)})
    {
        grids[dir] = DiscreteGrid{parameters, position};
    }
    DiscreteVectorField df{"df", parameters, grids, Grid::dual, 3};

    df.multiFab(Direction::xDir).setVal(0.0);
    #if AMREX_SPACEDIM>1
    df.multiFab(Direction::yDir).setVal(0.0);
    #endif
    #if AMREX_SPACEDIM == 3
    df.multiFab(Direction::zDir).setVal(0.0);
    #endif
    fill_vector_field_with_one_two_three(df);
    EXPECT_EQ(df.multiFab(Direction::xDir).norm0(), 1.0);
    EXPECT_EQ(df.multiFab(xDir).norm1(), GEMPIC_D_MULT (df.discrete_grid(xDir).size(xDir) ,
                                             df.discrete_grid(xDir).size(yDir) ,
                                             df.discrete_grid(xDir).size(zDir)));
    #if AMREX_SPACEDIM>1
    EXPECT_EQ(df.multiFab(Direction::yDir).norm0(), 2.0);
    EXPECT_EQ(df.multiFab(yDir).norm1(), 2 * GEMPIC_D_MULT (df.discrete_grid(yDir).size(xDir) ,
                                             df.discrete_grid(yDir).size(yDir) ,
                                             df.discrete_grid(yDir).size(zDir)));
    #endif
    #if AMREX_SPACEDIM==3                                         
    EXPECT_EQ(df.multiFab(Direction::zDir).norm0(), 3.0);
    EXPECT_EQ(df.multiFab(zDir).norm1(), 3 * GEMPIC_D_MULT (df.discrete_grid(zDir).size(xDir) ,
                                             df.discrete_grid(zDir).size(yDir) ,
                                             df.discrete_grid(zDir).size(zDir)));
    #endif                                             
}

TEST_F(DiscreteFieldsTest, setGhostCellsScalarField)
{
    Gempic::Io::Parameters parameters;
    DiscreteField df{
        "df", parameters,
        DiscreteGrid{parameters, {AMREX_D_DECL(DiscreteGrid::Cell, DiscreteGrid::Node, DiscreteGrid::Cell)}},
        Grid::dual, 3};
    DiscreteField res{
        "df", parameters,
        DiscreteGrid{parameters, {AMREX_D_DECL(DiscreteGrid::Cell, DiscreteGrid::Node, DiscreteGrid::Cell)}},
        Grid::dual, 3};

    fill_scalar_field_with_sin(df);
    fill_scalar_field_with_sin(res);
    for (Direction dir : {AMREX_D_DECL(Direction::xDir, Direction::yDir, Direction::zDir)})
    {
        EXPECT_EQ(df.multiFab().nGrow(dir), 0);
        df.set_ghost_size(1, dir);
        EXPECT_EQ(df.multiFab().nGrow(dir), 1);
    }
    EXPECT_LT(L_inf_error(df, res), 1.0e-13);
}

TEST_F(DiscreteFieldsTest, setGhostCellsVectorField)
{
    Gempic::Io::Parameters parameters;

    std::array<DiscreteGrid::Position, AMREX_SPACEDIM> position{
        {AMREX_D_DECL(DiscreteGrid::Cell, DiscreteGrid::Node, DiscreteGrid::Cell)}};
    std::array<DiscreteGrid, AMREX_SPACEDIM> grids{};
    for (Direction dir : {AMREX_D_DECL(Direction::xDir, Direction::yDir, Direction::zDir)})
    {
        grids[dir] = DiscreteGrid{parameters, position};
    }
    DiscreteVectorField df{"df", parameters, grids, Grid::dual, 3};
    DiscreteVectorField res{"df", parameters, grids, Grid::dual, 3};

    fill_vector_field_with_sin(df);
    fill_vector_field_with_sin(res);
    for (Direction dir1 : {AMREX_D_DECL(Direction::xDir, Direction::yDir, Direction::zDir)})
    {
        for (Direction dir2 : {AMREX_D_DECL(Direction::xDir, Direction::yDir, Direction::zDir)})
        {
            df.set_ghost_size(dir1 + dir2, dir1);
            EXPECT_EQ(df.multiFab(dir2).nGrow(dir1), dir1 + dir2);
        }
    }
    std::array<amrex::Real, AMREX_SPACEDIM> LInfError{L_inf_error(df, res)};
    EXPECT_LT(LInfError[xDir], 1.0e-13);
    #if AMREX_SPACEDIM>1
    EXPECT_LT(LInfError[yDir], 1.0e-13);
    #endif
    #if AMREX_SPACDIM == 3
    EXPECT_LT(LInfError[zDir], 1.0e-13);
    #endif
}

class LinearAlgebraTest : public ::testing::Test
{
public:
    LinearAlgebraTest ()
    {
        amrex::ParmParse pp;
        pp.addarr("ComputationalDomain.domainLo", domainLo);
        pp.addarr("ComputationalDomain.k", k);
        pp.addarr("ComputationalDomain.nCell", nCell);
        pp.addarr("ComputationalDomain.maxGridSize", maxGridSize);
        pp.addarr("ComputationalDomain.isPeriodic", isPeriodic);
    }
    amrex::Vector<amrex::Real> const domainLo{AMREX_D_DECL(-M_PI + 0.3, -M_PI + 0.6, -M_PI + 0.4)};
    amrex::Vector<amrex::Real> const k{AMREX_D_DECL(1.0, 1.0, 1.0)};
    amrex::Vector<int> const nCell{AMREX_D_DECL(9, 8, 7)};
    amrex::Vector<int> const maxGridSize{AMREX_D_DECL(3, 4, 5)};
    amrex::Vector<int> const isPeriodic{AMREX_D_DECL(0, 0, 0)};
};

TEST_F(LinearAlgebraTest, AddAssignDiscreteVectorFieldVectorField)
{
    Gempic::Io::Parameters parameters;
    std::array<DiscreteGrid::Position, AMREX_SPACEDIM> position{
        {AMREX_D_DECL(DiscreteGrid::Cell, DiscreteGrid::Node, DiscreteGrid::Cell)}};
    std::array<DiscreteGrid, AMREX_SPACEDIM> grids{};
    for (Direction dir : {AMREX_D_DECL(Direction::xDir, Direction::yDir, Direction::zDir)})
    {
        grids[dir] = DiscreteGrid{parameters, position};
    }
    DiscreteVectorField df{"df", parameters, grids, Grid::dual, 3};
    DiscreteVectorField res{"df", parameters, grids, Grid::dual, 3};
    for (Direction dir : {Direction::xDir, Direction::yDir, Direction::zDir})
    {
        df.multiFab(dir).setVal(1);
        res.multiFab(dir).setVal(0);
    }
    res += df;

    for (Direction dir : {Direction::xDir, Direction::yDir, Direction::zDir})
    {
        EXPECT_EQ(res.multiFab(dir).norm0(), 1);
        EXPECT_EQ(
            res.multiFab(dir).norm1(0, Gempic::Impl::to_amrex_periodicty(res.discrete_grid(dir))),
            res.discrete_grid(dir).size(Direction::xDir) *
                res.discrete_grid(dir).size(Direction::yDir) *
                res.discrete_grid(dir).size(Direction::zDir));
    }
}

TEST_F(LinearAlgebraTest, MultiplyAssignScalarVectorField)
{
    Gempic::Io::Parameters parameters;
    std::array<DiscreteGrid::Position, AMREX_SPACEDIM> position{
        {AMREX_D_DECL(DiscreteGrid::Cell, DiscreteGrid::Node, DiscreteGrid::Cell)}};
    std::array<DiscreteGrid, AMREX_SPACEDIM> grids{};
    for (Direction dir : {AMREX_D_DECL(Direction::xDir, Direction::yDir, Direction::zDir)})
    {
        grids[dir] = DiscreteGrid{parameters, position};
    }
    DiscreteVectorField df{"df", parameters, grids, Grid::dual, 3};
    DiscreteVectorField res{"df", parameters, grids, Grid::dual, 3};
    for (Direction dir : {Direction::xDir, Direction::yDir, Direction::zDir})
    {
        res.multiFab(dir).setVal(1);
    }
    res *= 2.0;

    for (Direction dir : {Direction::xDir, Direction::yDir, Direction::zDir})
    {
        EXPECT_EQ(res.multiFab(dir).norm0(), 2);
        EXPECT_EQ(
            res.multiFab(dir).norm1(0, Gempic::Impl::to_amrex_periodicty(res.discrete_grid(dir))),
            2 * GEMPIC_D_MULT(res.discrete_grid(dir).size(Direction::xDir) ,
                res.discrete_grid(dir).size(Direction::yDir) ,
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

    FieldsTest () : m_infra{Gempic::Test::Utils::get_compdom(s_gSize)} {}
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
