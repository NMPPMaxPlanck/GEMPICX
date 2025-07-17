
#include <gtest/gtest.h>

#include "GEMPIC_ComputationalDomain.H"
#include "GEMPIC_Config.H"
#include "GEMPIC_Parameters.H"

using namespace Gempic;

TEST(DiscreteGridTest, Node)
{
    amrex::Vector<amrex::Real> domainLo{AMREX_D_DECL(0.0, 0.0, 0.0)};
    amrex::Vector<amrex::Real> domainHi{AMREX_D_DECL(2.0, 2.5, 4.0)};
    amrex::Vector<int> nCell{AMREX_D_DECL(5, 6, 7)};
    amrex::Vector<int> isPeriodic{AMREX_D_DECL(1, 1, 1)};
    Io::Parameters parameters{};
    parameters.set("ComputationalDomain.domainLo", domainLo);
    parameters.set("ComputationalDomain.domainHi", domainHi);
    parameters.set("ComputationalDomain.nCell", nCell);
    parameters.set("ComputationalDomain.isPeriodic", isPeriodic);

    DiscreteGrid discreteGrid{
        parameters, {AMREX_D_DECL(DiscreteGrid::Node, DiscreteGrid::Node, DiscreteGrid::Node)}};
    for (auto dir : {AMREX_D_DECL(xDir, yDir, zDir)})
    {
        EXPECT_EQ(discreteGrid.position(dir), DiscreteGrid::Node);
        EXPECT_DOUBLE_EQ(discreteGrid.location_1d(dir, 0), discreteGrid.min(dir));
        EXPECT_DOUBLE_EQ(discreteGrid.location_1d(dir, discreteGrid.size(dir) - 1),
                         discreteGrid.max(dir));
        EXPECT_DOUBLE_EQ(discreteGrid.max(dir) - discreteGrid.min(dir), discreteGrid.length(dir));
        EXPECT_DOUBLE_EQ(discreteGrid.location_1d(dir, 2) - discreteGrid.location_1d(dir, 1),
                         discreteGrid.dx(dir));
    }
}

TEST(DiscreteGridTest, Cell)
{
    amrex::Vector<amrex::Real> domainLo{AMREX_D_DECL(0.0, 0.0, 0.0)};
    amrex::Vector<amrex::Real> domainHi{AMREX_D_DECL(2.0, 2.5, 4.0)};
    amrex::Vector<int> nCell{AMREX_D_DECL(5, 6, 7)};
    amrex::Vector<int> isPeriodic{AMREX_D_DECL(1, 1, 1)};
    Io::Parameters parameters{};
    parameters.set("ComputationalDomain.domainLo", domainLo);
    parameters.set("ComputationalDomain.domainHi", domainHi);
    parameters.set("ComputationalDomain.nCell", nCell);
    parameters.set("ComputationalDomain.isPeriodic", isPeriodic);

    DiscreteGrid discreteGrid{
        parameters, {AMREX_D_DECL(DiscreteGrid::Cell, DiscreteGrid::Cell, DiscreteGrid::Cell)}};
    for (auto dir : {AMREX_D_DECL(xDir, yDir, zDir)})
    {
        EXPECT_EQ(discreteGrid.position(dir), DiscreteGrid::Cell);
        EXPECT_DOUBLE_EQ(discreteGrid.location_1d(dir, 0),
                         discreteGrid.min(dir) + discreteGrid.dx(dir) / 2.0);
        EXPECT_DOUBLE_EQ(discreteGrid.location_1d(dir, discreteGrid.size(dir) - 1),
                         discreteGrid.max(dir) - discreteGrid.dx(dir) / 2.0);
        EXPECT_DOUBLE_EQ(discreteGrid.max(dir) - discreteGrid.min(dir), discreteGrid.length(dir));
        EXPECT_DOUBLE_EQ(discreteGrid.location_1d(dir, 2) - discreteGrid.location_1d(dir, 1),
                         discreteGrid.dx(dir));
    }
}
