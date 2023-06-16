#include <AMReX.H>
#include <GEMPIC_Config.H>
#include <GEMPIC_particle_mesh_coupling.H>
#include <GEMPIC_particle_mesh_coupling_C2.H>
#include "gtest/gtest.h"
#include "test_utils/GEMPIC_test_utils.H"

using namespace Particles;

class DegOne {
    public:
    static constexpr int deg{1};
};

class DegTwo {
    public:
    static constexpr int deg{2};
};

class DegThree {
    public:
    static constexpr int deg{3};
};

class DegFour {
    public:
    static constexpr int deg{4};
};

class DegFive {
    public:
    static constexpr int deg{5};
};

class DegSix {
    public:
    static constexpr int deg{6};
};

/*
template <typename X, typename Y, typename Z>
class SplineTest : public ::testing::test {
    public:
        static const int degx{X::deg};
        static const int degy{Y::deg};
        static const int degz{Z::deg};
        splines_at_particles<X::deg, Y::deg, Z::deg> spline;
};
*/
namespace {
template <class X>
class SplineTest : public testing::Test {
    protected:
        static const int degx{X::deg};
        spline1d_at_particles<X::deg> spline1D;
        splines_at_particles<X::deg, 1, 1> splineX;
        splines_at_particles<1, X::deg, 1> splineY;
        splines_at_particles<1, 1, X::deg> splineZ;

        amrex::GpuArray<amrex::Real, GEMPIC_SPACEDIM> const plo{AMREX_D_DECL(0.0, 0.0, 0.0)};
        amrex::GpuArray<amrex::Real, GEMPIC_SPACEDIM + 1> const dxi{AMREX_D_DECL(1.0, 1.0, 1.0), 1.0};
        
        void ConsistencyTests(amrex::GpuArray<amrex::Real, GEMPIC_SPACEDIM> &position) {
            splineX.init_particles(position, plo, dxi);
            splineY.init_particles(position, plo, dxi);
            splineZ.init_particles(position, plo, dxi);
            spline1D.init_position(position[0], plo[0], dxi[0]);
            
AMREX_D_TERM(EXPECT_DOUBLE_EQ(spline1D.box, splineX.box[0]);,
             EXPECT_DOUBLE_EQ(spline1D.box, splineY.box[1]);,
             EXPECT_DOUBLE_EQ(spline1D.box, splineZ.box[2]);)

            std::stringstream tmpSS;
            tmpSS << "Degree " << degx;
            for (int i{0}; i < degx; i++) {
AMREX_D_TERM(   EXPECT_DOUBLE_EQ(spline1D.spline[i], splineX.spline_node[0][i]) << tmpSS.str();,
                EXPECT_DOUBLE_EQ(spline1D.spline[i], splineY.spline_node[1][i]) << tmpSS.str();,
                EXPECT_DOUBLE_EQ(spline1D.spline[i], splineZ.spline_node[2][i]) << tmpSS.str();)
            }
/*
            if (degx < 6) {
                splines_at_particles<X::deg + 1, 1, 1> splineXCell;
                splineXCell.init_particles(position, plo, dxi);
                for (int i{0}; i < degx; i++) {
                    EXPECT_DOUBLE_EQ(splineX.spline_node[0][i], splineXCell.spline_cell[0][i])
                                    << tmpSS.str();
                }
            }
            */
        }
};

using MyTypes = ::testing::Types<DegOne, DegTwo, DegThree, DegFour, DegFive, DegSix>;
TYPED_TEST_SUITE(SplineTest, MyTypes);

TYPED_TEST(SplineTest, SplineConsistencyCheck) {
    // this-> syntax is necessary for typed tests
    auto plo{this->plo};
    auto dxi{this->dxi};
    const int numPos{4};

    amrex::GpuArray<amrex::GpuArray<amrex::Real, GEMPIC_SPACEDIM>, numPos> positions{
        // 1. Bottom of a cell
        plo,
        // 2. Middle of a cell
        {AMREX_D_DECL(plo[0] + 0.5/dxi[0], plo[1] + 0.5/dxi[1], plo[2] + 0.5/dxi[2])},
        // 3. Top of a cell
        {AMREX_D_DECL(plo[0] + 1.0/dxi[0], plo[0] + 1.0/dxi[1], plo[0] + 1.0/dxi[2])},
        // 4. Position closer to one node than the other
        {AMREX_D_DECL(plo[0] + 0.25/dxi[0], plo[1] + 0.25/dxi[1], plo[2] + 0.25/dxi[2])}};

    for (int i{0}; i < numPos; i++) {
        std::stringstream tmpSS;
        tmpSS << "Error for position vector number " << i + 1;
        SCOPED_TRACE(tmpSS.str());
        this->ConsistencyTests(positions[i]);
    }

    constexpr int degreeX{this->degx};
    EXPECT_LT(degreeX, 7);
    EXPECT_GT(degreeX, 0);
}

/*
TEST_P(SplineTest, NullTest) {
    ASSERT_EQ(1, true);
    EXPECT_EQ(1, degx);
    //EXPECT_EQ(1, degy);
    //EXPECT_EQ(1, degz);
}

INSTANTIATE_TEST_SUITE_P(
    Low3Degs, SplineTest,
    //::testing::Combine(::testing::Range(1,3), ::testing::Range(1,3), ::testing::Range(1,3)));
    //::testing::Combine(::testing::Values(1,2),::testing::Values(1,2),::testing::Values(1,2)));
    ::testing::Values(1,2));

);
*/
} //namespace
