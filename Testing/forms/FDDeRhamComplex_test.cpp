#include <AMReX.H>
#include "gtest/gtest.h"
#include "GEMPIC_computational_domain.H"
#include "GEMPIC_FDDeRhamComplex.H"
#include "GEMPIC_Fields.H"
#include "GEMPIC_Params.H"
#include "GEMPIC_parameters.H"
#include "test_utils/GEMPIC_test_utils.H"

// E = one form
// rho = three form
// phi = zero form

#define checkField(...) GEMPIC_TestUtils::checkField(__FILE__, __LINE__, __VA_ARGS__)

using namespace Gempic;
using namespace GEMPIC_FDDeRhamComplex;

namespace {
    class FDDeRhamComplexTest : public testing::Test {
        protected:

        // Linear splines is ok, and lower dimension Hodge is good enough
        static const int vDim{3};
        static const int numSpec{1};
        // Spline degreesx
        static const int degX{2};
        static const int degY{2};
        static const int degZ{2};
        //
        static const int degmw{2};
        static const int propagator{3};

        static const int hodgeDegree{6};
        // Number of ghost cells in mesh
        const int Nghost{GEMPIC_TestUtils::initNGhost(degX, degY, degZ)};
        Parameters params;
        computational_domain infra;

        // virtual void SetUp() will be called before each test is run.
        void SetUp() override {
            /* Initialize the infrastructure */
            const amrex::RealBox realBox({AMREX_D_DECL(0.0, 0.0, 0.0)},
                                         {AMREX_D_DECL(10.0, 10.0, 10.0)});
            const amrex::IntVect nCell{AMREX_D_DECL(10, 10, 10)};
            const amrex::IntVect maxGridSize{AMREX_D_DECL(10, 10, 10)};
            const amrex::Array<int, GEMPIC_SPACEDIM> isPeri{AMREX_D_DECL(1, 1, 1)};
            const amrex::IntVect isPeriodic{AMREX_D_DECL(1, 1, 1)};

            params = Parameters(realBox, nCell, maxGridSize, isPeri, hodgeDegree);

            infra.initialize_computational_domain(nCell, maxGridSize, isPeriodic, realBox);
        }
    };

    TEST_F(FDDeRhamComplexTest, MatrixMultTestDeg2) {
        const std::string analyticalFunc = "1.0";

        const int nVar = GEMPIC_SPACEDIM + 1;  // x, y, z, t
        amrex::ParserExecutor<nVar> func;
        amrex::Parser parser;

        parser.define(analyticalFunc);
        parser.registerVariables({AMREX_D_DECL("x", "y", "z"), "t"});
        func = parser.compile<nVar>();

        // Initialize the De Rham Complex
        auto deRham = std::make_shared<FDDeRhamComplex>(params);
        
        DeRhamField<Grid::dual, Space::cell> rho(deRham, func);
        DeRhamField<Grid::primal, Space::node> phi(deRham);

        EXPECT_NEAR(1, Gempic::Utils::gempic_norm(rho.data, infra, 2), 1e-12);

        // Select stencil according to degree
        auto [stencilNodeToCell, stencilCellToNode] =
            getHodgeStencils<degX, HodgeScheme::FDHodge>();

        const amrex::Geometry geom = params.geometry();
        matrixMult<xDir>(geom, stencilCellToNode, rho.data, phi.data);

        bool loopRun{false};

        for (amrex::MFIter mfi(phi.data); mfi.isValid(); ++mfi)
        {
            loopRun = true;

            checkField((phi.data[mfi]).array(), infra.n_cell.dim3(),
                    // Expect only one node of rhoarr (0, 0, 0) to be non-zero
                    {[] (AMREX_D_DECL(int a, int b, int c)) {return AMREX_D_TERM(a == 0,
                                                                              && b == 0,
                                                                              && c == 0);}},
                    // and receiving full weight of particle (1)
                    {1},
                    // with the remaining entries being 0
                    1);
        }
        ASSERT_TRUE(loopRun);
    }

    TEST_F(FDDeRhamComplexTest, MatrixMultTestDeg4) {
        const std::string analyticalFunc = "1.0";

        const int nVar = GEMPIC_SPACEDIM + 1;  // x, y, z, t
        amrex::ParserExecutor<nVar> func;
        amrex::Parser parser;

        parser.define(analyticalFunc);
        parser.registerVariables({AMREX_D_DECL("x", "y", "z"), "t"});
        func = parser.compile<nVar>();

        // Initialize the De Rham Complex
        auto deRham = std::make_shared<FDDeRhamComplex>(params);
        
        DeRhamField<Grid::dual, Space::cell> rho(deRham, func);
        DeRhamField<Grid::primal, Space::node> phi(deRham);

        EXPECT_NEAR(1, Gempic::Utils::gempic_norm(rho.data, infra, 2), 1e-12);

        // Select stencil according to degree
        auto [stencilNodeToCell, stencilCellToNode] =
            getHodgeStencils<4, HodgeScheme::FDHodge>();

        const amrex::Geometry geom = params.geometry();
        matrixMult<xDir>(geom, stencilCellToNode, rho.data, phi.data);

        bool loopRun{false};

        for (amrex::MFIter mfi(phi.data); mfi.isValid(); ++mfi)
        {
            loopRun = true;

            // Expect all entires to be 1
            checkField((phi.data[mfi]).array(), infra.n_cell.dim3(), {}, {}, 1);
        }
        ASSERT_TRUE(loopRun);
    }

    TEST_F(FDDeRhamComplexTest, MatrixMultTestDeg6) {

        // Select stencil according to degree
        const int stencilLength = 6 - 1;
        amrex::GpuArray<amrex::Real, stencilLength> stencilNodeToCell;
        amrex::GpuArray<amrex::Real, stencilLength> stencilCellToNode;

        const std::string analyticalFunc = "1.0";

        const int nVar = GEMPIC_SPACEDIM + 1;  // x, y, z, t
        amrex::ParserExecutor<nVar> func;
        amrex::Parser parser;

        parser.define(analyticalFunc);
        parser.registerVariables({AMREX_D_DECL("x", "y", "z"), "t"});
        func = parser.compile<nVar>();

        // Initialize the De Rham Complex
        auto deRham = std::make_shared<FDDeRhamComplex>(params);
        
        DeRhamField<Grid::dual, Space::cell> rho(deRham, func);
        DeRhamField<Grid::primal, Space::node> phi(deRham);

        EXPECT_NEAR(1, Gempic::Utils::gempic_norm(rho.data, infra, 2), 1e-12);

        std::tie(stencilNodeToCell, stencilCellToNode) =
            getHodgeStencils<6, HodgeScheme::FDHodge>();

        const amrex::Geometry geom = params.geometry();
        matrixMult<xDir>(geom, stencilCellToNode, rho.data, phi.data);

        bool loopRun{false};

        for (amrex::MFIter mfi(phi.data); mfi.isValid(); ++mfi)
        {
            loopRun = true;
            
            // Expect all nodes to be 1
            checkField((phi.data[mfi]).array(), infra.n_cell.dim3(), {}, {}, 1);
        }
        ASSERT_TRUE(loopRun);
    }

    TEST_F(FDDeRhamComplexTest, HodgeFDThreeFormZeroFormTest) {

        // Initialize the De Rham Complex
        auto deRham = std::make_shared<FDDeRhamComplex>(params);
        
        DeRhamField<Grid::dual, Space::cell> rho(deRham);
        DeRhamField<Grid::primal, Space::node> phi(deRham);
            
        ASSERT_EQ(0,Gempic::Utils::gempic_norm(rho.data, infra, 2));

        // Select stencil according to degree
        auto [stencilNodeToCell, stencilCellToNode] =
            getHodgeStencils<degX, HodgeScheme::FDHodge>();

        const amrex::Geometry geom = params.geometry();
        // matrixMult<xDir>(geom, stencilCellToNode, rho.data, phi.data);

        bool loopRun{false};

        for (amrex::MFIter mfi(phi.data); mfi.isValid(); ++mfi)
        {
            loopRun = true;

            // Expect all entires to be 0
            checkField((phi.data[mfi]).array(), infra.n_cell.dim3(), {}, {}, 0);
        }
        ASSERT_TRUE(loopRun);
    }

    TEST_F(FDDeRhamComplexTest, HodgeFDThreeFormZeroFormTestII) {

        const std::string analyticalFunc = "1.0";

        const int nVar = GEMPIC_SPACEDIM + 1;  // x, y, z, t
        amrex::ParserExecutor<nVar> func;
        amrex::Parser parser;

        parser.define(analyticalFunc);
        parser.registerVariables({AMREX_D_DECL("x", "y", "z"), "t"});
        func = parser.compile<nVar>();

        // Initialize the De Rham Complex
        auto deRham = std::make_shared<FDDeRhamComplex>(params);
        
        DeRhamField<Grid::dual, Space::cell> rho(deRham, func);
        DeRhamField<Grid::primal, Space::node> phi(deRham);

        EXPECT_NEAR(1, Gempic::Utils::gempic_norm(rho.data, infra, 2), 1e-12);

        // Select stencil according to degree
        auto [stencilNodeToCell, stencilCellToNode] =
            getHodgeStencils<degX, HodgeScheme::FDHodge>();

        const amrex::Geometry geom = params.geometry();
        matrixMult<xDir>(geom, stencilCellToNode, rho.data, phi.data);

        bool loopRun{false};

        for (amrex::MFIter mfi(phi.data); mfi.isValid(); ++mfi)
        {
            loopRun = true;

            // Expect all entires to be 1
            checkField((phi.data[mfi]).array(), infra.n_cell.dim3(), {}, {}, 1);
        }
        ASSERT_TRUE(loopRun);
    }

    TEST_F(FDDeRhamComplexTest, HodgeFDThreeFormZeroFormTestIII) {

        const std::string analyticalFunc = "1.0";

        const int nVar = GEMPIC_SPACEDIM + 1;  // x, y, z, t
        amrex::ParserExecutor<nVar> func;
        amrex::Parser parser;

        parser.define(analyticalFunc);
        parser.registerVariables({AMREX_D_DECL("x", "y", "z"), "t"});
        func = parser.compile<nVar>();

        // Initialize the De Rham Complex
        auto deRham = std::make_shared<FDDeRhamComplex>(params);

        DeRhamField<Grid::dual, Space::cell> rho(deRham);
        DeRhamField<Grid::primal, Space::node> phi(deRham, func);

//        const amrex::BoxArray &nba{amrex::convert(infra.grid, amrex::IntVect::TheNodeVector())};

//        rho.data.define(nba, infra.distriMap, Ncomp, Nghost);

//        phi.data.define(nba, infra.distriMap, Ncomp, Nghost);

        ASSERT_EQ(0,Gempic::Utils::gempic_norm(rho.data, infra, 2));

        // Select stencil according to degree
        auto [stencilNodeToCell, stencilCellToNode] =
            getHodgeStencils<degX, HodgeScheme::FDHodge>();

        const amrex::Geometry geom = params.geometry();
        matrixMult<xDir>(geom, stencilCellToNode, rho.data, phi.data);
    
        bool loopRun{false};
        
        for (amrex::MFIter mfi(phi.data); mfi.isValid(); ++mfi)
        {
            loopRun = true;

            // Expect all entires to be 0
            checkField((phi.data[mfi]).array(), infra.n_cell.dim3(), {}, {}, 0);
        }
        ASSERT_TRUE(loopRun);
    }
}
