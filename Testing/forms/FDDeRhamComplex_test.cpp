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

using namespace Gempic;
using namespace GEMPIC_FDDeRhamComplex;

namespace {

        /* Helper function to check entries of rho given a series of conditions and a default
         * value. Check order is prioritized, so a set of indices only fulfill the first succesful
         * condition.
         * 
         * Parameters:
         * ----------
         * @param line: int, the line from which the function was called
         * @param rhoarr: amrex::Array4, array containing rho values in an easily reached accessor
         * @param top: Dim3, top boundaries of box for rhoarray
         * @param condVec: vector<condLambda>, Vector of lambdas that check if the {SPACEDIM} indices fulfill a given condition.
         * @param checks: vector<amrex::Real>, Vector of values to compare to if indices fulfill the corresponding condVec condition.
         * @param defCheck: amrex::Real, Default value for all indices not fulfilling any of the given conditions.
         */
        using condLambda = bool(*)(AMREX_D_DECL(int, int, int));
        void checkRho(int line,
                      amrex::Array4<amrex::Real> const& rhoarr,
                      amrex::Dim3 const&& top,
                      std::vector<condLambda>&& condVec,
                      std::vector<amrex::Real>&& checks,
                      amrex::Real defCheck) {
            // Expect only one node of rhoarr (0, 0, 0) to be non-zero and receiving full weight of particle (1)
            for (int i{0}; i <= top.x; i++) { 
                for (int j{0}; j <= top.y; j++) {
                    for (int k{0}; k <= top.z; k++) {
                        int condNum{0};
                        amrex::Array<int, GEMPIC_SPACEDIM> idx{AMREX_D_DECL(i, j, k)};
                        for (auto cond : condVec) {
                            if (cond(AMREX_D_DECL(i, j, k))) {
                                EXPECT_NEAR(checks[condNum], *rhoarr.ptr(AMREX_D_DECL(i, j, k), 0), 1e-8) <<
                                   "LINE:" << line << ": Failed condition " << condNum <<
                                   ".\nIndices: " << GEMPIC_TestUtils::stringArray(idx, GEMPIC_SPACEDIM);
                                   break;
                            }
                            condNum++;
                        }
                        if (condNum == condVec.size()) {
                            EXPECT_NEAR(defCheck, *rhoarr.ptr(AMREX_D_DECL(i, j, k), 0), 1e-8) <<
                                "LINE:" << line << ": Failed default value check:" << defCheck <<
                                ".\nIndices: " << GEMPIC_TestUtils::stringArray(idx, GEMPIC_SPACEDIM);
                        }
                    }
                }
            }
        }

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

        static const int hodgeDegree{2};
        // Number of ghost cells in mesh
        const int Nghost{GEMPIC_TestUtils::initNGhost(1, 1, 1)};
        Parameters params;
        computational_domain infra;
        amrex::MultiFab rhoData;
        amrex::MultiFab phiData;

        // virtual void SetUp() will be called before each test is run.
        void SetUp() override {
            /* Initialize the infrastructure */
            const amrex::RealBox realBox({AMREX_D_DECL(0.0, 0.0, 0.0)},
                                         {AMREX_D_DECL(10.0, 10.0, 10.0)});
            const amrex::IntVect nCell{AMREX_D_DECL(10, 10, 10)};
            const amrex::IntVect maxGridSize{AMREX_D_DECL(10, 10, 10)};
            const amrex::Array<int, 3> isPeri{AMREX_D_DECL(1, 1, 1)};
            const amrex::IntVect isPeriodic{AMREX_D_DECL(1, 1, 1)};

            Parameters params(realBox, nCell, maxGridSize, isPeri, hodgeDegree);

            infra.initialize_computational_domain(nCell, maxGridSize, isPeriodic, realBox);
            // Setup rho. This is  the special part of this text fixture.
            // node centered BA:
            const amrex::BoxArray &nba{amrex::convert(infra.grid, amrex::IntVect::TheNodeVector())};
            int Ncomp{1};            

            rhoData.define(nba, params.distriMap(), Ncomp, {AMREX_D_DECL(1, 1, 1)});
            //rhoData.define(nba, infra.distriMap, Ncomp, {AMREX_D_DECL(1, 1, 1)});
            rhoData.setVal(1.0);

            phiData.define(nba, params.distriMap(), Ncomp, {AMREX_D_DECL(1, 1, 1)});
            //phiData.define(nba, infra.distriMap, Ncomp, {AMREX_D_DECL(1, 1, 1)});
            phiData.setVal(0.0);

            // Ensure rho exists and is 0 everywhere
            // ASSERT_EQ(0,Gempic::Utils::gempic_norm(rho, infra, 2));
        }
    };

    TEST_F(FDDeRhamComplexTest, MatrixMultTestDeg2) {

        // Select stencil according to degree
        const int stencilLength = degX - 1;
        amrex::GpuArray<amrex::Real, stencilLength> stencilNodeToCell;
        amrex::GpuArray<amrex::Real, stencilLength> stencilCellToNode;

        std::tie(stencilNodeToCell, stencilCellToNode) =
            getHodgeStencils<degX, 0, stencilLength>();

        const amrex::Geometry geom = params.geometry();
        matrixMult<stencilLength, 0>(geom, stencilCellToNode, rhoData, phiData);

        bool loopRun{false};

        for (amrex::MFIter mfi(phiData); mfi.isValid(); ++mfi)
        {
            loopRun = true;

            checkRho(__LINE__, (phiData[mfi]).array(), infra.n_cell.dim3(),
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

        // Select stencil according to degree
        const int stencilLength = 4 - 1;
        amrex::GpuArray<amrex::Real, stencilLength> stencilNodeToCell;
        amrex::GpuArray<amrex::Real, stencilLength> stencilCellToNode;

        std::tie(stencilNodeToCell, stencilCellToNode) =
            getHodgeStencils<4, 0, stencilLength>();

        const amrex::Geometry geom = params.geometry();
        matrixMult<stencilLength, 0>(geom, stencilCellToNode, rhoData, phiData);

        bool loopRun{false};

        for (amrex::MFIter mfi(phiData); mfi.isValid(); ++mfi)
        {
            loopRun = true;

            checkRho(__LINE__, (phiData[mfi]).array(), infra.n_cell.dim3(),
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

    TEST_F(FDDeRhamComplexTest, MatrixMultTestDeg6) {

        // Select stencil according to degree
        const int stencilLength = 6 - 1;
        amrex::GpuArray<amrex::Real, stencilLength> stencilNodeToCell;
        amrex::GpuArray<amrex::Real, stencilLength> stencilCellToNode;

        std::tie(stencilNodeToCell, stencilCellToNode) =
            getHodgeStencils<6, 0, stencilLength>();

        const amrex::Geometry geom = params.geometry();
        matrixMult<stencilLength, 0>(geom, stencilCellToNode, rhoData, phiData);

        bool loopRun{false};

        for (amrex::MFIter mfi(phiData); mfi.isValid(); ++mfi)
        {
            loopRun = true;

            checkRho(__LINE__, (phiData[mfi]).array(), infra.n_cell.dim3(),
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

    TEST_F(FDDeRhamComplexTest, HodgeFDThreeFormZeroFormTest) {

        // Initialize the De Rham Complex
        auto deRham = std::make_shared<FDDeRhamComplex>(params);
        
        DeRhamField<Grid::dual, Space::cell> rho(deRham);
        DeRhamField<Grid::primal, Space::node> phi(deRham);
            
        ASSERT_EQ(0,Gempic::Utils::gempic_norm(rho.data, infra, 2));

        // Select stencil according to degree
        const int stencilLength = degX - 1;
        amrex::GpuArray<amrex::Real, stencilLength> stencilNodeToCell;
        amrex::GpuArray<amrex::Real, stencilLength> stencilCellToNode;

        std::tie(stencilNodeToCell, stencilCellToNode) =
            getHodgeStencils<degX, 0, stencilLength>();

        const amrex::Geometry geom = params.geometry();
        // matrixMult<stencilLength, 0>(geom, stencilCellToNode, rho.data, phi.data);

        bool loopRun{false};

        for (amrex::MFIter mfi(phi.data); mfi.isValid(); ++mfi)
        {
            loopRun = true;

            checkRho(__LINE__, (phi.data[mfi]).array(), infra.n_cell.dim3(),
                    // Expect only one node of rhoarr (0, 0, 0) to be non-zero
                    {[] (AMREX_D_DECL(int a, int b, int c)) {return AMREX_D_TERM(a == 0,
                                                                              && b == 0,
                                                                              && c == 0);}},
                    // and receiving full weight of particle (1)
                    {0},
                    // with the remaining entries being 0
                    0);
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

        ASSERT_EQ(0,Gempic::Utils::gempic_norm(rho.data, infra, 2));

        // Select stencil according to degree
        const int stencilLength = degX - 1;
        amrex::GpuArray<amrex::Real, stencilLength> stencilNodeToCell;
        amrex::GpuArray<amrex::Real, stencilLength> stencilCellToNode;

        std::tie(stencilNodeToCell, stencilCellToNode) =
            getHodgeStencils<degX, 0, stencilLength>();

        const amrex::Geometry geom = params.geometry();
        matrixMult<stencilLength, 0>(geom, stencilCellToNode, rho.data, phi.data);

        bool loopRun{false};

        for (amrex::MFIter mfi(phi.data); mfi.isValid(); ++mfi)
        {
            loopRun = true;

            checkRho(__LINE__, (phi.data[mfi]).array(), infra.n_cell.dim3(),
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

        const amrex::BoxArray &nba{amrex::convert(infra.grid, amrex::IntVect::TheNodeVector())};

//        rho.data.define(nba, infra.distriMap, Ncomp, Nghost);

//        phi.data.define(nba, infra.distriMap, Ncomp, Nghost);

        ASSERT_EQ(0,Gempic::Utils::gempic_norm(rho.data, infra, 2));

        // Select stencil according to degree
        const int stencilLength = degX - 1;
        amrex::GpuArray<amrex::Real, stencilLength> stencilNodeToCell;
        amrex::GpuArray<amrex::Real, stencilLength> stencilCellToNode;

        std::tie(stencilNodeToCell, stencilCellToNode) =
            getHodgeStencils<degX, 0, stencilLength>();

        const amrex::Geometry geom = params.geometry();
        matrixMult<stencilLength, 0>(geom, stencilCellToNode, rho.data, phi.data);
    
        bool loopRun{false};
        
        for (amrex::MFIter mfi(phi.data); mfi.isValid(); ++mfi)
        {
            loopRun = true;

            checkRho(__LINE__, (phi.data[mfi]).array(), infra.n_cell.dim3(),
                    // Expect only one node of rhoarr (0, 0, 0) to be non-zero
                    {[] (AMREX_D_DECL(int a, int b, int c)) {return AMREX_D_TERM(a == 0,
                                                                              && b == 0,
                                                                              && c == 0);}},
                    // and receiving full weight of particle (1)
                    {0},
                    // with the remaining entries being 0
                    0);
        }
        ASSERT_TRUE(loopRun);
    }
}
