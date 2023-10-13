#include <AMReX.H>
#include <GEMPIC_FDDeRhamComplex.H>
#include <GEMPIC_Fields.H>
#include <GEMPIC_particle_mesh_coupling.H>
#include "gtest/gtest.h"
#include "test_utils/GEMPIC_test_utils.H"
#include "GEMPIC_Spline_Class.H"

using namespace Particles;
using namespace GEMPIC_FDDeRhamComplex;
using namespace GEMPIC_Fields;

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
                    const amrex::IntVect idx{AMREX_D_DECL(i, j, k)};
                    for (auto cond : condVec) {
                        if (cond(AMREX_D_DECL(i, j, k))) {
                            EXPECT_NEAR(checks[condNum], *rhoarr.ptr(idx, 0), 1e-8) <<
                                "LINE:" << line << ": Failed condition " << condNum <<
                                ".\nIndices: " << GEMPIC_TestUtils::stringArray(idx, GEMPIC_SPACEDIM);
                                break;
                        }
                        condNum++;
                    }
                    if (condNum == condVec.size()) {
                        EXPECT_NEAR(defCheck, *rhoarr.ptr(idx, 0), 1e-8) <<
                            "LINE:" << line << ": Failed default value check:" << defCheck <<
                            ".\nIndices: " << GEMPIC_TestUtils::stringArray(idx, GEMPIC_SPACEDIM);
                    }
                }
            }
        }
    }

    // When using amrex::ParallelFor you have to create a standalone helper function that does the execution on GPU and call that function from the unit test because of how GTest creates tests within a TEST_F fixture.
    template <int vDim, int degX, int degY, int degZ, int pDim>
    void accumulateJUpdateVC2ParallelFor(amrex::ParIter<0, 0, vDim + 1, 0>& pti,
                                 DeRhamField<Grid::primal, Space::face>& B,
                                 DeRhamField<Grid::dual, Space::face>& J,
                                 computational_domain& infra,
                                 amrex::Real weight,
                                 amrex::GpuArray<amrex::Real, GEMPIC_SPACEDIM> const dx,
                                 amrex::GpuArray<amrex::Real, 2>& bfields) {

        const long np{pti.numParticles()};

        const auto& particles{pti.GetArrayOfStructs()};
        const auto partData{particles().data()};

        amrex::GpuArray<amrex::Array4<amrex::Real>, vDim> jA;
        for (int cc = 0; cc < vDim; cc++) jA[cc] = (J.data[cc])[pti].array();

        amrex::AsyncArray aaBfields(&bfields, 1);
        auto *bfieldsGPU = aaBfields.data();

        amrex::GpuArray<amrex::Array4<amrex::Real>, int(vDim / 2.5) * 2 + 1> bA;
        for (int cc = 0; cc < (int(vDim / 2.5) * 2 + 1); cc++) bA[cc] = (B.data[cc])[pti].array();

        amrex::ParallelFor(np, [=] AMREX_GPU_DEVICE(long pp)
        {

            amrex::GpuArray<amrex::Real, GEMPIC_SPACEDIM> pos_start;
            for (unsigned int d{0}; d < GEMPIC_SPACEDIM; ++d)
                pos_start[d] = partData[0].pos(d);

            amrex::Real x_end = 0;

            Spline::SplineWithPrimitive<degX, degY, degZ> spline(pos_start, infra.plo, infra.dxi);

            spline.template update1DSplines<pDim>(x_end, infra.plo[0], infra.dxi[0]);
            spline.template update1DPrimitive<pDim>(x_end, infra.plo[0], infra.dxi[0]);

            accumulate_J_integrate_B<Spline::SplineWithPrimitive<degX, degY, degZ>, vDim, pDim>(spline, weight, dx, bA, jA, *bfieldsGPU);
        });

        aaBfields.copyToHost(&bfields, 1);
    }

    class AccumulateJUpdateVC2Test : public testing::Test {
        protected:

        static const int degX{1};
        static const int degY{1};
        static const int degZ{1};

        static const int numSpec{1};
        static const int vDim{3};
        static const int spec{0};
        const int Nghost{GEMPIC_TestUtils::initNGhost(degX, degY, degZ)};
        Parameters params;

        double charge{1};
        double mass{1};

        amrex::GpuArray<amrex::Array4<amrex::Real>, vDim> jA;
        amrex::GpuArray<amrex::Array4<amrex::Real>, int(vDim / 2.5) * 2 + 1> bA;

        computational_domain infra;
        amrex::GpuArray<std::unique_ptr<particle_groups<vDim>>, numSpec> particleGroup;

        static const int degP{1};
        static const int degP1{1};
        static const int degP2{1};

        static const int numParticles{1};

        static const int pDim{1};

        amrex::Real weight = 1.0;

        amrex::GpuArray<amrex::Real, 2> bfields{0., 0.};
        amrex::GpuArray<amrex::Real, std::max(degX, std::max(degY, degZ)) + 4> primitive;

        void SetUp() override {
            if constexpr(GEMPIC_SPACEDIM != 3) {
                GTEST_SKIP() << "This function barely works in 3D, let alone lower dimensions.";
            }
            /* Initialize the infrastructure */
            const amrex::RealBox realBox({AMREX_D_DECL(0.0, 0.0, 0.0)},
                                            {AMREX_D_DECL(10.0, 10.0, 10.0)});
            const amrex::IntVect nCell{AMREX_D_DECL(10, 10, 10)};
            const amrex::IntVect maxGridSize{AMREX_D_DECL(10, 10, 10)};
            const amrex::Array<int, GEMPIC_SPACEDIM> isPeriodic{AMREX_D_DECL(1, 1, 1)};
            const int hodgeDegree{2};

            // This class does the same as GEMPIC_parameters.H and needs to be urgently redesigned.
            // {1, 1, 1} represents periodicity, has different types than Params and gempic_parameters.
            infra.initialize_computational_domain(nCell, maxGridSize, {AMREX_D_DECL(1, 1, 1)}, realBox);

            params = Parameters(realBox, nCell, maxGridSize, isPeriodic, hodgeDegree);
            
            // particles
            for (int species{0}; species < numSpec; species++)
            {
                particleGroup[species] =
                    std::make_unique<particle_groups<vDim>>(charge, mass, infra);
            }
        }
    };

    TEST_F(AccumulateJUpdateVC2Test, NullTest) {
        // Adding particle to one cell
        const int numParticles{1};
        amrex::Array<amrex::GpuArray<amrex::Real, GEMPIC_SPACEDIM>, numParticles> positions{*infra.geom.ProbLo()};
        amrex::Array<amrex::Real, numParticles> weights{1};
        GEMPIC_TestUtils::addSingleParticles<vDim, numSpec, numParticles>(particleGroup, infra, weights, positions);

        // (default) charge correctly transferred from addSingleParticles
        EXPECT_EQ(1, particleGroup[0]->getCharge()); 

        const amrex::Array<std::string, 3> analyticalFuncB = {"0.0", 
                                                              "0.0",
                                                              "0.0"};

        const amrex::Array<std::string, 3> analyticalFuncJ = {"0.0", 
                                                              "0.0",
                                                              "0.0"};

        const int nVar{4}; //x, y, z, t
        amrex::Array<amrex::ParserExecutor<nVar>, GEMPIC_SPACEDIM> funcB;
        amrex::Array<amrex::ParserExecutor<nVar>, GEMPIC_SPACEDIM> funcJ; 
        amrex::Array<amrex::Parser, 6> parser;

        for (int i{0}; i<3; ++i)
        {
            parser[i].define(analyticalFuncB[i]);
            parser[i].registerVariables({"x", "y", "z", "t"});
            funcB[i] = parser[i].compile<4>();
        }

        for (int i{0}; i<3; ++i)
        {
            parser[i+3].define(analyticalFuncJ[i]);
            parser[i+3].registerVariables({"x", "y", "z", "t"});
            funcJ[i] = parser[i+3].compile<4>();
        }

        // Initialize the De Rham Complex
        auto deRham{std::make_shared<FDDeRhamComplex>(params)};

        DeRhamField<Grid::primal, Space::face> B(deRham, funcB);

        DeRhamField<Grid::dual, Space::face> J(deRham, funcJ);

        particleGroup[0]->Redistribute();  // assign particles to the tile they are in
        // Particle iteration ... over one particle. Hopefully.

        for (amrex::ParIter<0, 0, vDim + 1, 0> pti(*particleGroup[spec], 0); pti.isValid(); ++pti)
        {

            const long np{pti.numParticles()};
            EXPECT_EQ(1, np); // Only one particle added by addSingleParticles

            amrex::GpuArray<amrex::Real, 2> bfields{0., 0.};

            accumulateJUpdateVC2ParallelFor<vDim, degX, degY, degZ, pDim>(pti, B, J, infra, weight, infra.dx, bfields);

            EXPECT_EQ(bfields[0], 0);
            EXPECT_EQ(bfields[1], 0);

            // Expect all nodes to be 0
            checkRho(__LINE__, (J.data[0]).array(pti), infra.n_cell.dim3(), {}, {}, 0);
            checkRho(__LINE__, (J.data[1]).array(pti), infra.n_cell.dim3(), {}, {}, 0);
            checkRho(__LINE__, (J.data[2]).array(pti), infra.n_cell.dim3(), {}, {}, 0);
        }
    }

    TEST_F(AccumulateJUpdateVC2Test, SingleParticleMiddle) {
        // Adding particle to one cell
        const int numParticles{1};
        amrex::Array<amrex::GpuArray<amrex::Real, GEMPIC_SPACEDIM>, numParticles> positions{AMREX_D_DECL(infra.geom.ProbHi()[0] - 5.5*infra.dx[0],
                      infra.geom.ProbHi()[1] - 5.5*infra.dx[1],
                      infra.geom.ProbHi()[2] - 5.5*infra.dx[2])};
        amrex::Array<amrex::Real, numParticles> weights{1};
        GEMPIC_TestUtils::addSingleParticles<vDim, numSpec, numParticles>(particleGroup, infra, weights, positions);

        // (default) charge correctly transferred from addSingleParticles
        EXPECT_EQ(1, particleGroup[0]->getCharge()); 

        const amrex::Array<std::string, 3> analyticalFuncB = {"1.0", 
                                                              "1.0",
                                                              "1.0"};

        const amrex::Array<std::string, 3> analyticalFuncJ = {"1.0", 
                                                              "1.0",
                                                              "1.0"};

        const int nVar{4}; //x, y, z, t
        amrex::Array<amrex::ParserExecutor<nVar>, GEMPIC_SPACEDIM> funcB;
        amrex::Array<amrex::ParserExecutor<nVar>, GEMPIC_SPACEDIM> funcJ; 
        amrex::Array<amrex::Parser, 6> parser;

        for (int i{0}; i<3; ++i)
        {
            parser[i].define(analyticalFuncB[i]);
            parser[i].registerVariables({"x", "y", "z", "t"});
            funcB[i] = parser[i].compile<4>();
        }

        for (int i{0}; i<3; ++i)
        {
            parser[i+3].define(analyticalFuncJ[i]);
            parser[i+3].registerVariables({"x", "y", "z", "t"});
            funcJ[i] = parser[i+3].compile<4>();
        }

        // Initialize the De Rham Complex
        auto deRham{std::make_shared<FDDeRhamComplex>(params)};

        DeRhamField<Grid::primal, Space::face> B(deRham, funcB);

        DeRhamField<Grid::dual, Space::face> J(deRham, funcJ);

        particleGroup[0]->Redistribute();  // assign particles to the tile they are in
        // Particle iteration ... over one particle. Hopefully.

        for (amrex::ParIter<0, 0, vDim + 1, 0> pti(*particleGroup[spec], 0); pti.isValid(); ++pti)
        {
            const long np{pti.numParticles()};
            EXPECT_EQ(1, np); // Only one particle added by addSingleParticles

            amrex::GpuArray<amrex::Real, 2> bfields{0., 0.};

            accumulateJUpdateVC2ParallelFor<vDim, degX, degY, degZ, pDim>(pti, B, J, infra, weight, infra.dx, bfields);

            EXPECT_NEAR(bfields[0], -4.5, 1e-15);
            EXPECT_NEAR(bfields[1], -4.5, 1e-15);

            checkRho(__LINE__, (J.data[pDim]).array(pti), infra.n_cell.dim3(), 
                    // Expect the eight nearest nodes (4/5, 4/5, 4/5) to be non-zero 
                    {[] (AMREX_D_DECL(int a, int b, int c)) {return AMREX_D_TERM(
                                                            (a == 4 || a == 5),
                                                          && b == 4,
                                                         && (c == 4 || c == 5));},
                    [] (AMREX_D_DECL(int a, int b, int c)) {return AMREX_D_TERM(
                                                            (a == 4 || a == 5),
                                                         &&  b <= 3,
                                                         && (c == 4 || c == 5));}},
                    // getting an eight of the particle weight times the primitive, plus the original 1
                    {1 - 1./8, 1 - 0.25},
                    // with the remaining entries being 1
                    1);
            checkRho(__LINE__, (J.data[(pDim + 1) % 3]).array(pti), infra.n_cell.dim3(), {}, {}, 1);
            checkRho(__LINE__, (J.data[(pDim + 2) % 3]).array(pti), infra.n_cell.dim3(), {}, {}, 1);
        }
    }

    TEST_F(AccumulateJUpdateVC2Test, SingleParticleUnevenNodeSplit) {
        // Adding particle to one cell
        const int numParticles{1};
        amrex::Array<amrex::GpuArray<amrex::Real, GEMPIC_SPACEDIM>, numParticles> positions{AMREX_D_DECL(infra.geom.ProbHi()[0] - 5.25*infra.dx[0],
                      infra.geom.ProbHi()[1] - 5.25*infra.dx[1],
                      infra.geom.ProbHi()[2] - 5.25*infra.dx[2])};
        amrex::Array<amrex::Real, numParticles> weights{1};
        GEMPIC_TestUtils::addSingleParticles<vDim, numSpec, numParticles>(particleGroup, infra, weights, positions);

        // (default) charge correctly transferred from addSingleParticles
        EXPECT_EQ(1, particleGroup[0]->getCharge()); 

        const amrex::Array<std::string, 3> analyticalFuncB = {"1.0", 
                                                              "1.0",
                                                              "1.0"};

        const amrex::Array<std::string, 3> analyticalFuncJ = {"1.0", 
                                                              "1.0",
                                                              "1.0"};

        const int nVar{4}; //x, y, z, t
        amrex::Array<amrex::ParserExecutor<nVar>, GEMPIC_SPACEDIM> funcB;
        amrex::Array<amrex::ParserExecutor<nVar>, GEMPIC_SPACEDIM> funcJ; 
        amrex::Array<amrex::Parser, 6> parser;

        for (int i{0}; i<3; ++i)
        {
            parser[i].define(analyticalFuncB[i]);
            parser[i].registerVariables({"x", "y", "z", "t"});
            funcB[i] = parser[i].compile<4>();
        }

        for (int i{0}; i<3; ++i)
        {
            parser[i+3].define(analyticalFuncJ[i]);
            parser[i+3].registerVariables({"x", "y", "z", "t"});
            funcJ[i] = parser[i+3].compile<4>();
        }

        // Initialize the De Rham Complex
        auto deRham{std::make_shared<FDDeRhamComplex>(params)};

        DeRhamField<Grid::primal, Space::face> B(deRham, funcB);

        DeRhamField<Grid::dual, Space::face> J(deRham, funcJ);

        particleGroup[0]->Redistribute();  // assign particles to the tile they are in
        // Particle iteration ... over one particle. Hopefully.

        for (amrex::ParIter<0, 0, vDim + 1, 0> pti(*particleGroup[spec], 0); pti.isValid(); ++pti)
        {
            const long np{pti.numParticles()};
            EXPECT_EQ(1, np); // Only one particle added by addSingleParticles

            amrex::GpuArray<amrex::Real, 2> bfields{0., 0.};

            accumulateJUpdateVC2ParallelFor<vDim, degX, degY, degZ, pDim>(pti, B, J, infra, weight, infra.dx, bfields);

            EXPECT_NEAR(bfields[0], -4.75, 1e-15);
            EXPECT_NEAR(bfields[1], -4.75, 1e-15);

            checkRho(__LINE__, (J.data[pDim]).array(pti), infra.n_cell.dim3(), 
                    // Expect the eight nearest nodes (4/5, 4/5, 4/5) to be non-zero 
                    {[] (AMREX_D_DECL(int a, int b, int c)) {return AMREX_D_TERM(a == 4,
                                                                              && b == 4,
                                                                              && c == 4);},
                    [] (AMREX_D_DECL(int a, int b, int c)) {return AMREX_D_TERM(a == 4,
                                                                             && b <= 3,
                                                                             && c == 4);},
                    [] (AMREX_D_DECL(int a, int b, int c)) {return AMREX_D_TERM(a == 5,
                                                                             && b == 4,
                                                                             && c == 5);},
                    [] (AMREX_D_DECL(int a, int b, int c)) {return AMREX_D_TERM(a == 5,
                                                                             && b <= 3,
                                                                             && c == 5);},
                    [] (AMREX_D_DECL(int a, int b, int c)) {return AMREX_D_TERM(
                                                            (a == 4 || a == 5),
                                                         &&  b == 4,
                                                         && (c == 4 || c == 5) && c != a);},
                    [] (AMREX_D_DECL(int a, int b, int c)) {return AMREX_D_TERM(
                                                            (a == 4 || a == 5),
                                                         &&  b <= 3,
                                                         && (c == 4 || c == 5) && c != a);}},
                    // getting an eight of the particle weight times the primitive, plus the original 1
                    {1 - 3./64, 1 - 1./16, 1 - 27./64, 1 - 9./16, 1 - 9./64, 1 - 3./16},
                    //{},{},
                    // with the remaining entries being 1
                    1);
            checkRho(__LINE__, (J.data[(pDim + 1) % 3]).array(pti), infra.n_cell.dim3(), {}, {}, 1);
            checkRho(__LINE__, (J.data[(pDim + 2) % 3]).array(pti), infra.n_cell.dim3(), {}, {}, 1);
        }
    }

    TEST_F(AccumulateJUpdateVC2Test, DoubleParticleSeparate) {
        const int numParticles{2};
        // Particles in different cells to check that they don't interfere with each other
        amrex::Array<amrex::GpuArray<amrex::Real, GEMPIC_SPACEDIM>, numParticles> positions{{{
            AMREX_D_DECL(0, 0, 0)},
            {AMREX_D_DECL(
            infra.geom.ProbLo(0) + 5.5*infra.dx[0],
            infra.geom.ProbLo(1) + 5.5*infra.dx[1],
            infra.geom.ProbLo(2) + 5.5*infra.dx[2])}}};
        amrex::Array<amrex::Real, numParticles> weights{1, 1};
        GEMPIC_TestUtils::addSingleParticles<vDim, numSpec, numParticles>(particleGroup, infra, weights, positions);

        // (default) charge correctly transferred from addSingleParticles
        EXPECT_EQ(1, particleGroup[0]->getCharge()); 

        const amrex::Array<std::string, 3> analyticalFuncB = {"1.0", 
                                                              "1.0",
                                                              "1.0"};

        const amrex::Array<std::string, 3> analyticalFuncJ = {"1.0", 
                                                              "1.0",
                                                              "1.0"};

        const int nVar{4}; //x, y, z, t
        amrex::Array<amrex::ParserExecutor<nVar>, GEMPIC_SPACEDIM> funcB;
        amrex::Array<amrex::ParserExecutor<nVar>, GEMPIC_SPACEDIM> funcJ; 
        amrex::Array<amrex::Parser, 6> parser;

        for (int i{0}; i<3; ++i)
        {
            parser[i].define(analyticalFuncB[i]);
            parser[i].registerVariables({"x", "y", "z", "t"});
            funcB[i] = parser[i].compile<4>();
        }

        for (int i{0}; i<3; ++i)
        {
            parser[i+3].define(analyticalFuncJ[i]);
            parser[i+3].registerVariables({"x", "y", "z", "t"});
            funcJ[i] = parser[i+3].compile<4>();
        }

        // Initialize the De Rham Complex
        auto deRham{std::make_shared<FDDeRhamComplex>(params)};

        DeRhamField<Grid::primal, Space::face> B(deRham, funcB);

        DeRhamField<Grid::dual, Space::face> J(deRham, funcJ);

        particleGroup[0]->Redistribute();  // assign particles to the tile they are in
        // Particle iteration ... over one particle. Hopefully.

        for (amrex::ParIter<0, 0, vDim + 1, 0> pti(*particleGroup[spec], 0); pti.isValid(); ++pti)
        {
            const long np{pti.numParticles()};
            EXPECT_EQ(2, np); // Only one particle added by addSingleParticles

            amrex::GpuArray<amrex::Real, 2> bfields{0., 0.};

            accumulateJUpdateVC2ParallelFor<vDim, degX, degY, degZ, pDim>(pti, B, J, infra, weight, infra.dx, bfields);

            EXPECT_EQ(bfields[0], 0);
            EXPECT_EQ(bfields[1], 0);

            // Expect all nodes to be 1
            checkRho(__LINE__, (J.data[pDim]).array(pti), infra.n_cell.dim3(), {}, {}, 1);
            checkRho(__LINE__, (J.data[(pDim + 1) % 3]).array(pti), infra.n_cell.dim3(), {}, {}, 1);
            checkRho(__LINE__, (J.data[(pDim + 2) % 3]).array(pti), infra.n_cell.dim3(), {}, {}, 1);
        }
    }

    TEST_F(AccumulateJUpdateVC2Test, DoubleParticleOverlap) {
        const int numParticles{2};
        // Particles in different cells to check that they don't interfere with each other
        amrex::Array<amrex::GpuArray<amrex::Real, GEMPIC_SPACEDIM>, numParticles> positions{{{
            AMREX_D_DECL(0, 0, 0)},
            {AMREX_D_DECL(
            infra.geom.ProbLo(0) + 0.5*infra.dx[0],
            infra.geom.ProbLo(1) + 0.5*infra.dx[1],
            infra.geom.ProbLo(2) + 0.5*infra.dx[2])}}};
        amrex::Array<amrex::Real, numParticles> weights{1, 1};
        GEMPIC_TestUtils::addSingleParticles<vDim, numSpec, numParticles>(particleGroup, infra, weights, positions);

        // (default) charge correctly transferred from addSingleParticles
        EXPECT_EQ(1, particleGroup[0]->getCharge()); 

        const amrex::Array<std::string, 3> analyticalFuncB = {"1.0", 
                                                              "1.0",
                                                              "1.0"};

        const amrex::Array<std::string, 3> analyticalFuncJ = {"1.0", 
                                                              "1.0",
                                                              "1.0"};

        const int nVar{4}; //x, y, z, t
        amrex::Array<amrex::ParserExecutor<nVar>, GEMPIC_SPACEDIM> funcB;
        amrex::Array<amrex::ParserExecutor<nVar>, GEMPIC_SPACEDIM> funcJ; 
        amrex::Array<amrex::Parser, 6> parser;

        for (int i{0}; i<3; ++i)
        {
            parser[i].define(analyticalFuncB[i]);
            parser[i].registerVariables({"x", "y", "z", "t"});
            funcB[i] = parser[i].compile<4>();
        }

        for (int i{0}; i<3; ++i)
        {
            parser[i+3].define(analyticalFuncJ[i]);
            parser[i+3].registerVariables({"x", "y", "z", "t"});
            funcJ[i] = parser[i+3].compile<4>();
        }

        // Initialize the De Rham Complex
        auto deRham{std::make_shared<FDDeRhamComplex>(params)};

        DeRhamField<Grid::primal, Space::face> B(deRham, funcB);

        DeRhamField<Grid::dual, Space::face> J(deRham, funcJ);

        particleGroup[0]->Redistribute();  // assign particles to the tile they are in
        // Particle iteration ... over one particle. Hopefully.

        for (amrex::ParIter<0, 0, vDim + 1, 0> pti(*particleGroup[spec], 0); pti.isValid(); ++pti)
        {
            const long np{pti.numParticles()};
            EXPECT_EQ(2, np); // Only one particle added by addSingleParticles

            amrex::GpuArray<amrex::Real, 2> bfields{0., 0.};

            accumulateJUpdateVC2ParallelFor<vDim, degX, degY, degZ, pDim>(pti, B, J, infra, weight, infra.dx, bfields);

            EXPECT_EQ(bfields[0], 0);
            EXPECT_EQ(bfields[1], 0);

            // Expect all nodes to be 1
            checkRho(__LINE__, (J.data[pDim]).array(pti), infra.n_cell.dim3(), {}, {}, 1);
            checkRho(__LINE__, (J.data[(pDim + 1) % 3]).array(pti), infra.n_cell.dim3(), {}, {}, 1);
            checkRho(__LINE__, (J.data[(pDim + 2) % 3]).array(pti), infra.n_cell.dim3(), {}, {}, 1);
        }
    }

}