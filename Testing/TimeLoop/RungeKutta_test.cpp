#include <cmath>
#include <vector>

#include <gtest/gtest.h>

#include <AMReX.H>
#include <AMReX_SPACE.H>

#include "GEMPIC_Diagnostics.H"
#include "GEMPIC_Fields.H"
#include "GEMPIC_GempicNorm.H"
#include "GEMPIC_Interpolation.H"
#include "GEMPIC_RungeKutta.H"
#include "TestUtils/GEMPIC_TestUtils.H"

using namespace Gempic;
using namespace Forms;
using namespace Particle;
using namespace TimeLoop;
constexpr int hodgeDegree = 2;

ComputationalDomain get_compdom ()
{
    std::array<amrex::Real, AMREX_SPACEDIM> const domainLo{AMREX_D_DECL(0.0, 0.0, 0.0)};
    std::array<amrex::Real, AMREX_SPACEDIM> const domainHi{AMREX_D_DECL(10, 10, 10)};
    amrex::IntVect const nCell{AMREX_D_DECL(4, 4, 24)};
    amrex::IntVect const maxGridSize{AMREX_D_DECL(4, 4, 12)};
    std::array<int, AMREX_SPACEDIM> const isPeriodic{AMREX_D_DECL(1, 1, 1)};

    return ComputationalDomain(domainLo, domainHi, nCell, maxGridSize, isPeriodic);
}

class RungeKuttaTest : public testing::Test
{
protected:
    // Degree of splines in each direction
    static int const s_degX{1};
    static int const s_degY{1};
    static int const s_degZ{1};
    static constexpr unsigned int s_ndata = 10;
    // Number of species (second species only used for DoubleParticleMultipleSpecies)
    static int const s_numSpec{1};
    // Number of velocity dimensions.
    static int const s_vDim{3};
    // Number of ghost cells in mesh
    int const m_nghost{Gempic::Test::Utils::init_n_ghost(s_degX, s_degY, s_degZ)};
    amrex::IntVect const m_nghosts{AMREX_D_DECL(m_nghost, m_nghost, m_nghost)};
    amrex::IntVect const m_dstNGhosts{AMREX_D_DECL(0, 0, 0)};

    Io::Parameters m_params{};

    amrex::Array<amrex::Real, s_numSpec> m_charge{1};
    amrex::Array<amrex::Real, s_numSpec> m_mass{1};

    ComputationalDomain m_infra;
    std::vector<std::shared_ptr<ParticleGroups<s_vDim, s_ndata>>> m_particleGroup;
    std::shared_ptr<FDDeRhamComplex> m_deRham;
    amrex::Geometry const m_geom = m_infra.m_geom;

    RungeKuttaTest() : m_infra{get_compdom()}
    {
        // particle settings
        double charge{-1};
        double mass{1};

        m_params.set("Particle.species0.charge", charge);
        m_params.set("Particle.species0.mass", mass);

        // Full diagnostics
        m_params.set("FullDiagnostics.enable", true); // 1 for true, 0 for false
        amrex::Vector<std::string> diagsNames = {"field"};
        m_params.set("FullDiagnostics.groupNames", diagsNames);
        amrex::Vector<std::string> fieldNames = {"Ex", "Jx", "By"};
        m_params.set("FullDiagnostics.field.varNames", fieldNames);
        std::string cellCenterFunctor = "CellCenter";
        m_params.set("FullDiagnostics.field.outputProcessor", cellCenterFunctor);
        int fieldSave{1};
        m_params.set("FullDiagnostics.field.saveInterval", fieldSave);

        int const hodgeDegree{2};
        int const maxSplineDegree{std::max(std::max(s_degX, s_degY), s_degZ)};

        // Initialize the De Rham Complex
        m_deRham = std::make_shared<FDDeRhamComplex>(m_infra, hodgeDegree, maxSplineDegree);

        // particles
        m_particleGroup.resize(s_numSpec);
        for (int spec{0}; spec < s_numSpec; spec++)
        {
            m_particleGroup[spec] =
                std::make_unique<ParticleGroups<s_vDim, s_ndata>>(spec, m_infra);
        }
    }
};

namespace
{

#if AMREX_SPACEDIM == 3
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
#endif

TEST_F(RungeKuttaTest, TestConstantRHS)
{
    // Runge-Kutta solver
    // default stages=5
    RungeKutta rkSolver(m_infra, m_deRham);
    int const nStage = rkSolver.get_stages();
    amrex::Vector<amrex::Real> vect{1, 2, 3};
    amrex::Vector<amrex::Real> s1{0, 0, 0}, s2{0, 0, 0};
    amrex::Real dt{0.5};

    // 5-stage Runge-Kutta solver
    // S1 contains so
    for (int i = 0; i < nStage; i++)
    {
        for (int j = 0; j < 3; j++)
        {
            s2[j] = rkSolver.get_coeff_a(i) * s2[j] + dt * vect[j];
            s1[j] += rkSolver.get_coeff_b(i) * s2[j];
        }
    }
    EXPECT_NEAR(dt, s1[0], 1e-12);
    EXPECT_NEAR(2 * dt, s1[1], 1e-12);
    EXPECT_NEAR(3 * dt, s1[2], 1e-12);
}
TEST_F(RungeKuttaTest, OneKineticParticle)
{
#if AMREX_SPACEDIM != 3
    GTEST_SKIP();
#else
    //define initial conditions
    amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> R{1, 0, 0};    //position vector
    amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> V{0, -1, 0.1}; //velocity vector
    amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> E{0, 0, 0}, B{0, 0, -1};
    amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> solutionPosition{0, 0, 0};
    amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> solutionVelocity{0, 0, 0};
    amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> s2R{0, 0, 0};
    amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> s2V{0, 0, 0};

    double qElec = -1;
    double qom = -1;
    double tElec = 2 * M_PI / abs(qElec) * 1;
    double h = 0.1;
    double tend = 10 * tElec;
    int nSteps = tend / h;

    std::vector<std::vector<double>> positionsElec(nSteps, std::vector<double>(3, 0.0));

    std::vector<std::vector<double>> velocityElec(nSteps, std::vector<double>(3, 0.0));

    std::vector<std::vector<double>> analysisPosition(nSteps, std::vector<double>(3, 0.0));

    std::vector<std::vector<double>> analysisVelocity(nSteps, std::vector<double>(3, 0.0));

    for (int j = 0; j < 3; j++)
    {
        positionsElec[0][j] = R[j];
        velocityElec[0][j] = V[j];
    }

    for (int i = 0; i < nSteps; i++)
    {
        analysisPosition[i][0] = cos(i * h);
        analysisPosition[i][1] = -sin(i * h);
        analysisPosition[i][2] = i * h * V[2];

        analysisVelocity[i][0] = -sin(i * h);
        analysisVelocity[i][1] = -cos(i * h);
        analysisVelocity[i][2] = V[2];
    }
    // constexpr int stages = 5;
    RungeKutta rkSolver(m_infra, m_deRham);
    int const stages = rkSolver.get_stages();

    for (int i = 0; i < nSteps - 1; i++)
    {
        for (int j = 0; j < 3; j++)
        {
            solutionPosition[j] = positionsElec[i][j];
            solutionVelocity[j] = velocityElec[i][j];
        }
        for (int stageIndex = 0; stageIndex < stages; ++stageIndex)
        {
            amrex::Real const coeffsAStageIndex = rkSolver.get_coeff_a(stageIndex);
            amrex::Real const coeffsBStageIndex = rkSolver.get_coeff_b(stageIndex);
            Gempic::TimeLoop::Impl::lsrk_stage_kinetic_particle(
                solutionPosition, solutionVelocity, s2R, s2V, E, B, qom, h, coeffsAStageIndex,
                coeffsBStageIndex);
        }

        for (int k = 0; k < 3; k++)
        {
            positionsElec[i + 1][k] = solutionPosition[k];
            velocityElec[i + 1][k] = solutionVelocity[k];
        }
    }

    EXPECT_NEAR(positionsElec[nSteps - 1][0], analysisPosition[nSteps - 1][0], 1e-3);
    EXPECT_NEAR(positionsElec[nSteps - 1][1], analysisPosition[nSteps - 1][1], 1e-3);
    EXPECT_NEAR(positionsElec[nSteps - 1][2], analysisPosition[nSteps - 1][2], 1e-3);
    EXPECT_NEAR(velocityElec[nSteps - 1][0], analysisVelocity[nSteps - 1][0], 1e-3);
    EXPECT_NEAR(velocityElec[nSteps - 1][1], analysisVelocity[nSteps - 1][1], 1e-3);
    EXPECT_NEAR(velocityElec[nSteps - 1][2], analysisVelocity[nSteps - 1][2], 1e-3);
#endif
}
TEST_F(RungeKuttaTest, KineticParticleGroup)
{
#if AMREX_SPACEDIM != 3
    GTEST_SKIP();
#else
    // const int stages = 5;
    RungeKutta rkSolver(m_infra, m_deRham);
    int const stages = rkSolver.get_stages();

    // Initialize the fields
    DeRhamField<Grid::primal, Space::face> B(m_deRham);
    DeRhamField<Grid::primal, Space::edge> E(m_deRham);
    DeRhamField<Grid::dual, Space::face> J(m_deRham);

    constexpr int numParticles{1};
    amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> R{AMREX_D_DECL(6.0, 5.0, 0)};
    amrex::GpuArray<amrex::Real, 3> V{{0, -1, 0.1}};
    amrex::Real weight{1};
    amrex::Array<amrex::GpuArray<amrex::Real, AMREX_SPACEDIM>, numParticles> positions{R};
    amrex::Array<amrex::GpuArray<amrex::Real, 3>, numParticles> velocities{V};
    amrex::Array<amrex::Real, numParticles> weights = {weight};

    amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> rEnd;

    double qElec = -1;
    double tElec = 2 * M_PI / abs(qElec) * 1;
    double h = 0.1;
    double tend = 1.0 * tElec;
    int nSteps = 2.0 * tend / h;

    std::vector<std::vector<double>> analysisPosition(nSteps, std::vector<double>(3, 0.0));
    std::vector<std::vector<double>> analysisVelocity(nSteps, std::vector<double>(3, 0.0));
    std::vector<std::vector<double>> positionsElec(nSteps, std::vector<double>(3, 0.0));
    std::vector<std::vector<double>> velocityElec(nSteps, std::vector<double>(3, 0.0));

    for (int i = 0; i < nSteps; i++)
    {
        analysisPosition[i][0] = 5.0 + cos(i * h);
        analysisPosition[i][1] = 5.0 - sin(i * h);
        analysisPosition[i][2] = 0.1 * i * h;

        analysisVelocity[i][0] = -sin(i * h);
        analysisVelocity[i][1] = -cos(i * h);
        analysisVelocity[i][2] = 0.1;
    }

    B.m_data[xDir].setVal(0.0);
    B.m_data[yDir].setVal(0.0);
    // Bz=-1, since the multifab B is defined as a flux
    B.m_data[zDir].setVal(-1.0 * m_infra.geometry().CellSize(xDir) *
                          m_infra.geometry().CellSize(yDir));
    E.m_data[xDir].setVal(0.0);
    E.m_data[yDir].setVal(0.0);
    E.m_data[zDir].setVal(0.0);

    // Adding particle to one cell
    Gempic::Test::Utils::add_single_particles(m_particleGroup[0].get(), m_infra, weights, positions,
                                              velocities);

    for (int i = 0; i < nSteps - 1; i++)
    {
        for (int stageIndex = 0; stageIndex < stages; stageIndex++)
        {
            amrex::Real const coeffsAStageIndex = rkSolver.get_coeff_a(stageIndex);
            amrex::Real const coeffsBStageIndex = rkSolver.get_coeff_b(stageIndex);
            rkSolver.template lsrk_stage_push_particle_deposit_j<s_degX, s_degY, s_degZ>(
                m_particleGroup, E, B, J, h, coeffsAStageIndex, coeffsBStageIndex);
        }
    } //end of time loop

    for (auto& particleGrid : *m_particleGroup[0])
    {
        auto const ptd = particleGrid.GetParticleTile().getParticleTileData();
        auto const ii = m_particleGroup[0]->get_data_indices();

        AMREX_D_EXPR(rEnd[xDir] = ptd.rdata(ii.m_iposx)[0], rEnd[yDir] = ptd.rdata(ii.m_iposy)[0],
                     rEnd[zDir] = ptd.rdata(ii.m_iposz)[0]);
    }
    EXPECT_NEAR(rEnd[0], analysisPosition[nSteps - 1][0], 1e-3);
    EXPECT_NEAR(rEnd[1], analysisPosition[nSteps - 1][1], 1e-3);
    EXPECT_NEAR(rEnd[2], analysisPosition[nSteps - 1][2], 1e-3);
#endif
}

TEST_F(RungeKuttaTest, DriftKineticParticleGroup)
{
#if AMREX_SPACEDIM != 3
    GTEST_SKIP();
#else
    // const int stages = 5;
    RungeKutta rkSolver(m_infra, m_deRham);
    int const stages = rkSolver.get_stages();

    // Initialize the fields
    DeRhamField<Grid::primal, Space::face> B(m_deRham);
    DeRhamField<Grid::primal, Space::edge> E(m_deRham);
    DeRhamField<Grid::dual, Space::face> J(m_deRham);

    constexpr int numParticles{1};
    amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> r0{AMREX_D_DECL(0.0, 0.0, 0.0)};
    amrex::GpuArray<amrex::Real, 3> v0{{1.0, 1.0, 0.1}};
    amrex::Real weight{1};
    amrex::Array<amrex::GpuArray<amrex::Real, AMREX_SPACEDIM>, numParticles> positions{r0};
    amrex::Array<amrex::GpuArray<amrex::Real, 3>, numParticles> velocities{v0};
    amrex::Array<amrex::Real, numParticles> weights = {weight};
    amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> rEnd;
    amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> vEnd;

    double qElec = -1;
    double tElec = 2 * M_PI / abs(qElec) * 1;
    double h = 0.1;
    double tend = 1.0 * tElec;
    int nSteps = 2.0 * tend / h;

    std::vector<std::vector<double>> analysisPosition(nSteps, std::vector<double>(3, 0.0));
    std::vector<std::vector<double>> analysisVelocity(nSteps, std::vector<double>(3, 0.0));
    std::vector<std::vector<double>> positionsElec(nSteps, std::vector<double>(3, 0.0));
    std::vector<std::vector<double>> velocityElec(nSteps, std::vector<double>(3, 0.0));

    for (int i = 0; i < nSteps; i++)
    {
        analysisPosition[i][0] = fmod(r0[0] + (i * h), 10.0);
        analysisPosition[i][1] = r0[1];
        analysisPosition[i][2] = r0[2] + v0[2] * i * h;

        analysisVelocity[i][0] = v0[0];
        analysisVelocity[i][1] = v0[1];
        analysisVelocity[i][2] = v0[2];
    }
    // Bz=1, since the multifab B is defined as a flux
    B.m_data[xDir].setVal(0.0);
    B.m_data[yDir].setVal(0.0);
    B.m_data[zDir].setVal(1.0 * m_infra.geometry().CellSize(xDir) *
                          m_infra.geometry().CellSize(yDir));
    // Ey=1, multifab E is defined as an integral
    E.m_data[xDir].setVal(0.0);
    E.m_data[yDir].setVal(1.0 * m_infra.geometry().CellSize(yDir));
    E.m_data[zDir].setVal(0.0);

    // Adding particle to one cell
    Gempic::Test::Utils::add_single_particles(m_particleGroup[0].get(), m_infra, weights, positions,
                                              velocities);

    for (int i = 0; i < nSteps - 1; i++)
    {
        for (int stageIndex = 0; stageIndex < stages; stageIndex++)
        {
            amrex::Real const coeffsAStageIndex = rkSolver.get_coeff_a(stageIndex);
            amrex::Real const coeffsBStageIndex = rkSolver.get_coeff_b(stageIndex);
            rkSolver.template lsrk_stage_dk_push_particle_deposit_j<s_degX, s_degY, s_degZ>(
                m_particleGroup, E, B, J, h, coeffsAStageIndex, coeffsBStageIndex);
        }

        for (auto& particleGrid : *m_particleGroup[0])
        {
            auto const ptd = particleGrid.GetParticleTile().getParticleTileData();
            auto const ii = m_particleGroup[0]->get_data_indices();

            vEnd = {ptd.rdata(ii.m_ivelx)[0], ptd.rdata(ii.m_ively)[0], ptd.rdata(ii.m_ivelz)[0]};
            AMREX_D_EXPR(rEnd[xDir] = ptd.rdata(ii.m_iposx)[0],
                         rEnd[yDir] = ptd.rdata(ii.m_iposy)[0],
                         rEnd[zDir] = ptd.rdata(ii.m_iposz)[0]);
        }

    } //end of time loop
    EXPECT_NEAR(rEnd[0], analysisPosition[nSteps - 1][0], 1e-3);
    EXPECT_NEAR(rEnd[1], analysisPosition[nSteps - 1][1], 1e-3);
    EXPECT_NEAR(rEnd[2], analysisPosition[nSteps - 1][2], 1e-3);
    // Velocity does not change
    EXPECT_NEAR(vEnd[0], analysisVelocity[nSteps - 1][0], 1e-6);
    EXPECT_NEAR(vEnd[1], analysisVelocity[nSteps - 1][1], 1e-6);
    EXPECT_NEAR(vEnd[2], analysisVelocity[nSteps - 1][2], 1e-6);
#endif
}
TEST_F(RungeKuttaTest, test_LSRK_maxwell)
{
#if AMREX_SPACEDIM != 3
    GTEST_SKIP();
#else
    amrex::Array<std::string, 3> const analyticalE = {
        "cos(3.141592653589793*t-0.2*3.141592653589793*z)", //x component :cos(wt-kz)
        "0", "0"};
    int const nVar = AMREX_SPACEDIM + 1; // x, y, z, t
    amrex::Array<amrex::ParserExecutor<nVar>, 3> funcE;
    amrex::Array<amrex::Parser, 3> parserE;
    for (int i = 0; i < 3; ++i)
    {
        parserE[i].define(analyticalE[i]);
        parserE[i].registerVariables({AMREX_D_DECL("x", "y", "z"), "t"});
        funcE[i] = parserE[i].compile<nVar>();
    }

    amrex::Array<std::string, 3> const analyticalB = {
        "0",
        "cos(3.141592653589793*t-0.2*3.141592653589793*z)", //y component :cos(wt-kz)
        "0"};

    amrex::Array<amrex::ParserExecutor<nVar>, 3> funcB;
    amrex::Array<amrex::Parser, 3> parserB;
    for (int i = 0; i < 3; ++i)
    {
        parserB[i].define(analyticalB[i]);
        parserB[i].registerVariables({AMREX_D_DECL("x", "y", "z"), "t"});
        funcB[i] = parserB[i].compile<nVar>();
    }

    RungeKutta rkSolver(m_infra, m_deRham);
    DeRhamField<Grid::primal, Space::edge> E(m_deRham, funcE, "E");
    DeRhamField<Grid::primal, Space::edge> eSolution(m_deRham, "eSolution");
    DeRhamField<Grid::primal, Space::edge> eError(m_deRham, "eError");
    DeRhamField<Grid::dual, Space::face> D(m_deRham, funcE);
    DeRhamField<Grid::primal, Space::face> B(m_deRham, funcB, "B");
    DeRhamField<Grid::primal, Space::face> bSolution(m_deRham, funcB, "bSolution");
    DeRhamField<Grid::dual, Space::edge> H(m_deRham, funcB);
    DeRhamField<Grid::dual, Space::face> J(m_deRham, "J");

    amrex::Real dt = 0.1;
    double T = 10;       //T=2*pi/w w=0.2*pi
    double tend = 1 * T; // how many periods
    int nSteps = tend / dt;

    auto nGhost = m_deRham->get_n_ghost();

    m_deRham->projection(funcE, 0, eSolution, 6);
    m_deRham->projection(funcB, 0, bSolution, 6);

    for (int i = 0; i < nSteps; i++)
    {
        rkSolver.template lsrk_vlasov_maxwell<s_degX, s_degY, s_degZ>(m_particleGroup, E, D, B, H,
                                                                      J, dt);
        m_deRham->projection(funcE, dt * (i + 1), eSolution, 6);
        m_deRham->projection(funcB, dt * (i + 1), bSolution, 6);
        m_deRham->hodge(eError, D); //get eError from D
        eError -= eSolution;
    }

    amrex::Real errorMax = 0;
    auto dr{m_infra.cell_size_array()};
    for (int comp = 0; comp < 3; ++comp)
    {
        errorMax += max_error_midpoint<hodgeDegree>(m_geom, funcE[comp], E.m_data[comp], dr, 1,
                                                    false, comp, nSteps * dt);
    }

    eSolution -= E;
    ASSERT_NEAR(0, Utils::gempic_norm(eSolution.m_data[xDir], m_infra, 0), 10 * 1e-2);
    ASSERT_NEAR(0, Utils::gempic_norm(E.m_data[yDir], m_infra, 0), 1e-10);
    ASSERT_NEAR(0, Utils::gempic_norm(E.m_data[zDir], m_infra, 0), 1e-10);
#endif
}

TEST_F(RungeKuttaTest, test_LSRK_maxwell_hodgeDK)
{
#if AMREX_SPACEDIM != 3
    GTEST_SKIP();
#else
    // eigen frequency omega=k/sqrt(2)
    amrex::Array<std::string, 3> const analyticalE = {
        "cos(0.4442882938158367*t-0.2*3.141592653589793*z)", //x component, cos(wt-kz)
        "0", "0"};
    amrex::Array<std::string, 3> const analyticalD = {
        "2*cos(0.4442882938158367*t-0.2*3.141592653589793*z)", //x component, 2cos(wt-kz)
        "0", "0"};
    int const nVar = AMREX_SPACEDIM + 1; // x, y, z, t
    amrex::Array<amrex::ParserExecutor<nVar>, 3> funcE;
    amrex::Array<amrex::ParserExecutor<nVar>, 3> funcD;
    amrex::Array<amrex::Parser, 3> parserE;
    amrex::Array<amrex::Parser, 3> parserD;
    for (int i = 0; i < 3; ++i)
    {
        parserE[i].define(analyticalE[i]);
        parserE[i].registerVariables({AMREX_D_DECL("x", "y", "z"), "t"});
        funcE[i] = parserE[i].compile<nVar>();

        parserD[i].define(analyticalD[i]);
        parserD[i].registerVariables({AMREX_D_DECL("x", "y", "z"), "t"});
        funcD[i] = parserD[i].compile<nVar>();
    }

    amrex::Array<std::string, 3> const analyticalB = {
        "0",
        "1.4142135623730951*cos(0.4442882938158367*t-0.2*3.141592653589793*z)", //y component
                                                                                //sqrt(2)*cos(wt-kz)
        "0"};

    amrex::Array<amrex::ParserExecutor<nVar>, 3> funcB;
    amrex::Array<amrex::Parser, 3> parserB;
    for (int i = 0; i < 3; ++i)
    {
        parserB[i].define(analyticalB[i]);
        parserB[i].registerVariables({AMREX_D_DECL("x", "y", "z"), "t"});
        funcB[i] = parserB[i].compile<nVar>();
    }

    RungeKutta rkSolver(m_infra, m_deRham);
    DeRhamField<Grid::primal, Space::edge> E(m_deRham, funcE, "E");
    DeRhamField<Grid::primal, Space::edge> eSolution(m_deRham, "eSolution");
    DeRhamField<Grid::primal, Space::edge> eError(m_deRham, "eError");
    DeRhamField<Grid::dual, Space::face> D(m_deRham, funcD, "D");
    DeRhamField<Grid::primal, Space::face> B(m_deRham, funcB, "B");
    DeRhamField<Grid::primal, Space::face> bSolution(m_deRham, funcB, "bSolution");
    DeRhamField<Grid::primal, Space::face> bError(m_deRham, "bError");
    DeRhamField<Grid::dual, Space::edge> H(m_deRham, funcB, "H");
    DeRhamField<Grid::dual, Space::face> J(m_deRham, "J");

    // Compute inverse of dielectric tensor going from D to E
    DeRhamField<Grid::primal, Space::edge> tensor(m_deRham,
                                                  3); // tensor including polarization for DK
    amrex::GpuArray<amrex::Real, 3> val;

    // initialize tensor
    for (amrex::MFIter mfi(tensor.m_data[xDir], true); mfi.isValid(); ++mfi)
    {
        val = {0.5, 0, 0};
        initialize_tensor(mfi, tensor.m_data[xDir], val);
    }
    for (amrex::MFIter mfi(tensor.m_data[yDir], true); mfi.isValid(); ++mfi)
    {
        val = {0, 0.5, 0};
        initialize_tensor(mfi, tensor.m_data[yDir], val);
    }
    for (amrex::MFIter mfi(tensor.m_data[zDir], true); mfi.isValid(); ++mfi)
    {
        val = {0, 0, 1};
        initialize_tensor(mfi, tensor.m_data[zDir], val);
    }

    amrex::Real dt = 0.2;
    double T = 14;       //T=2*pi/w, w=0.2*pi/sqrt(2)=0.1414213562373095*pi
    double tend = 2 * T; // how many periods
    int nSteps = tend / dt;

    auto nGhost = m_deRham->get_n_ghost();

    m_deRham->projection(funcE, 0, eSolution, 6);
    m_deRham->projection(funcB, 0, bSolution, 6);

    for (int i = 0; i < nSteps; i++)
    {
        rkSolver.template lsrk_dk_vlasov_maxwell<s_degX, s_degY, s_degZ>(m_particleGroup, E, D, B,
                                                                         H, J, tensor, dt);
        m_deRham->projection(funcE, dt * (i + 1), eSolution, 6);
        m_deRham->projection(funcB, dt * (i + 1), bSolution, 6);
        m_deRham->hodge_dk(eError, D, tensor); // get eError from D
        eError -= eSolution;
    }

    amrex::Real errorMax = 0;
    auto dr{m_infra.cell_size_array()};
    for (int comp = 0; comp < 3; ++comp)
    {
        errorMax += max_error_midpoint<hodgeDegree>(m_geom, funcE[comp], E.m_data[comp], dr, 1,
                                                    false, comp, nSteps * dt);
    }
    eSolution -= E;

    ASSERT_NEAR(0, Utils::gempic_norm(eSolution.m_data[xDir], m_infra, 0), 10 * 1e-2);
    ASSERT_NEAR(0, Utils::gempic_norm(E.m_data[yDir], m_infra, 0), 1e-10);
    ASSERT_NEAR(0, Utils::gempic_norm(E.m_data[zDir], m_infra, 0), 1e-10);
#endif
}
} //namespace
