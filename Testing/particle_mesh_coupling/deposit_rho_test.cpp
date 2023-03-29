/** Testing for deposit_rho function
 *  \todo: Move helper functions out. Consider mocking particles.
*/

#include <AMReX.H>
#include <GEMPIC_amrex_init.H>
#include <GEMPIC_parameters.H>
#include <GEMPIC_computational_domain.H>
#include <GEMPIC_gempic_norm.H>
#include <GEMPIC_particle_mesh_coupling_C2.H>
#include <gtest/gtest.h>
#include <algorithm> // std::all_of, for add_particles function
#include <stdexcept>

/** Global setup/teardown test enviroment configuration for unit tests
 * (stolen from AMR_wind)
 *
 *  This class is registered with GoogleTest infrastructure to perform global
 *  setup/teardown tasks. The base implementation calls the amrex::Initialize
 *  and amrex::Finalize calls.
 *
 *  During the AmrexTestEnv::SetUp call, it also finalizes the amrex::ParmParse
 *  global instance so that each test can utilize a clean "input file". The user
 *  can disable this feature by passing `utest.keep_parameters=1` at the command
 *  line.
 *
 */
class AmrexTestEnv : public ::testing::Environment
{
public:
    AmrexTestEnv(int& argc, char**& argv) : m_argc(argc), m_argv(argv) {}

    ~AmrexTestEnv() override = default;

    void SetUp() override
    {
        amrex::Initialize(m_argc, m_argv, true, MPI_COMM_WORLD, []() {
            amrex::ParmParse pp("amrex");
            if (!(pp.contains("v") || pp.contains("verbose"))) {
                pp.add("verbose", -1);
                pp.add("v", -1);
            }

            pp.add("throw_exception", 1);
            pp.add("signal_handling", 0);
        });

        // Save managed memory flag for future use
        {
            amrex::ParmParse pp("amrex");
            pp.query("the_arena_is_managed", m_has_managed_memory);
        }

        // Call ParmParse::Finalize immediately to allow unit tests to start
        // with a clean "input file". However, allow user to override this
        // behavior through command line arguments.
        {
            amrex::ParmParse pp("utest");
            bool keep_parameters = false;
            pp.query("keep_parameters", keep_parameters);

            if (!keep_parameters) {
                amrex::ParmParse::Finalize();
            }
        }
    }

    void TearDown() override { amrex::Finalize(); }

    bool has_managed_memory() const { return m_has_managed_memory; }

protected:
    int& m_argc;
    char**& m_argv;

    bool m_has_managed_memory{true};
};

//Basics first
namespace {
    // Nghost helper function
    const int init_Nghost(int degx, int degy, int degz)
    {
        amrex::Array<int, GEMPIC_SPACEDIM> degs = {AMREX_D_DECL(degx, degy, degz)};
        const int maxdeg = *(std::max_element(degs.begin(), degs.end()));
        return maxdeg;
    }

    // Stream arr helper function
    template<typename T>
    std::string string_array(const T& valArray, int length) {
        std::stringstream stream;
        stream << "(" << valArray[0];
        for (int i = 1; i < length; i++) {
            stream << ", " << valArray[i];
        }
        stream << ")";
        return stream.str();
    }

    // Add-particle helper function. User must supply coordinates (but not necessarily velocities)
    // of all particles to be added.
    template<int vdim, int numspec, int numparticles>
    void add_single_particles(
        amrex::GpuArray<std::unique_ptr<particle_groups<vdim>>, numspec>& part_gr,
        computational_domain& infra,
        amrex::Array<amrex::Real, numparticles>& weights,
        amrex::Array<amrex::Array<amrex::Real, GEMPIC_SPACEDIM>, numparticles>& positions,
        int spec = 0,
        amrex::Array<amrex::Array<amrex::Real, GEMPIC_SPACEDIM>, numparticles>&& velocities = {0})
        {
            amrex::Array<bool, numparticles> checklist = {false};

            for (amrex::MFIter mfi = part_gr[0]->MakeMFIter(0); mfi.isValid(); ++mfi)
            {
                const amrex::Box& bx = mfi.validbox();
                amrex::IntVect lo = {bx.smallEnd()};
                amrex::IntVect hi = {bx.bigEnd()};

                amrex::ParticleTile<0, 0, vdim + 1, 0>& particles =
                    part_gr[spec]->GetParticles(0)[std::make_pair(mfi.index(), mfi.LocalTileIndex())];
                for (int i = 0; i < numparticles; i++) {
                    // make sure particle is within box (and not previously added? Why?)
                    if (!checklist[i] GEMPIC_D_COND(
                        && lo[0]*infra.dx[0] <= positions[i][0] && 
                                                positions[i][0] < (hi[0] + 1)*infra.dx[0],
                        && lo[1]*infra.dx[1] <= positions[i][1] &&
                                                positions[i][1] < (hi[1] + 1)*infra.dx[1],
                        && lo[2]*infra.dx[2] <= positions[i][2] &&
                                                positions[i][2] < (hi[2] + 1)*infra.dx[2]))
                    {
                        part_gr[spec]->add_particle({AMREX_D_DECL(positions[i][0],
                                                                  positions[i][1],
                                                                  positions[i][2])},
                                                    {AMREX_D_DECL(velocities[i][0],
                                                                  velocities[i][1],
                                                                  velocities[i][2])},
                                                    weights[i], particles);
                        checklist[i] = true;
                    }
                }
            }

            // Check that all particles have been added
            int idx{-1};
            auto result1 = std::find_if(std::begin(checklist), std::end(checklist), [&idx] (bool i) {idx++; return !i;});
            if (result1 != std::end(checklist)) {
                std::cerr << "Invalid position given to add_particle function:\n" << 
                            string_array(positions[idx], GEMPIC_SPACEDIM) << 
                            " is out of simulation bounds.\nLower: " << 
                            string_array(infra.geom.ProbLo(), GEMPIC_SPACEDIM) << 
                            "\nUpper: " << string_array(infra.geom.ProbHi(), GEMPIC_SPACEDIM);
            }
        }

    // Test fixture. Sets up clean environment before each test.
    class DepositRhoTest : public testing::Test {
        protected:

        // Degree of splines in each direction
        static const int degx{1};
        static const int degy{1};
        static const int degz{1};

        // Number of species (second species only used for DoubleParticleMultipleSpecies)
        static const int numspec = 2;
        // Number of velocity dimensions. Really ought to be 0, but then add_single_particles doesn't work.
        static const int vdim = 3;
        // Number of ghost cells in mesh
        const int Nghost = init_Nghost(degx, degy, degz);

        amrex::Array<amrex::Real, numspec> charge{1, -1};
        amrex::Array<amrex::Real, numspec> mass{1, 0.1};

        computational_domain infra;
        amrex::GpuArray<std::unique_ptr<particle_groups<vdim>>, numspec> part_gr;
        amrex::MultiFab rho; 

        // virtual void SetUp() will be called before each test is run.
        void SetUp() override {
            /* Initialize the infrastructure */
            const amrex::RealBox realBox({AMREX_D_DECL(0.0, 0.0, 0.0)},
                                            {AMREX_D_DECL(10.0, 10.0, 10.0)});
            const amrex::IntVect nCell{AMREX_D_DECL(10, 10, 10)};
            const amrex::IntVect maxGridSize{AMREX_D_DECL(10, 10, 10)};
            const amrex::IntVect isPeriodic{1, 1, 1};

            infra.initialize_computational_domain(nCell, maxGridSize, isPeriodic, realBox);

            // Setup rho. This is  the special part of this text fixture.
            // node centered BA:
            const amrex::BoxArray &nba = amrex::convert(infra.grid, amrex::IntVect::TheNodeVector());
            int Ncomp = 1;            

            rho.define(nba, infra.distriMap, Ncomp, Nghost);
            rho.setVal(0.0);
            // Ensure rho exists and is 0 everywhere
            ASSERT_EQ(0,Gempic::Utils::gempic_norm(rho, infra, 2));
            
            // particle groups
            for (int spec = 0; spec < numspec; spec++)
            {
                part_gr[spec] =
                    std::make_unique<particle_groups<vdim>>(charge[spec], mass[spec], infra);
            }
        }
    };

    /** Single particle tests. The only reason most of these maneuvres are necessary is because of
    /*  amrex::Array4<amrex::Real> const& rhoarr = rho[pti].array();
    /*  which is required for the connection between MultiFab rho and deposit_rho function. This in
    /*  turn requires the pti iterator, which means actual particles must be added, instead of
    /*  simply supplying positions directly.
    */

    // Adds a particle with 0 weight. Checks that rho is unchanged.
    TEST_F(DepositRhoTest, NullTest) {
        // Adding particle to one cell
        const int numparticles{1};
        amrex::Array<amrex::Array<amrex::Real, GEMPIC_SPACEDIM>, numparticles> positions{*infra.geom.ProbLo()};
        
        amrex::Array<amrex::Real, numparticles> weights{1};
        add_single_particles<vdim, numspec, numparticles>(part_gr, infra, weights, positions);

        // (default) charge correctly transferred from add_single_particles
        EXPECT_EQ(1, part_gr[0]->getCharge()); 
        
        // rho unchanged by add_single_particles
        EXPECT_EQ(0,Gempic::Utils::gempic_norm(rho, infra, 2));

        part_gr[0]->Redistribute();  // assign particles to the tile they are in
        // Particle iteration ... over one particle.
        bool particle_loop_run=false;
        for (amrex::ParIter<0, 0, vdim + 1, 0> pti(*part_gr[0], 0); pti.isValid(); ++pti)
        {
            particle_loop_run=true;

            const long np = pti.numParticles();

            EXPECT_EQ(numparticles, np); // Only one particle added by add_single_particles

            const auto& particles = pti.GetArrayOfStructs();
            const auto partData = particles().data();
            const auto weight = pti.GetStructOfArrays().GetRealData(vdim).data();

            EXPECT_EQ(1, weight[0]); // weight correctly transferred from add_single_particles

            amrex::Array4<amrex::Real> const& rhoarr = rho[pti].array();
            splines_at_particles<degx, degy, degz> spline;
            amrex::GpuArray<amrex::Real, GEMPIC_SPACEDIM> position;
            for (unsigned int d = 0; d < GEMPIC_SPACEDIM; ++d)
                position[d] = partData[0].pos(d);
            spline.init_particles(position, infra.plo, infra.dxi);
            // Needs at least max(degx, degy, degz) ghost cells
            gempic_deposit_rho_C3<degx, degy, degz>(spline, 0, rhoarr);
        }
        ASSERT_TRUE(particle_loop_run);
        
        EXPECT_EQ(0,Gempic::Utils::gempic_norm(rho, infra, 2));
    }

    // Adds one particle exactly on a node
    TEST_F(DepositRhoTest, SingleParticleOnNode) {
        // Adding particle to one cell
        const int numparticles{1};

        // Particle at position (0,0,0) in box (0,0,0)
        amrex::Array<amrex::Array<amrex::Real, GEMPIC_SPACEDIM>, numparticles> positions{*infra.geom.ProbLo()};
        amrex::Array<amrex::Real, numparticles> weights{1};
        add_single_particles<vdim, numspec, numparticles>(part_gr, infra, weights, positions);

        part_gr[0]->Redistribute();  // assign particles to the tile they are in
        // Particle iteration ... over one particle.
        for (amrex::ParIter<0, 0, vdim + 1, 0> pti(*part_gr[0], 0); pti.isValid(); ++pti)
        {
            const long np = pti.numParticles();
            EXPECT_EQ(numparticles, np); // Only one particle added

            const auto& particles = pti.GetArrayOfStructs();
            const auto partData = particles().data();
            const auto weight = pti.GetStructOfArrays().GetRealData(vdim).data();
            amrex::Array4<amrex::Real> const& rhoarr = rho[pti].array();

            amrex::ParallelFor(np, [=] AMREX_GPU_DEVICE(long pp)
            {
                splines_at_particles<degx, degy, degz> spline;
                amrex::GpuArray<amrex::Real, GEMPIC_SPACEDIM> position;
                for (unsigned int d = 0; d < GEMPIC_SPACEDIM; ++d)
                    position[d] = partData[pp].pos(d);
                spline.init_particles(position, infra.plo, infra.dxi);
                // Needs at least max(degx, degy, degz) ghost cells
                gempic_deposit_rho_C3<degx, degy, degz>(
                    spline, weight[pp],
                    rhoarr);
            });

            // Expect only one node of rhoarr (0, 0, 0) to be non-zero and receiving full weight of particle (1)
            amrex::Dim3 top = infra.n_cell.dim3();
            for (int i{0}; i <= top.x; i++) { 
                for (int j{0}; j <= top.y; j++) {
                    for (int k{0}; k <= top.z; k++) {
                        amrex::Array<int, GEMPIC_SPACEDIM> idx{AMREX_D_DECL(i, j, k)};
                        if (i == 0 && j == 0 && k == 0)
                            EXPECT_EQ(1, *rhoarr.ptr(AMREX_D_DECL(i, j, k), 0)) <<
                                "Indices: " << string_array(idx, GEMPIC_SPACEDIM);
                        else
                            EXPECT_EQ(0, *rhoarr.ptr(AMREX_D_DECL(i, j, k), 0)) <<
                                "Indices: " << string_array(idx, GEMPIC_SPACEDIM);
                    }
                }
            }
        }
        rho.SumBoundary(0, 1, {Nghost, Nghost, Nghost}, {0, 0, 0}, infra.geom.periodicity());
        rho.FillBoundary(infra.geom.periodicity());

        // Maximum occurs on node (0,0,0) and contains all of the 1 charge 1 particle of weight 1
        EXPECT_EQ(1, rho.norm0());
        EXPECT_EQ(1, rho.norm1(0, infra.geom.periodicity()));
    }

    // Adds one particle exactly between two nodes
    TEST_F(DepositRhoTest, SingleParticleMiddle) {
        const int numparticles{1};

        // Add particle in the middle of final cell to check periodic boundary conditions
        amrex::Array<amrex::Array<amrex::Real, GEMPIC_SPACEDIM>, numparticles> positions{AMREX_D_DECL(infra.geom.ProbHi()[0] - 0.5*infra.dx[0],
                      infra.geom.ProbHi()[1] - 0.5*infra.dx[1],
                      infra.geom.ProbHi()[2] - 0.5*infra.dx[2])};
        amrex::Array<amrex::Real, numparticles> weights{3};
        // Expect the 2^GEMPIC_SPACEDIM nearest nodes of rhoarr (9/10, 9/10, 9/10) to be non-zero and receiving 1/2^GEMPIC_SPACEDIM the weight of the particle (3)
        const auto charge = part_gr[0]->getCharge();
        amrex::Real expectedVal{charge * infra.dxi[GEMPIC_SPACEDIM] * weights[0] * pow(0.5, GEMPIC_SPACEDIM)};

        add_single_particles<vdim, numspec, numparticles>(part_gr, infra, weights, positions);
        part_gr[0]->Redistribute();  // assign particles to the tile they are in
        // Particle iteration ... over one particle.
        for (amrex::ParIter<0, 0, vdim + 1, 0> pti(*part_gr[0], 0); pti.isValid(); ++pti)
        {
            const long np = pti.numParticles();
            EXPECT_EQ(numparticles, np); // Only one particle added

            const auto& particles = pti.GetArrayOfStructs();
            const auto partData = particles().data();
            const auto weight = pti.GetStructOfArrays().GetRealData(vdim).data();
            amrex::Array4<amrex::Real> const& rhoarr = rho[pti].array();

            amrex::ParallelFor(np, [=] AMREX_GPU_DEVICE(long pp)
            {
                splines_at_particles<degx, degy, degz> spline;
                amrex::GpuArray<amrex::Real, GEMPIC_SPACEDIM> position;
                for (unsigned int d = 0; d < GEMPIC_SPACEDIM; ++d)
                    position[d] = partData[pp].pos(d);
                spline.init_particles(position, infra.plo, infra.dxi);
                // Needs at least max(degx, degy, degz) ghost cells
                gempic_deposit_rho_C3<degx, degy, degz>(
                    spline, charge * infra.dxi[GEMPIC_SPACEDIM] * weight[pp],
                    rhoarr);
            });

            // Expect the eight nearest nodes of rhoarr (9/10, 9/10, 9/10) to be non-zero and receiving 1/8 the weight of the particle (3)
            amrex::Dim3 top = infra.n_cell.dim3();
            for (int i{0}; i <= top.x; i++) { 
                for (int j{0}; j <= top.y; j++) {
                    for (int k{0}; k <= top.z; k++) {
                        amrex::Array<int, GEMPIC_SPACEDIM> idx{AMREX_D_DECL(i, j, k)};
                        if (GEMPIC_D_COND((i >= top.x - 1),
                                       && (j >= top.y - 1),
                                       && (k >= top.z - 1)))
                            EXPECT_EQ(expectedVal, *rhoarr.ptr(AMREX_D_DECL(i, j, k), 0)) <<
                                "Indices: " << string_array(idx, GEMPIC_SPACEDIM);

                        else
                            EXPECT_EQ(0, *rhoarr.ptr(AMREX_D_DECL(i, j, k), 0)) <<
                                "Indices: " << string_array(idx, GEMPIC_SPACEDIM);
                    }
                }
            }
        }
        rho.SumBoundary(0, 1, {Nghost, Nghost, Nghost}, {0, 0, 0}, infra.geom.periodicity());
        rho.FillBoundary(infra.geom.periodicity());
        
        // Maximum occurs evenly split between 2^GEMPIC_SPACEDIM nodes. The sum is still 1.
        EXPECT_EQ(expectedVal, rho.norm0());
        EXPECT_EQ(weights[0], rho.norm1(0, infra.geom.periodicity()));
    }

    // Adds one particle closer to on node than the other
    TEST_F(DepositRhoTest, SingleParticleUnevenNodeSplit) { 
        const int numparticles{1};

        amrex::Array<amrex::Array<amrex::Real, GEMPIC_SPACEDIM>, numparticles> positions{AMREX_D_DECL(infra.geom.ProbLo(0) + 0.25*infra.dx[0],
                      infra.geom.ProbLo(1) + 0.25*infra.dx[1],
                      infra.geom.ProbLo(2) + 0.25*infra.dx[2])};
        amrex::Array<amrex::Real, numparticles> weights{1};

        add_single_particles<vdim, numspec, numparticles>(part_gr, infra, weights, positions);
        part_gr[0]->Redistribute();  // assign particles to the tile they are in
        // Particle iteration ... over one particle.
        for (amrex::ParIter<0, 0, vdim + 1, 0> pti(*part_gr[0], 0); pti.isValid(); ++pti)
        {
            const long np = pti.numParticles();
            EXPECT_EQ(numparticles, np); // Only one particle added

            const auto& particles = pti.GetArrayOfStructs();
            const auto partData = particles().data();
            const auto weight = pti.GetStructOfArrays().GetRealData(vdim).data();
            const auto charge = part_gr[0]->getCharge();
            amrex::Array4<amrex::Real> const& rhoarr = rho[pti].array();

            amrex::ParallelFor(np, [=] AMREX_GPU_DEVICE(long pp)
            {
                splines_at_particles<degx, degy, degz> spline;
                amrex::GpuArray<amrex::Real, GEMPIC_SPACEDIM> position;
                for (unsigned int d = 0; d < GEMPIC_SPACEDIM; ++d)
                    position[d] = partData[pp].pos(d);
                spline.init_particles(position, infra.plo, infra.dxi);
                // Needs at least max(degx, degy, degz) ghost cells
                gempic_deposit_rho_C3<degx, degy, degz>(
                    spline, charge * infra.dxi[GEMPIC_SPACEDIM] * weight[pp],
                    rhoarr);
            });

            // Expect the 2^GEMPIC_SPACEDIM nearest nodes of rhoarr (0/1, 0/1, 0/1) to be non-zero and  0 nodes receiving (3/4) and 1 nodes receiving (1/4) the weight of the particle (1)
            amrex::Dim3 top = infra.n_cell.dim3();
            for (int i{0}; i <= top.x; i++) { 
                for (int j{0}; j <= top.y; j++) {
                    for (int k{0}; k <= top.z; k++) {
                        amrex::Array<int, GEMPIC_SPACEDIM> idx{AMREX_D_DECL(i, j, k)};
                        if (GEMPIC_D_COND((i <= 1),
                                       && (j <= 1),
                                       && (k <= 1))) {
                            amrex::Real expectedVal{GEMPIC_D_COND(std::abs(i-0.75),
                                                                 *std::abs(j-0.75),
                                                                 *std::abs(k-0.75))};
                            EXPECT_EQ(expectedVal, *rhoarr.ptr(AMREX_D_DECL(i, j, k), 0)) <<
                                "Indices: " << string_array(idx, GEMPIC_SPACEDIM);
                        }
                        else
                            EXPECT_EQ(0, *rhoarr.ptr(AMREX_D_DECL(i, j, k), 0)) <<
                                "Indices: " << string_array(idx, GEMPIC_SPACEDIM);
                    }
                }
            }
        }
        rho.SumBoundary(0, 1, {Nghost, Nghost, Nghost}, {0, 0, 0}, infra.geom.periodicity());
        rho.FillBoundary(infra.geom.periodicity());
        
        // Maximum occurs on node (0, 0, 0) with value (3/4)^GEMPIC_SPACEDIM. The sum is still 1.
        EXPECT_EQ(pow(0.75,GEMPIC_SPACEDIM), rho.norm0());
        EXPECT_EQ(1, rho.norm1(0, infra.geom.periodicity()));
    }

    // Adds two particles in different cells to check that they don't interfere with each other
    TEST_F(DepositRhoTest, DoubleParticleSeparate) {
        const int numparticles{2};
        amrex::Array<amrex::Array<amrex::Real, GEMPIC_SPACEDIM>, numparticles> positions{{{
            AMREX_D_DECL(0, 0, 0)},
            {AMREX_D_DECL(
            infra.geom.ProbLo(0) + 5.5*infra.dx[0],
            infra.geom.ProbLo(1) + 5.5*infra.dx[1],
            infra.geom.ProbLo(2) + 5.5*infra.dx[2])}}};
        
        amrex::Array<amrex::Real, numparticles> weights{1, 3};
        amrex::Real expectedValA{1}, expectedValB{3*pow(0.5, GEMPIC_SPACEDIM)};

        add_single_particles<vdim, numspec, numparticles>(part_gr, infra, weights, positions);
        
        part_gr[0]->Redistribute();  // assign particles to the tile they are in
        // Particle iteration ... over two distant particles.
        for (amrex::ParIter<0, 0, vdim + 1, 0> pti(*part_gr[0], 0); pti.isValid(); ++pti)
        {
            const long np = pti.numParticles();
            EXPECT_EQ(numparticles, np); // Two particles added

            const auto& particles = pti.GetArrayOfStructs();
            const auto partData = particles().data();
            const auto weight = pti.GetStructOfArrays().GetRealData(vdim).data();
            const auto charge = part_gr[0]->getCharge();
            amrex::Array4<amrex::Real> const& rhoarr = rho[pti].array();

            amrex::ParallelFor(np, [=] AMREX_GPU_DEVICE(long pp)
            {
                splines_at_particles<degx, degy, degz> spline;
                amrex::GpuArray<amrex::Real, GEMPIC_SPACEDIM> position;
                for (unsigned int d = 0; d < GEMPIC_SPACEDIM; ++d)
                    position[d] = partData[pp].pos(d);
                spline.init_particles(position, infra.plo, infra.dxi);
                // Needs at least max(degx, degy, degz) ghost cells
                gempic_deposit_rho_C3<degx, degy, degz>(
                    spline, charge * infra.dxi[GEMPIC_SPACEDIM] * weight[pp],
                    rhoarr);
            });

            // See SingleParticle test for explanation of expectations
            amrex::Dim3 top = infra.n_cell.dim3();
            for (int i{0}; i <= top.x; i++) { 
                for (int j{0}; j <= top.y; j++) {
                    for (int k{0}; k <= top.z; k++) {
                        amrex::Array<int, GEMPIC_SPACEDIM> idx{AMREX_D_DECL(i, j, k)};
                        if (GEMPIC_D_COND((i == 0),
                                       && (j == 0),
                                       && (k == 0)))
                            EXPECT_EQ(expectedValA, *rhoarr.ptr(AMREX_D_DECL(i, j, k), 0)) <<
                                "Indices: " << string_array(idx, GEMPIC_SPACEDIM);
                        else if (GEMPIC_D_COND((i == 5 || i == 6),
                                            && (j == 5 || j == 6),
                                            && (k == 5 || k == 6)))
                            EXPECT_EQ(expectedValB, *rhoarr.ptr(AMREX_D_DECL(i, j, k), 0)) <<
                                "Indices: " << string_array(idx, GEMPIC_SPACEDIM);
                        else
                            EXPECT_EQ(0, *rhoarr.ptr(AMREX_D_DECL(i, j, k), 0)) <<
                                "Indices: " << string_array(idx, GEMPIC_SPACEDIM);
                    }
                }
            }
        }
        rho.SumBoundary(0, 1, {Nghost, Nghost, Nghost}, {0, 0, 0}, infra.geom.periodicity());
        rho.FillBoundary(infra.geom.periodicity());
        
        EXPECT_EQ(expectedValA, rho.norm0());
        // Total charge added is the sum of each weight*charge, here 1 + 3
        EXPECT_EQ(4, rho.norm1(0, infra.geom.periodicity()));
    }

    // Adds particles in the same cell to check that they add up correctly
    TEST_F(DepositRhoTest, DoubleParticleOverlap) {
        const int numparticles{2};
        amrex::Array<amrex::Array<amrex::Real, GEMPIC_SPACEDIM>, numparticles> positions{{{
            AMREX_D_DECL(0, 0, 0)},
            {AMREX_D_DECL(
            infra.geom.ProbLo(0) + 0.5*infra.dx[0],
            infra.geom.ProbLo(1) + 0.5*infra.dx[1],
            infra.geom.ProbLo(2) + 0.5*infra.dx[2])}}};
        amrex::Array<amrex::Real, numparticles> weights{1, 3};
        
        amrex::Real expectedValA{1 + 3*pow(0.5, GEMPIC_SPACEDIM)};
        amrex::Real expectedValB{3*pow(0.5, GEMPIC_SPACEDIM)};

        add_single_particles<vdim, numspec, numparticles>(part_gr, infra, weights, positions);
        
        part_gr[0]->Redistribute();  // assign particles to the tile they are in
        // Particle iteration ... over two close particles.
        for (amrex::ParIter<0, 0, vdim + 1, 0> pti(*part_gr[0], 0); pti.isValid(); ++pti)
        {
            const long np = pti.numParticles();
            EXPECT_EQ(numparticles, np); // Two particles added

            const auto& particles = pti.GetArrayOfStructs();
            const auto partData = particles().data();
            const auto weight = pti.GetStructOfArrays().GetRealData(vdim).data();
            const auto charge = part_gr[0]->getCharge();
            amrex::Array4<amrex::Real> const& rhoarr = rho[pti].array();

            amrex::ParallelFor(np, [=] AMREX_GPU_DEVICE(long pp)
            {
                splines_at_particles<degx, degy, degz> spline;
                amrex::GpuArray<amrex::Real, GEMPIC_SPACEDIM> position;
                for (unsigned int d = 0; d < GEMPIC_SPACEDIM; ++d)
                    position[d] = partData[pp].pos(d);
                spline.init_particles(position, infra.plo, infra.dxi);
                // Needs at least max(degx, degy, degz) ghost cells
                gempic_deposit_rho_C3<degx, degy, degz>(
                    spline, charge * infra.dxi[GEMPIC_SPACEDIM] * weight[pp],
                    rhoarr);
            });

            // See SingleParticle test for explanation of expectations
            amrex::Dim3 top = infra.n_cell.dim3();
            for (int i{0}; i <= top.x; i++) { 
                for (int j{0}; j <= top.y; j++) {
                    for (int k{0}; k <= top.z; k++) {
                        amrex::Array<int, GEMPIC_SPACEDIM> idx{AMREX_D_DECL(i, j, k)};
                        if (GEMPIC_D_COND((i == 0),
                                       && (j == 0),
                                       && (k == 0)))
                            EXPECT_EQ(expectedValA, *rhoarr.ptr(AMREX_D_DECL(i, j, k), 0)) <<
                                "Indices: " << string_array(idx, GEMPIC_SPACEDIM);
                        else if (GEMPIC_D_COND((i <= 1),
                                            && (j <= 1),
                                            && (k <= 1)))
                            EXPECT_EQ(expectedValB, *rhoarr.ptr(AMREX_D_DECL(i, j, k), 0)) <<
                                "Indices: " << string_array(idx, GEMPIC_SPACEDIM);
                        else
                            EXPECT_EQ(0, *rhoarr.ptr(AMREX_D_DECL(i, j, k), 0)) <<
                                "Indices: " << string_array(idx, GEMPIC_SPACEDIM);
                    }
                }
            }
        }
        rho.SumBoundary(0, 1, {Nghost, Nghost, Nghost}, {0, 0, 0}, infra.geom.periodicity());
        rho.FillBoundary(infra.geom.periodicity());
        
        EXPECT_EQ(expectedValA, rho.norm0());
        EXPECT_EQ(4, rho.norm1(0, infra.geom.periodicity()));
    }

    // Adds particles of different species in the same cell
    TEST_F(DepositRhoTest, DoubleParticleMultipleSpecies) {
        const int numparticles{1};
        amrex::Array<amrex::Array<amrex::Real, GEMPIC_SPACEDIM>, numparticles> pPos{{
            AMREX_D_DECL(0, 0, 0)}};
        amrex::Array<amrex::Array<amrex::Real, GEMPIC_SPACEDIM>, numparticles> ePos{{
            AMREX_D_DECL(infra.geom.ProbLo(0) + 0.5*infra.dx[0],
                         infra.geom.ProbLo(1) + 0.5*infra.dx[1],
                         infra.geom.ProbLo(2) + 0.5*infra.dx[2])}};
        amrex::Array<amrex::Real, numparticles> pWeights{1};
        amrex::Array<amrex::Real, numparticles> eWeights{3};
        int pSpec{0}, eSpec{1};
        
        add_single_particles<vdim, numspec, numparticles>(part_gr, infra, pWeights, pPos, pSpec);
        add_single_particles<vdim, numspec, numparticles>(part_gr, infra, eWeights, ePos, eSpec);

        const auto pCharge{part_gr[pSpec]->getCharge()};
        const auto eCharge{part_gr[eSpec]->getCharge()};

        amrex::Real expectedValA{pCharge + eCharge*3*pow(0.5, GEMPIC_SPACEDIM)};
        amrex::Real expectedValB{eCharge*3*pow(0.5, GEMPIC_SPACEDIM)};
        
        for (int spec = 0; spec < numspec; spec++)
        {
            part_gr[spec]->Redistribute();  // assign particles to the tile they are in
            const auto charge{part_gr[spec]->getCharge()};
            // Particle iteration
            for (amrex::ParIter<0, 0, vdim + 1, 0> pti(*part_gr[spec], 0); pti.isValid(); ++pti)
            {
                const long np = pti.numParticles();
                EXPECT_EQ(numparticles, np); // Two particles added

                const auto& particles = pti.GetArrayOfStructs();
                const auto partData = particles().data();
                const auto weight = pti.GetStructOfArrays().GetRealData(vdim).data();
                amrex::Array4<amrex::Real> const& rhoarr = rho[pti].array();

                amrex::ParallelFor(np, [=] AMREX_GPU_DEVICE(long pp)
                {
                    splines_at_particles<degx, degy, degz> spline;
                    amrex::GpuArray<amrex::Real, GEMPIC_SPACEDIM> position;
                    for (unsigned int d = 0; d < GEMPIC_SPACEDIM; ++d)
                        position[d] = partData[pp].pos(d);
                    spline.init_particles(position, infra.plo, infra.dxi);
                    // Needs at least max(degx, degy, degz) ghost cells
                    gempic_deposit_rho_C3<degx, degy, degz>(
                        spline, charge * infra.dxi[GEMPIC_SPACEDIM] * weight[pp],
                        rhoarr);
                });

                if (spec == numspec - 1) {
                    // See SingleParticle test for explanation of expectations
                    amrex::Dim3 top = infra.n_cell.dim3();
                    for (int i{0}; i <= top.x; i++) { 
                        for (int j{0}; j <= top.y; j++) {
                            for (int k{0}; k <= top.z; k++) {
                                amrex::Array<int, GEMPIC_SPACEDIM> idx{AMREX_D_DECL(i, j, k)};
                                if (GEMPIC_D_COND((i == 0),
                                               && (j == 0),
                                               && (k == 0)))
                                    EXPECT_EQ(expectedValA, *rhoarr.ptr(AMREX_D_DECL(i, j, k), 0)) <<
                                        "Indices: " << string_array(idx, GEMPIC_SPACEDIM);
                                else if (GEMPIC_D_COND((i <= 1),
                                                    && (j <= 1),
                                                    && (k <= 1)))
                                    EXPECT_EQ(expectedValB, *rhoarr.ptr(AMREX_D_DECL(i, j, k), 0)) <<
                                        "Indices: " << string_array(idx, GEMPIC_SPACEDIM);
                                else
                                    EXPECT_EQ(0, *rhoarr.ptr(AMREX_D_DECL(i, j, k), 0)) <<
                                        "Indices: " << string_array(idx, GEMPIC_SPACEDIM);
                            }
                        }
                    }
                }
            }
        }
            rho.SumBoundary(0, 1, {Nghost, Nghost, Nghost}, {0, 0, 0}, infra.geom.periodicity());
            rho.FillBoundary(infra.geom.periodicity());
            
            EXPECT_EQ(expectedValA, rho.norm0());
            
            // Probably not GPU safe. Second argument of sum_unique is bool local, which decides if parallel reduction is done
            EXPECT_EQ(pCharge*pWeights[0] + eCharge*eWeights[0], rho.sum_unique(0, 0, infra.geom.periodicity()));
    }
}

int main(int argc, char* argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
  // gtest takes ownership of the TestEnvironment ptr - we don't delete it.
  auto utest_env = new AmrexTestEnv(argc, argv);
  ::testing::AddGlobalTestEnvironment(utest_env);
  return RUN_ALL_TESTS();
}