#include <iostream>
#include <random>

#include <AMReX.H>
#include <AMReX_Particles.H>
#include <AMReX_PlotFileUtil.H>
#include <AMReX_Print.H>
#include <AMReX_Random.H>

#include "GEMPIC_AmrexInit.H"
#include "GEMPIC_ComputationalDomain.H"
#include "GEMPIC_Config.H"
#include "GEMPIC_Diagnostics.H"
#include "GEMPIC_FDDeRhamComplex.H"
#include "GEMPIC_Fields.H"
#include "GEMPIC_Parameters.H"
#include "GEMPIC_ParticleGroups.H"
#include "GEMPIC_PoissonSolver.H"
#include "GEMPIC_Sampler.H"
#include "GEMPIC_SplineClass.H"
#include "GEMPIC_Splitting.H"

using namespace Gempic;
using namespace Forms;
using namespace Particle;
using namespace ParticleMeshCoupling;
using namespace TimeLoop;
using namespace Utils;
using namespace std;

int main (int argc, char* argv[])
{
    amrex::Initialize(argc, argv);

    constexpr int vdim{3};
    constexpr int ndata{1};

    // Node spline degrees (smoothing spline degree is one less in each direction)
    constexpr int degx{1};
    constexpr int degy{1};
    constexpr int degz{1};
    constexpr int maxSplineDegree{0};
    //
    constexpr int hodgeDegree{2};

    {
        Io::Parameters parameters{};
        // Initialize computational_domain
        ComputationalDomain infra;

        // Initialize the De Rham Complex
        auto deRham = std::make_shared<FDDeRhamComplex>(infra, hodgeDegree, maxSplineDegree,
                                                        HodgeScheme::FDHodge);

        auto [parseBBackground, funcBBackground] =
            Utils::parse_functions<3>({"BBackgroundX", "BBackgroundY", "BBackgroundZ"});
        auto [parseDensityField, funcDensityField] = Utils::parse_function("DensityField");
        auto [parseDensityFieldInv, funcDensityFieldInv] = Utils::parse_function("DensityFieldInv");

        DeRhamField<Grid::primal, Space::edge> E(deRham, "E");
        DeRhamField<Grid::dual, Space::face> D(deRham, "D");
        DeRhamField<Grid::primal, Space::face> B(deRham, "B");
        DeRhamField<Grid::dual, Space::edge> H(deRham, "H");
        DeRhamField<Grid::primal, Space::edge> jField(deRham, "JField");
        DeRhamField<Grid::dual, Space::face> jFieldT(deRham, "JFieldT");
        // temporary fields
        DeRhamField<Grid::primal, Space::face> auxPrimalF2(deRham);
        DeRhamField<Grid::dual, Space::face> auxDualF2(deRham);

        // Initialize needed propagators
        TimeLoop::OperatorHamilton<vdim, degx, degy, degz> operatorHamilton;

        // Initialize particle groups
        std::vector<std::shared_ptr<ParticleGroups<vdim, ndata>>> partGr;

        {
            // Initialize diagnostics and write initial time step
            Io::Parameters params("TimeLoop");
            amrex::Real dt;
            params.get("dt", dt);
            int nSteps;
            params.get("nSteps", nSteps);
            auto diagnostics = Io::make_diagnostics<degx, degy, degz>(infra, deRham, partGr);

            // Initialize noisy electric field
            std::mt19937 gen(123);
            std::uniform_real_distribution<> unif(-1.0, 1.0);
            std::normal_distribution<> norm01(0, 1);

            int comp = 2; //noise only in z-component
            for (amrex::MFIter mfi(E.m_data[comp], true); mfi.isValid(); ++mfi)
            {
                const amrex::Box& bx = mfi.tilebox();
                amrex::IntVect lo = {bx.smallEnd()};
                amrex::IntVect hi = {bx.bigEnd()};

                amrex::Array4<amrex::Real> const& form = (E.m_data[comp])[mfi].array();

                amrex::InitRandom(123 + GEMPIC_D_ADD(lo[xDir], lo[yDir] * 10, lo[zDir] * 100));

                ParallelForRNG(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k,
                                                        amrex::RandomEngine const& gen) noexcept
                               { form(i, j, k) = 0.1 * (amrex::Random(gen) - 0.5); });
            }

            E.fill_boundary();
            deRham->hodge(D, E);

            // Write initial time step
            amrex::Real simTime{0.0};
            diagnostics.compute_and_write_to_file(0, simTime);

            for (int tStep = 0; tStep < nSteps; tStep++)
            {
                // Hb
                operatorHamilton.apply_h_b(D, deRham, B, H, 0.5 * dt);

                // He,field (also computes E from D, needed in He,particle)
                operatorHamilton.apply_h_e_field(B, E, deRham, D, 0.5 * dt);
                deRham->hodge(auxDualF2, E); // Copy?? -> diagonal, same cost?
                auxDualF2 *= -0.5 * dt;
                auxDualF2 *= funcDensityField;
                jFieldT -= auxDualF2;

                // Hj
                apply_h_j(jFieldT, D, deRham, funcBBackground, dt);

                //He,field
                operatorHamilton.apply_h_e_field(B, E, deRham, D, 0.5 * dt);
                deRham->hodge(auxDualF2, E);
                auxDualF2 *= -0.5 * dt;
                auxDualF2 *= funcDensityField;
                jFieldT -= auxDualF2;

                //Hb
                operatorHamilton.apply_h_b(D, deRham, B, H, 0.5 * dt);

                // Compute and scale primal field for cold plasma energy
                deRham->hodge(jField, jFieldT);
                jField *= funcDensityFieldInv;

                //write outputs
                simTime = dt * (tStep + 1);
                diagnostics.compute_and_write_to_file(tStep + 1, simTime);

                if (tStep % 10 == 0)
                {
                    std::cout << "Time Step: " << tStep + 1 << std::endl;
                }
            }
        } // end of "time loop" scope
    }
    amrex::Finalize();
}
