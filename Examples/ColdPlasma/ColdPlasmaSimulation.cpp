#include <iostream>
#include <random>

#include <AMReX.H>
#include <AMReX_ParmParse.H>
#include <AMReX_Particles.H>
#include <AMReX_PlotFileUtil.H>
#include <AMReX_Print.H>
#include <AMReX_Random.H>

#include "GEMPIC_AmrexInit.H"
#include "GEMPIC_ComputationalDomain.H"
#include "GEMPIC_Config.H"
#include "GEMPIC_FDDeRhamComplex.H"
#include "GEMPIC_Fields.H"
#include "GEMPIC_MultiFullDiagnostics.H"
#include "GEMPIC_MultiReducedDiagnostics.H"
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

    // Spline degrees
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
        TimeLoop::OperatorHamilton<vdim, degx, degy, degz, hodgeDegree> operatorHamilton;

        // Initialize particle groups
        std::vector<std::shared_ptr<ParticleGroups<vdim, ndata>>> partGr;

        {
            // Initialize full diagnostics and write initial time step
            Io::Parameters params("TimeLoop");
            amrex::Real dt;
            params.get("dt", dt);
            int nSteps;
            params.get("nSteps", nSteps);
            Io::Parameters paramsSim("Sim");
            auto nGhost = deRham->get_n_ghost();
            Io::MultiDiagnostics<vdim, ndata> fullDiagn(dt);
            fullDiagn.init_data(infra, deRham->m_fieldsDiagnostics, deRham->m_fieldsScaling, partGr,
                                nGhost);

            // Initialize noisy electric field
            std::mt19937 gen(123);
            std::uniform_real_distribution<> unif(-1.0, 1.0);
            std::normal_distribution<> norm01(0, 1);

            int comp = 2;  //noise only in z-component
            for (amrex::MFIter mfi(E.m_data[comp], true); mfi.isValid(); ++mfi)
            {
                const amrex::Box& bx = mfi.tilebox();
                amrex::IntVect lo = {bx.smallEnd()};
                amrex::IntVect hi = {bx.bigEnd()};

                amrex::Array4<amrex::Real> const& form = (E.m_data[comp])[mfi].array();

                amrex::InitRandom(123 + lo[xDir] + lo[yDir] * 10 + lo[zDir] * 100);

                ParallelForRNG(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k,
                                                        amrex::RandomEngine const& gen) noexcept
                               { form(i, j, k) = 0.1 * (amrex::Random(gen) - 0.5); });
            }

            E.fill_boundary();
            E.average_sync();
            deRham->hodge(E, D);

            // Initialize reduced diagnostics and write initial time step
            Io::MultiReducedDiagnostics<vdim, degx, degy, degz, hodgeDegree, ndata> redDiagn(
                deRham);

            // Write initial time step
            redDiagn.compute_diags(infra, deRham->m_fieldsDiagnostics, partGr);
            redDiagn.write_to_file(0, dt);
            fullDiagn.filter_compute_pack_flush(0);

            for (int tStep = 0; tStep < nSteps; tStep++)
            {
                // Hb
                operatorHamilton.apply_h_b(D, deRham, B, H, 0.5 * dt);

                // He,field (also computes E from D, needed in He,particle)
                operatorHamilton.apply_h_e_field(B, deRham, E, D, 0.5 * dt);
                deRham->hodge(E, auxDualF2);  // Copy?? -> diagonal, same cost?
                auxDualF2 *= -0.5 * dt;
                auxDualF2 *= funcDensityField;
                jFieldT -= auxDualF2;

                // Hj
                apply_h_j(deRham, D, jFieldT, funcBBackground, dt);

                //He,field
                operatorHamilton.apply_h_e_field(B, deRham, E, D, 0.5 * dt);
                deRham->hodge(E, auxDualF2);
                auxDualF2 *= -0.5 * dt;
                auxDualF2 *= funcDensityField;
                jFieldT -= auxDualF2;

                //Hb
                operatorHamilton.apply_h_b(D, deRham, B, H, 0.5 * dt);

                // Compute and scale primal field for cold plasma energy
                deRham->hodge(jFieldT, jField);
                jField *= funcDensityFieldInv;

                //write outputs
                redDiagn.compute_diags(infra, deRham->m_fieldsDiagnostics, partGr);
                redDiagn.write_to_file(tStep + 1, dt);
                fullDiagn.filter_compute_pack_flush(tStep + 1);

                if (tStep % 10 == 0)
                {
                    std::cout << "Time Step: " << tStep + 1 << std::endl;
                }
            }
        }  // end of "time loop" scope
    }
    amrex::Finalize();
}
