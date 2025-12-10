#include <iostream>
#include <random>

#include <AMReX.H>
#include <AMReX_ParmParse.H>
#include <AMReX_Particles.H>
#include <AMReX_PlotFileUtil.H>
#include <AMReX_Print.H>

#include "GEMPIC_AmrexInit.H"
#include "GEMPIC_BilinearFilter.H"
#include "GEMPIC_ComputationalDomain.H"
#include "GEMPIC_Config.H"
#include "GEMPIC_Diagnostics.H"
#include "GEMPIC_FDDeRhamComplex.H"
#include "GEMPIC_Fields.H"
#include "GEMPIC_Parameters.H"
#include "GEMPIC_Particle.H"
#include "GEMPIC_ParticleMeshCoupling.H"
#include "GEMPIC_PoissonSolver.H"
#include "GEMPIC_QuasineutralSolver.H"
#include "GEMPIC_Sampler.H"
#include "GEMPIC_SplineClass.H"
#include "GEMPIC_Splitting.H"

using namespace Gempic;
using namespace Forms;
using namespace Particle;
using namespace ParticleMeshCoupling;
using namespace FieldSolvers;

int main (int argc, char* argv[])
{
    amrex::Initialize(argc, argv);

    constexpr int vdim{3};
    constexpr int ndata{1};

    // Node spline degrees (smoothing spline degree is one less in each direction)
    // A spline degree of atleast 2 is required for accurate Del dot S
    constexpr int degx{4};
    constexpr int degy{4};
    constexpr int degz{4};
    constexpr int maxSplineDegree{std::max(std::max(degx, degy), degz)};

    constexpr int hodgeDegree{2};

    {
        BL_PROFILE("QuasineutralVlasovMaxwellMain()");
        Io::Parameters parameters{};
        // Initialize computational_domain
        ComputationalDomain infra;

        // Initialize the De Rham Complex
        auto deRham = std::make_shared<FDDeRhamComplex>(infra, hodgeDegree, maxSplineDegree,
                                                        HodgeScheme::FDHodge);

        DeRhamField<Grid::primal, Space::edge> E(deRham, "E");
        DeRhamField<Grid::dual, Space::face> D(deRham, "D");
        amrex::Real c1 = -1.0, c2 = 1.0;

        DeRhamField<Grid::primal, Space::face> B(deRham, "B");
        DeRhamField<Grid::primal, Space::face> bOld(deRham, "B_old");
        DeRhamField<Grid::primal, Space::cell> divB(deRham);
        DeRhamField<Grid::dual, Space::edge> H(deRham, "H");
        DeRhamField<Grid::dual, Space::face> curlH(deRham);
        DeRhamField<Grid::dual, Space::face> jMinusCurlH(deRham);

        DeRhamField<Grid::dual, Space::face> A(deRham);
        DeRhamField<Grid::dual, Space::cell> divA(deRham);
        DeRhamField<Grid::primal, Space::edge> aPrimal(deRham);

        DeRhamField<Grid::dual, Space::face> jcrossB(
            deRham); // This is really sum(species) ((q/m) multiplied by J x B)
        DeRhamField<Grid::dual, Space::face> delDotS(
            deRham); // This is really sum(species) ((q/m) multiplied by Del.S)
        DeRhamField<Grid::dual, Space::face> rhs(deRham);

        DeRhamField<Grid::dual, Space::cell> rho(deRham, "rho");

        Io::Parameters iCparams("ICs");

        amrex::Vector<amrex::Real> bmean;
        iCparams.get("Bmean", bmean);
        amrex::GpuArray<amrex::Real, 3> dx{GEMPIC_D_PAD_ONE(infra.geometry().CellSize(xDir),
                                                            infra.geometry().CellSize(yDir),
                                                            infra.geometry().CellSize(zDir))};
        bmean[xDir] *= dx[yDir] * dx[zDir];
        bmean[yDir] *= dx[xDir] * dx[zDir];
        bmean[zDir] *= dx[xDir] * dx[yDir];

        auto [parserIdealRho, funcIdealRho] = Utils::parse_function("idealRho");
        DeRhamField<Grid::dual, Space::cell> rhoIdeal(deRham, funcIdealRho, "rho_ideal");

        DeRhamField<Grid::dual, Space::face> J(deRham);
        DeRhamField<Grid::dual, Space::face> jOld(deRham);
        DeRhamField<Grid::dual, Space::cell> divJ(deRham);
        DeRhamField<Grid::primal, Space::node> phiCorr(deRham);

        std::vector<std::shared_ptr<ParticleSpecies<vdim, ndata>>> particles;
        init_particles(particles, infra);

        // Initializing filter
        std::unique_ptr<Filter::Filter> biFilter = std::make_unique<Filter::BilinearFilter>();

        QuasineutralSolver<hodgeDegree, vdim, ndata, degx, degy, degz> hypreQNLinearSystem(infra,
                                                                                           deRham);

        { //"Time Loop" scope
            Io::Parameters params("TimeLoop");
            amrex::Real dt;
            params.get("dt", dt);
            int nSteps;
            params.get("nSteps", nSteps);

            auto diagnostics = Io::make_diagnostics<degx, degy, degz>(infra, deRham, particles);

            amrex::Real simTime{0.0};
            diagnostics.compute_and_write_to_file(0, simTime);

            amrex::Print() << "\nSTARTING TIME LOOP:\n";

            for (int tStep = 0; tStep < nSteps; tStep++)
            {
                // Correction of divJ, Javg
                deposit_current_density<degx, degy, degz, vdim, ndata>(J, particles, infra);
                copy(jOld, J);
                hypreQNLinearSystem.push_particles_and_correct_div_j(J, phiCorr, particles, dt);
                check_j_change_norms(jOld, J, deRham, infra);
                copy(jOld, J);
                hypreQNLinearSystem.push_particles_and_correct_javg(J, rho, particles);
                check_j_change_norms(jOld, J, deRham, infra);
                amrex::Print() << "\n";

                amrex::Print() << "Obtain B(" << tStep << ") -> ";
                // Calculate B and add Bmean
                hypreQNLinearSystem.solve_negative_poisson_equation(J, A);
                hodge(aPrimal, A, deRham->scaling_dto_e());
                copy(bOld, B);
                curl(B, aPrimal);
                B += bmean;
                if (tStep != 0) check_b_calculation(E, bOld, B, deRham, dt, infra);
                hodge(H, B, deRham->scaling_bto_h());

                // B-related error checks
                check_b_related_norms(J, divJ, A, divA, B, divB, H, curlH, jMinusCurlH, deRham,
                                      infra);
                check_energies(B, H, particles, deRham, infra);

                // Deposit JxB, Del.S using X, V
                amrex::Print() << "Deposit JxB,Del.S(" << tStep << ") -> ";
                deposit_rho_j_jcrossb_deldots<degx, degy, degz, vdim, ndata>(
                    rho, J, B, jcrossB, delDotS, particles, infra);

                // Obtain E(0), using B(0) and V(0), X(0)
                amrex::Print() << "Obtain E(" << tStep << ") from QN eqn -> ";
                linear_combination(rhs, c1, jcrossB, c2, delDotS);

                hypreQNLinearSystem.solve_curlcurl_plus_particle_charge_e(rhs, E, particles);

                hodge(D, E, deRham->scaling_eto_d());

                // E-related error checks
                check_e_related_norms<vdim, ndata, degx, degy, degz>(E, B, H, J, rho, rhoIdeal,
                                                                     deRham, rhs, particles, infra);

                amrex::Print() << "Obtain V(" << tStep << " + 1/2) and X(" << tStep + 1 << ") -> ";
                // Update V using E and B and X using V
                hypreQNLinearSystem.update_particle_positions_and_velocities(
                    E, B, particles, dt, RotatePushOrder::PushvThenRotatev);

                amrex::Print() << "Obtain V(" << tStep + 1 << ") -> ";
                // Update V using E and B
                hypreQNLinearSystem.update_particle_positions_and_velocities(
                    E, B, particles, dt, RotatePushOrder::RotatevThenPushv);

                amrex::Print() << "Done." << std::endl;

                simTime = dt * (tStep + 1);
                diagnostics.compute_and_write_to_file(tStep + 1, simTime);

                amrex::Print() << "finished time-step: " << tStep << std::endl;
            }

            amrex::Print() << "\nFINISHED TIME LOOP.\n";
        }
    }
    amrex::Finalize();
}