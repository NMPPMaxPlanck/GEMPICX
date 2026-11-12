/**************************************************************************************************
 * Copyright (c) 2021 GEMPICX                                                                     *
 * SPDX-License-Identifier: BSD-3-Clause                                                          *
 **************************************************************************************************/
#include <iostream>
#include <random>

#include <AMReX.H>
#include <AMReX_ParmParse.H>
#include <AMReX_Particles.H>
#include <AMReX_PlotFileUtil.H>
#include <AMReX_Print.H>

#include "GEMPIC_ComputationalDomain.H"
#include "GEMPIC_Config.H"
#include "GEMPIC_Diagnostics.H"
#include "GEMPIC_FDDeRhamComplex.H"
#include "GEMPIC_FieldRegistry.H"
#include "GEMPIC_Fields.H"
#include "GEMPIC_Filter.H"
#include "GEMPIC_Initialize.H"
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
    Gempic::initialize(argc, argv);

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
        Io::Parameters parameters{};
        // Initialize computational_domain
        ComputationalDomain infra;

        // Initialize the De Rham Complex
        auto deRham = std::make_shared<FDDeRhamComplex>(infra, hodgeDegree, maxSplineDegree,
                                                        HodgeScheme::FDHodge);
        Gempic::Io::FieldRegistry fieldRegistry;
        // Field declarations (labels removed from constructors)
        auto E = Gempic::Io::registered_form<DeRhamField<Forms::Grid::primal, Forms::Space::edge>>(
            fieldRegistry, "E", deRham);
        auto D = Gempic::Io::registered_form<DeRhamField<Forms::Grid::dual, Forms::Space::face>>(
            fieldRegistry, "D", deRham);

        auto B = Gempic::Io::registered_form<DeRhamField<Forms::Grid::primal, Forms::Space::face>>(
            fieldRegistry, "B", deRham);
        DeRhamField<Forms::Grid::primal, Forms::Space::face> bOld(deRham);
        DeRhamField<Forms::Grid::primal, Forms::Space::cell> divB(deRham);
        auto H = Gempic::Io::registered_form<DeRhamField<Forms::Grid::dual, Forms::Space::edge>>(
            fieldRegistry, "H", deRham);
        DeRhamField<Forms::Grid::dual, Forms::Space::face> curlH(deRham);
        DeRhamField<Forms::Grid::dual, Forms::Space::face> jMinusCurlH(deRham);

        DeRhamField<Forms::Grid::dual, Forms::Space::face> A(deRham);
        DeRhamField<Forms::Grid::dual, Forms::Space::cell> divA(deRham);
        DeRhamField<Forms::Grid::primal, Forms::Space::edge> aPrimal(deRham);

        DeRhamField<Forms::Grid::dual, Forms::Space::face> jcrossB(
            deRham); // This is really sum(species) ((q/m) J x B)
        DeRhamField<Forms::Grid::dual, Forms::Space::face> delDotS(
            deRham); // This is really sum(species) ((q/m) Del.S)
        DeRhamField<Forms::Grid::dual, Forms::Space::face> rhs(deRham);

        auto rho = Gempic::Io::registered_form<DeRhamField<Forms::Grid::dual, Forms::Space::cell>>(
            fieldRegistry, "rho", deRham);

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
        DeRhamField<Forms::Grid::dual, Forms::Space::cell> rhoIdeal(deRham, funcIdealRho);
        Gempic::Io::register_form(fieldRegistry, "rho_ideal", rhoIdeal);

        DeRhamField<Forms::Grid::dual, Forms::Space::face> J(deRham);
        DeRhamField<Forms::Grid::dual, Forms::Space::face> jOld(deRham);
        DeRhamField<Forms::Grid::dual, Forms::Space::cell> divJ(deRham);
        DeRhamField<Forms::Grid::primal, Forms::Space::node> phiCorr(deRham);

        std::vector<std::shared_ptr<ParticleSpecies<vdim, ndata>>> partGr;
        init_particles(partGr, infra);

        std::vector<DeRhamField<Forms::Grid::dual, Forms::Space::cell>> rhoVec;
        std::vector<DeRhamField<Forms::Grid::dual, Forms::Space::cell>> rhoMassVec;

        for (int i = 0; i < partGr.size(); i++)
        {
            rhoVec.emplace_back(deRham);
            rhoMassVec.emplace_back(deRham);
        }

        for (int i = 0; i < partGr.size(); i++)
        {
            std::string name = "rho_" + std::to_string(i);
            Gempic::Io::register_form(fieldRegistry, name, rhoVec[i]);
            name = "rhoMass_" + std::to_string(i);
            Gempic::Io::register_form(fieldRegistry, name, rhoMassVec[i]);
        } // Separate loop to avoid registry pointing to changed memory location due to vector
          // reallocation

        // Initializing filter
        std::unique_ptr<Filter::Filter> biFilter = Filter::make_filter(infra);

        QuasineutralSolver<hodgeDegree, vdim, ndata, degx, degy, degz> qn(infra);

        { //"Time Loop" scope
            Io::Parameters params("TimeLoop");
            amrex::Real dt;
            params.get("dt", dt);
            int nSteps;
            params.get("nSteps", nSteps);
            std::string chargeMatrixType{"particle"};
            params.get("chargeMatrixType", chargeMatrixType);
            bool printChecks{false};
            params.get_or_set("printChecks", printChecks);

            auto diagnostics =
                Io::make_diagnostics<degx, degy, degz>(infra, deRham, fieldRegistry, partGr);

            amrex::Real simTime{0.0};
            diagnostics.compute_and_write_to_file(0, simTime);

            amrex::Print() << "\nSTARTING TIME LOOP:\n";

            for (int tStep = 0; tStep < nSteps; tStep++)
            {
                // Correction of divJ, Javg
                qn.push_particles_and_correct_div_j(J, phiCorr, partGr, dt);
                if (printChecks) check_j_change_norms_qn(jOld, J, deRham, infra);
                copy(jOld, J);
                qn.push_particles_and_correct_javg(J, rhoVec, rho, rhoMassVec, partGr);
                if (printChecks) check_j_change_norms_qn(jOld, J, deRham, infra);
                amrex::Print() << "\n";

                amrex::Print() << "Obtain B(" << tStep << ") -> ";
                // Calculate B and add Bmean
                qn.solve_negative_poisson_equation(J, A);
                hodge(aPrimal, A, deRham->scaling_dto_e());
                copy(bOld, B);
                curl(B, aPrimal);
                B += bmean;
                if (tStep != 0)
                {
                    if (printChecks) check_b_calculation_qn(E, bOld, B, deRham, dt, infra);
                }
                hodge(H, B, deRham->scaling_bto_h());

                // B-related error checks
                if (printChecks)
                {
                    check_b_related_norms_qn(J, divJ, A, divA, B, divB, H, curlH, jMinusCurlH,
                                             infra);
                }
                if (printChecks) check_energies_qn(B, H, partGr, infra);

                // Deposit JxB, Del.S using X, V
                amrex::Print() << "Deposit JxB,Del.S(" << tStep << ") -> ";
                deposit_rho_j_jcrossb_deldots<degx, degy, degz, vdim, ndata>(
                    rhoVec, rho, J, B, jcrossB, delDotS, partGr, infra);

                // Obtain E(0), using B(0) and V(0), X(0)
                amrex::Print() << "Obtain E(" << tStep << ") from QN eqn -> ";
                linear_combination(rhs, -1.0, jcrossB, 1.0, delDotS);
                rhs.post_particle_loop_sync();

                if (chargeMatrixType == "particle")
                {
                    qn.solve_curlcurl_plus_particle_charge_e(rhs, E, partGr);
                }
                else if (chargeMatrixType == "field_qbym")
                {
                    qn.solve_curlcurl_plus_qbym_field_charge_e(rhs, E, partGr, rhoVec);
                }

                hodge(D, E, deRham->scaling_eto_d());

                // E-related error checks
                if (printChecks)
                {
                    check_e_related_norms_qn<vdim, ndata, degx, degy, degz>(
                        E, rho, rhoIdeal, deRham, rhs, partGr, infra);
                }

                amrex::Print() << "Obtain V(" << tStep << " + 1/2) and X(" << tStep + 1 << ") -> ";
                // Update V using E and B and X using V
                qn.update_particle_positions_and_velocities(E, B, partGr, dt,
                                                            RotatePushOrder::PushvThenRotatev);

                amrex::Print() << "Obtain V(" << tStep + 1 << ") -> ";
                // Update V using E and B
                qn.update_particle_positions_and_velocities(E, B, partGr, dt,
                                                            RotatePushOrder::RotatevThenPushv);

                amrex::Print() << "Done." << std::endl;
                simTime = dt * (tStep + 1);
                diagnostics.compute_and_write_to_file(tStep + 1, simTime);
            }

            amrex::Print() << "\nFINISHED TIME LOOP.\n";
        }
    }
    Gempic::finalize();
}
