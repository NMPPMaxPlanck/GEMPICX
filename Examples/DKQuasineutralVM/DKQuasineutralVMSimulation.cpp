/**************************************************************************************************
 * Copyright (c) 2021 GEMPICX                                                                     *
 * SPDX-License-Identifier: BSD-3-Clause                                                          *
 **************************************************************************************************/
#include <iostream>
#include <random>

#include <Kokkos_Profiling_ScopedRegion.hpp>

#include <AMReX.H>
#include <AMReX_ParmParse.H>
#include <AMReX_Particles.H>
#include <AMReX_PlotFileUtil.H>
#include <AMReX_Print.H>

#include "GEMPICX.H"
#include "GEMPIC_AmrexInit.H"
#include "GEMPIC_ComputationalDomain.H"
#include "GEMPIC_Config.H"
#include "GEMPIC_DKQuasineutralSolver.H"
#include "GEMPIC_Diagnostics.H"
#include "GEMPIC_FDDeRhamComplex.H"
#include "GEMPIC_Fields.H"
#include "GEMPIC_Parameters.H"
#include "GEMPIC_Particle.H"
#include "GEMPIC_ParticleMeshCoupling.H"
#include "GEMPIC_ParticleUtils.H"
#include "GEMPIC_PoissonSolver.H"
#include "GEMPIC_Sampler.H"
#include "GEMPIC_SplineClass.H"
#include "GEMPIC_Splitting.H"

using namespace Gempicx;

int main (int argc, char* argv[])
{
    Gempic::initialize(argc, argv);

    constexpr int vdim{3};
    constexpr int ndata{9}; // default 7 : Weight + 6 auxiliary variables // 9 for DeltaF

    // Node spline degrees (smoothing spline degree is one less in each direction)
    // A spline degree of atleast 2 is required for accurate Del dot S
    constexpr int degx{4};
    constexpr int degy{4};
    constexpr int degz{4};
    constexpr int maxSplineDegree{std::max(std::max(degx, degy), degz)};

    constexpr int hodgeDegree{2};

    {
        Kokkos::Profiling::ScopedRegion region("DKQuasineutralVlasovMaxwellMain()");
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

        auto estar =
            Gempic::Io::registered_form<DeRhamField<Forms::Grid::primal, Forms::Space::edge>>(
                fieldRegistry, "Estar", deRham);

        auto [parseB, funcB] = Utils::parse_functions<3>({"Bx", "By", "Bz"});
        auto B = Gempic::Io::registered_form<DeRhamField<Forms::Grid::primal, Forms::Space::face>>(
            fieldRegistry, "B", deRham, funcB); // initially zero.

        auto H = Gempic::Io::registered_form<DeRhamField<Forms::Grid::dual, Forms::Space::edge>>(
            fieldRegistry, "H", deRham);
        DeRhamField<Grid::dual, Space::face> curlH(deRham);

        auto rho = Gempic::Io::registered_form<DeRhamField<Forms::Grid::dual, Forms::Space::cell>>(
            fieldRegistry, "rho", deRham);
        std::vector<DeRhamField<Grid::dual, Space::cell>> rhoVec;
        std::vector<DeRhamField<Grid::dual, Space::cell>> rhoMassVec;

        DeRhamField<Grid::dual, Space::face> J(deRham);
        DeRhamField<Grid::dual, Space::cell> divJ(deRham);

        auto dx{infra.cell_size_3darray()};

        // Obtaining Bmean from input file and updating B and H
        Io::Parameters iCparams("ICs");
        amrex::Vector<amrex::Real> bmean;
        iCparams.get("Bmean", bmean);
        bmean[xDir] *= dx[yDir] * dx[zDir];
        bmean[yDir] *= dx[xDir] * dx[zDir];
        bmean[zDir] *= dx[xDir] * dx[yDir];

        B += bmean;
        hodge(H, B, deRham->scaling_bto_h());

        // Calculating bdir, pdir[0], pdir[1]
        int bdir{-1};
        for (int i = 0; i < vdim; ++i)
        {
            if (bmean[i] != 0.0)
            {
                bdir = i;
                break;
            }
        }
        GEMPIC_ALWAYS_ASSERT_WITH_MESSAGE(bdir != -1, "BMean is 0 in all directions.");

        std::array<int, 2> pdir; // perpendicular directions
        pdir[0] = (bdir + 1) % vdim;
        pdir[1] = (bdir + 2) % vdim;

        std::vector<std::shared_ptr<ParticleSpecies<vdim, ndata>>> partGr;
        init_particles(partGr, infra);

        std::vector<amrex::Real> rhoAnVec(partGr.size());

        int index = 0;
        for (auto& particleSpecies : partGr)
        {
            std::string specString = "Particle." + particleSpecies->get_name();
            Io::Parameters paramsSpec(specString);
            rhoAnVec[index] = particleSpecies->get_charge() * infra.cell_volume();
            index++;
        }

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

        DKQuasineutralSolver<hodgeDegree, vdim, ndata, degx, degy, degz> dkqn(infra, deRham);
        DKQuasineutralCalculationE<hodgeDegree, vdim, ndata, degx, degy, degz> dkqnEcalc(infra,
                                                                                         deRham);

        { //"Time Loop" scope
            Io::Parameters params("TimeLoop");
            amrex::Real dt;
            params.get("dt", dt);
            int nSteps;
            params.get("nSteps", nSteps);

            bool printChecks{false};
            params.get_or_set("printChecks", printChecks);

            auto diagnostics =
                Io::make_diagnostics<degx, degy, degz>(infra, deRham, fieldRegistry, partGr);

            // Initial E-field calculation begins
            // For the E_perp equation
            DeRhamField<Grid::dual, Space::face> rhsEperpEqn(deRham);

            // For the E_parallel equation
            DeRhamField<Grid::dual, Space::face> rhsEparallelEqn(deRham);

            amrex::Print() << "\nCalculating initial E-field:\n";
            // Initialize E to zero before calculating it
            for (int i = 0; i < vdim; i++)
            {
                E.m_data[i].setVal(0.0);
            }

            deposit_individual_particle_density<degx, degy, degz>(rhoVec, rho, rhoMassVec, partGr,
                                                                  infra);

            deposit_current_density_drift_kinetic<degx, degy, degz,
                                                  ElectricFieldUse::IncludeElectricField>(
                J, partGr, infra, E, B, bdir);

            if (printChecks)
            {
                check_energies_dkqn(B, H, partGr, infra, bdir);
                check_rho_related_norms_dkqn(rho, rhoAnVec, J, divJ, infra);
            }

            // Calculate RHS of the E_perp equation on dual face; solve for E_perp(t_0); check if
            // E_perp was calculated correctly
            dkqnEcalc.de_ki_calculate_rhs_eperp_eqn(rhsEperpEqn, E, B, H, curlH, J, partGr, bdir);
            dkqnEcalc.solve_de_fi_model_eperp(E, rhsEperpEqn, B, partGr, bdir, pdir, rhoVec);

            // Calculate RHS of the E_par equation on dual face; solve for E_par(t_n); check if
            // E_par was calculated correctly
            dkqnEcalc.de_ki_calculate_rhs_epar_eqn(rhsEparallelEqn, E, B, partGr, bdir, pdir);
            dkqnEcalc.solve_de_fi_model_epar(E, rhsEparallelEqn, partGr, bdir, rhoVec);

            // Calculate D(t_0)
            hodge(D, E, deRham->scaling_eto_d());
            // Initial E-field calculation ends

            amrex::Real simTime{0.0};
            diagnostics.compute_and_write_to_file(0, simTime);

            amrex::Print() << "\nSTARTING TIME LOOP:\n";

            for (int tStep = 0; tStep < nSteps; tStep++)
            {
                deposit_individual_particle_density<degx, degy, degz>(rhoVec, rho, rhoMassVec,
                                                                      partGr, infra);

                deposit_current_density_drift_kinetic<degx, degy, degz,
                                                      ElectricFieldUse::IncludeElectricField>(
                    J, partGr, infra, E, B, bdir);

                if (printChecks)
                {
                    check_energies_dkqn(B, H, partGr, infra, bdir);
                    check_rho_related_norms_dkqn(rho, rhoAnVec, J, divJ, infra);
                }

                dkqn.time_stepping_quasineutral_drift_electrons_kinetic_ions(
                    E, estar, D, B, H, curlH, J, partGr, bdir, pdir, dt, rhoVec, rhoMassVec, rho);

                simTime = dt * (tStep + 1);
                diagnostics.compute_and_write_to_file(tStep + 1, simTime);
                amrex::Print() << "====================" << tStep << std::endl;
                amrex::Print() << "finished time-step: " << tStep << std::endl;
                amrex::Print() << "====================" << tStep << std::endl;
            }

            amrex::Print() << "\nFINISHED TIME LOOP.\n";
        }
    }
    amrex::Finalize();
}