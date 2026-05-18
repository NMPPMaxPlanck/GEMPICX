/**************************************************************************************************
 * Copyright (c) 2021 GEMPICX                                                                     *
 * SPDX-License-Identifier: BSD-3-Clause                                                          *
 **************************************************************************************************/
#include <AMReX.H>
#include <AMReX_FFT.H>
#include <AMReX_GpuComplex.H>
#include <AMReX_Particles.H>
#include <AMReX_PlotFileUtil.H>
#include <AMReX_Print.H>

#include "GEMPIC_AmrexInit.H"
#include "GEMPIC_ComputationalDomain.H"
#include "GEMPIC_Config.H"
#include "GEMPIC_Diagnostics.H"
#include "GEMPIC_FDDeRhamComplex.H"
#include "GEMPIC_Fields.H"
#include "GEMPIC_Filter.H"
#include "GEMPIC_MatrixOperations.H"
#include "GEMPIC_NumericalIntegrationDifferentiation.H"
#include "GEMPIC_Parameters.H"
#include "GEMPIC_Particle.H"
#include "GEMPIC_ParticleMeshCoupling.H"
#include "GEMPIC_PoissonSolver.H"
#include "GEMPIC_Sampler.H"
#include "GEMPIC_SplineClass.H"
#include "GEMPIC_Splitting.H"
#include "GEMPIC_VlasovMaxwellEV.H"
#include "GEMPIC_VlasovMaxwellSemiImplicitFFT.H"
#include "GEMPIC_VlasovMaxwellSemiImplicitOperators.H"

using namespace Gempic;
using namespace Forms;
using namespace Particle;
using namespace ParticleMeshCoupling;
using namespace FieldSolvers;
using namespace TimeLoop;

int main (int argc, char* argv[])
{
    amrex::Initialize(argc, argv);

    constexpr int vdim{3};

    constexpr int degx{3};
    constexpr int degy{3};
    constexpr int degz{3};
    constexpr int maxSplineDegree{std::max(std::max(degx, degy), degz)};
    constexpr int hodgeDegree{2};

    // Stencil offsets and bandwidths for the particle Mass matrix
    static constexpr std::array<int, 3> sSGdecOffx{
        (hodgeDegree - 2), (hodgeDegree - 1),
        (hodgeDegree - 1)}; // x,y,z-field offsets in x-direction
    static constexpr std::array<int, 3> sSGdecOffy{
        (hodgeDegree - 1), (hodgeDegree - 2),
        (hodgeDegree - 1)}; // x,y,z-field offsets in y-direction
    static constexpr std::array<int, 3> sSGdecOffz{
        (hodgeDegree - 1), (hodgeDegree - 1),
        (hodgeDegree - 2)}; // x,y,z-field offsets in z-direction

    static constexpr std::array<int, 3> sSGdecBWx{
        (2 * sSGdecOffx[xDir] + 1), (2 * sSGdecOffx[yDir] + 1),
        (2 * sSGdecOffx[zDir] + 1)}; // x,y,z-field bandwidths in x-direction
    static constexpr std::array<int, 3> sSGdecBWy{
        (2 * sSGdecOffy[xDir] + 1), (2 * sSGdecOffy[yDir] + 1),
        (2 * sSGdecOffy[zDir] + 1)}; // x,y,z-field bandwidths in y-direction
    static constexpr std::array<int, 3> sSGdecBWz{
        (2 * sSGdecOffz[xDir] + 1), (2 * sSGdecOffz[yDir] + 1),
        (2 * sSGdecOffz[zDir] + 1)}; // x,y,z-field bandwidths in z-direction

    static constexpr std::array<int, 3> sSNGdecCoeffs{
        GEMPIC_D_MULT(sSGdecBWx[xDir], sSGdecBWy[xDir],
                      sSGdecBWz[xDir]), // total number of x-field coefficients
        GEMPIC_D_MULT(sSGdecBWx[yDir], sSGdecBWy[yDir],
                      sSGdecBWz[yDir]), // total number of y-field coefficients
        GEMPIC_D_MULT(sSGdecBWx[zDir], sSGdecBWy[zDir],
                      sSGdecBWz[zDir])}; // total number of z-field coefficients

    {
        BL_PROFILE("VlasovMaxwellSemiImplicitMain()");
        Io::Parameters parameters{};
        // Initialize computational_domain
        ComputationalDomain infra;

        // Initialize the De Rham Complex
        auto deRham = std::make_shared<FDDeRhamComplex>(infra, hodgeDegree, maxSplineDegree,
                                                        HodgeScheme::GDECHodge);

        Gempic::Io::FieldRegistry fieldRegistry;

        auto [parseB, funcB] = Utils::parse_functions<3>({"Bx", "By", "Bz"});
        auto [parseE, funcE] = Utils::parse_functions<3>({"Ex", "Ey", "Ez"});

        auto E = Gempic::Io::registered_form<DeRhamField<Grid::primal, Space::edge>>(
            fieldRegistry, "E", deRham, funcE);
        //DeRhamField<Grid::primal, Space::edge> eFiltered(deRham);
        DeRhamField<Grid::primal, Space::edge> e1(deRham);
        DeRhamField<Grid::primal, Space::edge> e2(deRham);
        auto B = Gempic::Io::registered_form<DeRhamField<Grid::primal, Space::face>>(
            fieldRegistry, "B", deRham, funcB);
        auto phi = Gempic::Io::registered_form<DeRhamField<Grid::primal, Space::node>>(
            fieldRegistry, "phi", deRham);
        DeRhamField<Grid::primal, Space::edge> ephi(deRham);

        auto divD = Gempic::Io::registered_form<DeRhamField<Grid::dual, Space::cell>>(
            fieldRegistry, "divD", deRham);
        DeRhamField<Grid::dual, Space::face> eStar(deRham);
        DeRhamField<Grid::dual, Space::face> eStarParticleMatrixTimeEPart(deRham);
        DeRhamField<Grid::dual, Space::face> es1(deRham);
        DeRhamField<Grid::dual, Space::face> es2(deRham);
        auto D = Gempic::Io::registered_form<DeRhamField<Grid::dual, Space::face>>(
            fieldRegistry, "D", deRham, funcE);
        auto H = Gempic::Io::registered_form<DeRhamField<Grid::dual, Space::edge>>(
            fieldRegistry, "H", deRham, funcB);
        auto J = Gempic::Io::registered_form<DeRhamField<Grid::dual, Space::face>>(fieldRegistry,
                                                                                   "J", deRham);
        //DeRhamField<Grid::dual, Space::face> jFiltered(deRham); // Filtered J for diagnostics
        auto rho = Gempic::Io::registered_form<DeRhamField<Grid::dual, Space::cell>>(fieldRegistry,
                                                                                     "rho", deRham);
        DeRhamField<Grid::dual, Space::cell> rhoFiltered(deRham); // Filtered rho for diagnostics

        // DeRhamField storing non-zero components of particle mass matrix
        DeRhamField<Grid::primal, Space::edge> particleMassMatrix(deRham, sSNGdecCoeffs);

        std::vector<std::shared_ptr<ParticleSpecies<vdim>>> partGr;
        init_particles(partGr, infra);
        amrex::Real rhoBackground{0.0};
        parameters.get_or_set("rhoBackground", rhoBackground);

        auto poisson{make_poisson_solver(deRham, infra)};

        // Create the FFT solver for Maxwell equations
        auto maxwellFFTSolver =
            std::make_unique<Gempic::FieldSolvers::VlasovMaxwellSemiImplicitFFTSolver>(
                infra, deRham, hodgeDegree);

        // Create EV hypre solver
        VlasovMaxwellEV<hodgeDegree, degx, degy, degz, vdim> hypreVlasovMaxwellEv(infra, deRham);

        // Initializing filter
        std::unique_ptr<Filter::Filter> filter = Gempic::Filter::make_filter(infra);

        { //"Time Loop" scope

            Io::Parameters params("TimeLoop");
            amrex::Real dt;
            params.get("dt", dt);
            int nSteps;
            params.get("nSteps", nSteps);

            auto diagnostics =
                Io::make_diagnostics<degx, degy, degz>(infra, deRham, fieldRegistry, partGr);

            auto dr = infra.cell_size_array();

            // Deposit initial charge
            rho.m_data.setVal(0.0);
            deposit_particle_density_gdec<hodgeDegree, degx, degy, degz>(rho, partGr, infra);

            // Add background charge (needs to be done after post_particle_loop_sync)
            rho += rhoBackground * infra.cell_volume();

            // solve Poisson
            poisson->solve(phi, rho);

            a_times_grad(ephi, phi, -1);
            E += ephi;
            hodge(D, E, deRham->scaling_eto_d());
            // compute div D for diagnostics
            div(divD, D);

            hodge(H, B, deRham->scaling_bto_h());

            amrex::Real simTime{0.0};
            diagnostics.compute_and_write_to_file(0, simTime);

            // Here begin the time-steps
            // We use the Strang splitting here

            for (int tStep = 0; tStep < nSteps; tStep++)
            {
                // *********************************
                // This is the implicit FFT field solver for E and B, corresponding to the operator
                // 3 on page 9 in [Energy conserving PIC, Kormann & Sonnendruecker, 2021, JCP].
                maxwellFFTSolver->solve_implicit_step(E, B, 0.5 * dt);
                // ******************************************
                // Here begins the substeps, corresponding to the operator 1, 2, 4 on page 9 in
                // [Energy conserving PIC, Kormann & Sonnendruecker, 2021, JCP].
                // initialize J to 0
                J.m_data[0].setVal(0.0);
                J.m_data[1].setVal(0.0);
                J.m_data[2].setVal(0.0);
                // initialize particle mass matrix times e
                eStarParticleMatrixTimeEPart.m_data[0].setVal(0.0);
                eStarParticleMatrixTimeEPart.m_data[1].setVal(0.0);
                eStarParticleMatrixTimeEPart.m_data[2].setVal(0.0);
                particleMassMatrix.m_data[0].setVal(0.0);
                particleMassMatrix.m_data[1].setVal(0.0);
                particleMassMatrix.m_data[2].setVal(0.0);
                semi_implicit_gdec_particle_loop<hodgeDegree, degx, degy, degz>(
                    E, eStar, dt, infra, partGr, B, J, eStarParticleMatrixTimeEPart,
                    particleMassMatrix, hypreVlasovMaxwellEv);
                // This is the implicit FFT field solver for E and B, corresponding to the operator
                // 3 on page 9 in [Energy conserving PIC, Kormann & Sonnendruecker, 2021, JCP].
                maxwellFFTSolver->solve_implicit_step(E, B, 0.5 * dt);

                // prepare for the diagnostics
                hodge(D, E, deRham->scaling_eto_d());
                // Deposit charge for diagnostics
                rho.m_data.setVal(0.0);
                deposit_particle_density_gdec<hodgeDegree, degx, degy, degz>(rho, partGr, infra);
                // compute div D for diagnostics
                div(divD, D);
                hodge(H, B, deRham->scaling_bto_h());
                simTime = dt * (tStep + 1);
                diagnostics.compute_and_write_to_file(tStep + 1, simTime);
                amrex::Print() << "finished time-step: " << tStep << std::endl;
            }
        }
    }
    amrex::Finalize();
}
