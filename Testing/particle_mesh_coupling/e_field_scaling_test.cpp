#include <AMReX.H>
#include <GEMPIC_Fields.H>
#include <GEMPIC_FDDeRhamComplex.H>
#include <GEMPIC_parameters.H>
#include <GEMPIC_Params.H>
#include <GEMPIC_computational_domain.H>
#include "gtest/gtest.h"
#include "test_utils/GEMPIC_test_utils.H"
#include <GEMPIC_sampler.H>

using namespace Gempic;
using namespace Sampling;

namespace {
    class EFieldScalingTest : public testing::Test {
        protected:
        // Linear splines is ok, and lower dimension Hodge is good enough
        static const int vdim{3};
        static const int numspec{1};
        // Spline degrees
        static const int degx{1};
        static const int degy{1};
        static const int degz{1};

        static const int degmw{2};
        static const int hodgeDegree{2};
    };

    TEST_F(EFieldScalingTest, NullTest) {

        gempic_parameters<vdim, numspec> parametersBernstein;
        parametersBernstein.read_pp_params();
        parametersBernstein.set_computed_params();
        const amrex::Array<int, GEMPIC_SPACEDIM> isPeriodic = {AMREX_D_DECL(
                                                        parametersBernstein.is_periodic[0],
                                                        parametersBernstein.is_periodic[1],
                                                        parametersBernstein.is_periodic[2])};

        Parameters params(parametersBernstein.real_box,
                        parametersBernstein.n_cell,
                        parametersBernstein.max_grid_size,
                        isPeriodic,
                        hodgeDegree);

        // Initialize the De Rham Complex
        auto deRham = std::make_shared<GEMPIC_FDDeRhamComplex::FDDeRhamComplex>(params);

        DeRhamField<Grid::primal, Space::edge> E(deRham);
        DeRhamField<Grid::dual, Space::cell> rho(deRham);
        DeRhamField<Grid::primal, Space::node> phi(deRham);
        
        // Parse analytical fields and and initialize parserEval. Has to be the same as Bx,By,Bz and Ex, Ey, Ez
        const amrex::Array<std::string, 3> analyticalFuncE = {"1.0", "1.0", "1.0"};

        const int nVar = 4;  // x, y, z, t
        amrex::Array<amrex::ParserExecutor<nVar>, GEMPIC_SPACEDIM> funcE;
        amrex::Parser parser;

        for (int i = 0; i < 3; ++i)
        {
            parser.define(analyticalFuncE[i]);
            parser.registerVariables({"x", "y", "z", "t"});
            funcE[i] = parser.compile<4>();
        }

        deRham->projection(funcE, 0.0, E);

        // Initialize particle groups
        amrex::GpuArray<std::unique_ptr<particle_groups<vdim>>, numspec> ions;

        // In order to use the particle sampler we need to also initialize computational_domain
        // This class does the same as GEMPIC_parameters.H and needs to be urgently redesigned. 
        // {1, 1, 1} represents periodicity, has different types than Params and gempic_parameters.
        computational_domain infra;
        infra.initialize_computational_domain(parametersBernstein.n_cell, parametersBernstein.max_grid_size, parametersBernstein.is_periodic, parametersBernstein.real_box);

        for (int spec = 0; spec < numspec; spec++)
        {
            ions[spec] =
                std::make_unique<particle_groups<vdim>>(parametersBernstein.charge[spec], parametersBernstein.mass[spec], infra);
        }

        // initialize particles & loop preparation:
        // FIRST SPECIES
        amrex::Vector<amrex::Vector<amrex::Real>> meanVelocity = {{0.0, 0.0, 0.0}};
        amrex::Vector<amrex::Vector<amrex::Real>> vThermal = {{1.0, 1.0, 1.0}};
        amrex::Vector<amrex::Real> vWeight = {1.0};

        init_particles_full_domain<vdim, numspec>(infra,
                                                ions,
                                                parametersBernstein.n_part_per_cell,
                                                meanVelocity,
                                                vThermal, vWeight, 0,
                                                parametersBernstein.densityEval[0]);

        const int ndata = 1; // Needs to be 1 so that the correct ParIter type is defined. Putting 4 gets a non-defined type

        deRham->hodgeFD<degmw>(rho, phi);

        deRham->grad(phi, E);

        E *= -1.0;

        for (amrex::ParIter<0, 0, vdim + ndata, 0> pti(*ions[0], 0); pti.isValid(); ++pti)
        {
            amrex::Particle<0, 0>* AMREX_RESTRICT particles = &(pti.GetArrayOfStructs()[0]);
            const long np = pti.numParticles();
            auto particle_attributes = &pti.GetStructOfArrays();
            amrex::ParticleReal* const AMREX_RESTRICT velx =
                particle_attributes->GetRealData(0).data();
            amrex::ParticleReal* const AMREX_RESTRICT vely =
                particle_attributes->GetRealData(1).data();
            amrex::ParticleReal* const AMREX_RESTRICT velz =
                particle_attributes->GetRealData(2).data();

            const auto weight = pti.GetStructOfArrays().GetRealData(vdim).data();

            amrex::Array4<amrex::Real> const& rhoarr = rho.data[pti].array();

            amrex::GpuArray<amrex::Array4<amrex::Real>, vdim> eArray;

            for (int cc = 0; cc < vdim; cc++)
            {
                eArray[cc] = (E.data[cc])[pti].array();
            }

            amrex::Real charge = ions[0]->getCharge();
            amrex::Real chargemass = charge / ions[0]->getMass();
            amrex::Real dt = parametersBernstein.dt;

            amrex::ParallelFor(np, [=] AMREX_GPU_DEVICE(long pp)
            {
                // Read out particle position
                amrex::GpuArray<amrex::Real, GEMPIC_SPACEDIM> positionParticle;
                for (unsigned int d = 0; d < GEMPIC_SPACEDIM; ++d)
                    positionParticle[d] = particles[pp].pos(d);

                amrex::GpuArray<amrex::Real, vdim> vel{velx[pp], vely[pp], velz[pp]};

                Spline::SplineWithPrimitive<degx, degy, degz> spline(positionParticle, infra.plo, infra.dxi);

                // evaluate the electric field
                amrex::GpuArray<amrex::Real, vdim> efield =
                    spline.template evalField<vdim, 1>(eArray);

                // push v with the electric field
                amrex::GpuArray<amrex::Real, vdim> newPosE =
                    push_v_efield<vdim>(vel, dt * 0.5, chargemass, efield);

                velx[pp] = newPosE[0];
                vely[pp] = newPosE[1];
                velz[pp] = newPosE[2];

                for (unsigned int d = 0; d < GEMPIC_SPACEDIM; ++d) {
                    positionParticle[d] = positionParticle[d] + 0.5 * dt * velx[pp];
                    particles[pp].pos(d) = positionParticle[d];
                }

                Spline::SplineBase<degx, degy, degz> splinePos(positionParticle, infra.plo, infra.dxi);

                gempic_deposit_rho_C3_new_splines<degx, degy, degz>(
                    splinePos, charge * infra.dxi[GEMPIC_SPACEDIM] * weight[pp], rhoarr);
            });
        }
        ions[0] -> Redistribute();
    }
}