#include <AMReX.H>
#include <AMReX_ParmParse.H>
#include <AMReX_Particles.H>
#include <AMReX_PlotFileUtil.H>
#include <AMReX_Print.H>
#include <GEMPIC_Config.H>
#include <GEMPIC_amrex_init.H>
#include <GEMPIC_parameters.H>
#include <GEMPIC_Params.H>
#include <GEMPIC_computational_domain.H>
#include <GEMPIC_particle_groups.H>
#include <GEMPIC_particle_mesh_coupling.H>
#include <GEMPIC_sampler.H>
#include <GEMPIC_hs_zigzag.H>
#include <GEMPIC_Fields.H>
#include <GEMPIC_FDDeRhamComplex.H>
#include <GEMPIC_PoissonSolver.H>

#include <random>
#include <iostream>

using namespace Gempic;
using namespace Sampling;

void write_rho(DeRhamField<Grid::dual, Space::cell> &rho, computational_domain& infra, double time, int step) {
        // save fields
        // MultiFab Info -------------------------------------------------------------
        std::string plotfilename{"Plotfiles/" + amrex::Concatenate("rho", step)};
 
        amrex::Vector<std::string> varnames{{"rho"}};
 
        amrex::WriteSingleLevelPlotfile(plotfilename, rho.data, varnames, infra.geom, time, 0);
}

void write_phi(DeRhamField<Grid::primal, Space::node> &phi, computational_domain& infra, double time, int step) {
        // save fields
        // MultiFab Info -------------------------------------------------------------
        std::string plotfilename{"Plotfiles/" + amrex::Concatenate("phi", step)};
 
        amrex::Vector<std::string> varnames{"phi"};
 
        amrex::WriteSingleLevelPlotfile(plotfilename, phi.data, varnames, infra.geom, time, 0);
}

void write_Ex(DeRhamField<Grid::primal, Space::edge> &E, computational_domain& infra, double time, int step) {
        // save fields
        // MultiFab Info -------------------------------------------------------------
        std::string plotfilename{"Plotfiles/" + amrex::Concatenate("E", step)};
 
        amrex::Vector<std::string> varnames{{"Ex"}};
 
        amrex::WriteSingleLevelPlotfile(plotfilename, E.data[0], varnames, infra.geom, time, 0);
}


// apply npass times a cubic spline filter to rho
void filter(DeRhamField<Grid::dual, Space::cell> &rho, DeRhamField<Grid::dual, Space::cell> &rhoTemp, int npass){
    
    auto nghost = rho.deRham->getNGhost();
    for (int pass = 0; pass < npass; pass++)
    {  
        // amrex::Print() << "filt pass " << pass << std::endl;
        for (int dimension = 0; dimension < 3; dimension++)
        {
            for (amrex::MFIter mfi(rho.data); mfi.isValid(); ++mfi) // Loop over grids
            {
                const amrex::Box& bx = mfi.validbox();
                amrex::Array4<amrex::Real> const & rhoArr = rho.data[mfi].array();
                amrex::Array4<amrex::Real> const & rhoTempArr = rhoTemp.data[mfi].array();

                ParallelFor(bx,
                            [=] AMREX_GPU_DEVICE(int i, int j, int k)
                            {
                                amrex::GpuArray<amrex::Real,3> coef{1./6., 2./3., 1./6.};
                                amrex::Real val = 0.0;
                                for (int d = 0; d < 3; d++)
                                {
                                    val += coef[d] * rhoArr(i + (dimension == 0 ? d - 1 : 0),
                                                            j + (dimension == 1 ? d - 1 : 0),
                                                            k + (dimension == 2 ? d - 1 : 0));
                                }
                                rhoTempArr(i, j, k) = val;
                            });                    
            }
            // Copy into rho
            rhoTemp.fillBoundary();
            amrex::Copy(rho.data, rhoTemp.data, 0, 0, 1, nghost);  
        }
    }
}


int main(int argc, char* argv[])
{
    amrex::Initialize(argc, argv);

    // Linear splines is ok, and lower dimension Hodge is good enough
    constexpr int vdim{3};
    constexpr int numspec{1};
    // Spline degrees
    constexpr int degx{1};
    constexpr int degy{1};
    constexpr int degz{1};
    //
    constexpr int hodgeDegree{2};

{
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
    auto deRham = std::make_shared<FDDeRhamComplex>(params);

    DeRhamField<Grid::primal, Space::face> B(deRham);
    DeRhamField<Grid::primal, Space::edge> E(deRham);
    DeRhamField<Grid::dual, Space::cell> rho(deRham);
    DeRhamField<Grid::dual, Space::cell> rhoTemp(deRham);
    DeRhamField<Grid::primal, Space::node> phi(deRham);

    amrex::Array<amrex::ParserExecutor<GEMPIC_SPACEDIM + 1>, 3> funcB{parametersBernstein.BxEval, parametersBernstein.ByEval, parametersBernstein.BzEval};
    deRham->projection(funcB, 0.0, B);
    // For the moment we consider only a constant background field.
    amrex::Real Bz = parametersBernstein.BzEval(AMREX_D_DECL(0., 0., 0.), 0.);
    amrex::Print() << "Bz " << Bz << std::endl;

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
    const int npass = 3; // Number of filter passes
    // Needed for SumBoundary
    auto nGhost = deRham->getNGhost();
    amrex::IntVect zeroVect{AMREX_D_DECL(0,0,0)};

    // Deposit initial charge
    for (int spec = 0; spec < numspec; spec++) {

        amrex::Real charge = ions[spec]->getCharge();

        for (amrex::ParIter<0, 0, vdim + ndata, 0> pti(*ions[spec], 0); pti.isValid(); ++pti)
        {
            const long np = pti.numParticles();
            const auto particles = pti.GetArrayOfStructs()().data();
            const auto weight = pti.GetStructOfArrays().GetRealData(vdim).data();

            amrex::Array4<amrex::Real> const& rhoarr = rho.data[pti].array();

            amrex::ParallelFor(np,
                            [=] AMREX_GPU_DEVICE(long pp)
                            {
                                amrex::GpuArray<amrex::Real, GEMPIC_SPACEDIM> positionParticle;
                                for (unsigned int d = 0; d < GEMPIC_SPACEDIM; ++d)
                                    positionParticle[d] = particles[pp].pos(d);
                                Spline::SplineBase<degx, degy, degz> spline(positionParticle, infra.plo, infra.dxi);
                                // Needs at least max(degx, degy, degz) ghost cells
                                gempic_deposit_rho<degx, degy, degz>(
                                    spline, charge * weight[pp], rhoarr);
                                    //spline, charge * infra.dxi[GEMPIC_SPACEDIM] * weight[pp],
                                    //rhoarr);
                            });
        }
    }

   (rho.data).SumBoundary(0, 1, nGhost, zeroVect, params.geometry().periodicity());
    rho.averageSync();
    rho.fillBoundary();

    filter(rho, rhoTemp, npass);

    deRham->hodgeFD<hodgeDegree>(rho, phi);
    deRham->grad(phi, E);
    E *= -1.0;

    amrex::Real dt = parametersBernstein.dt;
    int nSteps = parametersBernstein.n_steps;

    write_rho(rho, infra, 0, 0);
    write_Ex(E, infra, 0, 0);

    for (int tStep = 0; tStep < nSteps; tStep++) {

        for (int spec = 0; spec < numspec; spec++)
        {
            amrex::Real charge = ions[spec]->getCharge();
            amrex::Real chargemass = charge / ions[spec]->getMass();
            amrex::Real a = 0.5 * chargemass * dt * Bz;

            rho.data.setVal(0.0);

            for (amrex::ParIter<0, 0, vdim + ndata, 0> pti(*ions[spec], 0); pti.isValid(); ++pti)
            {
                const long np = pti.numParticles();
                const auto& particles = pti.GetArrayOfStructs()().data();
                const auto velx = pti.GetStructOfArrays().GetRealData(0).data();
                const auto vely = pti.GetStructOfArrays().GetRealData(1).data();
                const auto velz = pti.GetStructOfArrays().GetRealData(2).data();
                const auto weight = pti.GetStructOfArrays().GetRealData(vdim).data();

                amrex::Array4<amrex::Real> const& rhoarr = rho.data[pti].array();

                amrex::ParallelFor(np, [=] AMREX_GPU_DEVICE(long pp)
                {
                    // Local arrays for particle position and velocities
                    amrex::GpuArray<amrex::Real, GEMPIC_SPACEDIM> positionParticle;
                    amrex::GpuArray<amrex::Real, vdim> vel{velx[pp], vely[pp], velz[pp]};
                    for (unsigned int d = 0; d < GEMPIC_SPACEDIM; ++d)
                    {
                        // positionParticle data structure needed for spline
                        positionParticle[d] = particles[pp].pos(d) + 0.5 * dt * vel[d];
                        particles[pp].pos(d) = positionParticle[d];
                    }

                    Spline::SplineBase<degx, degy, degz> spline(positionParticle, infra.plo, infra.dxi);

                    gempic_deposit_rho<degx, degy, degz>(
                        spline, charge * weight[pp], rhoarr);
                        //spline, charge * infra.dxi[GEMPIC_SPACEDIM] * weight[pp], rhoarr);
                });

            }

            ions[spec] -> Redistribute();

            (rho.data).SumBoundary(0, 1, nGhost, zeroVect, params.geometry().periodicity());
            rho.averageSync();
            rho.fillBoundary();
            filter(rho, rhoTemp, npass);

            deRham->hodgeFD<hodgeDegree>(rho, phi);
            deRham->grad(phi, E);
            E *= -1.0;

            rho.data.setVal(0.0);

            for (amrex::ParIter<0, 0, vdim + ndata, 0> pti(*ions[spec], 0); pti.isValid(); ++pti)
            {
                const long np = pti.numParticles();
                const auto& particles = pti.GetArrayOfStructs()().data();
                const auto velx = pti.GetStructOfArrays().GetRealData(0).data();
                const auto vely = pti.GetStructOfArrays().GetRealData(1).data();
                const auto velz = pti.GetStructOfArrays().GetRealData(2).data();
                const auto weight = pti.GetStructOfArrays().GetRealData(vdim).data();

                amrex::Array4<amrex::Real> const& rhoarr = rho.data[pti].array();
                amrex::GpuArray<amrex::Array4<amrex::Real>, vdim> eA;

                // Extract E
                for (int cc = 0; cc < vdim; cc++)
                {
                    eA[cc] = (E.data[cc])[pti].array();
                }

                amrex::ParallelFor(np, [=] AMREX_GPU_DEVICE(long pp)
                {
                    // Read out particle position
                    amrex::GpuArray<amrex::Real, GEMPIC_SPACEDIM> positionParticle;
                    for (unsigned int d = 0; d < GEMPIC_SPACEDIM; ++d)
                        positionParticle[d] = particles[pp].pos(d);

                    amrex::GpuArray<amrex::Real, vdim> vel{velx[pp], vely[pp], velz[pp]};

                    Spline::SplineBase<degx, degy, degz> spline(positionParticle, infra.plo, infra.dxi);

                    // evaluate the electric field
                    amrex::GpuArray<amrex::Real, vdim> efield =
                        spline.template evalSplineField<vdim, 1>(eA);

                    // push v with the electric field over dt/2
                    push_v_efield<vdim>(vel, dt * 0.5, chargemass, efield);


                    // rotate v with magnetic field over dt
                    amrex::Real vx = vel[0];
                    amrex::Real vy = vel[1];

                    vel[0] = (vx*(1.-a*a) + 2.*a*vy)/(1.+a*a);
                    vel[1] = (vy*(1.-a*a) - 2.*a*vx)/(1.+a*a);

                    // push v with the electric field over dt/2
                    push_v_efield<vdim>(vel, dt * 0.5, chargemass, efield);

                    // update global particle velocities arrays    
                    velx[pp] = vel[0];
                    vely[pp] = vel[1];
                    velz[pp] = vel[2];

                    for (unsigned int d = 0; d < GEMPIC_SPACEDIM; ++d)
                    {
                        positionParticle[d] = particles[pp].pos(d) + 0.5 * dt * vel[d];
                        particles[pp].pos(d) = positionParticle[d];
                    }

                    Spline::SplineBase<degx, degy, degz> splineNew(positionParticle, infra.plo, infra.dxi);

                    gempic_deposit_rho<degx, degy, degz>(
                        splineNew, charge * weight[pp], rhoarr);
                        // splineNew, charge * infra.dxi[GEMPIC_SPACEDIM] * weight[pp], rhoarr);
                });
            }
            ions[spec] -> Redistribute();
        }

        (rho.data).SumBoundary(0, 1, nGhost, zeroVect, params.geometry().periodicity());

        rho.averageSync();
        rho.fillBoundary();
        filter(rho, rhoTemp, npass);

        if (tStep%parametersBernstein.save_fields == 0) {
            write_rho(rho, infra, (tStep+1)*parametersBernstein.dt, tStep+1);
            write_Ex(E, infra, (tStep+1)*parametersBernstein.dt, tStep+1);
        }
        if (tStep%1 == 0) {
            std::cout << "Time Step: " << tStep+1 << std::endl;
        }
    }
}
    amrex::Finalize();
}
