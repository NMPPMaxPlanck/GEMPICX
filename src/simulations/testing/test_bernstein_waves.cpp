#include <AMReX.H>
#include <AMReX_ParmParse.H>
#include <AMReX_Particles.H>
#include <AMReX_PlotFileUtil.H>
#include <AMReX_Print.H>
#include <GEMPIC_Config.H>
#include <GEMPIC_amrex_init.H>
#include <GEMPIC_parameters.H>
#include <GEMPIC_computational_domain.H>
#include <GEMPIC_particle_groups.H>
#include <GEMPIC_particle_mesh_coupling.H>
#include <GEMPIC_sampler.H>
#include <GEMPIC_Fields.H>
#include <GEMPIC_FDDeRhamComplex.H>
//#include <GEMPIC_PoissonSolver.H>
#include <BilinearFilter.H>

#include <random>
#include <iostream>
#include <MultiFullDiagnostics.H>
#include <MultiReducedDiagnostics.H>

using namespace Gempic;
using namespace CompDom;
using namespace Sampling;
using namespace GEMPIC_Fields;
using namespace GEMPIC_FDDeRhamComplex;

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
 
        amrex::WriteSingleLevelPlotfile(plotfilename, E.data[xDir], varnames, infra.geom, time, 0);
}


// apply npass times a cubic spline filter to rho
void filter(DeRhamField<Grid::dual, Space::cell> &rho, DeRhamField<Grid::dual, Space::cell> &rhoTemp, int npass){
    
    auto nghost = rho.deRham->getNGhost();
    for (int pass = 0; pass < npass; pass++)
    {  
        // amrex::Print() << "filt pass " << pass << std::endl;
        for (int direction = 0; direction < 3; direction++)
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
                                    val += coef[d] * rhoArr(i + (direction == xDir ? d - 1 : 0),
                                                            j + (direction == yDir ? d - 1 : 0),
                                                            k + (direction == zDir ? d - 1 : 0));
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
    constexpr int ndata{1};
    // Spline degrees
    constexpr int degx{1};
    constexpr int degy{1};
    constexpr int degz{1};
    constexpr int maxSplineDegree{std::max(std::max(degx, degy), degz)};
    //
    constexpr int hodgeDegree{2};

{
    //Parameters::setPrintOutput();  // uncomment to print an output file
    Parameters parameters{};


    // Initialize computational_domain
    computational_domain infra;

    // Initialize the De Rham Complex
    auto deRham = std::make_shared<FDDeRhamComplex>(infra, hodgeDegree, maxSplineDegree, HodgeScheme::FDHodge);

    auto [parseB, funcB] = Utils::parseFunctions<3>({"Bx", "By", "Bz"});

    DeRhamField<Grid::primal, Space::face> B(deRham, funcB, "B");
    DeRhamField<Grid::dual, Space::edge> H(deRham, funcB, "H");
    DeRhamField<Grid::primal, Space::edge> E(deRham, "E");
    DeRhamField<Grid::dual, Space::face> D(deRham, "D");
    DeRhamField<Grid::dual, Space::cell> rho(deRham, "rho");
    DeRhamField<Grid::dual, Space::cell> divD(deRham, "divD");
    DeRhamField<Grid::dual, Space::cell> rhoTemp(deRham);
    DeRhamField<Grid::primal, Space::node> phi(deRham, "phi");
    DeRhamField<Grid::dual, Space::face> currentDensity(deRham, "J");

    // Initialize particle groups
    amrex::GpuArray<std::shared_ptr<particle_groups<vdim>>, numspec> ions;

    //Initializing filter
    std::unique_ptr<Filter> biFilter = std::make_unique<BilinearFilter> ();

    // initialize particles & loop preparation:
    // FIRST SPECIES
    for (int spec = 0; spec < numspec; spec++)
    {
        ions[spec] =
            std::make_unique<particle_groups<vdim>>(spec, infra);
        init_particles_full_domain<vdim, numspec>(infra, ions, spec);
    }

    // For the moment we consider only a constant background field.
    amrex::Real Bz = funcB[zDir](AMREX_D_DECL(0., 0., 0.), 0.);

    {// "Time Loop" scope. Should be a separate function
        const int ndata = 1; // Needs to be 1 so that the correct ParIter type is defined. Putting 4 gets a non-defined type
        const int npass = 0; // Number of filter passes

        // Initialize full diagnostics and write initial time step
        
        Parameters params("time_loop");
        amrex::Real dt;
        params.get("dt", dt);
        int nSteps;
        params.get("n_steps", nSteps);
        Parameters paramsSim("sim");
        amrex::Real Te; // electron temperature
        paramsSim.get("Te", Te);
        auto nGhost = deRham->getNGhost();
        MultiDiagnostics<vdim, numspec, ndata> fullDiagn(dt);
        fullDiagn.InitData(infra, deRham->fieldsDiagnostics, deRham->fieldsScaling, ions, nGhost);

        // Initialize reduced diagnostics and write initial time step
        MultiReducedDiagnostics<vdim, numspec, degx, degy, degz, hodgeDegree, 1> redDiagn(deRham);
    
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
                                    gempic_deposit_rho(spline, charge * weight[pp], rhoarr);
                                });
            }
        }

        rho.postParticleLoopSync();

        //filter(rho, rhoTemp, npass);
        biFilter->ApplyStencil(rhoTemp.data, rho.data, 0, 0, 1);
        
        for (int component = 0; component < vdim; component++)
            currentDensity.data[component].SumBoundary(0, 1, nGhost, amrex::IntVect(AMREX_D_DECL(0,0,0)),  infra.geom.periodicity());
        currentDensity.averageSync();
        currentDensity.fillBoundary();

        deRham->hodge(rho, phi);
        deRham->grad(phi, E);
        //E *= -1.0;
        E *= - Te;

        // D is also needed to compute energy
        deRham->hodge(E, D);

        // computing rho from D to check Gauss law // Makes no sense for quasi-neutral model
        // deRham->div(D,rho_gauss_law); //must be done before writing reduced diagnostics

        // Write initial time step
        redDiagn.ComputeDiags(infra, deRham->fieldsDiagnostics, ions);
        redDiagn.WriteToFile(0, dt);
        fullDiagn.FilterComputePackFlush(0); 

        // write_rho(rho, infra, 0, 0);
        // write_Ex(E, infra, 0, 0);

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

                        gempic_deposit_rho(spline, charge * weight[pp], rhoarr);
                    });

                }

                ions[spec] -> Redistribute();

                rho.postParticleLoopSync();
                //filter(rho, rhoTemp, npass);
                biFilter->ApplyStencil(rhoTemp.data, rho.data, 0, 0, 1);

                deRham->hodge(rho, phi);
                deRham->grad(phi, E);
                //E *= -1.0;
                E *= - Te;
                // D is also needed to compute energy
                deRham->hodge(E, D);

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
                            spline.template evalSplineField<Field::PrimalOneForm>(eA);

                        // push v with the electric field over dt/2
                        push_v_efield(vel, dt * 0.5, chargemass, efield);


                        // rotate v with magnetic field over dt
                        amrex::Real vx = vel[xDir];
                        amrex::Real vy = vel[yDir];

                        vel[xDir] = (vx*(1.-a*a) + 2.*a*vy)/(1.+a*a);
                        vel[yDir] = (vy*(1.-a*a) - 2.*a*vx)/(1.+a*a);

                        // push v with the electric field over dt/2
                        push_v_efield(vel, dt * 0.5, chargemass, efield);

                        // update global particle velocities arrays    
                        velx[pp] = vel[xDir];
                        vely[pp] = vel[yDir];
                        velz[pp] = vel[zDir];

                        for (unsigned int d = 0; d < GEMPIC_SPACEDIM; ++d)
                        {
                            positionParticle[d] = particles[pp].pos(d) + 0.5 * dt * vel[d];
                            particles[pp].pos(d) = positionParticle[d];
                        }

                        Spline::SplineBase<degx, degy, degz> splineNew(positionParticle, infra.plo, infra.dxi);

                        gempic_deposit_rho(splineNew, charge * weight[pp], rhoarr);
                    });
                }
                ions[spec] -> Redistribute();
            }

            rho.postParticleLoopSync();

            //filter(rho, rhoTemp, npass);
            biFilter->ApplyStencil(rhoTemp.data, rho.data, 0, 0, 1);

            redDiagn.ComputeDiags(infra, deRham->fieldsDiagnostics, ions);
            redDiagn.WriteToFile(tStep + 1, dt);
            fullDiagn.FilterComputePackFlush(tStep+1); 
            // if (tStep%saveFields == 0) {
            //     write_rho(rho, infra, (tStep+1)*dt, tStep+1);
            //     write_Ex(E, infra, (tStep+1)*dt, tStep+1);
            // }
            if (tStep%10 == 0) {
                std::cout << "Time Step: " << tStep+1 << std::endl;
            }
        }
    } // end of "time loop" scope
}
    amrex::Finalize();
}
