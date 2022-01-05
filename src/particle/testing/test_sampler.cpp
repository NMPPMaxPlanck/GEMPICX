/**
 * @file test_sampler.cpp
 * @author sonnen@ipp.mpg.de
 * @brief Tests GEMPIC_sampler.H
 * @version 0.1
 * @date 2021-12-30
 * @details GEMPIC_sampler.H samples functions of the type (for vdim=3) !!! Formula only for 1 Gaussian 
 * @f$f(x,v)= n_0(x) \left( \frac{vWeight[0]}{(2\pi v_{th})^{3/2}}\exp{(v_0-u_0)^2(2v_{th,0})}\exp{(v_1-u_1)^2(2v_{th,1})} \exp{(v_2-u_2)^2(2v_{th,2})} @f$ 
 * @copyright Copyright (c) 2021
 * 
 */

#include <AMReX.H>
#include <AMReX_ParmParse.H>
#include <AMReX_Particles.H>
#include <AMReX_PlotFileUtil.H>
#include <AMReX_Print.H>

//#include <GEMPIC_Config.H>
#include <GEMPIC_sampler.H>
#include <GEMPIC_computational_domain.H>
#include <GEMPIC_particle_groups.H>
#include <GEMPIC_parameters.H>

//using namespace std;
//using namespace amrex;
using namespace Gempic;

//using namespace Field_solvers;
using namespace Particles;
using namespace Sampling;

// wave function
AMREX_GPU_HOST_DEVICE amrex::Real wave_function(amrex::Real x, amrex::Real y, amrex::Real z)
{
    amrex::Real val = 1.0;
        return val;
}

template<int vdim, int numspec>
void print_particles(const particle_groups<vdim, numspec> & part_gr, const int species) {
    std::ofstream ofs("particles.out", std::ofstream::out);
        for (amrex::ParIter<vdim+1,0,0,0> pti(*part_gr.mypc[species], 0); pti.isValid(); ++pti) {
            auto& particles = pti.GetArrayOfStructs(); // get particles
            const long np  = pti.numParticles();
            amrex::Print(ofs) << "number of particles " << np << "\n"; 
            for (int pp=0; pp < np; pp++){
                amrex::Print(ofs) << pp << " "; 
                for (int i=0; i < GEMPIC_SPACEDIM; i++) {
                    amrex::Print(ofs) << particles[pp].pos(i) << " ";
                }
                for (int i=0; i <= vdim; i++){
                    amrex::Print(ofs) << particles[pp].rdata(i) << " ";
                } // rdata[vdim] is the particle weight
                amrex::Print(ofs) << "\n";
            }
        }
        ofs.close();
}

template <int vdim, int numspec>
void print_vMoments(const particle_groups<vdim, numspec> & part_gr, const int species) {
    // compute the first three moments of f(x,v), only one species 
    amrex::GpuArray<amrex::Real,vdim+2> vMoment;
    amrex::Real vMoment_tmp;
    // 1) \int f(x,v) dx dv
    vMoment_tmp =  amrex::ReduceSum( *part_gr.mypc[species],
                            [=] AMREX_GPU_HOST_DEVICE (const amrex::Particle<vdim+1,0>& p) -> amrex::Real {
                auto w  = p.rdata(vdim); // particle weight
                return (w);
        });
    // reduce sum over MPI ranks
    amrex::ParallelDescriptor::ReduceRealSum(vMoment_tmp, amrex::ParallelDescriptor::IOProcessorNumber());
    vMoment[0] = vMoment_tmp;   

    // 2) \int v f(x,v) dx dv
    for (int cmp=0;cmp<vdim;cmp++) {
        // reduce sum over one MPI rank
        vMoment_tmp = amrex::ReduceSum( *part_gr.mypc[0],
                            [=] AMREX_GPU_HOST_DEVICE (const amrex::Particle<vdim+1,0>& p) -> amrex::Real {
                auto w  = p.rdata(vdim); // particle weight
                auto vel = p.rdata(cmp); // velocity component
                return (w*vel);
        });
        // reduced sum over MPI ranks
        amrex::ParallelDescriptor::ReduceRealSum(vMoment_tmp, amrex::ParallelDescriptor::IOProcessorNumber());
        vMoment[cmp+1] = vMoment_tmp;
    }
    // 3) \int v^2 f(x,v) dx dv
    vMoment_tmp =  amrex::ReduceSum( *part_gr.mypc[0],
                            [=] AMREX_GPU_HOST_DEVICE (const amrex::Particle<vdim+1,0>& p) -> amrex::Real {
                auto w  = p.rdata(vdim); // particle weight
                auto v2 = std::pow(p.rdata(0),2)+std::pow(p.rdata(1),2)+std::pow(p.rdata(2),2);
                return (w*v2);
        });
    // reduced sum over MPI ranks
    amrex::ParallelDescriptor::ReduceRealSum(vMoment_tmp, amrex::ParallelDescriptor::IOProcessorNumber());
    vMoment[vdim+1] = vMoment_tmp;   
    
    amrex::AllPrintToFile("test_sampler.tmp") << vMoment[0]; 
    for (int i=0; i < vdim; i++) {
        amrex::AllPrintToFile("test_sampler.tmp") << " " <<vMoment[i+1];  
    }
    amrex::AllPrintToFile("test_sampler.tmp") << " " << vMoment[vdim+1] << "\n";
}

template<int vdim, int numspec>
void main_main ()
{
    //------------------------------------------------------------------------------
    gempic_parameters<vdim, numspec> gpParam;
    //gpParam.init_Nghost(1, 1, 1);
    amrex::IntVect is_periodic = {AMREX_D_DECL(1,1,1)};
    amrex::IntVect num_cells = {AMREX_D_DECL(4,4,4)};
    std::array<int, numspec> n_part_per_cell = {1000};
    int species = 0; // all particles are same species for now

    std::array<std::vector<amrex::Real>, vdim> vMean{};
    std::array<std::vector<amrex::Real>, vdim> vThermal{}; 
    std::array<std::vector<amrex::Real>, vdim> vWeight{};  // ???? relative weight of Gaussian? Why 3 components
    amrex::GpuArray<amrex::Real,vdim+2> vMoment;

    // only 1 Gaussian
    for (int i = 0; i < vdim; i++) {
        vMean[i].push_back(0.0);
        vThermal[i].push_back(2.0);
        vWeight[i].push_back(1.0);
    }

    gpParam.set_params("sampler_ctest", num_cells, n_part_per_cell);
    gpParam.set_computed_params();
    computational_domain domain;
    gpParam.real_box = amrex::RealBox(AMREX_D_DECL(0.0, 0.0, 0.0),AMREX_D_DECL(1.0, 1.0, 1.0));
    domain.initialize_computational_domain(gpParam.n_cell, gpParam.max_grid_size, gpParam.is_periodic, gpParam.real_box);
    amrex::Print() << "domain " << *gpParam.real_box.lo() << " " << *gpParam.real_box.hi() << "\n";

    //------------------------------------------------------------------------------
    //Initialize Particle Groups
    particle_groups<vdim, numspec> part_gr_cell(gpParam.charge, gpParam.mass, domain);
    init_particles_cellwise<vdim, numspec>(domain, part_gr_cell,n_part_per_cell, vMean, vThermal, vWeight, species, wave_function);

    particle_groups<vdim, numspec> part_gr_full(gpParam.charge, gpParam.mass, domain);
    init_particles_full_domain<vdim,numspec>(domain, part_gr_full,n_part_per_cell, vMean, vThermal, vWeight, species, wave_function);

    //std::string wave_function = "kvarx*x + kvary*y + kvarz*z"; 
    std::string wave_function_string = "1";
    particle_groups<vdim, numspec> part_gr_full_str(gpParam.charge, gpParam.mass, domain);
    init_particles_full_domain<vdim,numspec>(domain, part_gr_full_str,n_part_per_cell, gpParam.k, wave_function_string, vMean, vThermal, vWeight, species);
 
    // Print particles data
    bool printPart = false;
    if (printPart) {
        print_particles<vdim,numspec>(part_gr_cell, species);
        print_particles<vdim,numspec>(part_gr_full, species);
        print_particles<vdim,numspec>(part_gr_full_str, species);
    }

    amrex::AllPrintToFile("test_sampler.tmp") << "\n"; 
    print_vMoments<vdim,numspec>(part_gr_cell, species);    
    print_vMoments<vdim,numspec>(part_gr_full, species);
    print_vMoments<vdim,numspec>(part_gr_full_str, species);
}

int main(int argc, char* argv[])
{
    amrex::Initialize(argc,argv);

    if (amrex::ParallelDescriptor::MyProc()==0) remove("test_sampler.tmp.0");
    //if (amrex::ParallelDescriptor::MyProc()==0) amrex::FileSystem::Remove("rm test_sampler.output");

    const int vdim = 3;
    const int numspec = 1;
    main_main<vdim, numspec>();

    if (amrex::ParallelDescriptor::MyProc()==0) std::rename("test_sampler.tmp.0", "test_sampler.output");

    amrex::Finalize();
}



