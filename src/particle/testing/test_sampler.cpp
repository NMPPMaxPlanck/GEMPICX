/**
 * @file test_sampler.cpp
 * @author sonnen@ipp.mpg.de
 * @brief Tests GEMPIC_sampler.H
 * @version 0.1
 * @date 2021-12-30
 * @details GEMPIC_sampler.H samples functions of the type (for vdim=3) !!! Formula only for 1
 * species @n
 * @f[f(x,v)= n_0(x) \left(
 * \prod_{j=1}^{vdim}\frac{w_j}{\sqrt{2\pi v_{th,j}}}\exp(-\frac{(v_j-u_j)^2}{2v_{th,j}}
 * \right) @f] @n
 * Analytical solution: @n
 * @f[\int f dx\,dv = \int n_0\, dx, \quad
 *  \int f v_j dx\,dv = u_j \int n_0\, dx, \quad
 *  \int f (v_1^2+v_2^2+v_3^2) dx\,dv = \sum_{j=1}^{vdim} (u_j^2+v_{th,j}^2) \int n_0\, dx
 * @f]
 * @copyright Copyright (c) 2021
 *
 */

#include <AMReX.H>
#include <AMReX_ParmParse.H>
#include <AMReX_Particles.H>
#include <AMReX_PlotFileUtil.H>
#include <AMReX_Print.H>
#include <GEMPIC_amrex_init.H>
//#include <GEMPIC_Config.H>
#include <GEMPIC_computational_domain.H>
#include <GEMPIC_parameters.H>
#include <GEMPIC_particle_groups.H>
#include <GEMPIC_sampler.H>

// using namespace amrex;
using namespace Gempic;

// using namespace Field_solvers;
using namespace Particles;
using namespace Sampling;

template <unsigned int vdim, unsigned int numspec>
void print_particles(amrex::GpuArray<std::shared_ptr<particle_groups<vdim>>, numspec>& part_gr,
                     const int species)
{
    std::ofstream ofs("particles.out", std::ofstream::out);
    for (amrex::ParIter<0, 0, vdim + 1, 0> pti(*part_gr[species], 0); pti.isValid(); ++pti)
    {
        auto& particles = pti.GetArrayOfStructs();  // get particles
        const long np = pti.numParticles();
        auto& particle_attributes = pti.GetStructOfArrays();
        amrex::Print(ofs) << "number of particles " << np << "\n";
        for (int pp = 0; pp < np; pp++)
        {
            amrex::Print(ofs) << pp << " ";
            for (int i = 0; i < GEMPIC_SPACEDIM; i++)
            {
                amrex::Print(ofs) << particles[pp].pos(i) << " ";
            }
            for (int i = 0; i <= vdim; i++)
            {
                amrex::Print(ofs) << particle_attributes.GetRealData(i)[pp] << " ";
            }  // particle_attributes(vdim) is the particle weight
            amrex::Print(ofs) << "\n";
        }
    }
    ofs.close();
}

/**
 * @brief Compute velocity moments based on particle distribution.
 * Note that here the concept of SuperParticleType which agregates AoS and SoA in
 * Particle<NStructReal+NArrayReal, NStructInt+NArrayInt> is used
 *
 * @tparam vdim
 * @tparam numspec
 * @param part_gr
 * @param species
 */
template <unsigned int vdim, unsigned int numspec>
void print_vMoments(const amrex::GpuArray<std::shared_ptr<particle_groups<vdim>>, numspec>& part_gr,
                    const int species)
{
    // compute the first three moments of f(x,v), only one species
    amrex::GpuArray<amrex::Real, vdim + 2> vMoment;
    amrex::Real vMoment_tmp;
    // 1) \int f(x,v) dx dv
    vMoment_tmp = amrex::ReduceSum(
        *part_gr[species],
        [=] AMREX_GPU_HOST_DEVICE(const amrex::Particle<vdim + 1, 0>& p) -> amrex::Real
        {
            auto w = p.rdata(vdim);  // particle weight
            return (w);
        });
    // reduce sum over MPI ranks
    amrex::ParallelDescriptor::ReduceRealSum(vMoment_tmp,
                                             amrex::ParallelDescriptor::IOProcessorNumber());
    vMoment[0] = vMoment_tmp;

    // 2) \int v f(x,v) dx dv
    for (int cmp = 0; cmp < vdim; cmp++)
    {
        // reduce sum over one MPI rank
        vMoment_tmp = amrex::ReduceSum(
            *part_gr[species],
            [=] AMREX_GPU_HOST_DEVICE(const amrex::Particle<vdim + 1, 0>& p) -> amrex::Real
            {
                auto w = p.rdata(vdim);   // particle weight
                auto vel = p.rdata(cmp);  // velocity component
                return (w * vel);
            });
        // reduced sum over MPI ranks
        amrex::ParallelDescriptor::ReduceRealSum(vMoment_tmp,
                                                 amrex::ParallelDescriptor::IOProcessorNumber());
        vMoment[cmp + 1] = vMoment_tmp;
    }
    // 3) \int v^2 f(x,v) dx dv
    vMoment_tmp = amrex::ReduceSum(
        *part_gr[species],
        [=] AMREX_GPU_HOST_DEVICE(const amrex::Particle<vdim + 1, 0>& p) -> amrex::Real
        {
            auto w = p.rdata(vdim);  // particle weight
            auto v2 = std::pow(p.rdata(0), 2) + std::pow(p.rdata(1), 2) + std::pow(p.rdata(2), 2);
            return (w * v2);
        });
    // reduced sum over MPI ranks
    amrex::ParallelDescriptor::ReduceRealSum(vMoment_tmp,
                                             amrex::ParallelDescriptor::IOProcessorNumber());
    vMoment[vdim + 1] = vMoment_tmp;

    amrex::PrintToFile("test_sampler.tmp") << vMoment[0];
    for (int i = 0; i < vdim; i++)
    {
        amrex::PrintToFile("test_sampler.tmp") << " " << vMoment[i + 1];
    }
    amrex::PrintToFile("test_sampler.tmp") << " " << vMoment[vdim + 1] << "\n";
}

template <unsigned int vdim, unsigned int numspec>
void main_main()
{
    //------------------------------------------------------------------------------
    Parameters parameters{};
    int species = 0;  // only one species

    double twopi = 4 * asin(1.0);
    const amrex::Vector<amrex::Real> k{AMREX_D_DECL(twopi, twopi, twopi)};
    parameters.set("k", k);
    computational_domain domain;
    amrex::Print() << "domain " << domain.real_box.lo() << " " << domain.real_box.hi() << "\n";

    //------------------------------------------------------------------------------
    // Initialize Particle Groups
    amrex::GpuArray<std::shared_ptr<particle_groups<vdim>>, numspec> part_gr_cell;
    for (int spec = 0; spec < numspec; spec++)
    {
        part_gr_cell[spec] = std::make_shared<particle_groups<vdim>>(spec, domain);
    }
    init_particles_cellwise(domain, part_gr_cell, species);

    amrex::GpuArray<std::shared_ptr<particle_groups<vdim>>, numspec> part_gr_full;
    for (int spec = 0; spec < numspec; spec++)
    {
        part_gr_full[spec] = std::make_shared<particle_groups<vdim>>(spec, domain);
    }
    init_particles_full_domain(domain, part_gr_full, species);

    amrex::GpuArray<std::shared_ptr<particle_groups<vdim>>, numspec> part_gr_full_gpu;
    for (int spec = 0; spec < numspec; spec++)
    {
        part_gr_full_gpu[spec] = std::make_shared<particle_groups<vdim>>(spec, domain);
    }
    init_particles_full_domain_gpu(domain, part_gr_full_gpu, species);

    // Print particles data
    bool printPart = false;
    if (printPart)
    {
        print_particles(part_gr_cell, species);
        print_particles(part_gr_full, species);
        // print_particles(part_gr_full_gpu, species);
    }

    amrex::PrintToFile("test_sampler.tmp") << "\n";
    // Print analytical solution
    amrex::PrintToFile("test_sampler.tmp") << "1";

    Parameters params("particle.species0");
    amrex::Vector<amrex::Vector<amrex::Real>> vMean{};
    amrex::Vector<amrex::Vector<amrex::Real>> vThermal{};
    amrex::Vector<amrex::Real> vWeight{}; 
    int num_gaussian;
    params.get("num_gaussians", num_gaussian);

    for (int i{0}; i < num_gaussian; ++i)
    {
        amrex::Real vw;
        std::string vwString = "vWeight_g" + std::to_string(i);
        params.get(vwString, vw);
        vWeight.push_back(vw);

        amrex::Vector<amrex::Real> vm, vt;
        std::string vmString = "vMean_g" + std::to_string(i);
        params.get(vmString, vm);
        vMean.push_back(vm);
        
        std::string vtString = "vThermal_g" + std::to_string(i);
        params.get(vtString, vt);
        vThermal.push_back(vt);
    }

    amrex::Real mom2 = 0;
    for (int i = 0; i < vdim; i++)
    {
        amrex::Real mom1 = 0;
        for (int j = 0; j < num_gaussian; j++)
        {

            mom1 += vWeight[j] * vMean[j][i];
            mom2 += vWeight[j] * (std::pow(vThermal[j][i], 2) + std::pow(vMean[j][i], 2));
        }
        amrex::PrintToFile("test_sampler.tmp") << " " << mom1;
    }
    amrex::PrintToFile("test_sampler.tmp") << " " << mom2 << "\n";
    // Print computed solutions
    print_vMoments(part_gr_cell, species);
    print_vMoments(part_gr_full, species);
    // print_vMoments(part_gr_full_gpu, species);
}

int main(int argc, char* argv[])
{
    const bool build_parm_parse = true;
    amrex::Initialize(argc, argv, build_parm_parse, MPI_COMM_WORLD,
                      overwrite_amrex_parser_defaults);

    if (amrex::ParallelDescriptor::MyProc() == 0) remove("test_sampler.tmp.0");

    const unsigned int vdim = 3;
    const unsigned int numspec = 1;
    main_main<vdim, numspec>();

    if (amrex::ParallelDescriptor::MyProc() == 0)
        std::rename("test_sampler.tmp.0", "test_sampler.output");

    amrex::Finalize();
}
