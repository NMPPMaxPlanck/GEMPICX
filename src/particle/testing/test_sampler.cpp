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

// wave function
AMREX_GPU_HOST_DEVICE amrex::Real wave_function(amrex::Real x, amrex::Real y, amrex::Real z,
                                                amrex::Real t)
{
    amrex::Real val = 1.0;
    return val;
}

template <int vdim, int numspec>
void print_particles(amrex::GpuArray<std::unique_ptr<particle_groups<vdim>>, numspec>& part_gr,
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

template <int vdim, int numspec>
void print_vMoments(const amrex::GpuArray<std::unique_ptr<particle_groups<vdim>>, numspec>& part_gr,
                    const int species)
{
    // compute the first three moments of f(x,v), only one species
    amrex::GpuArray<amrex::Real, vdim + 2> vMoment;
    amrex::Real vMoment_tmp;
    // 1) \int f(x,v) dx dv
    vMoment_tmp = amrex::ReduceSum(
        *part_gr[species],
        [=] AMREX_GPU_HOST_DEVICE(const amrex::Particle<vdim + 1, 0>& p) -> amrex::Real {
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
            [=] AMREX_GPU_HOST_DEVICE(const amrex::Particle<vdim + 1, 0>& p) -> amrex::Real {
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
        [=] AMREX_GPU_HOST_DEVICE(const amrex::Particle<vdim + 1, 0>& p) -> amrex::Real {
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

template <int vdim, int numspec>
void main_main()
{
    //------------------------------------------------------------------------------
    gempic_parameters<vdim, numspec> gpParam;
    // gpParam.init_Nghost(1, 1, 1);
    amrex::IntVect num_cells = {AMREX_D_DECL(4, 4, 4)};
    amrex::GpuArray<int, numspec> n_part_per_cell = {1000};
    int species = 0;  // only one species

    amrex::Vector<amrex::Vector<amrex::Real>> vMean{};
    amrex::Vector<amrex::Vector<amrex::Real>> vThermal{};
    amrex::Vector<amrex::Real> vWeight{};

    int num_gaussian = 2;  // velocity distribution is sum of 2 Gaussians
    vMean = {{0.0, 0.0, 0.0}, {2.0, 2.0, 2.0}};
    vThermal = {{2.0, 2.0, 2.0}, {1.0, 1.0, 1.0}};
    vWeight = {0.75, 0.25};

    gpParam.set_params("sampler_ctest", num_cells, n_part_per_cell);
    //gpParam.density[0] = "1 + 0.5 * sin(kvarx*x + kvary*y + kvarz*z)";
    gpParam.density[0] = "1";
    double twopi = 4 * asin(1.0);
    gpParam.k = {twopi, twopi, twopi};
    gpParam.set_computed_params();
    computational_domain domain;
    domain.initialize_computational_domain(gpParam.n_cell, gpParam.max_grid_size,
                                           gpParam.is_periodic, gpParam.real_box);
    amrex::Print() << "domain " << *gpParam.real_box.lo() << " " << *gpParam.real_box.hi() << "\n";

    //------------------------------------------------------------------------------
    // Initialize Particle Groups
    amrex::GpuArray<std::unique_ptr<particle_groups<vdim>>, numspec> part_gr_cell;
    for (int spec = 0; spec < numspec; spec++)
    {
        part_gr_cell[spec] = std::make_unique<particle_groups<vdim>>(gpParam.charge[spec],
                                                                     gpParam.mass[spec], domain);
    }
    init_particles_cellwise<vdim, numspec>(domain, part_gr_cell, n_part_per_cell, vMean, vThermal,
                                           vWeight, species, wave_function);

    amrex::GpuArray<std::unique_ptr<particle_groups<vdim>>, numspec> part_gr_full;
    for (int spec = 0; spec < numspec; spec++)
    {
        part_gr_full[spec] = std::make_unique<particle_groups<vdim>>(gpParam.charge[spec],
                                                                     gpParam.mass[spec], domain);
    }
    init_particles_full_domain<vdim, numspec>(domain, part_gr_full, n_part_per_cell, vMean,
                                              vThermal, vWeight, species, wave_function);

    amrex::GpuArray<std::unique_ptr<particle_groups<vdim>>, numspec> part_gr_full_str;
    for (int spec = 0; spec < numspec; spec++)
    {
        part_gr_full_str[spec] = std::make_unique<particle_groups<vdim>>(
            gpParam.charge[spec], gpParam.mass[spec], domain);
    }
    init_particles_full_domain<vdim, numspec>(domain, part_gr_full_str, n_part_per_cell, vMean,
                                              vThermal, vWeight, species,
                                              gpParam.densityEval[species]);

    amrex::GpuArray<std::unique_ptr<particle_groups<vdim>>, numspec> part_gr_full_gpu;
    for (int spec = 0; spec < numspec; spec++)
    {
        part_gr_full_gpu[spec] = std::make_unique<particle_groups<vdim>>(gpParam.charge[spec],
                                                                     gpParam.mass[spec], domain);
    }
    init_particles_full_domain_gpu<vdim, numspec>(domain, part_gr_full_gpu, n_part_per_cell, vMean,
                                                  vThermal, vWeight, species,
                                                  gpParam.densityEval[species]);

    // Print particles data
    bool printPart = false;
    if (printPart)
    {
        print_particles<vdim, numspec>(part_gr_cell, species);
        print_particles<vdim, numspec>(part_gr_full, species);
        print_particles<vdim, numspec>(part_gr_full_str, species);
    }

    amrex::PrintToFile("test_sampler.tmp") << "\n";
    // Print analytical solution
    amrex::PrintToFile("test_sampler.tmp") << "1";
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
    print_vMoments<vdim, numspec>(part_gr_cell, species);
    print_vMoments<vdim, numspec>(part_gr_full, species);
    print_vMoments<vdim, numspec>(part_gr_full_str, species);
    print_vMoments<vdim, numspec>(part_gr_full_gpu, species);
}

int main(int argc, char* argv[])
{
    const bool build_parm_parse = true;
    amrex::Initialize(argc, argv, build_parm_parse, MPI_COMM_WORLD,
                      overwrite_amrex_parser_defaults);

    if (amrex::ParallelDescriptor::MyProc() == 0) remove("test_sampler.tmp.0");

    const int vdim = 3;
    const int numspec = 1;
    main_main<vdim, numspec>();

    if (amrex::ParallelDescriptor::MyProc() == 0)
        std::rename("test_sampler.tmp.0", "test_sampler.output");

    amrex::Finalize();
}
