/* Copyright 2019-2020 Yinjian Zhao
 *
 * This file is part of WarpX.
 *
 * License: BSD-3-Clause-LBNL
 */

#ifndef REDUCEDDIAGS_MOMENTUM_H_
#define REDUCEDDIAGS_MOMENTUM_H_

#include "ReducedDiags.H"
#include <GEMPIC_checkpoint.H>
#include <GEMPIC_computational_domain.H>
#include <GEMPIC_gempic_norm.H>
#include <GEMPIC_maxwell_yee.H>
#include <GEMPIC_particle_groups.H>
#include <GEMPIC_PlotFile.H>
#include <GEMPIC_profiling.H>

#include <string>

/**
 *  This class mainly contains a function that
 *  computes the Momentum.
 */
template <int vdim, int numspec, int degx, int degy, int degz, int degree, int ndata>
class Momentum : public ReducedDiags<vdim, numspec, degx, degy, degz, degree, ndata>
{
public:

    /**
     * constructor
     * @param[in] rd_name reduced diags names
     */
    Momentum(std::string rd_name)
    : ReducedDiags<vdim, numspec, degx, degy, degz, degree, ndata>(rd_name)
    {
        if (amrex::ParallelDescriptor::IOProcessor())
        {
            if ( this->m_IsNotRestart )
            {
                // open file
                std::ofstream ofs{this->m_path + this->m_rd_name + "." + this->m_extension, std::ofstream::out};
                // write header row
                int c = 0;
                ofs << "#";
                ofs << "[" << c++ << "]step()";
                ofs << this->m_sep;
                ofs << "[" << c++ << "]time(s)";
                ofs << this->m_sep;
                ofs << "[" << c++ << "]PX" + std::string("(Ns)");
                ofs << this->m_sep;
                ofs << "[" << c++ << "]PY" + std::string("(Ns)");
                ofs << this->m_sep;
                ofs << "[" << c++ << "]PZ" + std::string("(Ns)");
                ofs << std::endl;
                // close file
                ofs.close();
            }
        }
    }
    // end constructor

    /**
     * This function computes the Momentum (Particle momentum)
     *
     * @param[in] step current time step
     */
    void ComputeDiags(int step, computational_domain infra, maxwell_yee<vdim> *mw_yee,
                        amrex::GpuArray<std::unique_ptr<particle_groups<vdim, ndata>>, numspec> &part_gr) override final
    {   
        constexpr int noutputs = 3;
        this->m_data.resize(noutputs, 0.0);
        
        for (int spec = 0; spec < numspec; spec++)
        {
            for (int cmp = 0; cmp < vdim; cmp++)
                {
                auto mom_tmp =
                    amrex::ReduceSum(*part_gr[spec],
                                     [=] AMREX_GPU_HOST_DEVICE(
                                         const amrex::Particle<vdim + ndata, 0> &p) -> amrex::Real
                                     {
                                         auto weight = p.rdata(vdim);
                                         auto vel = p.rdata(cmp);
                                         return (weight * vel);
                                     });

                // reduced sum over mpi ranks
                amrex::ParallelDescriptor::ReduceRealSum(
                    mom_tmp, amrex::ParallelDescriptor::IOProcessorNumber());

                // save results for this species i_s into m_data
                this->m_data[cmp] += part_gr[spec]->getMass() * mom_tmp / this->vol;
            }
        }
    }
    // end void Momentum::ComputeDiags

};

#endif
