/* Copyright 2019-2020 Yinjian Zhao
 *
 * This file is part of WarpX.
 *
 * License: BSD-3-Clause-LBNL
 */

#ifndef REDUCEDDIAGS_PARTICLEENERGY_H_
#define REDUCEDDIAGS_PARTICLEENERGY_H_

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
 *  computes the particle energy.
 */
template <int vdim, int numspec, int degx, int degy, int degz, int degree, int ndata>
class ParticleEnergy : public ReducedDiags<vdim, numspec, degx, degy, degz, degree, ndata>
{
public:

    /**
     * constructor
     * @param[in] rd_name reduced diags names
     */
    ParticleEnergy(std::string rd_name)
    : ReducedDiags<vdim, numspec, degx, degy, degz, degree, ndata>(rd_name)
    {
        //amrex::ParmParse pp_rd_name(rd_name);
        //rd_name.query("domain_volume", vol);

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
                ofs << "[" << c++ << "KE" + std::string("(J)");
                ofs << std::endl;
                // close file
                ofs.close();
            }
        }
    }
    // end constructor

    /**
     * This function computes the Kinetic energy (KE)
     * @param[in] step current time step
     */
    void ComputeDiags(int step, computational_domain infra, maxwell_yee<vdim> *mw_yee,
                        amrex::GpuArray<std::unique_ptr<particle_groups<vdim, ndata>>, numspec> &part_gr) override final
    {   
        constexpr int noutputs = 1;
        this->m_data.resize(noutputs, 0.0);
        // Judge if the diags should be done
        //if (!m_intervals.contains(step+1)) { return; }

        // Particle Kinetic energy
        for (int spec = 0; spec < numspec; spec++)
        {
            BL_PROFILE_VAR("diagnostics::kinetic energy", kin);
            auto kin_tmp = amrex::ReduceSum(
                *part_gr[spec],
                [=] AMREX_GPU_HOST_DEVICE(const amrex::Particle<vdim + ndata, 0> &p) -> amrex::Real
                {
                    amrex::Real weight = p.rdata(vdim);
                    amrex::Real vel2 = pow(p.rdata(0), 2) + pow(p.rdata(1), 2) + pow(p.rdata(2), 2);
                    return (weight * vel2);
                });

            // reduced sum over mpi ranks
            amrex::ParallelDescriptor::ReduceRealSum(
                kin_tmp, amrex::ParallelDescriptor::IOProcessorNumber());

            this->m_data[0] = 0.5 * part_gr[spec]->getMass() * kin_tmp / this->vol;
        }
    }
    // end void ParticleEnergy::ComputeDiags

};

#endif
