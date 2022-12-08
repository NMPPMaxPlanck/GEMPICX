/* Copyright 2019-2020 Yinjian Zhao
 *
 * This file is part of WarpX.
 *
 * License: BSD-3-Clause-LBNL
 */

#ifndef REDUCEDDIAGS_GAUSSERROR_H_
#define REDUCEDDIAGS_GAUSSERROR_H_

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
 *  computes the field energy.
 */
template <int vdim, int numspec, int degx, int degy, int degz, int degree, int ndata>
class GaussError: public ReducedDiags<vdim, numspec, degx, degy, degz, degree, ndata>
{
public:

    /**
     * constructor
     * @param[in] rd_name reduced diags names
     */
    GaussError(std::string rd_name)
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
                ofs << "[" << c++ << "]";
                ofs << std::endl;
                // close file
                ofs.close();
            }
        }
    }
    // end constructor

    /**
     * This function computes the Gauss Error
     *
     * @param[in] step current time step
     */
    void ComputeDiags(int step, computational_domain infra, maxwell_yee<vdim> *mw_yee,
                        amrex::GpuArray<std::unique_ptr<particle_groups<vdim, ndata>>, numspec> &part_gr) override final
    {   
        constexpr int noutputs = 1;
        this->m_data.resize(noutputs, 0.0);
        this->m_data[0] =
            mw_yee->template compute_gauss_error<numspec, degx, degy, degz, vdim, ndata>(infra,
                                                                                         part_gr);
    }
    // end void FieldEnergy::ComputeDiags

};

#endif
