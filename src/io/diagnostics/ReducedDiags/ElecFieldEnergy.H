/* Copyright 2019-2020 Yinjian Zhao
 *
 * This file is part of WarpX.
 *
 * License: BSD-3-Clause-LBNL
 */

#ifndef REDUCEDDIAGS_ELECFIELDENERGY_H_
#define REDUCEDDIAGS_ELECFIELDENERGY_H_

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
class ElecFieldEnergy : public ReducedDiags<vdim, numspec, degx, degy, degz, degree, ndata>
{
public:

    /**
     * constructor
     * @param[in] rd_name reduced diags names
     */
    ElecFieldEnergy(std::string rd_name)
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
                ofs << "[" << c++ << "]Ex" + std::string("(J)");
                ofs << this->m_sep;
                ofs << "[" << c++ << "]Ey" + std::string("(J)");
                ofs << this->m_sep;
                ofs << "[" << c++ << "]Ez" + std::string("(J)");
                ofs << std::endl;
                // close file
                ofs.close();
            }
        }
    }
    // end constructor

    /**
     * This function computes the electric field energy (EF) component wise:
     * epref_ep0 is the vacuum permittivity
     *
     * @param[in] step current time step
     */
    void ComputeDiags(int step, computational_domain infra, maxwell_yee<vdim> *mw_yee,
                        amrex::GpuArray<std::unique_ptr<particle_groups<vdim, ndata>>, numspec> &part_gr) override final
    {   
        constexpr int noutputs = 3;
        this->m_data.resize(noutputs, 0.0);

        // Judge if the diags should be done
        //if (!m_intervals.contains(step+1)) { return; }

        // E field
        mw_yee->template hodge_full<degree>(infra, mw_yee->E_Array, mw_yee->HE_Array, true);
        for (int cc = 0; cc < vdim; cc++)
        {
            amrex::Real val = amrex::MultiFab::Dot(*(mw_yee->E_Masks[cc]), *(mw_yee->E_Array[cc]),
                                                   0, *(mw_yee->HE_Array[cc]), 0, 1, 0);
            val *= infra.dx[0] / infra.Length[0]
#if (GEMPIC_SPACEDIM > 1)
                   * infra.dx[1] / infra.Length[1]
#endif
#if (GEMPIC_SPACEDIM > 2)
                   * infra.dx[2] / infra.Length[2]
#endif
                ;
            this->m_data[cc] = 0.5 * mw_yee->epref_ep0 * val;
        }
    }
    // end void ElecFieldEnergy::ComputeDiags

};

#endif
