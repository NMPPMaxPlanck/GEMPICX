/* Copyright 2019-2020 Yinjian Zhao
 *
 * This file is part of WarpX.
 *
 * License: BSD-3-Clause-LBNL
 */
/*
* Modified for GEMPICX
*/

#ifndef REDUCEDDIAGS_PARTICLE_H_
#define REDUCEDDIAGS_PARTICLE_H_

#include <AMReX_Reduce.H>
#include <AMReX_Particle.H>

#include "ReducedDiagnostics.H"
#include <GEMPIC_computational_domain.H>
#include <GEMPIC_gempic_norm.H>
#include <GEMPIC_particle_groups.H>
#include <GEMPIC_profiling.H>

#include <string>

/**
 *  This class mainly contains a function that
 *  computes the Momentum and Kinetic Energy.
 */
template <int vdim, int numspec, int degx, int degy, int degz, int degree, int ndata>
class Particle : public ReducedDiagnostics<vdim, numspec, degx, degy, degz, degree, ndata>
{
public:

    /**
     * constructor
     * @param[in] rd_name reduced diags names
     */
    Particle(std::string rd_name)
    : ReducedDiagnostics<vdim, numspec, degx, degy, degz, degree, ndata>(rd_name)
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
                ofs << this->m_sep;
                ofs << "[" << c++ << "]KE" + std::string("(J)");
                ofs << std::endl;
                // close file
                ofs.close();
            }
        }
    }
    // end constructor

    /**
     * This function computes the Momentum (Particle momentum) and Particle Kinetic energy
     */
    void ComputeDiags(Gempic::CompDom::computational_domain& infra, std::map<std::string, amrex::MultiFab*>& fieldsDiagnostics, 
                        amrex::GpuArray<std::shared_ptr<Gempic::Particles::particle_groups<vdim, ndata>>, numspec> &part_gr) override final
    {   
        constexpr int noutputs = 4;
        this->m_data.resize(noutputs,0);
        for (int i = 0; i < noutputs; i++)
        {
            this->m_data[i] = 0.0;
        }
        
        for (int spec = 0; spec < numspec; spec++)
        {
            amrex::ReduceOps<amrex::ReduceOpSum, amrex::ReduceOpSum, amrex::ReduceOpSum, amrex::ReduceOpSum> reduce_ops;
            auto particle_tmp =
                amrex::ParticleReduce<amrex::ReduceData<amrex::Real, amrex::Real, amrex::Real, amrex::Real>>(
                    *part_gr[spec],
                    [=] AMREX_GPU_HOST_DEVICE(
                        const amrex::ConstParticleTileData<amrex::Particle<0, 0>, vdim + ndata, 0> & pt, const int pp) -> amrex::GpuTuple<amrex::Real, amrex::Real, amrex::Real, amrex::Real>
                    {
                        auto weight = pt.m_rdata[vdim][pp];
                        auto velx = pt.m_rdata[xDir][pp];
                        auto vely = pt.m_rdata[yDir][pp];
                        auto velz = pt.m_rdata[zDir][pp];
                        auto energy = 0.5 * (velx*velx + vely*vely + velz*velz);
                        return {weight * velx, weight * vely, weight * velz, weight * energy};
                    },
                    reduce_ops);

            amrex::Real Px = amrex::get<xDir>(particle_tmp);
            amrex::Real Py = amrex::get<yDir>(particle_tmp);
            amrex::Real Pz = amrex::get<zDir>(particle_tmp);
            amrex::Real PE = amrex::get<3>(particle_tmp);

            // reduced sum over mpi ranks
            amrex::ParallelDescriptor::ReduceRealSum(
                {Px, Py, Pz, PE}, amrex::ParallelDescriptor::IOProcessorNumber());

            // save results for this species i_s into m_data
            this->m_data[xDir] += part_gr[spec]->getMass() * Px;
            this->m_data[yDir] += part_gr[spec]->getMass() * Py;
            this->m_data[zDir] += part_gr[spec]->getMass() * Pz;
            this->m_data[3] += part_gr[spec]->getMass() * PE;

        }
    }
    // end void Particle::ComputeDiags

};

#endif
