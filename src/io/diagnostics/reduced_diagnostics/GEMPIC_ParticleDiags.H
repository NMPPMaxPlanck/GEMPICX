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

#include <string>

#include <AMReX_Particle.H>
#include <AMReX_Reduce.H>

#include "GEMPIC_ComputationalDomain.H"
#include "GEMPIC_GempicNorm.H"
#include "GEMPIC_ParticleGroups.H"
#include "GEMPIC_Profiling.H"
#include "GEMPIC_ReducedDiagnostics.H"

namespace Gempic::Io
{
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
     * @param[in] rdName reduced diags names
     */
    Particle(std::string rdName) :
        ReducedDiagnostics<vdim, numspec, degx, degy, degz, degree, ndata>(rdName)
    {
        if (amrex::ParallelDescriptor::IOProcessor())
        {
            if (this->m_isNotRestart)
            {
                // open file
                std::ofstream ofs{this->m_path + this->m_rdName + "." + this->m_extension,
                                  std::ofstream::out};
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
    void compute_diags (
        ComputationalDomain& infra,
        std::map<std::string, amrex::MultiFab*>& fieldsDiagnostics,
        amrex::GpuArray<std::shared_ptr<Gempic::Particle::ParticleGroups<vdim, ndata>>, numspec>&
            partGr) final
    {
        constexpr int noutputs = 4;
        this->m_data.resize(noutputs, 0);
        for (int i = 0; i < noutputs; i++)
        {
            this->m_data[i] = 0.0;
        }

        for (int spec = 0; spec < numspec; spec++)
        {
            amrex::ReduceOps<amrex::ReduceOpSum, amrex::ReduceOpSum, amrex::ReduceOpSum,
                             amrex::ReduceOpSum>
                reduceOps;
            auto particleTmp = amrex::ParticleReduce<
                amrex::ReduceData<amrex::Real, amrex::Real, amrex::Real, amrex::Real>>(
                *partGr[spec],
                [=] AMREX_GPU_HOST_DEVICE(
                    const amrex::ConstParticleTileData<amrex::Particle<0, 0>, vdim + ndata, 0>& pt,
                    const int pp)
                    -> amrex::GpuTuple<amrex::Real, amrex::Real, amrex::Real, amrex::Real>
                {
                    auto weight = pt.m_rdata[vdim][pp];
                    auto velx = pt.m_rdata[xDir][pp];
                    auto vely = pt.m_rdata[yDir][pp];
                    auto velz = pt.m_rdata[zDir][pp];
                    auto energy = 0.5 * (velx * velx + vely * vely + velz * velz);
                    return {weight * velx, weight * vely, weight * velz, weight * energy};
                },
                reduceOps);

            amrex::Real Px = amrex::get<xDir>(particleTmp);
            amrex::Real Py = amrex::get<yDir>(particleTmp);
            amrex::Real Pz = amrex::get<zDir>(particleTmp);
            amrex::Real pe = amrex::get<3>(particleTmp);

            // reduced sum over mpi ranks
            amrex::ParallelDescriptor::ReduceRealSum(
                {Px, Py, Pz, pe}, amrex::ParallelDescriptor::IOProcessorNumber());

            // save results for this species i_s into m_data
            this->m_data[xDir] += partGr[spec]->get_mass() * Px;
            this->m_data[yDir] += partGr[spec]->get_mass() * Py;
            this->m_data[zDir] += partGr[spec]->get_mass() * Pz;
            this->m_data[3] += partGr[spec]->get_mass() * pe;
        }
    }
    // end void Particle::ComputeDiags
};

}  // namespace Gempic::Io

#endif
