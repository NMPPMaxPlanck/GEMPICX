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
#include "GEMPIC_FunctionParse.H"
#include "GEMPIC_GempicNorm.H"
#include "GEMPIC_ParticleGroups.H"
#include "GEMPIC_ParticleUtils.H"
#include "GEMPIC_Profiling.H"
#include "GEMPIC_ReducedDiagnostics.H"

namespace Gempic::Io
{
/**
 *  This class mainly contains a function that
 *  computes the Momentum and Kinetic Energy.
 */
template <int vdim, int degx, int degy, int degz, int ndata>
class Particle : public ReducedDiagnostics<vdim, degx, degy, degz, ndata>
{
public:
    /**
     * constructor
     * @param[in] rdName reduced diags names
     * @param[in] indexNum the index of the print field in the header
     * @param[in] isLastType is this the last type in the group?
     * @param[in] fileName default file name suggested by MultiReducedDiagnostics
     */
    Particle(std::string const& rdName,
             int& indexNum,
             bool isLastType,
             std::string const& fileName) :
        ReducedDiagnostics<vdim, degx, degy, degz, ndata>(rdName, indexNum, isLastType, fileName)
    {
        BL_PROFILE("Gempic::Io::Particle::Particle()");
        if (amrex::ParallelDescriptor::IOProcessor())
        {
            if (this->m_isNotRestart)
            {
                // open file
                std::ofstream ofs{this->m_path, std::ofstream::out | std::ofstream::app};
                // write header row
                if (this->m_isFirstType)
                {
                    ofs << "#";
                    ofs << "[" << ++indexNum << "]step()";
                    ofs << this->m_sep;
                    ofs << "[" << ++indexNum << "]time(s)";
                }
                ofs << this->m_sep;
                ofs << "[" << ++indexNum << "]PX" + std::string("(Ns)");
                ofs << this->m_sep;
                ofs << "[" << ++indexNum << "]PY" + std::string("(Ns)");
                ofs << this->m_sep;
                ofs << "[" << ++indexNum << "]PZ" + std::string("(Ns)");
                ofs << this->m_sep;
                ofs << "[" << ++indexNum << "]KE" + std::string("(J)");
                if (this->m_isLastType)
                {
                    ofs << std::endl;
                }
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
        ComputationalDomain const& infra,
        std::vector<std::shared_ptr<Gempic::Particle::ParticleGroups<vdim, ndata>>> const& partGr)
        final
    {
        BL_PROFILE("Gempic::Io::Particle::compute_diags()");
        constexpr int noutputs = 4;
        this->m_data.assign(noutputs, 0.0);

        for (auto& particleSpecies : partGr)
        {
            amrex::ReduceOps<amrex::ReduceOpSum, amrex::ReduceOpSum, amrex::ReduceOpSum,
                             amrex::ReduceOpSum>
                reduceOps;
            using PTDType = typename Gempic::Particle::ParticleGroups<
                vdim + ndata, 0>::ParticleTileType::ConstParticleTileDataType;
            auto const ii = particleSpecies->get_data_indices();
            auto particleTmp = amrex::ParticleReduce<
                amrex::ReduceData<amrex::Real, amrex::Real, amrex::Real, amrex::Real>>(
                *particleSpecies,
                [=] AMREX_GPU_HOST_DEVICE(PTDType const& pt, int const pp)
                    -> amrex::GpuTuple<amrex::Real, amrex::Real, amrex::Real, amrex::Real>
                {
                    auto const weight = pt.rdata(ii.m_iweight)[pp];
                    auto const velx = pt.rdata(ii.m_ivelx)[pp];
                    auto const vely = pt.rdata(ii.m_ively)[pp];
                    auto const velz = pt.rdata(ii.m_ivelz)[pp];
                    auto const energy = 0.5 * (velx * velx + vely * vely + velz * velz);
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
            this->m_data[0] += particleSpecies->get_mass() * Px;
            this->m_data[1] += particleSpecies->get_mass() * Py;
            this->m_data[2] += particleSpecies->get_mass() * Pz;
            this->m_data[3] += particleSpecies->get_mass() * pe;
        }
    }
    // end void Particle::ComputeDiags
};

/**
 *  This class mainly contains a function that
 *  computes the Momentum and Kinetic Energy for LinVlasov Model.
 */
template <int vdim, int degx, int degy, int degz, int ndata>
class ParticleLinVlasov : public ReducedDiagnostics<vdim, degx, degy, degz, ndata>
{
public:
    /**
     * constructor
     * @param[in] rdName reduced diags names
     * @param[in] indexNum the index of the print field in the header
     * @param[in] isLastType is this the last type in the group?
     * @param[in] fileName default file name suggested by MultiReducedDiagnostics
     */
    ParticleLinVlasov(std::string const& rdName,
                      int& indexNum,
                      bool isLastType,
                      std::string const& fileName) :
        ReducedDiagnostics<vdim, degx, degy, degz, ndata>(rdName, indexNum, isLastType, fileName)
    {
        if (amrex::ParallelDescriptor::IOProcessor())
        {
            if (this->m_isNotRestart)
            {
                // open file
                std::ofstream ofs{this->m_path, std::ofstream::out | std::ofstream::app};
                // write header row
                if (this->m_isFirstType)
                {
                    ofs << "#";
                    ofs << "[" << ++indexNum << "]step()";
                    ofs << this->m_sep;
                    ofs << "[" << ++indexNum << "]time(s)";
                }
                ofs << this->m_sep;
                ofs << "[" << ++indexNum << "]KEW" + std::string("(J)");
                ofs << this->m_sep;
                ofs << "[" << ++indexNum << "]KEP" + std::string("(J)");
                if (this->m_isLastType)
                {
                    ofs << std::endl;
                }
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
        ComputationalDomain const& infra,
        std::vector<std::shared_ptr<Gempic::Particle::ParticleGroups<vdim, ndata>>> const& partGr)
        final
    {
        constexpr int noutputs = 2;
        this->m_data.assign(noutputs, 0.0);

        std::shared_ptr<Gempic::Particle::ParticleGroupsLinVlasov<vdim, ndata>> partGrLinVlasov;

        for (int spec = 0; spec < partGr.size(); spec++)
        {
            partGrLinVlasov =
                std::dynamic_pointer_cast<Gempic::Particle::ParticleGroupsLinVlasov<vdim, ndata>>(
                    partGr[spec]);
            amrex::Real vThermalBackground = partGrLinVlasov->get_v_thermal_background();
            amrex::ReduceOps<amrex::ReduceOpSum, amrex::ReduceOpSum> reduceOps;
            using PTDType = typename Gempic::Particle::ParticleGroups<
                vdim + ndata, 0>::ParticleTileType::ConstParticleTileDataType;
            auto const ii = partGr[spec]->get_data_indices();
            auto particleTmp = amrex::ParticleReduce<amrex::ReduceData<amrex::Real, amrex::Real>>(
                *partGr[spec],
                [=] AMREX_GPU_HOST_DEVICE(PTDType const& pt,
                                          int const pp) -> amrex::GpuTuple<amrex::Real, amrex::Real>
                {
                    auto const weight = pt.rdata(ii.m_iweight)[pp];
                    auto const velx = pt.rdata(ii.m_ivelx)[pp];
                    auto const vely = pt.rdata(ii.m_ively)[pp];
                    auto const velz = pt.rdata(ii.m_ivelz)[pp];
                    auto const s0 = pt.rdata(ii.m_is0)[pp];
                    auto const energyWeight = 0.5 * s0 * weight * weight;
                    auto const energyKin = 0.5 * (velx * velx + vely * vely + velz * velz);
                    return {energyWeight, energyKin};
                },
                reduceOps);

            amrex::Real ew = amrex::get<xDir>(particleTmp);
            amrex::Real ekin = amrex::get<yDir>(particleTmp);

            // reduced sum over mpi ranks
            amrex::ParallelDescriptor::ReduceRealSum(
                {ew, ekin}, amrex::ParallelDescriptor::IOProcessorNumber());

            // save results for this species i_s into m_data
            this->m_data[0] +=
                partGr[spec]->get_mass() * vThermalBackground * vThermalBackground * ew;
            this->m_data[1] += partGr[spec]->get_mass() * ekin;
        }
    }
    // end void Particle::ComputeDiags
};

} // namespace Gempic::Io

#endif
