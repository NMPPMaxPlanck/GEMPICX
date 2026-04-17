/**************************************************************************************************
 * Copyright (c) 2021 GEMPICX                                                                     *
 * SPDX-License-Identifier: BSD-3-Clause                                                          *
 **************************************************************************************************/

#ifndef REDUCEDDIAGS_PARTICLE_H_
#define REDUCEDDIAGS_PARTICLE_H_

#include <string>

#include <AMReX_Particle.H>
#include <AMReX_Reduce.H>

#include "GEMPIC_ComputationalDomain.H"
#include "GEMPIC_FunctionParse.H"
#include "GEMPIC_GempicNorm.H"
#include "GEMPIC_Particle.H"
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
        ComputationalDomain const& /*infra*/,
        std::vector<std::shared_ptr<Gempic::Particle::ParticleSpecies<vdim, ndata>>> const&
            particles) final
    {
        BL_PROFILE("Gempic::Io::Particle::compute_diags()");
        constexpr int noutputs = 4;
        this->m_data.assign(noutputs, 0.0);

        for (auto& particleSpecies : particles)
        {
            amrex::ReduceOps<amrex::ReduceOpSum, amrex::ReduceOpSum, amrex::ReduceOpSum,
                             amrex::ReduceOpSum>
                reduceOps;
            using PTDType = typename Gempic::Particle::ParticleSpecies<
                vdim, ndata>::ParticleTileType::ConstParticleTileDataType;
            auto const ii = particleSpecies->get_data_indices();
            auto particleTmp = amrex::ParticleReduce<
                amrex::ReduceData<amrex::Real, amrex::Real, amrex::Real, amrex::Real>>(
                *particleSpecies, 0, particleSpecies->finestLevel(),
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
        ComputationalDomain const& /*infra*/,
        std::vector<std::shared_ptr<Gempic::Particle::ParticleSpecies<vdim, ndata>>> const&
            particles) final
    {
        constexpr int noutputs = 2;
        this->m_data.assign(noutputs, 0.0);

        std::shared_ptr<Gempic::Particle::ParticleSpeciesLinVlasov<vdim, ndata>> particlesLinVlasov;

        for (unsigned int spec = 0; spec < particles.size(); spec++)
        {
            particlesLinVlasov =
                std::dynamic_pointer_cast<Gempic::Particle::ParticleSpeciesLinVlasov<vdim, ndata>>(
                    particles[spec]);
            amrex::Real vThermalBackground = particlesLinVlasov->get_v_thermal_background();
            amrex::ReduceOps<amrex::ReduceOpSum, amrex::ReduceOpSum> reduceOps;
            using PTDType = typename Gempic::Particle::ParticleSpecies<
                vdim, ndata>::ParticleTileType::ConstParticleTileDataType;
            auto const ii = particles[spec]->get_data_indices();
            auto particleTmp = amrex::ParticleReduce<amrex::ReduceData<amrex::Real, amrex::Real>>(
                *particles[spec], 0, particles[spec]->finestLevel(),
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
                particles[spec]->get_mass() * vThermalBackground * vThermalBackground * ew;
            this->m_data[1] += particles[spec]->get_mass() * ekin;
        }
    }
    // end void Particle::ComputeDiags
};

/**
 *  This class mainly contains a function that
 *  computes the Kinetic Energy for the relativistic Model.
 */
template <int vdim, int degx, int degy, int degz, int ndata>
class ParticleRelativistic : public ReducedDiagnostics<vdim, degx, degy, degz, ndata>
{
public:
    /**
     * constructor
     * @param[in] rdName reduced diags names
     * @param[in] indexNum the index of the print field in the header
     * @param[in] isLastType is this the last type in the group?
     * @param[in] fileName default file name suggested by MultiReducedDiagnostics
     */
    ParticleRelativistic(std::string const& rdName,
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
        ComputationalDomain const& /*infra*/,
        std::vector<std::shared_ptr<Gempic::Particle::ParticleSpecies<vdim, ndata>>> const&
            particles) final
    {
        BL_PROFILE("Gempic::Io::Particle::compute_diags()");
        constexpr int noutputs = 4;
        this->m_data.assign(noutputs, 0.0);

        Io::Parameters parameters{};
        amrex::Real c{1.0};
        // if nothing else is defined in the input file, the speed of light is set to one
        parameters.get_or_set("c", c);

        for (auto& particleSpecies : particles)
        {
            amrex::ReduceOps<amrex::ReduceOpSum, amrex::ReduceOpSum, amrex::ReduceOpSum,
                             amrex::ReduceOpSum>
                reduceOps;

            using PTDType = typename Gempic::Particle::ParticleSpecies<
                vdim, ndata>::ParticleTileType::ConstParticleTileDataType;
            auto const ii = particleSpecies->get_data_indices();
            auto particleTmp = amrex::ParticleReduce<
                amrex::ReduceData<amrex::Real, amrex::Real, amrex::Real, amrex::Real>>(
                *particleSpecies, 0, particleSpecies->finestLevel(),
                [=] AMREX_GPU_HOST_DEVICE(PTDType const& pt, int const pp)

                    -> amrex::GpuTuple<amrex::Real, amrex::Real, amrex::Real, amrex::Real>
                {
                    auto const weight = pt.rdata(ii.m_iweight)[pp];
                    auto const ux = pt.rdata(ii.m_ivelx)[pp];
                    auto const uy = pt.rdata(ii.m_ively)[pp];
                    auto const uz = pt.rdata(ii.m_ivelz)[pp];
                    amrex::Real gamma = sqrt(1 + (ux * ux + uy * uy + uz * uz) /
                                                     (c * c)); // Lorentz factor: sqrt(1+u^2/c^2)

                    auto const energy = (gamma - 1) * c * c;
                    return {weight * ux / gamma, weight * uy / gamma, weight * uz / gamma,
                            weight * energy};
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
 *  computes the relativeRMS error of different particle quantities
 *  base on their variance and the number of particles.
 */
template <int vdim, int degx, int degy, int degz, int ndata>
class ParticleVariance : public ReducedDiagnostics<vdim, degx, degy, degz, ndata>
{
private:
    std::string m_speciesName; // index of the particle species for which the variance is computed
public:
    /**
     * constructor
     * @param[in] speciesName name of particle species
     * @param[in] rdName reduced diags names
     * @param[in] indexNum the index of the print field in the header
     * @param[in] isLastType is this the last type in the group?
     * @param[in] fileName default file name suggested by MultiReducedDiagnostics
     */
    ParticleVariance(std::string const& speciesName,
                     std::string const& rdName,
                     int& indexNum,
                     bool isLastType,
                     std::string const& fileName) :
        ReducedDiagnostics<vdim, degx, degy, degz, ndata>(rdName, indexNum, isLastType, fileName),
        m_speciesName{speciesName}
    {
        BL_PROFILE("Gempic::Io::ParticleVariance::ParticleVariance()");
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
                    ofs << "[" << ++indexNum << "]time";
                }
                ofs << this->m_sep;
                ofs << "[" << ++indexNum << "]varN" << speciesName;
                ofs << this->m_sep;
                ofs << "[" << ++indexNum << "]varNuX" << speciesName;
                ofs << this->m_sep;
                ofs << "[" << ++indexNum << "]varNuY" << speciesName;
                ofs << this->m_sep;
                ofs << "[" << ++indexNum << "]varNuZ" << speciesName;
                ofs << this->m_sep;
                ofs << "[" << ++indexNum << "]varKinX" << speciesName;
                ofs << this->m_sep;
                ofs << "[" << ++indexNum << "]varKinY" << speciesName;
                ofs << this->m_sep;
                ofs << "[" << ++indexNum << "]varKinZ" << speciesName;
                ofs << this->m_sep;
                // variance of the first mode of the density in Fourier space
                ofs << "[" << ++indexNum << "]varNmode1" << speciesName;
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
     * This function computes the variance of different particle quantities, and saves the results
     * in m_data
     */
    void compute_diags (
        ComputationalDomain const& infra,
        std::vector<std::shared_ptr<Gempic::Particle::ParticleSpecies<vdim, ndata>>> const&
            particles) final
    {
        BL_PROFILE("Gempic::Io::ParticleVariance::compute_diags()");
        constexpr int noutputs = 8; // number of outputs for this reduced diagnostic: varN, varNuX,
                                    // varNuY, varNuZ, varKinX, varKinY, varKinZ, varNmode1
        this->m_data.assign(noutputs, 0.0);

        std::shared_ptr<Gempic::Particle::ParticleSpecies<vdim, ndata>> particleSpecies;
        std::string speciesNames = "(options: {";
        for (auto& species : particles)
        {
            if (m_speciesName == species->get_name())
            {
                particleSpecies = species;
                break;
            }
            speciesNames += " '" + species->get_name() + "' ";
        }
        if (not particleSpecies)
        {
            speciesNames += "})";
            std::string msg = "Unknown particle name '" + m_speciesName +
                              "' passed to Variance particle diagnostics " + speciesNames;
            GEMPIC_ERROR(msg);
        }

        amrex::ReduceOps<
            amrex::ReduceOpSum, amrex::ReduceOpSum, amrex::ReduceOpSum, amrex::ReduceOpSum,
            amrex::ReduceOpSum, amrex::ReduceOpSum, amrex::ReduceOpSum, amrex::ReduceOpSum,
            amrex::ReduceOpSum, amrex::ReduceOpSum, amrex::ReduceOpSum, amrex::ReduceOpSum,
            amrex::ReduceOpSum, amrex::ReduceOpSum, amrex::ReduceOpSum, amrex::ReduceOpSum>
            reduceOps;

        using PTDType = typename Gempic::Particle::ParticleSpecies<
            vdim, ndata>::ParticleTileType::ConstParticleTileDataType;
        auto const ii = particleSpecies->get_data_indices();
        long const np = particleSpecies->TotalNumberOfParticles();
        amrex::Real const kx =
            2.0 * M_PI / (infra.geometry().ProbHi(xDir) - infra.geometry().ProbLo(xDir));

        auto particleTmp = amrex::ParticleReduce<amrex::ReduceData<
            amrex::Real, amrex::Real, amrex::Real, amrex::Real, amrex::Real, amrex::Real,
            amrex::Real, amrex::Real, amrex::Real, amrex::Real, amrex::Real, amrex::Real,
            amrex::Real, amrex::Real, amrex::Real, amrex::Real>>(
            *particleSpecies, 0, particleSpecies->finestLevel(),
            [=] AMREX_GPU_HOST_DEVICE(PTDType const& pt, int const pp)
                -> amrex::GpuTuple<amrex::Real, amrex::Real, amrex::Real, amrex::Real, amrex::Real,
                                   amrex::Real, amrex::Real, amrex::Real, amrex::Real, amrex::Real,
                                   amrex::Real, amrex::Real, amrex::Real, amrex::Real, amrex::Real,
                                   amrex::Real>
            {
                // The weight contains the 1/np factor. This needs to be multiplied back to get the
                // correct variance value.
                auto const weight = pt.rdata(ii.m_iweight)[pp] * np;
                auto const velx = pt.rdata(ii.m_ivelx)[pp];
                auto const vely = pt.rdata(ii.m_ively)[pp];
                auto const velz = pt.rdata(ii.m_ivelz)[pp];
                auto const x = pt.rdata(ii.m_iposx)[pp];
                auto const mode1 = cos(kx * x); // first mode of the density in Fourier space
                return {weight,
                        weight * weight,
                        weight * velx,
                        (weight * velx) * (weight * velx),
                        weight * vely,
                        (weight * vely) * (weight * vely),
                        weight * velz,
                        (weight * velz) * (weight * velz),
                        weight * velx * velx,
                        (weight * velx * velx) * (weight * velx * velx),
                        weight * vely * vely,
                        (weight * vely * vely) * (weight * vely * vely),
                        weight * velz * velz,
                        (weight * velz * velz) * (weight * velz * velz),
                        weight * mode1,
                        (weight * mode1) * (weight * mode1)};
            },
            reduceOps);

        amrex::Real N = amrex::get<0>(particleTmp);
        amrex::Real n2 = amrex::get<1>(particleTmp);
        amrex::Real nuX = amrex::get<2>(particleTmp);
        amrex::Real nuX2 = amrex::get<3>(particleTmp);
        amrex::Real nuY = amrex::get<4>(particleTmp);
        amrex::Real nuY2 = amrex::get<5>(particleTmp);
        amrex::Real nuZ = amrex::get<6>(particleTmp);
        amrex::Real nuZ2 = amrex::get<7>(particleTmp);
        amrex::Real nuXsq = amrex::get<8>(particleTmp);
        amrex::Real nuXsq2 = amrex::get<9>(particleTmp);
        amrex::Real nuYsq = amrex::get<10>(particleTmp);
        amrex::Real nuYsq2 = amrex::get<11>(particleTmp);
        amrex::Real nuZsq = amrex::get<12>(particleTmp);
        amrex::Real nuZsq2 = amrex::get<13>(particleTmp);
        amrex::Real nmode1 = amrex::get<14>(particleTmp);
        amrex::Real nmode12 = amrex::get<15>(particleTmp);

        // reduced sum over mpi ranks
        amrex::ParallelDescriptor::ReduceRealSum(
            {N, n2, nuX, nuX2, nuY, nuY2, nuZ, nuZ2, nuXsq, nuXsq2, nuYsq, nuYsq2, nuZsq, nuZsq2,
             nmode1, nmode12},
            amrex::ParallelDescriptor::IOProcessorNumber());

        // save results for this species i_s into m_data
        // RMS error = sqrt(Unbiased empirical variance/ \int_f0)/sqrt(np)
        // the error is normalized by \int f_0 = sum weights = N, and the sqrt(np) factor is because
        // the variance computed here is the variance of the sum of particle quantities, while what
        // we want is the variance of the average (sum/np).
        this->m_data[0] = sqrt(((n2 - N * N / np) / (np - 1)) / N); // varN
        this->m_data[1] = sqrt((nuX2 - nuX * nuX / np) / (np - 1) / N);
        this->m_data[2] = sqrt((nuY2 - nuY * nuY / np) / (np - 1) / N);
        this->m_data[3] = sqrt((nuZ2 - nuZ * nuZ / np) / (np - 1) / N);
        this->m_data[4] = sqrt((nuXsq2 - nuXsq * nuXsq / np) / (np - 1) / N);
        this->m_data[5] = sqrt((nuYsq2 - nuYsq * nuYsq / np) / (np - 1) / N);
        this->m_data[6] = sqrt((nuZsq2 - nuZsq * nuZsq / np) / (np - 1) / N);
        this->m_data[7] = sqrt((nmode12 - nmode1 * nmode1 / np) / (np - 1) / N);
    }
    // end void ParticleVariance::ComputeDiags
};

} // namespace Gempic::Io

#endif
