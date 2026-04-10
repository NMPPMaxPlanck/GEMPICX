/**************************************************************************************************
 * Copyright (c) 2021 GEMPICX                                                                     *
 * SPDX-License-Identifier: BSD-3-Clause                                                          *
 **************************************************************************************************/
#ifndef GEMPICX_FLUSHFORMATPLOTFILE_H_
#define GEMPICX_FLUSHFORMATPLOTFILE_H_

#include <fstream>
#include <memory>

#include <AMReX.H>
#include <AMReX_Config.H>
#include <AMReX_MultiFab.H>
#include <AMReX_ParallelDescriptor.H>
#include <AMReX_PlotFileUtil.H>
#include <AMReX_Print.H>
#include <AMReX_REAL.H>
#include <AMReX_VisMF.H>

#include "GEMPIC_FlushFormat.H"
#include "GEMPIC_Profiling.H"

namespace
{
std::string const defaultLevelPrefix{"Level_"};
}

/**
 * \brief This class aims at dumping diags data to disk using the AMReX Plotfile format.
 * In particular, function WriteToFile takes fields and particles as input arguments,
 * and writes data to file.
 */
template <int vdim, int ndata>
class FlushFormatPlotfile : public FlushFormat<vdim, ndata>
{
public:
    FlushFormatPlotfile() {}

    /** Flush fields and particles to plotfile */
    void write_to_file (
        amrex::Vector<std::string> const& varnames,
        amrex::MultiFab const& mf,
        amrex::Geometry& geom,
        int const iteration,
        double const time,
        std::vector<std::shared_ptr<Gempic::Particle::ParticleSpecies<vdim, ndata>>> const&
            particles,
        int /*nlev*/,
        std::string const& prefix,
        int fileMinDigits,
        bool plotParticles,
        int /*bufferID = 1*/,
        int /*numBuffers = 1*/) const override
    {
        BL_PROFILE("FlushFormatPlotfile::write_to_file()");
        std::string filename{amrex::Concatenate(prefix, iteration, fileMinDigits)};
        amrex::VisMF::Header::Version const currentVersion = amrex::VisMF::GetHeaderVersion();

        amrex::Print() << "Writing plotfile " + filename << "\n";

        int levelStep{1};
        amrex::WriteSingleLevelPlotfile(filename, mf, varnames, geom,
                                        static_cast<amrex::Real>(time), levelStep);

        if (plotParticles)
        {
            // Particle Info -------------------------------------------------------------
            amrex::Vector<int> const& writeIntComp = {};
            amrex::Vector<std::string> const& intCompNames = {};
            amrex::Vector<int> const writeRealComp(vdim + ndata, 1);

            for (auto& particleSpecies : particles)
            {
                particleSpecies->WritePlotFile(filename, particleSpecies->get_name(), writeRealComp,
                                               writeIntComp, particleSpecies->get_comp_names(),
                                               intCompNames);
            }
        }
        write_job_info(filename, geom);

        amrex::VisMF::SetHeaderVersion(currentVersion);
    }

    /** Write general info of the run into the plotfile */
    void write_job_info (std::string const& dir, amrex::Geometry const& geom) const
    {
        BL_PROFILE("FlushFormatPlotfile::write_job_info()");
        if (amrex::ParallelDescriptor::IOProcessor())
        {
            // job_info file with details about the run
            std::ofstream jobInfoFile;
            std::string fullPathJobInfoFile = dir;

            std::string const prettyLine = std::string(78, '=') + "\n";

            fullPathJobInfoFile += "/gempic_job_info";
            jobInfoFile.open(fullPathJobInfoFile.c_str());

            // job information
            jobInfoFile << prettyLine;
            jobInfoFile << " GEMPICX Job Information\n";
            jobInfoFile << prettyLine;

            jobInfoFile << "number of MPI processes: " << amrex::ParallelDescriptor::NProcs()
                        << "\n";

            jobInfoFile << "\n\n";

            // grid information
            jobInfoFile << " Boundary conditions\n";

            std::array<std::string, 2> bcString = {"Non-periodic", "Periodic"};
            // versions <18 and 18 of clang-format disagree on the formatting here
            // clang-format off
            AMREX_D_TERM(jobInfoFile << "   -x: " << bcString[geom.isPeriodic(xDir)] << "\n";
                         jobInfoFile << "   +x: " << bcString[geom.isPeriodic(xDir)] << "\n";
                       , jobInfoFile << "   -y: " << bcString[geom.isPeriodic(yDir)] << "\n";
                         jobInfoFile << "   +y: " << bcString[geom.isPeriodic(yDir)] << "\n";
                       , jobInfoFile << "   -z: " << bcString[geom.isPeriodic(zDir)] << "\n";
                         jobInfoFile << "   +z: " << bcString[geom.isPeriodic(zDir)] << "\n";)
            // clang-format on

            jobInfoFile << "\n\n";

            // runtime parameters
            jobInfoFile << prettyLine;
            jobInfoFile << " Input File Parameters\n";
            jobInfoFile << prettyLine;

            amrex::ParmParse::dumpTable(jobInfoFile, true);

            jobInfoFile.close();
        }
    }

    ~FlushFormatPlotfile() override {}
};

#endif // GEMPICX_FLUSHFORMATPLOTFILE_H_
