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
const std::string defaultLevelPrefix{"Level_"};
}

/**
 * \brief This class aims at dumping diags data to disk using the AMReX Plotfile format.
 * In particular, function WriteToFile takes fields and particles as input arguments,
 * and writes data to file.
 */
template <int vdim, int numspec, int ndata>
class FlushFormatPlotfile : public FlushFormat<vdim, numspec, ndata>
{
public:
    FlushFormatPlotfile() {}

    /** Flush fields and particles to plotfile */
    void write_to_file (
        const amrex::Vector<std::string> varnames,
        const amrex::MultiFab& mf,
        amrex::Geometry& geom,
        const int iteration,
        const double time,
        amrex::GpuArray<std::shared_ptr<Gempic::Particle::ParticleGroups<vdim, ndata>>, numspec>
            particles,
        int nlev,
        const std::string prefix,
        int fileMinDigits,
        bool plotParticles,
        int bufferID = 1,
        int numBuffers = 1) const override
    {
        BL_PROFILE("FlushFormatPlotfile::WriteToFile()");
        std::string filename{amrex::Concatenate(prefix, iteration, fileMinDigits)};
        const amrex::VisMF::Header::Version currentVersion = amrex::VisMF::GetHeaderVersion();

        amrex::Print() << "Writing plotfile " + filename << "\n";

        int levelStep{1};
        amrex::WriteSingleLevelPlotfile(filename, mf, varnames, geom,
                                        static_cast<amrex::Real>(time), levelStep);

        if (plotParticles)
        {
            // Particle Info -------------------------------------------------------------
            const amrex::Vector<int>& writeIntComp = {};
            const amrex::Vector<std::string>& intCompNames = {};
            const amrex::Vector<int> writeRealComp(vdim + ndata, 1);
            amrex::Vector<std::string> realCompNames = {"vx", "vy", "vz", "weight"};
            if constexpr (vdim < 3)
            {
                realCompNames.resize(vdim + ndata);
                if constexpr (vdim + ndata > 0)
                {
                    realCompNames[vdim + ndata - 1] = "weight";
                }
            }
            for (int idx = 0; idx < numspec; idx++)
            {
                particles[idx]->WritePlotFile(filename, particles[idx]->get_name(), writeRealComp,
                                              writeIntComp, realCompNames, intCompNames);
            }
        }
        write_job_info(filename);

        write_warp_x_header(filename, geom);

        amrex::VisMF::SetHeaderVersion(currentVersion);
    }

    /** Write general info of the run into the plotfile */
    void write_job_info (const std::string& dir) const
    {
        if (amrex::ParallelDescriptor::IOProcessor())
        {
            // job_info file with details about the run
            std::ofstream jobInfoFile;
            std::string fullPathJobInfoFile = dir;

            const std::string prettyLine = std::string(78, '=') + "\n";
            //        std::string OtherLine = std::string(78, '-') + "\n";
            //        std::string SkipSpace = std::string(8, ' ') + "\n";

            fullPathJobInfoFile += "/gempic_job_info";
            jobInfoFile.open(fullPathJobInfoFile.c_str());

            // job information
            jobInfoFile << prettyLine;
            jobInfoFile << " WarpX Job Information\n";
            jobInfoFile << prettyLine;

            jobInfoFile << "number of MPI processes: " << amrex::ParallelDescriptor::NProcs()
                        << "\n";

            jobInfoFile << "\n\n";
            /*
                    // build information
                    jobInfoFile << PrettyLine;
                    jobInfoFile << " Build Information\n";
                    jobInfoFile << PrettyLine;

                    jobInfoFile << "build date:    " << buildInfoGetBuildDate() << "\n";
                    jobInfoFile << "build machine: " << buildInfoGetBuildMachine() << "\n";
                    jobInfoFile << "build dir:     " << buildInfoGetBuildDir() << "\n";
                    jobInfoFile << "AMReX dir:     " << buildInfoGetAMReXDir() << "\n";

                    jobInfoFile << "\n";

                    jobInfoFile << "COMP:          " << buildInfoGetComp() << "\n";
                    jobInfoFile << "COMP version:  " << buildInfoGetCompVersion() << "\n";

                    jobInfoFile << "\n";

                    jobInfoFile << "C++ compiler:  " << buildInfoGetCXXName() << "\n";
                    jobInfoFile << "C++ flags:     " << buildInfoGetCXXFlags() << "\n";

                    jobInfoFile << "\n";

                    jobInfoFile << "Fortran comp:  " << buildInfoGetFName() << "\n";
                    jobInfoFile << "Fortran flags: " << buildInfoGetFFlags() << "\n";

                    jobInfoFile << "\n";

                    jobInfoFile << "Link flags:    " << buildInfoGetLinkFlags() << "\n";
                    jobInfoFile << "Libraries:     " << buildInfoGetLibraries() << "\n";

                    jobInfoFile << "\n";

                    const char* githash1 = buildInfoGetGitHash(1);
                    const char* githash2 = buildInfoGetGitHash(2);
                    const char* githash3 = buildInfoGetGitHash(3);
                    if (strlen(githash1) > 0) {
                        jobInfoFile << "WarpX  git describe: " << githash1 << "\n";
                    }
                    if (strlen(githash2) > 0) {
                        jobInfoFile << "AMReX  git describe: " << githash2 << "\n";
                    }
                    if (strlen(githash3) > 0) {
                        jobInfoFile << "PICSAR git describe: " << githash3 << "\n";
                    }
            */
            jobInfoFile << "\n\n";

            // grid information
            /*
                    jobInfoFile << PrettyLine;
                    jobInfoFile << " Grid Information\n";
                    jobInfoFile << PrettyLine;

                    for (int i = 0; i <= 0; i++)
                    {
                        jobInfoFile << " level: " << i << "\n";
                        jobInfoFile << "   number of boxes = " << warpx.boxArray(i).size() << "\n";
                        jobInfoFile << "   maximum zones   = ";
                        for (int n = 0; n < AMREX_SPACEDIM; n++)
                        {
                            jobInfoFile << warpx.Geom(i).Domain().length(n) << " ";
                        }
                        jobInfoFile << "\n\n";
                    }
            */
            jobInfoFile << " Boundary conditions\n";

            AMREX_D_TERM(jobInfoFile << "   -x: "
                                     << "interior"
                                     << "\n";
                         jobInfoFile << "   +x: "
                                     << "interior"
                                     << "\n";
                         , jobInfoFile << "   -y: "
                                       << "interior"
                                       << "\n";
                         jobInfoFile << "   +y: "
                                     << "interior"
                                     << "\n";
                         , jobInfoFile << "   -z: "
                                       << "interior"
                                       << "\n";
                         jobInfoFile << "   +z: "
                                     << "interior"
                                     << "\n";)

            jobInfoFile << "\n\n";

            // runtime parameters
            jobInfoFile << prettyLine;
            jobInfoFile << " Inputs File Parameters\n";
            jobInfoFile << prettyLine;

            // amrex::ParmParse::dumpTable(jobInfoFile, true);

            jobInfoFile.close();
        }
    }

    /** Write WarpX-specific plotfile header */
    void write_warp_x_header (const std::string& name, amrex::Geometry& geom) const
    {
        if (amrex::ParallelDescriptor::IOProcessor())
        {
            amrex::VisMF::IO_Buffer ioBuffer(amrex::VisMF::IO_Buffer_Size);
            std::ofstream headerFile;
            headerFile.rdbuf()->pubsetbuf(ioBuffer.dataPtr(), ioBuffer.size());
            const std::string headerFileName(name + "/GEMPICXHeader");
            headerFile.open(headerFileName.c_str(),
                            std::ofstream::out | std::ofstream::trunc | std::ofstream::binary);
            if (!headerFile.good())
            {
                amrex::FileOpenFailed(headerFileName);
            }

            headerFile.precision(17);

            headerFile << "Checkpoint version: 1\n";
            /*
                    const int nlevels = warpx.finestLevel()+1;
                    HeaderFile << nlevels << "\n";

                    for (int i = 0; i < warpx.getistep().size(); ++i) {
                        HeaderFile << warpx.getistep(i) << " ";
                    }
                    HeaderFile << "\n";

                    for (int i = 0; i < warpx.getnsubsteps().size(); ++i) {
                        HeaderFile << warpx.getnsubsteps(i) << " ";
                    }
                    HeaderFile << "\n";

                    for (int i = 0; i < warpx.gett_new().size(); ++i) {
                        HeaderFile << warpx.gett_new(i) << " ";
                    }
                    HeaderFile << "\n";

                    for (int i = 0; i < warpx.gett_old().size(); ++i) {
                        HeaderFile << warpx.gett_old(i) << " ";
                    }
                    HeaderFile << "\n";

                    for (int i = 0; i < warpx.getdt().size(); ++i) {
                        HeaderFile << warpx.getdt(i) << " ";
                    }
                    HeaderFile << "\n";

                    HeaderFile << warpx.getmoving_window_x() << "\n";

                    HeaderFile << warpx.getis_synchronized() << "\n";

                    // Geometry
                    for (int i = 0; i < AMREX_SPACEDIM; ++i) {
                        HeaderFile << geom[0].ProbLo(i) << ' ';
                    }
                    HeaderFile << '\n';
                    for (int i = 0; i < AMREX_SPACEDIM; ++i) {
                        HeaderFile << geom[0].ProbHi(i) << ' ';
                    }
                    HeaderFile << '\n';

                    // BoxArray
                    for (int lev = 0; lev < nlevels; ++lev) {
                        warpx.boxArray(lev).writeOn(HeaderFile);
                        HeaderFile << '\n';
                    }

                    //warpx.GetPartContainer().WriteHeader(HeaderFile);

                    //HeaderFile << warpx.getdo_moving_window() << "\n";

                    //HeaderFile << warpx.time_of_last_gal_shift << "\n";
                */
        }
    }

    ~FlushFormatPlotfile() override {}
};

#endif  // GEMPICX_FLUSHFORMATPLOTFILE_H_
