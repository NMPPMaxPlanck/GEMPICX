#include <AMReX.H>
#include <AMReX_Box.H>
#include <AMReX_BoxArray.H>
#include <AMReX_CoordSys.H>
#include <AMReX_DistributionMapping.H>
#include <AMReX_FPC.H>
#include <AMReX_Geometry.H>
#include <AMReX_IntVect.H>
#include <AMReX_ParallelDescriptor.H>
#include <AMReX_PlotFileUtil.H>
#include <AMReX_Print.H>
#include <AMReX_REAL.H>
#include <AMReX_RealBox.H>
#include <AMReX_Utility.H>

#include <GEMPIC_infrastructure.H>
#include <GEMPIC_maxwell_yee.H>
#include <GEMPIC_particle_groups.H>

using namespace Field_solvers;
using namespace Infra;
using namespace Particles;

#ifndef CHECKPOINT_H_
#define CHECKPOINT_H_

namespace Gempic {
namespace Utils {

// utility to skip to next line in Header
void
Gempic_GotoNextLine (std::istream& is)
{
    constexpr std::streamsize bl_ignore_max { 100000 };
    is.ignore(bl_ignore_max, '\n');
}

void
Gempic_WriteCheckpointFile (maxwell_yee *mw_yee, infrastructure *infra,
                            const std::string& dirname, const int step,
                            int header_precision)
{
    // dirnamestep            write a checkpoint file with this root directory
    // dirnamestep/Header     this contains information you need to save (e.g., finest_level, t_new, etc.) and also
    //                        the BoxArrays at each level
    // dirnamestep/Level_0/
    // dirnamestep/Level_1/
    // etc.                   these subdirectories will hold the MultiFab data at each level of refinement

    const std::string& checkpointname = amrex::Concatenate(dirname,step);

    amrex::Print() << "Writing checkpoint " << checkpointname << "\n";

    // ---- prebuild a hierarchy of directories
    // ---- dirName is built first.  if dirName exists, it is renamed.  then build
    // ---- dirName/subDirPrefix_0 .. dirName/subDirPrefix_nlevels-1
    // ---- if callBarrier is true, call ParallelDescriptor::Barrier()
    // ---- after all directories are built
    // ---- ParallelDescriptor::IOProcessor() creates the directories
    const int nlevels = 1;
    amrex::PreBuildDirectorHierarchy(checkpointname, "Level_", nlevels, true);

    // write Header file
    if (amrex::ParallelDescriptor::IOProcessor()) {

        std::string HeaderFileName(checkpointname + "/Header");
        amrex::VisMF::IO_Buffer io_buffer(amrex::VisMF::IO_Buffer_Size);
        std::ofstream HeaderFile;
        HeaderFile.rdbuf()->pubsetbuf(io_buffer.dataPtr(), io_buffer.size());
        HeaderFile.open(HeaderFileName.c_str(), std::ofstream::out   |
                        std::ofstream::trunc |
                        std::ofstream::binary);
        if( ! HeaderFile.good()) {
            amrex::FileOpenFailed(HeaderFileName);
        }

        HeaderFile.precision(header_precision);

        // write out title line
        HeaderFile << "Checkpoint file\n";

        // write out finest_level
        HeaderFile << nlevels-1 << "\n";

        // write out array of step (only one level so only one entry)
        HeaderFile << step << " ";
        HeaderFile << "\n";

        // write out array of dt (only one level so only one entry)
        HeaderFile << mw_yee->dt << " ";
        HeaderFile << "\n";

        // write out array of t_new (only one level so only one entry)
        HeaderFile << (mw_yee->dt)*step << " ";
        HeaderFile << "\n";

        // write the BoxArray at each level
        amrex::BoxArray boxArray(infra->grid, *(mw_yee->E_Index[0]));
        boxArray.writeOn(HeaderFile);
        HeaderFile << '\n';

    }

    // write the MultiFab data to, e.g., dirnamestep/Level_0/
    amrex::VisMF::Write(*(mw_yee->J_Array[0]),
            amrex::MultiFabFileFullPrefix(nlevels-1, checkpointname, "Level_", "Jx"));

}


void
Gempic_ReadCheckpointFile (maxwell_yee *mw_yee,
                           const std::string& dirname, const int step)
{

    const std::string& restart_chkfile = amrex::Concatenate(dirname,step);
    amrex::Print() << "Restart from checkpoint " << restart_chkfile << "\n";

    // Header
    std::string File(restart_chkfile + "/Header");

    amrex::VisMF::IO_Buffer io_buffer(amrex::VisMF::GetIOBufferSize());

    amrex::Vector<char> fileCharPtr;
    amrex::ParallelDescriptor::ReadAndBcastFile(File, fileCharPtr);
    std::string fileCharPtrString(fileCharPtr.dataPtr());
    std::istringstream is(fileCharPtrString, std::istringstream::in);

    std::string line, word;

    // read in title line
    std::getline(is, line);

    // read in finest_level
    int finest_level;
    is >> finest_level;
    Utils::Gempic_GotoNextLine(is);

    // read in array of istep
    int step_read;
    is >> step_read;
    Utils::Gempic_GotoNextLine(is);

    // read in array of dt
    is >> mw_yee->dt;
    Utils::Gempic_GotoNextLine(is);

    // read in array of t_new
    amrex::Real time;
    is >> time;
    Utils::Gempic_GotoNextLine(is);

    // read in level 'lev' BoxArray from Header
    amrex::BoxArray ba;
    ba.readFrom(is);
    Utils::Gempic_GotoNextLine(is);

    // create a distribution mapping
    amrex::DistributionMapping dm { ba, amrex::ParallelDescriptor::NProcs() };

    // set BoxArray grids and DistributionMapping dmap in AMReX_AmrMesh.H class
    // not using class so not needed?

    // build MultiFab and FluxRegister data
    int ncomp = 1;
    int nghost = 0;
    //phi_old[lev].define(grids[lev], dmap[lev], ncomp, nghost);
    //phi_new[lev].define(grids[lev], dmap[lev], ncomp, nghost);

    // read in the MultiFab data
    const int nlevels = 1;
    amrex::VisMF::Read(*(mw_yee->J_Array[0]),
                    amrex::MultiFabFileFullPrefix(nlevels-1, restart_chkfile, "Level_", "Jx"));

}


}

}

#endif
