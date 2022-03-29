#ifndef CHECKPOINT_H_
#define CHECKPOINT_H_

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
#include <GEMPIC_computational_domain.H>
#include <GEMPIC_maxwell_yee.H>
#include <GEMPIC_particle_groups.H>

using namespace Field_solvers;
using namespace CompDom;
using namespace Particles;

namespace Gempic
{
namespace Utils
{
// utility to skip to next line in Header
void Gempic_GotoNextLine(std::istream &is)
{
    constexpr std::streamsize bl_ignore_max{100000};
    is.ignore(bl_ignore_max, '\n');
}

template <int vdim>
void Gempic_WriteCheckpointFile_Field(maxwell_yee<vdim> *mw_yee, computational_domain *infra,
                                      const std::string &dirname, int field, const int step,
                                      amrex::IndexType IndexT, int header_precision)
{
    // dirnamestep            write a checkpoint file with this root directory
    // dirnamestep/Header     this contains information you need to save (e.g., finest_level, t_new,
    // etc.) and also
    //                        the BoxArrays at each level
    // dirnamestep/Level_0/
    // dirnamestep/Level_1/
    // etc.                   these subdirectories will hold the MultiFab data at each level of
    // refinement

    std::array<std::string, 9> fieldnames = {"Jx", "Jy", "Jz", "Ex", "Ey", "Ez", "Bx", "By", "Bz"};

    const std::string &checkpointname =
        amrex::Concatenate(dirname + "_" + fieldnames[field] + "_", step);

    // ---- prebuild a hierarchy of directories
    // ---- dirName is built first.  if dirName exists, it is renamed.  then build
    // ---- dirName/subDirPrefix_0 .. dirName/subDirPrefix_nlevels-1
    // ---- if callBarrier is true, call ParallelDescriptor::Barrier()
    // ---- after all directories are built
    // ---- ParallelDescriptor::IOProcessor() creates the directories
    const int nlevels = 1;
    amrex::PreBuildDirectorHierarchy(checkpointname, "Level_", nlevels, true);

    // write Header file
    if (amrex::ParallelDescriptor::IOProcessor())
    {
        std::string HeaderFileName(checkpointname + "/Header");
        amrex::VisMF::IO_Buffer io_buffer(amrex::VisMF::IO_Buffer_Size);
        std::ofstream HeaderFile;
        HeaderFile.rdbuf()->pubsetbuf(io_buffer.dataPtr(), io_buffer.size());
        HeaderFile.open(HeaderFileName.c_str(),
                        std::ofstream::out | std::ofstream::trunc | std::ofstream::binary);
        if (!HeaderFile.good())
        {
            amrex::FileOpenFailed(HeaderFileName);
        }

        HeaderFile.precision(header_precision);

        // write out title line
        HeaderFile << "Checkpoint file\n";

        // write out finest_level
        HeaderFile << nlevels - 1 << "\n";

        // write out array of step (only one level so only one entry)
        HeaderFile << step << " ";
        HeaderFile << "\n";

        // write out array of dt (only one level so only one entry)
        HeaderFile << mw_yee->dt << " ";
        HeaderFile << "\n";

        // write out array of t_new (only one level so only one entry)
        HeaderFile << (mw_yee->dt) * step << " ";
        HeaderFile << "\n";

        // write the BoxArray at each level
        amrex::BoxArray boxArray(infra->grid, IndexT);
        boxArray.writeOn(HeaderFile);
        HeaderFile << '\n';
    }

    // write the MultiFab data to, e.g., dirnamestep/Level_0/
    // using integer division to get floored result
    if (field / 3 == 0)
    {
        amrex::VisMF::Write(*(mw_yee->J_Array[field % 3]),
                            amrex::MultiFabFileFullPrefix(nlevels - 1, checkpointname, "Level_",
                                                          fieldnames[field]));
    }
    else if (field / 3 == 1)
    {
        amrex::VisMF::Write(*(mw_yee->E_Array[field % 3]),
                            amrex::MultiFabFileFullPrefix(nlevels - 1, checkpointname, "Level_",
                                                          fieldnames[field]));
    }
    else
    {
        amrex::VisMF::Write(*(mw_yee->B_Array[field % 3]),
                            amrex::MultiFabFileFullPrefix(nlevels - 1, checkpointname, "Level_",
                                                          fieldnames[field]));
    }
}

template <int vdim, int numspec>
void Gempic_WriteCheckpointFile(maxwell_yee<vdim> *mw_yee, particle_groups<vdim, numspec> *part_gr,
                                computational_domain *infra, const std::string &dirname,
                                const int step, int header_precision, bool particles = true)
{
    amrex::Print() << "Writing checkpoint folders " << amrex::Concatenate(dirname + "_field_", step)
                   << "\n";

    // J
    Gempic_WriteCheckpointFile_Field(mw_yee, infra, dirname, 0, step, *(mw_yee->E_Index[0]),
                                     header_precision);
    Gempic_WriteCheckpointFile_Field(mw_yee, infra, dirname, 1, step, *(mw_yee->E_Index[1]),
                                     header_precision);
    Gempic_WriteCheckpointFile_Field(mw_yee, infra, dirname, 2, step, *(mw_yee->E_Index[2]),
                                     header_precision);

    // E
    Gempic_WriteCheckpointFile_Field(mw_yee, infra, dirname, 3, step, *(mw_yee->E_Index[0]),
                                     header_precision);
    Gempic_WriteCheckpointFile_Field(mw_yee, infra, dirname, 4, step, *(mw_yee->E_Index[1]),
                                     header_precision);
    Gempic_WriteCheckpointFile_Field(mw_yee, infra, dirname, 5, step, *(mw_yee->E_Index[2]),
                                     header_precision);

    // B
    Gempic_WriteCheckpointFile_Field(mw_yee, infra, dirname, 6, step, *(mw_yee->B_Index[0]),
                                     header_precision);
    Gempic_WriteCheckpointFile_Field(mw_yee, infra, dirname, 7, step, *(mw_yee->B_Index[1]),
                                     header_precision);
    Gempic_WriteCheckpointFile_Field(mw_yee, infra, dirname, 8, step, *(mw_yee->B_Index[2]),
                                     header_precision);

    if (particles)
    {
        for (int spec = 0; spec < numspec; spec++)
        {
            part_gr->mypc[spec]
                ->Checkpoint("test_dir", "electrons", true, {"vx", "vy", "vz", "weight"}, {});
        }
    }
}

template <int vdim>
void Gempic_ReadCheckpointFile_Field(maxwell_yee<vdim> *mw_yee, const std::string &dirname,
                                     int field, const int step)
{
    std::array<std::string, 9> fieldnames = {"Jx", "Jy", "Jz", "Ex", "Ey", "Ez", "Bx", "By", "Bz"};
    const std::string &restart_chkfile =
        amrex::Concatenate(dirname + "_" + fieldnames[field] + "_", step);
    amrex::Print() << "Reading in checkpoint " << restart_chkfile << "\n";

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
    amrex::DistributionMapping dm{ba, amrex::ParallelDescriptor::NProcs()};

    // set BoxArray grids and DistributionMapping dmap in AMReX_AmrMesh.H class
    // not using class so not needed?

    // read in the MultiFab data
    const int nlevels = 1;
    if (field / 3 == 0)
    {
        amrex::VisMF::Read(*(mw_yee->J_Array[field % 3]),
                           amrex::MultiFabFileFullPrefix(nlevels - 1, restart_chkfile, "Level_",
                                                         fieldnames[field]));
    }
    else if (field / 3 == 1)
    {
        amrex::VisMF::Read(*(mw_yee->E_Array[field % 3]),
                           amrex::MultiFabFileFullPrefix(nlevels - 1, restart_chkfile, "Level_",
                                                         fieldnames[field]));
    }
    else
    {
        amrex::VisMF::Read(*(mw_yee->B_Array[field % 3]),
                           amrex::MultiFabFileFullPrefix(nlevels - 1, restart_chkfile, "Level_",
                                                         fieldnames[field]));
    }
}

template <int vdim, int numspec, int field_dim = vdim>
void Gempic_ReadCheckpointFile(maxwell_yee<field_dim> *mw_yee,
                               particle_groups<vdim, numspec> *part_gr, computational_domain *infra,
                               const std::string &dirname, const int step, bool particles = true)
{
    amrex::Print() << "Reading checkpoint folders " << amrex::Concatenate(dirname + "_field_", step)
                   << "\n";

    // J
    Gempic_ReadCheckpointFile_Field(mw_yee, dirname, 0, step);
    Gempic_ReadCheckpointFile_Field(mw_yee, dirname, 1, step);
    Gempic_ReadCheckpointFile_Field(mw_yee, dirname, 2, step);

    // E
    Gempic_ReadCheckpointFile_Field(mw_yee, dirname, 3, step);
    Gempic_ReadCheckpointFile_Field(mw_yee, dirname, 4, step);
    Gempic_ReadCheckpointFile_Field(mw_yee, dirname, 5, step);

    // B
    Gempic_ReadCheckpointFile_Field(mw_yee, dirname, 6, step);
    Gempic_ReadCheckpointFile_Field(mw_yee, dirname, 7, step);
    Gempic_ReadCheckpointFile_Field(mw_yee, dirname, 8, step);

    mw_yee->FillBD(*infra);

    // Particles
    if (particles)
    {
        for (int spec = 0; spec < numspec; spec++)
        {
            part_gr->mypc[spec]->Restart("test_dir", "electrons", true);
        }
    }
}

}  // namespace Utils

}  // namespace Gempic

#endif
