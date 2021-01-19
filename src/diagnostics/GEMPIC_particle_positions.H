#include <AMReX.H>
#include <AMReX_Box.H>
#include <AMReX_BoxArray.H>
#include <AMReX_CoordSys.H>
#include <AMReX_DistributionMapping.H>
#include <AMReX_FPC.H>
#include <AMReX_Geometry.H>
#include <AMReX_IntVect.H>
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

#ifndef PARTICLE_POSITION_H_
#define PARTICLE_POSITION_H_

namespace Gempic {
namespace Diagnostics_Output {

template<int vdim, int numspec>
void save_particle_positions (particle_groups<vdim, numspec> *ppg, std::string ID = "", int species = 0){

    auto all_parts = (*(*ppg).mypc[species]).ParticleBoxArray(0);
    std::ofstream ofsss("particle_output/x/W_" + ID + ".output", std::ofstream::out);
    amrex::RealDescriptor ParticleRealDescriptor = amrex::FPC::Native64RealDescriptor();

    //-------------------------------------------------------------
    const std::string& dir = "ParticlePlotfiles/plt00000";
    const std::string& name = "electrons";
    const amrex::Vector<int>& write_real_comp = {1,1,1,1}; // should this and next line include "obligatory" vals? does this need names?
    const amrex::Vector<int>& write_int_comp = {};
    const amrex::Vector<std::string>& real_comp_names = {"vx", "vy", "vz", "weight"};
    const amrex::Vector<std::string>&  int_comp_names = {};
    (*(*ppg).mypc[species]).WritePlotFile(dir, name, write_real_comp, write_int_comp, real_comp_names, int_comp_names);
    //--------------------------------------------------------------


    //(*(*ppg).mypc[species]).WritePlotFile("ParticlePlotfiles", "Particles_0");
    ofsss.close();
    const std::string& pltfile = "test_particle_output";
    (*(*ppg).mypc[species]).WriteAsciiFile(pltfile);
    std::ofstream ofss("particle_output/x/P_" + ID + ".output", std::ofstream::out);
    for (amrex::ParIter<vdim+1,0,0,0> pti(*(*ppg).mypc[species], 0); pti.isValid(); ++pti) {

        const auto& particles = pti.GetArrayOfStructs();
        const long np = pti.numParticles();
        for (int pp=0;pp<np;pp++) {
            amrex::Print(ofss) << particles[pp].pos(0) << "," <<
                      #if (GEMPIC_SPACEDIM > 1)
                                  particles[pp].pos(1) << "," <<
                      #endif
                      #if (GEMPIC_SPACEDIM > 2)
                                  particles[pp].pos(2) <<
                      #endif
                                  std::endl;

            amrex::AllPrintToFile("part_pos.") << particles[pp].pos(0) << "," <<
                                      #if (GEMPIC_SPACEDIM > 1)
                                                  particles[pp].pos(1) << "," <<
                                      #endif
                                      #if (GEMPIC_SPACEDIM > 2)
                                                  particles[pp].pos(2) <<
                                      #endif
                                                  std::endl;
        }
    }
    ofss.close();
}

template<int vdim, int numspec>
void save_particle_velocities (particle_groups<vdim, numspec> *ppg, std::string ID = "", int species = 0){
    std::ofstream ofss("particle_output/v/P_" + ID + ".output", std::ofstream::out);
    for (amrex::ParIter<vdim+1,0,0,0> pti(*(*ppg).mypc[species], 0); pti.isValid(); ++pti) {

        const auto& particles = pti.GetArrayOfStructs();
        const long np = pti.numParticles();
        for (int pp=0;pp<np;pp++) {
            amrex::Print(ofss) << particles[pp].rdata(0) << "," <<
                      #if (GEMPIC_SPACEDIM > 1)
                                  particles[pp].rdata(1) << "," <<
                      #endif
                      #if (GEMPIC_SPACEDIM > 2)
                                  particles[pp].rdata(2) <<
                      #endif
                                  std::endl;
        }
    }
    ofss.close();
}
}
}

template<int vdim, int numspec>
void
Gempic_WritePlotFile (particle_groups<vdim, numspec> *ppg, maxwell_yee<vdim> *mw_yee, infrastructure *infra, const std::string& plot_file, const int step)
{
    //TODO: other dims

    // MultiFab Info -------------------------------------------------------------
    const std::string& plotfilename = "ParticlePlotfiles/" + amrex::Concatenate(plot_file,step);
    amrex::Print() << "  Writing plotfile " << plotfilename << "\n";

    amrex::VisMF::Header::Version current_version = amrex::VisMF::GetHeaderVersion();
    amrex::VisMF::SetHeaderVersion(amrex::VisMF::Header::Version_v1);

    int bdim = int(vdim/2.5)*2+1;
    amrex::MultiFab tmp_b_mf((*infra).grid, (*infra).distriMap, 2*vdim+bdim, 0); // temporary MF to store all fields
    amrex::Vector<std::string> varnames;
    tmp_b_mf.Copy(tmp_b_mf, *(*mw_yee).E_Array[0], 0, 0, 1, 0); // srccomp, destcomp, numcomp, nghost
    varnames.push_back("E_x");
#if (vdim>1)
    tmp_b_mf.Copy(tmp_b_mf, *(*mw_yee).E_Array[1], 0, 1, 1, 0);
    varnames.push_back("E_y");
#endif
#if (vdim>2)
    tmp_b_mf.Copy(tmp_b_mf, *(*mw_yee).E_Array[2], 0, 2, 1, 0);
    varnames.push_back("E_z");
#endif
    tmp_b_mf.Copy(tmp_b_mf, *(*mw_yee).B_Array[0], 0, vdim, 1, 0); // srccomp, destcomp, numcomp, nghost
    varnames.push_back("B_x");
#if (bdim>1)
    tmp_b_mf.Copy(tmp_b_mf, *(*mw_yee).B_Array[1], 0, vdim+1, 1, 0);
    varnames.push_back("B_y");
#endif
#if (bdim>2)
    tmp_b_mf.Copy(tmp_b_mf, *(*mw_yee).B_Array[2], 0, vdim+2, 1, 0);
    varnames.push_back("B_z");
#endif
    tmp_b_mf.Copy(tmp_b_mf, *(*mw_yee).J_Array[0], 0, vdim+bdim, 1, 0); // srccomp, destcomp, numcomp, nghost
    varnames.push_back("J_x");
#if (vdim>1)
    tmp_b_mf.Copy(tmp_b_mf, *(*mw_yee).J_Array[1], 0, vdim+bdim+1, 1, 0);
    varnames.push_back("J_y");
#endif
#if (vdim>2)
    tmp_b_mf.Copy(tmp_b_mf, *(*mw_yee).J_Array[2], 0, vdim+bdim+2, 1, 0);
    varnames.push_back("J_z ");
#endif

    amrex::Real time = step*(*mw_yee).dt;
    WriteSingleLevelPlotfile(plotfilename, tmp_b_mf, varnames,
                             (*infra).geom, time, 0);

    // Particle Info -------------------------------------------------------------
    const std::string& name = "electrons";
    const amrex::Vector<int>& write_real_comp = {1,1
#if (vdim > 1)
                                                 ,1
#endif
#if (vdim>2)
                                                 ,1
#endif
                                                };
    const amrex::Vector<int>& write_int_comp = {};
    const amrex::Vector<std::string>& real_comp_names = {"vx"
#if (vdim > 1)
                                                         , "vy"
#endif
#if (vdim>2)
                                                         , "vz"
#endif
                                                         , "weight"};
    std::cout << "components of parts: " << (*(*ppg).mypc[0]).NumRealComps()
              << ", Num varnames passed: " << real_comp_names.size() << ", num bool passed: "
              << write_real_comp.size() << std::endl;
    const amrex::Vector<std::string>&  int_comp_names = {};
    int species = 0; //TODO: multispecies
    (*(*ppg).mypc[species]).WritePlotFile(plotfilename, name, write_real_comp, write_int_comp, real_comp_names, int_comp_names);

    // warpx_job_info -------------------------------------------------------------
    std::string FullPathJobInfoFile = plotfilename;
    FullPathJobInfoFile += "/warpx_job_info";
    std::ofstream jobInfoFile(FullPathJobInfoFile);
    amrex::Print(jobInfoFile) << "geometry.is_periodic = " << (*infra).is_periodic[0]
                             #if GEMPIC_SPACEDIM > 1
                              << " " << (*infra).is_periodic[1]
                             #endif
                             #if GEMPIC_SPACEDIM > 2
                              << " " << (*infra).is_periodic[2]
                             #endif
                              << std::endl;
    jobInfoFile.close();

    // WarpXHeader -------------------------------------------------------------
    std::string FullPathHeaderFile = plotfilename;
    FullPathHeaderFile += "/WarpXHeader";
    std::ofstream headerFile(FullPathHeaderFile);
    headerFile << "Checkpoint version: 1" << std::endl;
    headerFile << "1" << std::endl; //num_levels
    headerFile << "0" << std::endl; //i_step
    headerFile << "1" << std::endl; //numsubsteps
    headerFile << time << std::endl; //t_new
    headerFile << time-(*mw_yee).dt << std::endl; //t_old
    headerFile << (*mw_yee).dt << std::endl; //dt
    headerFile << std::numeric_limits<amrex::Real>::max() << std::endl; //moving_window_x
    headerFile << "1" << std::endl;//is_synchronised

    for (int i = 0; i < GEMPIC_SPACEDIM; ++i) {
        headerFile << (*infra).geom.ProbLo(i) << " "; //domain_low
    }
    headerFile << std::endl;

    for (int i = 0; i < GEMPIC_SPACEDIM; ++i) {
        headerFile << (*infra).geom.ProbHi(i) << " "; //domain_high
    }
    headerFile << std::endl;

    (*infra).grid.writeOn(headerFile);
    headerFile << '\n';

    amrex::Print(headerFile) << (*ppg).charge[species] << std::endl; //charge
    headerFile.close();

    amrex::VisMF::SetHeaderVersion(current_version);

}

template<int vdim, int numspec>
void
Gempic_WritePlotFile_rho_phi (particle_groups<vdim, numspec> *ppg, maxwell_yee<vdim> *mw_yee, infrastructure *infra, const std::string& plot_file, const int step)
{
    //TODO: other dims

    // MultiFab Info -------------------------------------------------------------
    const std::string& plotfilename = "ParticlePlotfiles/" + amrex::Concatenate(plot_file,step);
    amrex::Print() << "  Writing plotfile " << plotfilename << "\n";

    amrex::VisMF::Header::Version current_version = amrex::VisMF::GetHeaderVersion();
    amrex::VisMF::SetHeaderVersion(amrex::VisMF::Header::Version_v1);

    amrex::MultiFab tmp_b_mf((*infra).grid, (*infra).distriMap, 2, 0); // temporary MF to store all fields
    tmp_b_mf.Copy(tmp_b_mf, (*mw_yee).rho, 0, 0, 1, 0); // srccomp, destcomp, numcomp, nghost
    tmp_b_mf.Copy(tmp_b_mf, (*mw_yee).phi, 0, 1, 1, 0);

    amrex::Vector<std::string> varnames = {"rho", "phi"};
    amrex::Real time = step*(*mw_yee).dt;
    WriteSingleLevelPlotfile(plotfilename, tmp_b_mf, varnames,
                             (*infra).geom, time, 0);

    // Particle Info -------------------------------------------------------------
    const std::string& name = "electrons";
    const amrex::Vector<int>& write_real_comp = {1,1
#if (vdim > 1)
                                                 ,1
#endif
#if (vdim>2)
                                                 ,1
#endif
                                                };
    const amrex::Vector<int>& write_int_comp = {};
    const amrex::Vector<std::string>& real_comp_names = {"vx"
#if (vdim > 1)
                                                         , "vy"
#endif
#if (vdim>2)
                                                         , "vz"
#endif
                                                         , "weight"};
    const amrex::Vector<std::string>&  int_comp_names = {};
    int species = 0; //TODO: multispecies
    (*(*ppg).mypc[species]).WritePlotFile(plotfilename, name, write_real_comp, write_int_comp, real_comp_names, int_comp_names);

    // warpx_job_info -------------------------------------------------------------
    std::string FullPathJobInfoFile = plotfilename;
    FullPathJobInfoFile += "/warpx_job_info";
    std::ofstream jobInfoFile(FullPathJobInfoFile);
    amrex::Print(jobInfoFile) << "geometry.is_periodic = " << (*infra).is_periodic[0]
                             #if GEMPIC_SPACEDIM > 1
                              << " " << (*infra).is_periodic[1]
                             #endif
                             #if GEMPIC_SPACEDIM > 2
                              << " " << (*infra).is_periodic[2]
                             #endif
                              << std::endl;
    jobInfoFile.close();

    // WarpXHeader -------------------------------------------------------------
    std::string FullPathHeaderFile = plotfilename;
    FullPathHeaderFile += "/WarpXHeader";
    std::ofstream headerFile(FullPathHeaderFile);
    headerFile << "Checkpoint version: 1" << std::endl;
    headerFile << "1" << std::endl; //num_levels
    headerFile << "0" << std::endl; //i_step
    headerFile << "1" << std::endl; //numsubsteps
    headerFile << time << std::endl; //t_new
    headerFile << time-(*mw_yee).dt << std::endl; //t_old
    headerFile << (*mw_yee).dt << std::endl; //dt
    headerFile << std::numeric_limits<amrex::Real>::max() << std::endl; //moving_window_x
    headerFile << "1" << std::endl;//is_synchronised

    for (int i = 0; i < GEMPIC_SPACEDIM; ++i) {
        headerFile << (*infra).geom.ProbLo(i) << " "; //domain_low
    }
    headerFile << std::endl;

    for (int i = 0; i < GEMPIC_SPACEDIM; ++i) {
        headerFile << (*infra).geom.ProbHi(i) << " "; //domain_high
    }
    headerFile << std::endl;

    (*infra).grid.writeOn(headerFile);
    headerFile << '\n';

    amrex::Print(headerFile) << (*ppg).charge[species] << std::endl; //charge
    headerFile.close();

    amrex::VisMF::SetHeaderVersion(current_version);

}

#endif
