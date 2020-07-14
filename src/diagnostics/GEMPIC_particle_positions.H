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
void save_particle_positions (particle_groups *ppg, std::string ID = "", int species = 0){

    auto all_parts = (*(*ppg).mypc[species]).ParticleBoxArray(0);
    std::cout << all_parts[0] << " | " << all_parts[1] << std::endl;
    std::ofstream ofsss("particle_output/x/W_" + ID + ".output", std::ofstream::out);
    long unsigned int tmp_int = 10;
    amrex::RealDescriptor ParticleRealDescriptor = amrex::FPC::Native64RealDescriptor();
    //(*(*ppg).mypc[species]).WriteParticleRealData(tmp_int, ofsss, ParticleRealDescriptor);
    //(*(*ppg).mypc[species]).WriteParticles ();
    //(*(*ppg).mypc[species]).Checkpoint("tmpPar", "Particle0");

    //-------------------------------------------------------------
    const std::string& dir = "tmpPar";
    const std::string& name = "Electron";
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
    for (amrex::ParIter<GEMPIC_VDIM+1,0,0,0> pti(*(*ppg).mypc[species], 0); pti.isValid(); ++pti) {

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

void save_particle_velocities (particle_groups *ppg, std::string ID = "", int species = 0){
    std::ofstream ofss("particle_output/v/P_" + ID + ".output", std::ofstream::out);
    for (amrex::ParIter<GEMPIC_VDIM+1,0,0,0> pti(*(*ppg).mypc[species], 0); pti.isValid(); ++pti) {

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

void
WritePlotFile (particle_groups *ppg, maxwell_yee *mw_yee, infrastructure *infra, const std::string& plot_file, const int step)
{
    const std::string& plotfilename = amrex::Concatenate(plot_file,step);
    amrex::Print() << "  Writing plotfile " << plotfilename << "\n";

    amrex::Vector<std::string> rfs = {};
    amrex::VisMF::Header::Version current_version = amrex::VisMF::GetHeaderVersion();
    amrex::VisMF::SetHeaderVersion(amrex::VisMF::Header::Version_v1);
    amrex::Real time = 0.0;
    amrex::WriteSingleLevelPlotfile(plotfilename,
                                    (*(*mw_yee).E_Array[0]),
                                    {"Ex"},
                                    (*infra).geom,
                                    time, step,
                                    "PIC_Weibel",
                                    "Level_",
                                    "Cell",
                                    rfs);


    //-------------------------------------------------------------
    const std::string& dir = plotfilename;
    const std::string& name = "Electron";
    int species = 0;
    const amrex::Vector<int>& write_real_comp = {1,1,1,1}; // should this and next line include "obligatory" vals? does this need names?
    const amrex::Vector<int>& write_int_comp = {};
    const amrex::Vector<std::string>& real_comp_names = {"vx", "vy", "vz", "weight"};
    const amrex::Vector<std::string>&  int_comp_names = {};
    (*(*ppg).mypc[species]).WritePlotFile(dir, name, write_real_comp, write_int_comp, real_comp_names, int_comp_names);
    //--------------------------------------------------------------

    //WriteJobInfo(plotfilename); //TODO: nice to have

    amrex::VisMF::SetHeaderVersion(current_version);

}

/*
void
WritePlotFile (particle_groups *ppg, maxwell_yee *mw_yee, infrastructure *infra, const std::string& plot_file, const int step)
{
    const std::string& plotfilename = amrex::Concatenate(plot_file,step);
    amrex::Print() << "  Writing plotfile " << plotfilename << "\n";

    amrex::Vector<std::string> varnames = {"Ex", "Bx", "Jx", "Ey", "By", "Jy", "Ez", "Bz", "Jz"}; // Name of the written fields
    amrex::Vector<const amrex::MultiFab*> output_mf; // will point to the data to be written
    for (int comp=0; comp<3; comp++) {
        output_mf.push_back(&(*(*mw_yee).E_Array[comp]));
        output_mf.push_back(&(*(*mw_yee).B_Array[comp]));
        output_mf.push_back(&(*(*mw_yee).J_Array[comp]));
    }
    amrex::Vector<amrex::Geometry> output_geom;
    for (int i=0; i<9; i++) {
        output_geom.push_back((*infra).geom);
    }

    amrex::Vector<std::string> rfs;
    amrex::VisMF::Header::Version current_version = amrex::VisMF::GetHeaderVersion();
    amrex::VisMF::SetHeaderVersion(amrex::VisMF::Header::Version_v1); //TODO
    int n_levels = 1;
    amrex::Real time = 0.0;
    const amrex::Vector<int>& istep = {0};
    const amrex::Vector<amrex::IntVect>& ref_ratio = {};
    std::cout << "AA" << std::endl;
    amrex::WriteMultiLevelPlotfile(plotfilename, n_levels,
                                   output_mf, varnames, output_geom,
                                   time, istep, ref_ratio,
                                   "PIC_Weibel",
                                   "Level_",
                                   "Cell",
                                   rfs
                                   );
    std::cout << "BB" << std::endl;

    //-------------------------------------------------------------
    const std::string& dir = plotfilename;
    const std::string& name = "Electron";
    int species = 0;
    const amrex::Vector<int>& write_real_comp = {1,1,1,1}; // should this and next line include "obligatory" vals? does this need names?
    const amrex::Vector<int>& write_int_comp = {};
    const amrex::Vector<std::string>& real_comp_names = {"vx", "vy", "vz", "weight"};
    const amrex::Vector<std::string>&  int_comp_names = {};
    (*(*ppg).mypc[species]).WritePlotFile(dir, name, write_real_comp, write_int_comp, real_comp_names, int_comp_names);
    //--------------------------------------------------------------

    //WriteJobInfo(plotfilename); //TODO: nice to have

    amrex::VisMF::SetHeaderVersion(current_version);

}
*/

/*
void
prepareFields(int const step,
              amrex::Vector<std::string>& varnames,
              amrex::Vector<amrex::MultiFab>& mf_avg,
              amrex::Vector<const amrex::MultiFab*>& output_mf,
              amrex::Vector<amrex::Geometry>& output_geom)
{
    // Average the fields from the simulation grid to the cell centers
    const int ngrow = 0;
    AverageAndPackFields( varnames, mf_avg, ngrow );

    // Coarsen the fields, if requested by the user
    Vector<MultiFab> coarse_mf; // will remain empty if there is no coarsening
    if (plot_coarsening_ratio != 1) {
        coarsenCellCenteredFields( coarse_mf, output_geom, mf_avg, Geom(),
                                   plot_coarsening_ratio, finest_level );
        output_mf = amrex::GetVecOfConstPtrs(coarse_mf);
    } else {  // No averaging necessary, simply point to mf_avg
        output_mf = amrex::GetVecOfConstPtrs(mf_avg);
        output_geom = Geom();
    }
}

void
AverageAndPackFields (amrex::Vector<std::string>& varnames,
                      amrex::Vector<amrex::MultiFab>& mf_avg,
                      const int ngrow,
                      amrex::BoxArray grid,
                      amrex::DistributionMapping dmap)
{
    // Count how many different fields should be written (ncomp)
    amrex::Vector<std::string> fields_to_plot = {"Ex", "Ey", "Ez", "Bx", "By", "Bz", "Jx", "Jy", "Jz"};
    int ncomp = fields_to_plot.size();

    int nvecs = 3;

    // Allocate pointers to the `ncomp` fields that will be added
    mf_avg.push_back(amrex::MultiFab(grid, dmap, ncomp, ngrow));

    // For E, B and J: build cell-centered temporary MultiFab with 3 comps
    //amrex::MultiFab mf_tmp_E, mf_tmp_B, mf_tmp_J;
    // Allocate temp MultiFab with 3 components
    amrex::MultiFab mf_tmp_E(grid, dmap, nvecs, ngrow);
    amrex::MultiFab mf_tmp_B(grid, dmap, nvecs, ngrow);
    amrex::MultiFab mf_tmp_J(grid, dmap, nvecs, ngrow);
    // Fill MultiFab mf_tmp_E with averaged E
    AverageAndPackVectorField(mf_tmp_E, Efield_aux[lev], dmap, 0, ngrow);
    AverageAndPackVectorField(mf_tmp_B, Bfield_aux[lev], dmap, 0, ngrow);
    AverageAndPackVectorField(mf_tmp_J, current_fp[lev], dmap, 0, ngrow);
    int dcomp = 3;
    AverageAndPackVectorFieldComponents(mf_tmp_E, Efield_aux[lev], dmap, dcomp, ngrow);
    AverageAndPackVectorFieldComponents(mf_tmp_B, Bfield_aux[lev], dmap, dcomp, ngrow);
    AverageAndPackVectorFieldComponents(mf_tmp_J, current_fp[lev], dmap, dcomp, ngrow);

    // Go through the different fields in fields_to_plot, pack them into
    // mf_avg[lev] add the corresponding names to `varnames`.
    // plot_fine_patch and plot_coarse_patch are treated separately
    // (after this for loop).
    dcomp = 0;
    for (int ifield=0; ifield<fields_to_plot.size(); ifield++){
        std::string fieldname = fields_to_plot[ifield];
        if(lev==0) varnames.push_back(fieldname);
        if        (fieldname == "Ex"){
            MultiFab::Copy( mf_avg[lev], mf_tmp_E, 0, dcomp++, 1, ngrow);
            CopyVectorFieldComponentsToMultiFab(lev, mf_avg, mf_tmp_E, 0, dcomp, ngrow, "Er", varnames);
        } else if (fieldname == "Ey"){
            MultiFab::Copy( mf_avg[lev], mf_tmp_E, 1, dcomp++, 1, ngrow);
            CopyVectorFieldComponentsToMultiFab(lev, mf_avg, mf_tmp_E, 1, dcomp, ngrow, "Etheta", varnames);
        } else if (fieldname == "Ez"){
            MultiFab::Copy( mf_avg[lev], mf_tmp_E, 2, dcomp++, 1, ngrow);
            CopyVectorFieldComponentsToMultiFab(lev, mf_avg, mf_tmp_E, 2, dcomp, ngrow, "Ez", varnames);
        } else if (fieldname == "Bx"){
            MultiFab::Copy( mf_avg[lev], mf_tmp_B, 0, dcomp++, 1, ngrow);
            CopyVectorFieldComponentsToMultiFab(lev, mf_avg, mf_tmp_B, 0, dcomp, ngrow, "Br", varnames);
        } else if (fieldname == "By"){
            MultiFab::Copy( mf_avg[lev], mf_tmp_B, 1, dcomp++, 1, ngrow);
            CopyVectorFieldComponentsToMultiFab(lev, mf_avg, mf_tmp_B, 1, dcomp, ngrow, "Btheta", varnames);
        } else if (fieldname == "Bz"){
            MultiFab::Copy( mf_avg[lev], mf_tmp_B, 2, dcomp++, 1, ngrow);
            CopyVectorFieldComponentsToMultiFab(lev, mf_avg, mf_tmp_B, 2, dcomp, ngrow, "Bz", varnames);
        } else if (fieldname == "jx"){
            MultiFab::Copy( mf_avg[lev], mf_tmp_J, 0, dcomp++, 1, ngrow);
            CopyVectorFieldComponentsToMultiFab(lev, mf_avg, mf_tmp_J, 0, dcomp, ngrow, "jr", varnames);
        } else if (fieldname == "jy"){
            MultiFab::Copy( mf_avg[lev], mf_tmp_J, 1, dcomp++, 1, ngrow);
            CopyVectorFieldComponentsToMultiFab(lev, mf_avg, mf_tmp_J, 1, dcomp, ngrow, "jtheta", varnames);
        } else if (fieldname == "jz"){
            MultiFab::Copy( mf_avg[lev], mf_tmp_J, 2, dcomp++, 1, ngrow);
            CopyVectorFieldComponentsToMultiFab(lev, mf_avg, mf_tmp_J, 2, dcomp, ngrow, "jz", varnames);
        } else if (fieldname == "rho"){
            AverageAndPackScalarField( mf_avg[lev], *rho_fp[lev], dcomp++, ngrow );
            CopyScalarFieldComponentsToMultiFab(lev, mf_avg, *rho_fp[lev], dcomp, ngrow, n_rz_azimuthal_modes,
                                                fieldname, varnames);
        } else if (fieldname == "F"){
            AverageAndPackScalarField( mf_avg[lev], *F_fp[lev], dcomp++, ngrow );
            CopyScalarFieldComponentsToMultiFab(lev, mf_avg, *F_fp[lev], dcomp, ngrow, n_rz_azimuthal_modes,
                                                fieldname, varnames);
        } else if (fieldname == "part_per_cell") {
            MultiFab temp_dat(grids[lev],mf_avg[lev].DistributionMap(),1,0);
            temp_dat.setVal(0);
            // MultiFab containing number of particles in each cell
            mypc->Increment(temp_dat, lev);
            AverageAndPackScalarField( mf_avg[lev], temp_dat, dcomp++, ngrow );
        } else if (fieldname == "part_per_grid"){
            const Vector<long>& npart_in_grid = mypc->NumberOfParticlesInGrid(lev);
            // MultiFab containing number of particles per grid
            // (stored as constant for all cells in each grid)
#ifdef _OPENMP
#pragma omp parallel
#endif
            for (MFIter mfi(mf_avg[lev]); mfi.isValid(); ++mfi) {
                (mf_avg[lev])[mfi].setVal(static_cast<Real>(npart_in_grid[mfi.index()]));
            }
            dcomp++;
        } else if (fieldname == "part_per_proc"){
            const Vector<long>& npart_in_grid = mypc->NumberOfParticlesInGrid(lev);
            // MultiFab containing number of particles per process
            // (stored as constant for all cells in each grid)
            long n_per_proc = 0;
#ifdef _OPENMP
#pragma omp parallel reduction(+:n_per_proc)
#endif
            for (MFIter mfi(mf_avg[lev]); mfi.isValid(); ++mfi) {
                n_per_proc += npart_in_grid[mfi.index()];
            }
            mf_avg[lev].setVal(static_cast<Real>(n_per_proc),dcomp++,1);
        } else if (fieldname == "proc_number"){
            // MultiFab containing the Processor ID
#ifdef _OPENMP
#pragma omp parallel
#endif
            mf_avg[lev].setVal(static_cast<Real>(ParallelDescriptor::MyProc()),dcomp++,1);
        } else if (fieldname == "divB"){
            if (do_nodal) amrex::Abort("TODO: do_nodal && plot divb");
            ComputeDivB(mf_avg[lev], dcomp++,
                        {Bfield_aux[lev][0].get(),
                                Bfield_aux[lev][1].get(),
                                Bfield_aux[lev][2].get()},
                        WarpX::CellSize(lev) );
        } else if (fieldname == "divE"){
            if (do_nodal) amrex::Abort("TODO: do_nodal && plot dive");
            const BoxArray& ba = amrex::convert(boxArray(lev),IntVect::TheUnitVector());
            MultiFab dive(ba,DistributionMap(lev),1,0);
            ComputeDivE( dive, 0,
                         {Efield_aux[lev][0].get(),
                                 Efield_aux[lev][1].get(),
                                 Efield_aux[lev][2].get()},
                         WarpX::CellSize(lev) );
            AverageAndPackScalarField( mf_avg[lev], dive, dcomp++, ngrow );
        } else {
            amrex::Abort("unknown field in fields_to_plot: " + fieldname);
        }
    }
    if (plot_finepatch)
    {
        AverageAndPackVectorField( mf_avg[lev], Efield_fp[lev], dmap[lev], dcomp, ngrow );
        dcomp += 3;
        AverageAndPackVectorFieldComponents(mf_avg[lev], Efield_fp[lev], dmap[lev], dcomp, ngrow);
        if (lev == 0) {
            AddToVarNames(varnames, "E", "_fp");
            if (n_rz_azimuthal_modes > 1) {
                AddToVarNamesRZ(varnames, "E", ComponentName("_fp", 0, "real"));
                for (int mode=1 ; mode < n_rz_azimuthal_modes ; mode++) {
                    AddToVarNamesRZ(varnames, "E", ComponentName("_fp", mode, "real"));
                    AddToVarNamesRZ(varnames, "E", ComponentName("_fp", mode, "imag"));
                }
            }
        }
        AverageAndPackVectorField( mf_avg[lev], Bfield_fp[lev], dmap[lev], dcomp, ngrow );
        dcomp += 3;
        AverageAndPackVectorFieldComponents(mf_avg[lev], Bfield_fp[lev], dmap[lev], dcomp, ngrow);
        if (lev == 0) {
            AddToVarNames(varnames, "B", "_fp");
            if (n_rz_azimuthal_modes > 1) {
                AddToVarNamesRZ(varnames, "B", ComponentName("_fp", 0, "real"));
                for (int mode=1 ; mode < n_rz_azimuthal_modes ; mode++) {
                    AddToVarNamesRZ(varnames, "B", ComponentName("_fp", mode, "real"));
                    AddToVarNamesRZ(varnames, "B", ComponentName("_fp", mode, "imag"));
                }
            }
        }
    }

    if (plot_crsepatch)
    {
        if (lev == 0)
        {
            mf_avg[lev].setVal(0.0, dcomp, 3, ngrow);
        }
        else
        {
            if (do_nodal) amrex::Abort("TODO: do_nodal && plot_crsepatch");
            std::array<std::unique_ptr<MultiFab>, 3> E = getInterpolatedE(lev);
            AverageAndPackVectorField( mf_avg[lev], E, dmap[lev], dcomp, ngrow );

        }
        if (lev == 0) AddToVarNames(varnames, "E", "_cp");
        dcomp += 3;

        // now the magnetic field
        if (lev == 0)
        {
            mf_avg[lev].setVal(0.0, dcomp, 3, ngrow);
        }
        else
        {
            if (do_nodal) amrex::Abort("TODO: do_nodal && plot_crsepatch");
            std::array<std::unique_ptr<MultiFab>, 3> B = getInterpolatedB(lev);
            AverageAndPackVectorField( mf_avg[lev], B, dmap[lev], dcomp, ngrow );
        }
        if (lev == 0) AddToVarNames(varnames, "B", "_cp");
        dcomp += 3;
    }

    if (costs[0] != nullptr and plot_costs)
    {
        AverageAndPackScalarField( mf_avg[lev], *costs[lev], dcomp, ngrow );
        if(lev==0) varnames.push_back("costs");
        dcomp += 1;
    }

};
*/


#endif
