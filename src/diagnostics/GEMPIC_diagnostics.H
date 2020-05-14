#include <AMReX.H>
#include <AMReX_Print.H>
#include <AMReX_PlotFileUtil.H>
#include <AMReX_RealBox.H>
#include <AMReX_REAL.H>
#include <AMReX_BoxArray.H>
#include <AMReX_Geometry.H>
#include <AMReX_IntVect.H>
#include <AMReX_Box.H>
#include <AMReX_DistributionMapping.H>
#include <AMReX_CoordSys.H>

#include <GEMPIC_particle_groups.H>
#include <GEMPIC_infrastructure.H>
#include <GEMPIC_maxwell_yee.H>
#include <GEMPIC_gempic_norm.H>

#ifndef DIAGNOSTICS_H_
#define DIAGNOSTICS_H_

class diagnostics
{
public:
    int n_time; // number of timesteps
    // 9 static so far!! need to fix this
    std::array<std::array<amrex::Real,GEMPIC_BDIM>,100001> Bfield;
    std::array<std::array<amrex::Real,GEMPIC_VDIM>,100001> Efield, mom;
    std::array<amrex::Real,100001> kin;
    int freq_x;
    int freq_v;
    int freq_slice;
    std::string sim_name;

    // The default constructor.
    diagnostics ();

    // Explicit const
    diagnostics (int n_t, int fx, int fv, int fs, std::string sim_n){
        n_time = n_t;
        freq_x = fx;
        freq_v = fv;
        freq_slice = fs;
        sim_name = sim_n;
    }

    // save position and velocities
    void save_step_data(int step,
                        infrastructure infra,
                        maxwell_yee *mw_yee,
                        particle_groups *part_gr){
        amrex::Real vel;

        // E and B field
        for(int cc=0;cc<GEMPIC_VDIM;cc++){
            Efield[step][cc] = gempic_norm(&(*(*mw_yee).E_Array[cc]), infra, 2);
        }

        for(int cc=0;cc<GEMPIC_BDIM;cc++){
            Bfield[step][cc] = gempic_norm(&(*(*mw_yee).B_Array[cc]), infra, 2);
        }

        // compute kinetic energy norm and momentum
        kin[step] = 0;
        mom[step][0] = 0.0;
#if (GEMPIC_VDIM > 1)
        mom[step][1] = 0.0;
#endif
#if (GEMPIC_VDIM > 2)
        mom[step][2] = 0.0;
#endif

        for (int spec=0;spec<(*part_gr).n_species;spec++){
            for (amrex::ParIter<GEMPIC_VDIM+1,0,0,0> pti(*(*part_gr).mypc[spec], 0); pti.isValid(); ++pti) {
                auto& particles = pti.GetArrayOfStructs();
                const long np = pti.numParticles();

                for (int pp=0;pp<np;pp++) {
                    std::array<double,GEMPIC_VDIM> absvel = {AMREX_V_DECL(std::abs(particles[pp].rdata(0)),std::abs(particles[pp].rdata(1)),std::abs(particles[pp].rdata(2)))};
                    vel = std::accumulate(absvel.begin(), absvel.end(), 0.);
                    vel = std::abs(particles[pp].rdata(0))+std::abs(particles[pp].rdata(1))+std::abs(particles[pp].rdata(2));
                    kin[step] += particles[pp].rdata(GEMPIC_VDIM)*pow(vel,2);
                    for (int cmp=0;cmp<GEMPIC_VDIM;cmp++) {
                        mom[step][cmp] += particles[pp].rdata(GEMPIC_VDIM)*std::abs(particles[pp].rdata(cmp));
                    }
                }
            }
        }

        // save Plotfile
        if(step%freq_slice==0) {
            amrex::Vector<std::string> varnames = {"rho"};
            if (step == 0) {
                WriteSingleLevelPlotfile ("SliceData/rho_plotfile_0", (*mw_yee).rho, varnames,
                                          infra.geom, step*(*mw_yee).dt, 0); //level_step
            }

            varnames[0] = "Ex";
            WriteSingleLevelPlotfile ("SliceData/Ex_plotfile_" + std::to_string(step), (*(*mw_yee).E_Array[0]), varnames, infra.geom, 0.0, 0); // last two args: time and level_step
            varnames[0] = "E_solx";
            WriteSingleLevelPlotfile ("SliceData/E_solx_plotfile_" + std::to_string(step), (*(*mw_yee).E_sol_Array[0]), varnames, infra.geom, 0.0, 0);
            varnames[0] = "Jx";
            WriteSingleLevelPlotfile ("SliceData/Jx_plotfile_" + std::to_string(step), (*(*mw_yee).J_Array[0]), varnames, infra.geom, 0.0, 0);
            varnames[0] = "Bx";
            WriteSingleLevelPlotfile ("SliceData/Bx_plotfile_" + std::to_string(step), (*(*mw_yee).B_Array[0]), varnames, infra.geom, 0.0, 0);
            varnames[0] = "B_solx";
            WriteSingleLevelPlotfile ("SliceData/B_solx_plotfile_" + std::to_string(step), (*(*mw_yee).B_sol_Array[0]), varnames, infra.geom, 0.0, 0);
#if (GEMPIC_VDIM > 1)
            varnames[0] = "Ey";
            WriteSingleLevelPlotfile ("SliceData/Ey_plotfile_" + std::to_string(step), (*(*mw_yee).E_Array[1]), varnames, infra.geom, 0.0, 0);
            varnames[0] = "E_soly";
            WriteSingleLevelPlotfile ("SliceData/E_soly_plotfile_" + std::to_string(step), (*(*mw_yee).E_sol_Array[1]), varnames, infra.geom, 0.0, 0);
            varnames[0] = "Jy";
            WriteSingleLevelPlotfile ("SliceData/Jy_plotfile_" + std::to_string(step), (*(*mw_yee).J_Array[1]), varnames, infra.geom, 0.0, 0);
#endif
#if (GEMPIC_BDIM > 1)
            varnames[0] = "By";
            WriteSingleLevelPlotfile ("SliceData/By_plotfile_" + std::to_string(step), (*(*mw_yee).B_Array[1]), varnames, infra.geom, 0.0, 0);
            varnames[0] = "B_soly";
            WriteSingleLevelPlotfile ("SliceData/B_soly_plotfile_" + std::to_string(step), (*(*mw_yee).B_sol_Array[1]), varnames, infra.geom, 0.0, 0);
#endif
#if (GEMPIC_VDIM > 2)
            varnames[0] = "Ez";
            WriteSingleLevelPlotfile ("SliceData/Ez_plotfile_" + std::to_string(step), (*(*mw_yee).E_Array[2]), varnames, infra.geom, 0.0, 0);
            varnames[0] = "E_solz";
            WriteSingleLevelPlotfile ("SliceData/E_solz_plotfile_" + std::to_string(step), (*(*mw_yee).E_sol_Array[2]), varnames, infra.geom, 0.0, 0);
            varnames[0] = "Jz";
            WriteSingleLevelPlotfile ("SliceData/Jz_plotfile_" + std::to_string(step), (*(*mw_yee).J_Array[2]), varnames, infra.geom, 0.0, 0);
            varnames[0] = "Bz";
            WriteSingleLevelPlotfile ("SliceDataTMP/Bz_plotfile_" + std::to_string(step), (*(*mw_yee).B_Array[2]), varnames, infra.geom, 0.0, 0);
            varnames[0] = "B_solz";
            WriteSingleLevelPlotfile ("SliceData/B_solz_plotfile_" + std::to_string(step), (*(*mw_yee).B_sol_Array[2]), varnames, infra.geom, 0.0, 0);
#endif

        }
    }

    void save_all_to_textfile(amrex::Real dt, std::string filename = "PIC_save.output", int partial = 0){
        if (partial == 0){
            partial = n_time;
        }
        std::ofstream ofsfun(filename, std::ofstream::out);
        amrex::Print(ofsfun) << "# time   E components   B components   kinetic energy   momentum components " << std::endl;
        for (int step=0;step<=partial;step++){
            amrex::Print(ofsfun) << dt*(step) << " " << Efield[step][0]
                        #if (GEMPIC_VDIM > 1)
                                 << " " << Efield[step][1]
                        #endif
                        #if (GEMPIC_VDIM > 2)
                                 << " " << Efield[step][2]
                        #endif
                                 << " " << Bfield[step][0]
                        #if (GEMPIC_BDIM > 1)
                                 << " " << Bfield[step][1]
                        #endif
                        #if (GEMPIC_BDIM > 2)
                                 << " " << Bfield[step][2]
                        #endif
                                 << " " << kin[step] << " " << mom[step][0]
                        #if (GEMPIC_VDIM > 1)
                                 << " " << mom[step][1]
                        #endif
                        #if (GEMPIC_VDIM > 2)
                                 << " " << mom[step][2]
                        #endif
                                 << std::endl;
        }
        ofsfun.close();
    }
};

#endif
