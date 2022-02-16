#ifndef VLASOV_MAXWELL_H_
#define VLASOV_MAXWELL_H_

#include <AMReX.H>
#include <AMReX_Box.H>
#include <AMReX_BoxArray.H>
#include <AMReX_CoordSys.H>
#include <AMReX_DistributionMapping.H>
#include <AMReX_FillPatchUtil.H>
#include <AMReX_Geometry.H>
#include <AMReX_IndexType.H>
#include <AMReX_IntVect.H>
#include <AMReX_MFIter.H>
#include <AMReX_MLMG.H>
#include <AMReX_MLNodeLaplacian.H>
#include <AMReX_MultiFab.H>
#include <AMReX_ParmParse.H>
#include <AMReX_PlotFileUtil.H>
#include <AMReX_Print.H>
#include <AMReX_REAL.H>
#include <AMReX_RealBox.H>

#include <GEMPIC_gempic_norm.H>
#include <GEMPIC_computational_domain.H>
#include <GEMPIC_MLNodeLaplacian_FD.H>
#include <GEMPIC_particle_mesh_coupling.H>
#include <GEMPIC_particle_groups.H>

using namespace CompDom;
using namespace Particles;
using namespace Utils;

namespace Gempic {
namespace Vlasov_Maxwell {

template<int vdim, int numspec>
class gempic_parameters
{
public:

    // compile time parameters
    int Nghost;

    // parameters from main input file
    std::string sim_name = "simulation";
    amrex::IntVect n_cell;
    std::array<int, numspec> n_part_per_cell = {100};
    int n_steps = 10;
    int freq_x = 1000;
    int freq_v = 1000;
    int freq_slice = 1000;
    amrex::IntVect is_periodic;
    amrex::IntVect max_grid_size;
    amrex::Real dt = 0.01;
    amrex::GpuArray<amrex::Real, numspec> charge = {-1};
    amrex::GpuArray<amrex::Real, numspec> mass = {1};
    amrex::GpuArray<amrex::Real, 3> k = {1.25, 1.25, 1.25};
    amrex::Real kx, ky, kz;
    std::string WF = "1.0";
    std::string Bx = "0.0";
    std::string By = "0.0";
    std::string Bz = "0.0";
    std::string phi = "0.0";
    std::string rho = "0.0";
    int num_gaussians = 1;
    int propagator = 3;
    int strang_order = 2;
    bool time_staggered = false;
    amrex::Real tolerance_particles = 1e-10;
    int max_coarsening_level = 30;
    int restart = 0;
    std::string checkpoint_file = "";
    int curr_step = 0;

    // parameters from particle species file
    std::array<std::vector<amrex::Real>, vdim> VM{};
    std::array<std::vector<amrex::Real>, vdim> VD{};
    std::array<std::vector<amrex::Real>, vdim> VW{};

    // Parser
    amrex::ParmParse pp;

    // Params that used to live in initializer
    int NP_cell_tot;
    amrex::RealBox real_box;

    // Scaling parameters
    amrex::Real curlB_scaling = 1.;
    amrex::Real epref_ep0 = 1.;
    amrex::Real energy_b_scaling = 1.;

    // The default constructor.
    /**
     * Default Constructor.
     */
    gempic_parameters (){};

    void init_Nghost (int degx, int degy, int degz) {
        std::array<int, GEMPIC_SPACEDIM> degs = {AMREX_D_DECL(degx, degy, degz)};
        int maxdeg = *(std::max_element(degs.begin(), degs.end()));
        Nghost = maxdeg;
    }

    void read_pp_params () {

        std::array<int,GEMPIC_SPACEDIM> is_periodic_vector = {1,1,1};
        std::array<int,GEMPIC_SPACEDIM> max_grid_size_vector = {5,5,5};
        std::array<int,GEMPIC_SPACEDIM> n_cell_vector = {10, 10, 10};
        std::array<amrex::Real, numspec> charge_in = {-1};
        std::array<amrex::Real, numspec> mass_in = {1};
        std::array<amrex::Real, 3> k_in = {1.25, 1.25, 1.25};
        std::array<int, numspec> n_part_per_cell_in = {100};

        pp.query("sim_name",sim_name);
        pp.query("n_cell_vector",n_cell_vector);
        n_cell = amrex::IntVect{AMREX_D_DECL(n_cell_vector[0],n_cell_vector[1],n_cell_vector[2])};
        pp.query("n_part_per_cell",n_part_per_cell_in);
        for (int i = 0; i < numspec; i++) n_part_per_cell[i] = n_part_per_cell_in[i];
        pp.query("n_steps",n_steps);
        pp.query("freq_x",freq_x);
        pp.query("freq_v",freq_v);
        pp.query("freq_slice",freq_slice);
        pp.query("is_periodic_vector",is_periodic_vector);
        is_periodic = amrex::IntVect{AMREX_D_DECL(is_periodic_vector[0],is_periodic_vector[1],is_periodic_vector[2])};
        pp.query("max_grid_size_vector",max_grid_size_vector);
        max_grid_size = amrex::IntVect{AMREX_D_DECL(max_grid_size_vector[0],max_grid_size_vector[1],max_grid_size_vector[2])};
        pp.query("dt",dt);
        pp.query("charge",charge_in);
        for (int i = 0; i < numspec; i++) charge[i] = charge_in[i];
        pp.query("mass",mass_in);
        for (int i = 0; i < numspec; i++) mass[i] = mass_in[i];
        pp.query("k",k_in);
        for (int i = 0; i < numspec; i++) k[i] = k_in[i];
        pp.query("Bx",Bx);
        pp.query("By",By);
        pp.query("Bz",Bz);
        pp.query("phi",phi);
        pp.query("num_gaussians",num_gaussians);
        pp.query("propagator",propagator);
        pp.query("tolerance_particles", tolerance_particles);
        pp.query("max_coarsening_level", max_coarsening_level);
        pp.query("restart", restart);
        pp.query("checkpoint_file", checkpoint_file);
        pp.query("curr_step", curr_step);

        pp.query("epref_ep0", epref_ep0);
        pp.query("curlB_scaling",curlB_scaling);
        pp.query("energy_b_scaling", energy_b_scaling);

        pp.query("WF",WF);
        for (int spec=0; spec<numspec; spec++) {
            std::array<amrex::Real, 3> vm, vd, vw;
            std::string vms = "velocity_mean_"+std::to_string(spec);
            pp.query(vms.c_str(),vm);
            std::string vds = "velocity_deviation_"+std::to_string(spec);
            pp.query(vds.c_str(),vd);
            std::string vws = "velocity_weight_"+std::to_string(spec);
            pp.query(vws.c_str(),vw);
            for (int j = 0; j < vdim; j++){
                VM[j].push_back(vm[j]);
                VD[j].push_back(vd[j]);
                VW[j].push_back(vw[j]);
            }
        }

        set_prop_related();
    }

    void read_particle_spec (int spec) {
        VM = {};
        VD = {};
        VW = {};
        std::string line;
        std::ifstream myfile ("species_data_" + std::to_string(vdim) + "V_" + std::to_string(spec) + ".txt");
        int gaussian_num = -1;
        int comp = -1;
        int prop = -1; // 0 -> mean, 1 -> deviation, 2 -> weight
        int line_num = -1;
        bool WF_line = true;

        if (myfile.is_open())
        {
            while ( getline (myfile,line) )
            {
                if (WF_line) {
                    WF = line;
                    WF_line = false;
                } else {
                line_num++;
                if ((line.at(0) == '#') & (line.at(2) == 'G')) {
                    gaussian_num++;
                } else {
                    if (line.at(0) == '#') {
                        prop = (prop+1)%3;
                    } else {
                        comp = (comp+1)%vdim;
                        switch (prop) {
                        case 0:
                            VM[comp].push_back(stod(line));
                            break;
                        case 1:
                            VD[comp].push_back(stod(line));
                            break;
                        case 2:
                            VW[comp].push_back(stod(line));
                            break;
                        }
                    }
                }
                }
            }
            myfile.close();
        }
        else std::cout << "Unable to open file for species " << spec << std::endl;
    }

    void set_params (std::string sm = "Weibel",
                     amrex::IntVect n_cell_vector = {AMREX_D_DECL(24, 8, 8)}, //std::array<int,GEMPIC_SPACEDIM> n_cell_vector = {AMREX_D_DECL(24, 8, 8)},
                     std::array<int, numspec> n_part_per_cell_in = {100},
                     int NS = 10,
                     int FX = 12,
                     int FV = 12,
                     int FS = 12,
                     amrex::IntVect is_periodic_vector = {AMREX_D_DECL(1, 1, 1)},
                     amrex::IntVect max_grid_size_vector = {AMREX_D_DECL(4, 4, 4)},
                     amrex::Real DT = 0.02,
                     amrex::GpuArray<amrex::Real, numspec> C = {-1.0},
                     amrex::GpuArray<amrex::Real, numspec> M = {1.0},
                     amrex::Real K = 1.25,
                     std::string wf = "1.0",
                     std::string BX = "0.0",
                     std::string BY = "0.0",
                     std::string BZ = "1e-3 * cos(kvarx * x)",
                     std::string PHI = "4 * 0.5 * cos(0.5 * x)",
                     int NG = 1,
                     int PROP = 0,
                     amrex::Real TOL_P = 1.e-10,
                     int MAX_C = 30,
                     int RST = 0,
                     std::string CF = "",
                     int CD = 0
                     ) {
        sim_name = sm;
        n_cell = n_cell_vector;//amrex::IntVect{AMREX_D_DECL(n_cell_vector[0],n_cell_vector[1],n_cell_vector[2])};
        for (int i=0; i<GEMPIC_SPACEDIM; i++) n_part_per_cell[i] = n_part_per_cell_in[i];
        n_steps = NS;
        freq_x = FX;
        freq_v = FV;
        freq_slice = FS;
        //is_periodic_vector = PER;
        is_periodic = is_periodic_vector;//amrex::IntVect{AMREX_D_DECL(is_periodic_vector[0],is_periodic_vector[1],is_periodic_vector[2])};
        max_grid_size = max_grid_size_vector;//amrex::IntVect{AMREX_D_DECL(max_grid_size_vector[0],max_grid_size_vector[1],max_grid_size_vector[2])};
        dt = DT;
        for (int i = 0; i < numspec; i++) charge[i] = C[i];
        for (int i = 0; i < numspec; i++) mass[i] = M[i];
        k = {K,K,K};
        WF = wf;
        Bx = BX;
        By = BY;
        Bz = BZ;
        phi = PHI;
        num_gaussians = NG;
        propagator = PROP;
        tolerance_particles = TOL_P;
        restart = RST;
        checkpoint_file = CF;
        curr_step = CD;

        set_prop_related();
    }



        void set_prop_related () {
            switch (propagator) {
            case 0:
                time_staggered = true;
                break;
            case 1:
                time_staggered = false;
                Nghost += 3;
                break;
            case 2:
                time_staggered = false;
                Nghost += 3;
                break;
            case 3:
                time_staggered = false;
                Nghost += 3;
                break;
            default:
                break;
            }
        }

        void set_computed_params() {
            NP_cell_tot = 0;
            for (int spec = 0; spec < numspec; spec++) {
                NP_cell_tot += n_part_per_cell[spec];
            }

            double twopi = 4 * asin(1.0); // 2.0*3.14159265359;

            amrex::Real boxLo[GEMPIC_SPACEDIM] = {AMREX_D_DECL(0,0,0)};
            amrex::Real boxHi[GEMPIC_SPACEDIM] = {AMREX_D_DECL(twopi/k[0],twopi/k[1],twopi/k[2])};

            real_box.setLo(boxLo);
            real_box.setHi(boxHi);

            kx = k[0];
            ky = k[1];
            kz = k[2];
        }

        void init_particles_from_file(particle_groups<vdim, numspec> *part_gr,
                                      int species,
                                      std::string filename = "particle_input.txt") {
            for(amrex::MFIter mfi=(*(*part_gr).mypc[species]).MakeMFIter(0); mfi.isValid(); ++mfi) {
                if(mfi.index() == 0) {
                    //using ParticleType = amrex::Particle<vdim+1, 0>; // Particle template
                    amrex::ParticleTile<vdim+1, 0, 0, 0>& particles = (*(*part_gr).mypc[species]).GetParticles(0)[std::make_pair(mfi.index(), mfi.LocalTileIndex())];

                    std::string line;
                    std::ifstream myfile (filename);
                    if (myfile.is_open())
                    {
                        while ( getline (myfile,line) )
                        {
                            std::vector<std::string> readVal;
                            std::istringstream iss(line);
                            for(std::string line; iss >> line; ){
                                readVal.push_back(line);
                            }

                            amrex::GpuArray<amrex::Real,GEMPIC_SPACEDIM> position;
                            amrex::GpuArray<amrex::Real,vdim> velocity;
                            amrex::Real weight = stod(readVal[GEMPIC_SPACEDIM+vdim]);
                            for (int comp = 0; comp < vdim; comp++) {
                                position[comp] = stod(readVal[comp]);
                                velocity[comp] = stod(readVal[GEMPIC_SPACEDIM+comp]);
                            }
                            (*part_gr).add_particle(position, velocity, weight, particles);
                        }
                    }
                    else std::cout << "Unable to open particle input " << species << std::endl;

                }
            }
            (*(*part_gr).mypc[0]).Redistribute();
        }


};

}
}

#endif
