#include <tinyexpr.h>

#include <AMReX.H>
#include <AMReX_Print.H>
#include <AMReX_PlotFileUtil.H>
#include <AMReX_ParmParse.H>
#include <AMReX_Print.H>
#include <AMReX_Particles.H>

#include <GEMPIC_Config.H>
#include <GEMPIC_initializer.H>
#include <GEMPIC_loop_preparation.H>
#include <GEMPIC_maxwell_yee.H>
#include <GEMPIC_particle_groups.H>
#include <GEMPIC_particle_positions.H>
#include <GEMPIC_sampler.H>
#include <GEMPIC_time_loop_boris_fd.H>
#include <GEMPIC_time_loop_hs_fem.H>
#include <GEMPIC_time_loop_hsall_fem.H>

using namespace std;
using namespace amrex;
using namespace Gempic;

using namespace Diagnostics_Output;
using namespace Field_solvers;
using namespace Init;
using namespace Particles;
using namespace Sampling;
using namespace Time_Loop;

template<int vdim, int numspec, int degx, int degy, int degz, int electromagnetic=true>
void main_main (bool ctest)
{
    bool readinfile = false;
    // ------------------------------------------------------------------------------
    // ------------PARAMETERS--------------------------------------------------------

    // compile parameters
    std::array<int, GEMPIC_SPACEDIM> degs = {AMREX_D_DECL(degx, degy, degz)};
    int maxdeg = *(std::max_element(degs.begin(), degs.end()));
    int Nghost = maxdeg;

    // initialize parameters
    std::string sim_name;
    std::array<int,GEMPIC_SPACEDIM> n_cell_vector;
    std::array<int, numspec> n_part_per_cell;
    int n_steps;
    int freq_x;
    int freq_v;
    int freq_slice;
    std::array<int,GEMPIC_SPACEDIM> is_periodic_vector;
    int max_grid_size;
    amrex::Real dt;
    std::array<amrex::Real, numspec> charge;
    std::array<amrex::Real, numspec> mass;
    amrex::Real kx;
    amrex::Real ky;
    amrex::Real kz;
    std::array<amrex::Real,GEMPIC_SPACEDIM> k;
    std::string WF;
    std::string Bx;
    std::string By;
    std::string Bz;
    std::string phi;
    std::string rho = "0.0";
    int num_gaussians;
    int propagator;
    bool time_staggered;
    amrex::Real tolerance_particles;
    int restart;
    std::string checkpoint_file;
    int curr_step;

    // parse parameters
    amrex::ParmParse pp;

    std::array<std::vector<amrex::Real>, vdim> VM{};
    std::array<std::vector<amrex::Real>, vdim> VD{};
    std::array<std::vector<amrex::Real>, vdim> VW{};

    if (ctest) {
        sim_name = "Weibel";
        n_cell_vector[0] = 24;
        n_cell_vector[1] = 8;
        n_cell_vector[2] = 8;
        n_part_per_cell = {100};
        n_steps = 10;
        freq_x = 12;
        freq_v = 12;
        freq_slice = 12;
        is_periodic_vector[0] = 1;
        is_periodic_vector[1] = 1;
        is_periodic_vector[2] = 1;
        max_grid_size = 4;
        dt = 0.02;
        charge[0] = -1.0;
        mass[0] = 1.0;
        kx = 1.25;
        ky = 1.25;
        kz = 1.25;
        WF = "1.0 + 0.0 * cos(kvarx * x)";
        Bx = "0.0";
        By = "0.0";
        Bz = "1e-3 * cos(kvarx * x)";
        phi = "4 * 0.5 * cos(0.5 * x)";
        num_gaussians = 1;
        tolerance_particles = 1.e-10;

        for (int j=0; j<vdim; j++) {
            VM[j].push_back(0.0);
            VW[j].push_back(1.0);
        }
        VD[0].push_back(0.02/sqrt(2));
        VD[1].push_back(sqrt(12)*VD[0][0]);
        VD[2].push_back(VD[1][0]);
        restart = 0;
        checkpoint_file = "";
        curr_step = 0;

    } else {
        pp.get("sim_name",sim_name);
        pp.get("n_cell_vector",n_cell_vector);
        pp.get("n_part_per_cell",n_part_per_cell);
        pp.get("n_steps",n_steps);
        pp.get("freq_x",freq_x);
        pp.get("freq_v",freq_v);
        pp.get("freq_slice",freq_slice);
        pp.get("is_periodic_vector",is_periodic_vector);
        pp.get("max_grid_size",max_grid_size);
        pp.get("dt",dt);
        pp.get("charge",charge);
        pp.get("mass",mass);
        pp.get("kx",kx);
        pp.get("ky",ky);
        pp.get("kz",kz);
        pp.get("WF",WF);
        pp.get("Bx",Bx);
        pp.get("By",By);
        pp.get("Bz",Bz);
        pp.get("phi",phi);
        pp.get("num_gaussians",num_gaussians);
        pp.get("propagator",propagator);
        pp.get("tolerance_particles", tolerance_particles);
        pp.get("restart", restart);
        pp.get("checkpoint_file", checkpoint_file);
        pp.get("curr_step", curr_step);

        std::array<double, vdim> read_tmp_M;
        std::array<double, vdim> read_tmp_D;
        std::array<double, vdim> read_tmp_W;

        for (int i=0; i<num_gaussians; i++) {
            std::string name_str_M = "velocity_mean_" +  std::to_string(i);
            std::string name_str_D = "velocity_deviation_" +  std::to_string(i);
            std::string name_str_W = "velocity_weight_" +  std::to_string(i);
            const char *name_char_M = name_str_M.c_str();
            const char *name_char_D = name_str_D.c_str();
            const char *name_char_W = name_str_W.c_str();
            pp.get(name_char_M,read_tmp_M);
            pp.get(name_char_D,read_tmp_D);
            pp.get(name_char_W,read_tmp_W);
            for (int j=0; j<vdim; j++) {
                VM[j].push_back(read_tmp_M[j]);
                VD[j].push_back(read_tmp_D[j]);
                VW[j].push_back(read_tmp_W[j]);
            }
        }

        // Depending on which propagator is chosen, staggering in time is needed or not
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
        default:
            break;
        }

    }

    // initialize amrex data structures from parameters
    amrex::IntVect n_cell(AMREX_D_DECL(n_cell_vector[0],n_cell_vector[1],n_cell_vector[2]));
    amrex::IntVect is_periodic(AMREX_D_DECL(is_periodic_vector[0],is_periodic_vector[1],is_periodic_vector[2]));

    // functions
    double x, y, z;
    int err;
    te_variable read_vars[] = {{"x", &x}, {"y", &y}, {"z", &z}, {"kvarx", &kx}, {"kvary", &ky}, {"kvarz", &kz}};
    int varcount = 6;
    te_expr *WF_parse = te_compile(WF.c_str(), read_vars, varcount, &err);

    te_expr *Bx_parse = te_compile(Bx.c_str(), read_vars, varcount, &err);
    te_expr *By_parse = te_compile(By.c_str(), read_vars, varcount, &err);
    te_expr *Bz_parse = te_compile(Bz.c_str(), read_vars, varcount, &err);

    te_variable read_vars_poi[] = {{"x", &x}, {"y", &y}, {"z", &z}};
    varcount = 3;
    te_expr *rho_parse = te_compile(rho.c_str(), read_vars_poi, varcount, &err);
    te_expr *phi_parse = te_compile(phi.c_str(), read_vars_poi, varcount, &err);

    // ------------------------------------------------------------------------------
    // ------------INITIALIZE GEMPIC-STRUCTURES--------------------------------------

    //initializer
    initializer<vdim, numspec> init;
    k = {AMREX_D_DECL(kx,ky,kz)};
    init.initialize_from_parameters(n_cell,max_grid_size,is_periodic,Nghost,dt,n_steps,charge,mass,n_part_per_cell,k,
                                    VM,VD,VW,tolerance_particles);

    // infrastructure
    infrastructure infra;
    init.initialize_infrastructure(&infra);

    // maxwell_yee
    maxwell_yee<vdim> mw_yee(init, infra, init.Nghost);
    mw_yee.init_rho_phi(infra, phi_parse, rho_parse, &x, &y, &z);

    // particles
    particle_groups<vdim, numspec> part_gr(init, infra);

    diagnostics<vdim, numspec,degx,degy,degz> diagn(mw_yee.nsteps, freq_x, freq_v, freq_slice, sim_name);

    //------------------------------------------------------------------------------
    // initialize particles:
    if (restart == 0) {
        for (int spec=0; spec<numspec; spec++) {
                // Reading in species information
                std::array<std::vector<amrex::Real>, vdim> VM_tmp;
                std::array<std::vector<amrex::Real>, vdim> VD_tmp;
                std::array<std::vector<amrex::Real>, vdim> VW_tmp;
            if (!ctest) {
                string line;
                ifstream myfile ("species_data_" + to_string(vdim) + "V_" + to_string(spec) + ".txt");
                int gaussian_num = -1;
                int comp = -1;
                int prop = -1; // 0 -> mean, 1 -> deviation, 2 -> weight
                int line_num = -1;

                if (myfile.is_open())
                {
                    while ( getline (myfile,line) )
                    {
                        line_num++;
                        if (line == "# Gaussian") {
                            gaussian_num++;
                        } else {
                            if (line.at(0) == '#') {
                                prop = (prop+1)%3;
                            } else {
                                comp = (comp+1)%vdim;
                                switch (prop) {
                                case 0:
                                    VM_tmp[comp].push_back(stod(line));
                                    break;
                                case 1:
                                    VD_tmp[comp].push_back(stod(line));
                                    break;
                                case 2:
                                    VW_tmp[comp].push_back(stod(line));
                                    break;
                                }
                            }
                        }
                    }
                    myfile.close();
                }
                else cout << "Unable to open file for species " << spec << std::endl;
            } else {
                VM_tmp = VM;
                VD_tmp = VD;
                VW_tmp = VW;
            }

            if (readinfile) {

                int species = 0;
                for(amrex::MFIter mfi=(*(part_gr).mypc[species]).MakeMFIter(0); mfi.isValid(); ++mfi) {
                    if(mfi.index() == 0) {
                        using ParticleType = amrex::Particle<vdim+1, 0>; // Particle template
                        amrex::ParticleTile<vdim+1, 0, 0, 0>& particles = (*(part_gr).mypc[species]).GetParticles(0)[std::make_pair(mfi.index(), mfi.LocalTileIndex())];

                        string line;
                        string filename = "particle_input.txt";
                        ifstream myfile (filename);
                        if (myfile.is_open())
                        {
                            while ( getline (myfile,line) )
                            {
                                std::vector<std::string> readVal;
                                std::istringstream iss(line);
                                for(std::string line; iss >> line; ){
                                    readVal.push_back(line);
                                }

                                std::array<amrex::Real,GEMPIC_SPACEDIM> position;
                                std::array<amrex::Real,vdim> velocity;
                                amrex::Real weight = stod(readVal[GEMPIC_SPACEDIM+vdim]);
                                for (int comp = 0; comp < vdim; comp++) {
                                    position[comp] = stod(readVal[comp]);
                                    velocity[comp] = stod(readVal[GEMPIC_SPACEDIM+comp]);
                                }
                                (part_gr).add_particle(position, velocity, weight, particles);
                            }
                        }
                        else cout << "Unable to open particle input " << spec << std::endl;

                    }
                }
                (*(part_gr).mypc[0]).Redistribute();
            } else {
                init_particles_full_domain<vdim,numspec>(infra, part_gr, init, VM_tmp, VD_tmp, VW_tmp, spec, WF_parse, &x, &y, &z);
            }
        }

     /*   for (amrex::ParIter<vdim+1,0,0,0> pti(*(part_gr).mypc[0], 0); pti.isValid(); ++pti) {

            const auto& particles = pti.GetArrayOfStructs();
            const long np = pti.numParticles();
            for (int pp=0;pp<np;pp++) {
                std::cout << "(" << particles[pp].pos(0) << "," << particles[pp].pos(1) << "," << particles[pp].pos(2) << ") (" <<
                             particles[pp].rdata(0) << "," << particles[pp].rdata(1) << "," << particles[pp].rdata(2) << ") " <<
                             particles[pp].rdata(3) << std::endl;
            }
        }*/


        //------------------------------------------------------------------------------
        // solve:
        loop_preparation<vdim, numspec>(infra, &mw_yee, &part_gr, &diagn, Bx_parse, By_parse, Bz_parse, &x, &y, &z,time_staggered);
    } else {
        Gempic_ReadCheckpointFile (&mw_yee, &part_gr, &infra, checkpoint_file, curr_step);
    }
    std::ofstream ofs("vlasov_maxwell.output", std::ofstream::out);
    if (ctest) AllPrintToFile("test_output_pre_rename.output") << endl;
    switch (propagator) {
    case 0:
        time_loop_boris_fd<vdim, numspec, degx, degy, degz, electromagnetic>(infra, &mw_yee, &part_gr, &diagn, ctest, &ofs);
        break;
    case 1:
        time_loop_hs_fem<vdim, numspec, degx, degy, degz, electromagnetic>(infra, &mw_yee, &part_gr, &diagn, ctest, &ofs);
        break;
    case 2:
        time_loop_hsall_fem<vdim, numspec, degx, degy, degz, electromagnetic>(infra, &mw_yee, &part_gr, &diagn, ctest, &ofs);
        break;
    default:
        break;
    }
    if (ctest & (ParallelDescriptor::MyProc()==0)) std::rename("test_output_pre_rename.output.0", "vlasov_maxwell.output");
}

int main(int argc, char* argv[])
{
    amrex::Initialize(argc,argv);

#if (GEMPIC_SPACEDIM == 1)
    if (argc==0) {
        main_main<1, 1, 1, 1, 1>(argc==1); // run for ctest
    }
    main_main<2, GEMPIC_NUMSPEC, 1, 1, 1, GEMPIC_ELECTROMAGNETIC>(argc==1);
#elif (GEMPIC_SPACEDIM == 2)
    if (argc==0) {
        main_main<2, GEMPIC_NUMSPEC, 1, 1, 1>(argc==1); // run for ctest
    }
    main_main<3, GEMPIC_NUMSPEC, 1, 1, 1, GEMPIC_ELECTROMAGNETIC>(argc==1);
#elif (GEMPIC_SPACEDIM == 3)
    main_main<3, GEMPIC_NUMSPEC, 1, 1, 1, GEMPIC_ELECTROMAGNETIC>(argc==1);
#endif

    amrex::Finalize();
}



