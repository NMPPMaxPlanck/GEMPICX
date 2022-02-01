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

namespace Gempic
{
    namespace Vlasov_Maxwell
    {

        template <int vdim, int numspec>
        class gempic_parameters
        {
        public:
            // compile time parameters
            int Nghost;

            // parameters from main input file
            std::string sim_name = "simulation";
            // grid
            amrex::RealBox real_box;
            amrex::Vector<amrex::Real> domain_lo = {0.0, 0.0, 0.0};
            amrex::Vector<amrex::Real> domain_hi = {1.0, 1.0, 1.0};
            amrex::IntVect is_periodic = {1, 1, 1};
            amrex::IntVect n_cell = {8, 8, 8};
            amrex::IntVect max_grid_size = {4, 4, 4}; // for parallelisation
            // for periodic directions length of domain given by 2*pi/k[i]
            amrex::Vector<amrex::Real> k = {1.25, 1.25, 1.25};
            amrex::Real kx, ky, kz;

            // time discretisation
            int n_steps = 10;
            amrex::Real dt = 0.01;
            int propagator = 3;
            int strang_order = 2;
            bool time_staggered = false;

            // particles (for each species)
            int nPartCellTot;
            std::array<int, numspec> n_part_per_cell = {100};
            amrex::GpuArray<amrex::Real, numspec> charge = {-1};
            amrex::GpuArray<amrex::Real, numspec> mass = {1};
            // Initial velocity distribution (sum of gaussians)
            amrex::GpuArray<amrex::Real, numspec> num_gaussians;
            std::vector<std::vector<std::vector<amrex::Real>>> VM{}; // species, gaussian, vdim
            std::vector<std::vector<std::vector<amrex::Real>>> VD{};
            std::vector<std::vector<std::vector<amrex::Real>>> VW{};

            std::string WF = "1.0"; // particle density
            amrex::Parser WFparse;
            amrex::ParserExecutor<3> WFeval;

            std::string Bx = "0.0";
            amrex::Parser BxParse;
            amrex::ParserExecutor<3> BxEval;
            std::string By = "0.0";
            amrex::Parser ByParse;
            amrex::ParserExecutor<3> ByEval;
            std::string Bz = "0.0";
            amrex::Parser BzParse;
            amrex::ParserExecutor<3> BzEval;
            std::string phi = "0.0";
            amrex::Parser phiParse;
            amrex::ParserExecutor<3> phiEval;
            std::string rho = "0.0";
            amrex::Parser rhoParse;
            amrex::ParserExecutor<3> rhoEval;

            // code data
            amrex::Real tolerance_particles = 1e-10;
            int max_coarsening_level = 30;
            int restart = 0;
            std::string checkpoint_file = "";
            int curr_step = 0;

            // Diagnostics
            int freq_x = 1000; //
            int freq_v = 1000;
            int freq_slice = 1000; // time evolution

            // Scaling parameters
            amrex::Real curlB_scaling = 1.;
            amrex::Real epref_ep0 = 1.;
            amrex::Real energy_b_scaling = 1.;

            // The default constructor.
            /**
             * Default Constructor.
             */
            gempic_parameters(){};

            void init_Nghost(int degx, int degy, int degz)
            {
                std::array<int, GEMPIC_SPACEDIM> degs = {AMREX_D_DECL(degx, degy, degz)};
                int maxdeg = *(std::max_element(degs.begin(), degs.end()));
                Nghost = maxdeg;
            }

            void read_pp_params()
            {
                // Local variables
                amrex::ParmParse pp;
                std::array<int, GEMPIC_SPACEDIM> is_periodic_vector = {1, 1, 1};
                std::array<int, GEMPIC_SPACEDIM> max_grid_size_vector = {4, 4, 4};
                std::array<int, GEMPIC_SPACEDIM> n_cell_vector = {10, 10, 10};

                pp.query("sim_name", sim_name);
                // computational domain
                pp.queryarr("domain_lo", domain_lo);
                pp.queryarr("domain_hi", domain_hi);
                pp.queryarr("k", k);
                pp.query("is_periodic_vector", is_periodic_vector);
                is_periodic = amrex::IntVect{AMREX_D_DECL(is_periodic_vector[0], is_periodic_vector[1], is_periodic_vector[2])};
                pp.query("n_cell_vector", n_cell_vector);
                n_cell = amrex::IntVect{AMREX_D_DECL(n_cell_vector[0], n_cell_vector[1], n_cell_vector[2])};
                pp.query("max_grid_size_vector", max_grid_size_vector);
                max_grid_size = amrex::IntVect{AMREX_D_DECL(max_grid_size_vector[0], max_grid_size_vector[1], max_grid_size_vector[2])};

                // time stepping
                pp.query("n_steps", n_steps);
                pp.query("dt", dt);
                pp.query("propagator", propagator);
                pp.query("strang_order", dt);
                pp.query("time_staggered", dt);

                // particle data
                pp.query("WF", WF); // initial density

                for (int spec = 0; spec < numspec; spec++)
                {
                    std::vector<std::vector<amrex::Real>> vvm, vvt, vvw;
                    std::string specString = "species" + std::to_string(spec);
                    amrex::ParmParse ppSpec(specString);
                    ppSpec.query("n_part_per_cell", n_part_per_cell[spec]);
                    ppSpec.query("charge", charge[spec]);
                    ppSpec.query("mass", mass[spec]);
                    ppSpec.query("num_gaussian", num_gaussians[spec]);
                    for (int ng = 0; ng < num_gaussians[spec]; ng++)
                    {
                        std::vector<amrex::Real> vm, vt, vw;
                        std::string vmString = "vMean_g" + std::to_string(ng);
                        ppSpec.queryarr(vmString.c_str(), vm);
                        vvm.push_back(vm);
                        std::string vtString = "vThermal_g" + std::to_string(ng);
                        ppSpec.queryarr(vtString.c_str(), vt);
                        vvt.push_back(vt);
                        std::string vwString = "vWeight_g" + std::to_string(ng);
                        ppSpec.queryarr(vwString.c_str(), vw);
                        vvw.push_back(vw);
                    }
                    VM.push_back(vvm);
                    VD.push_back(vvt);
                    VW.push_back(vvw);
                }

                // initial fields
                pp.query("Bx", Bx);
                pp.query("By", By);
                pp.query("Bz", Bz);
                pp.query("phi", phi); // initial density
                pp.query("rho", rho); // initial density

                // code data
                pp.query("tolerance_particles", tolerance_particles);
                pp.query("max_coarsening_level", max_coarsening_level);
                pp.query("restart", restart);
                pp.query("checkpoint_file", checkpoint_file);
                pp.query("curr_step", curr_step);

                // diagnostics
                pp.query("freq_x", freq_x);
                pp.query("freq_v", freq_v);
                pp.query("freq_slice", freq_slice);

                // scaling parameters
                pp.query("epref_ep0", epref_ep0);
                pp.query("curlB_scaling", curlB_scaling);
                pp.query("energy_b_scaling", energy_b_scaling);

                // set dependent parameters
                set_computed_params();
            }

            void set_params(std::string sm = "Weibel",
                            amrex::IntVect n_cell_vector = {AMREX_D_DECL(24, 8, 8)}, // std::array<int,GEMPIC_SPACEDIM> n_cell_vector = {AMREX_D_DECL(24, 8, 8)},
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
                            amrex::GpuArray<int, numspec> NG = {1},
                            int PROP = 0,
                            amrex::Real TOL_P = 1.e-10,
                            int MAX_C = 30,
                            int RST = 0,
                            std::string CF = "",
                            int CD = 0)
            {
                sim_name = sm;
                n_cell = n_cell_vector; // amrex::IntVect{AMREX_D_DECL(n_cell_vector[0],n_cell_vector[1],n_cell_vector[2])};
                for (int i = 0; i < GEMPIC_SPACEDIM; i++)
                    n_part_per_cell[i] = n_part_per_cell_in[i];
                n_steps = NS;
                freq_x = FX;
                freq_v = FV;
                freq_slice = FS;
                // is_periodic_vector = PER;
                is_periodic = is_periodic_vector;     // amrex::IntVect{AMREX_D_DECL(is_periodic_vector[0],is_periodic_vector[1],is_periodic_vector[2])};
                max_grid_size = max_grid_size_vector; // amrex::IntVect{AMREX_D_DECL(max_grid_size_vector[0],max_grid_size_vector[1],max_grid_size_vector[2])};
                dt = DT;
                for (int i = 0; i < numspec; i++)
                    charge[i] = C[i];
                for (int i = 0; i < numspec; i++)
                    mass[i] = M[i];
                k = {K, K, K};
                WF = wf;
                Bx = BX;
                By = BY;
                Bz = BZ;
                phi = PHI;
                for (int i = 0; i < numspec; i++)
                    num_gaussians[i] = NG[i];
                propagator = PROP;
                tolerance_particles = TOL_P;
                restart = RST;
                checkpoint_file = CF;
                curr_step = CD;

                set_computed_params();
            }

            void set_params_Weibel()
            {
                sim_name = "Weibel";
                assert(numspec == 1); // Only electrons
                n_cell = {AMREX_D_DECL(24, 8, 8)};
                n_part_per_cell[0] = 100;
                n_steps = 10;
                freq_x = 12;
                freq_v = 12;
                freq_slice = 12;
                is_periodic = {AMREX_D_DECL(1, 1, 1)};   // amrex::IntVect{AMREX_D_DECL(is_periodic_vector[0],is_periodic_vector[1],is_periodic_vector[2])};
                max_grid_size = {AMREX_D_DECL(4, 4, 4)}; // amrex::IntVect{AMREX_D_DECL(max_grid_size_vector[0],max_grid_size_vector[1],max_grid_size_vector[2])};
                dt = 0.02;
                charge[0] = -1.0;
                mass[0] = 1.0;
                k = {AMREX_D_DECL(1.25, 1.25, 1.25)};
                WF = "1.0";
                Bx = "0.0";
                By = "0.0";
                Bz = "1e-3 * cos(kvarx * x)";
                phi = "4 * 0.5 * cos(0.5 * x)";
                propagator = 0;
                tolerance_particles = 1.e-10;
                restart = 0;
                checkpoint_file = "";
                curr_step = 0;
                num_gaussians[0] = 1;
                VM = {{{0.0, 0.0, 0.0}}};
                VD = {{{0.01414213562, 0.04898979486, 0.04898979486}}};
                VW = {{{1.0, 1.0, 1.0}}};
            }

            // void set_prop_related()
            // {
            //     switch (propagator)
            //     {
            //     case 0:
            //         time_staggered = true;
            //         break;
            //     case 1:
            //         time_staggered = false;
            //         Nghost += 3;
            //         break;
            //     case 2:
            //         time_staggered = false;
            //         Nghost += 3;
            //         break;
            //     case 3:
            //         time_staggered = false;
            //         Nghost += 3;
            //         break;
            //     default:
            //         break;
            //     }
            // }

            void
            set_computed_params()
            {
                switch (propagator)
                {
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

                nPartCellTot = 0;
                for (int spec = 0; spec < numspec; spec++)
                {
                    nPartCellTot += n_part_per_cell[spec];
                }

                // In periodic directions domain is[0, 2pi / k]
                for (int i = 0; i < GEMPIC_SPACEDIM; i++)
                {
                    if (is_periodic[i] == 1)
                    {
                        domain_lo[i] = 0.0;
                        domain_hi[i] = 2 * M_PI / k[i];
                    }
                }
                real_box = amrex::RealBox({AMREX_D_DECL(domain_lo[0], domain_lo[1], domain_lo[2])},
                                          {AMREX_D_DECL(domain_hi[0], domain_hi[1], domain_hi[2])});

                kx = k[0];
                ky = k[1];
                kz = k[2];

                // Parsers
                // --------
                WFparse = amrex::Parser(WF);
                WFparse.setConstant("kvarx", k[0]);
                WFparse.setConstant("kvary", k[1]);
                WFparse.setConstant("kvarz", k[2]);
                WFparse.registerVariables({"x", "y", "z"});
                WFeval = WFparse.compile<3>(); // 3 because there are three variables.

                // initial fields
                BxParse = amrex::Parser(Bx);
                BxParse.setConstant("kvarx", k[0]);
                BxParse.setConstant("kvary", k[1]);
                BxParse.setConstant("kvarz", k[2]);
                BxParse.registerVariables({"x", "y", "z"});
                BxEval = BxParse.compile<3>();

                ByParse = amrex::Parser(By);
                ByParse.setConstant("kvarx", k[0]);
                ByParse.setConstant("kvary", k[1]);
                ByParse.setConstant("kvarz", k[2]);
                ByParse.registerVariables({"x", "y", "z"});
                ByEval = BzParse.compile<3>();

                BzParse = amrex::Parser(Bz);
                BzParse.setConstant("kvarx", k[0]);
                BzParse.setConstant("kvary", k[1]);
                BzParse.setConstant("kvarz", k[2]);
                BzParse.registerVariables({"x", "y", "z"});
                BzEval = BzParse.compile<3>();

                phiParse = amrex::Parser(phi);
                phiParse.setConstant("kvarx", k[0]);
                phiParse.setConstant("kvary", k[1]);
                phiParse.setConstant("kvarz", k[2]);
                phiParse.registerVariables({"x", "y", "z"});
                phiEval = phiParse.compile<3>();

                rhoParse = amrex::Parser(rho);
                rhoParse.setConstant("kvarx", k[0]);
                rhoParse.setConstant("kvary", k[1]);
                rhoParse.setConstant("kvarz", k[2]);
                rhoParse.registerVariables({"x", "y", "z"});
                rhoEval = rhoParse.compile<3>();
            }

            void init_particles_from_file(particle_groups<vdim, numspec> *part_gr,
                                          int species,
                                          std::string filename = "particle_input.txt")
            {
                for (amrex::MFIter mfi = (*(*part_gr).mypc[species]).MakeMFIter(0); mfi.isValid(); ++mfi)
                {
                    if (mfi.index() == 0)
                    {
                        // using ParticleType = amrex::Particle<vdim+1, 0>; // Particle template
                        amrex::ParticleTile<vdim + 1, 0, 0, 0> &particles = (*(*part_gr).mypc[species]).GetParticles(0)[std::make_pair(mfi.index(), mfi.LocalTileIndex())];

                        std::string line;
                        std::ifstream myfile(filename);
                        if (myfile.is_open())
                        {
                            while (getline(myfile, line))
                            {
                                std::vector<std::string> readVal;
                                std::istringstream iss(line);
                                for (std::string line; iss >> line;)
                                {
                                    readVal.push_back(line);
                                }

                                amrex::GpuArray<amrex::Real, GEMPIC_SPACEDIM> position;
                                amrex::GpuArray<amrex::Real, vdim> velocity;
                                amrex::Real weight = stod(readVal[GEMPIC_SPACEDIM + vdim]);
                                for (int comp = 0; comp < vdim; comp++)
                                {
                                    position[comp] = stod(readVal[comp]);
                                    velocity[comp] = stod(readVal[GEMPIC_SPACEDIM + comp]);
                                }
                                (*part_gr).add_particle(position, velocity, weight, particles);
                            }
                        }
                        else
                            std::cout << "Unable to open particle input " << species << std::endl;
                    }
                }
                (*(*part_gr).mypc[0]).Redistribute();
            }
        };

    }
}

#endif
