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
#include <AMReX_Parser.H>
#include <AMReX_PlotFileUtil.H>
#include <AMReX_Print.H>
#include <AMReX_REAL.H>
#include <AMReX_RealBox.H>
//#include <GEMPIC_MLNodeLaplacian_FD.H>
#include <GEMPIC_computational_domain.H>
#include <GEMPIC_gempic_norm.H>
#include <GEMPIC_particle_groups.H>
#include <GEMPIC_particle_mesh_coupling.H>

using namespace CompDom;
using namespace Particles;
using namespace Utils;

void parseErrorMessage(std::string name)
{
    amrex::Print() << "Cannot find " + name + " input! Default value will be used\n";
}
template <unsigned int numspec>
class gempic_parameters
{
public:
    int Nghost;

    // parameters from main input file
    std::string sim_name = "simulation";
    // grid
    amrex::RealBox real_box;
    amrex::Vector<amrex::Real> domain_lo{AMREX_D_DECL(0.0, 0.0, 0.0)};
    amrex::Vector<amrex::Real> domain_hi{AMREX_D_DECL(1.0, 1.0, 1.0)};
    amrex::IntVect is_periodic{AMREX_D_DECL(1, 1, 1)};
    amrex::IntVect n_cell{AMREX_D_DECL(8, 8, 8)};
    amrex::IntVect max_grid_size{AMREX_D_DECL(4, 4, 4)}; 
    // for parallelisation
    // for periodic directions length of domain given by 2*pi/k[i]
    amrex::Vector<amrex::Real> k{AMREX_D_DECL(1.25, 1.25, 1.25)};
    amrex::Real kx, ky, kz;

    // time discretisation
    int n_steps = 10;
    amrex::Real dt = 0.01;
    int propagator = 3;
    int strang_order = 2;
    bool time_staggered = false;

    // particles (for each species)
    int nPartCellTot;
    amrex::GpuArray<int, numspec> n_part_per_cell = {100};
    amrex::GpuArray<amrex::Real, numspec> charge = {-1};
    amrex::GpuArray<amrex::Real, numspec> mass = {1};
    amrex::GpuArray<std::string, numspec> name = {"electrons"};
    // Initial velocity distribution (sum of gaussians)
    amrex::GpuArray<int, numspec> num_gaussians = {0};
    amrex::Vector<amrex::Vector<amrex::Vector<amrex::Real>>>
        meanVelocity{};  // species, gaussian, vdim
    amrex::Vector<amrex::Vector<amrex::Vector<amrex::Real>>> vThermal{};
    amrex::Vector<amrex::Vector<amrex::Real>> vWeight{};

    amrex::GpuArray<std::string, numspec> density = {"0"};  // particle density
    amrex::GpuArray<amrex::Parser, numspec> densityParse;
    amrex::GpuArray<amrex::ParserExecutor<GEMPIC_SPACEDIM + 1>, numspec> densityEval;

    std::string Bx = "0.0";
    amrex::Parser BxParse;
    amrex::ParserExecutor<GEMPIC_SPACEDIM + 1> BxEval;
    std::string By = "0.0";
    amrex::Parser ByParse;
    amrex::ParserExecutor<GEMPIC_SPACEDIM + 1> ByEval;
    std::string Bz = "0.0";
    amrex::Parser BzParse;
    amrex::ParserExecutor<GEMPIC_SPACEDIM + 1> BzEval;
    std::string Ex = "0.0";
    amrex::Parser ExParse;
    amrex::ParserExecutor<GEMPIC_SPACEDIM + 1> ExEval;
    std::string Ey = "0.0";
    amrex::Parser EyParse;
    amrex::ParserExecutor<GEMPIC_SPACEDIM + 1> EyEval;
    std::string Ez = "0.0";
    amrex::Parser EzParse;
    amrex::ParserExecutor<GEMPIC_SPACEDIM + 1> EzEval;
    std::string phi = "0.0";
    amrex::Parser phiParse;
    amrex::ParserExecutor<GEMPIC_SPACEDIM + 1> phiEval;
    std::string rho = "0.0";
    amrex::Parser rhoParse;
    amrex::ParserExecutor<GEMPIC_SPACEDIM + 1> rhoEval;

    // code data
    amrex::Real tolerance_particles = 1e-10;
    int max_coarsening_level = 30;
    int restart = 0;
    std::string checkpoint_file = "";
    int curr_step = 0;

    // Diagnostics
    int save_fields = 10000;  //
    int save_particles = 10000;
    int save_checkpoint = 10000;  // time evolution

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
        amrex::Array<int, GEMPIC_SPACEDIM> degs = {AMREX_D_DECL(degx, degy, degz)};
        int maxdeg = *(std::max_element(degs.begin(), degs.end()));
        Nghost = maxdeg;
    }

    void read_pp_params()
    {
        // Local variables
        amrex::ParmParse pp;
        amrex::Array<int, GEMPIC_SPACEDIM> is_periodic_vector{AMREX_D_DECL(1, 1, 1)};
        amrex::Array<int, GEMPIC_SPACEDIM> max_grid_size_vector{AMREX_D_DECL(4, 4, 4)};
        amrex::Array<int, GEMPIC_SPACEDIM> n_cell_vector{AMREX_D_DECL(10, 10, 10)};

        if (!pp.query("sim_name", sim_name))
        {
            parseErrorMessage("sim_name");
        }
        // computational domain
        if (!pp.query("is_periodic_vector", is_periodic_vector))
        {
            parseErrorMessage("is_periodic_vector");
        }
        is_periodic = amrex::IntVect{
            AMREX_D_DECL(is_periodic_vector[xDir], is_periodic_vector[yDir], is_periodic_vector[zDir])};
        int allPeriodic = 1;
        int nonePeriodic = 0;
        for (size_t i = 0; i < GEMPIC_SPACEDIM; i++)
        {
            allPeriodic *= is_periodic[i];
            nonePeriodic += is_periodic[i];
        }
        if (allPeriodic == 0)
        {
            if (!pp.queryarr("domain_lo", domain_lo))
            {
                parseErrorMessage("domain_lo");
            }
            if (!pp.queryarr("domain_hi", domain_hi))
            {
                parseErrorMessage("domain_hi");
            }
        }
        if (!(nonePeriodic == 0))
        {
            if (!pp.queryarr("k", k))
            {
                parseErrorMessage("k");
            }
        }

        if (!pp.query("n_cell_vector", n_cell_vector))
        {
            parseErrorMessage("n_cell_vector");
        }
        n_cell = amrex::IntVect{AMREX_D_DECL(n_cell_vector[xDir], n_cell_vector[yDir], n_cell_vector[zDir])};
        if (!pp.query("max_grid_size_vector", max_grid_size_vector))
        {
            parseErrorMessage("max_grid_size_vector");
        }
        max_grid_size = amrex::IntVect{AMREX_D_DECL(
            max_grid_size_vector[xDir], max_grid_size_vector[yDir], max_grid_size_vector[zDir])};

        // time stepping
        if (!pp.query("n_steps", n_steps))
        {
            parseErrorMessage("n_steps");
        }
        if (!pp.query("dt", dt))
        {
            parseErrorMessage("dt");
        }
        if (!pp.query("propagator", propagator))
        {
            parseErrorMessage("propagator");
        }
        if (!pp.query("strang_order", strang_order))
        {
            parseErrorMessage("strang_order");
        }

        // particle data

        for (int spec = 0; spec < numspec; spec++)
        {
            amrex::Vector<amrex::Vector<amrex::Real>> vvm, vvt;
            amrex::Vector<amrex::Real> vvw;
            std::string specString = "species" + std::to_string(spec);
            amrex::ParmParse ppSpec(specString);
            if (!ppSpec.query("n_part_per_cell", n_part_per_cell[spec]))
            {
                parseErrorMessage(specString + ".n_part_per_cell");
            }
            if (!ppSpec.query("charge", charge[spec]))
            {
                parseErrorMessage(specString + ".charge");
            }
            if (!ppSpec.query("mass", mass[spec]))
            {
                parseErrorMessage(specString + ".mass");
            }
            if (!ppSpec.query("name", name[spec]))
            {
                parseErrorMessage(specString + ".name");
            }
            if (!ppSpec.query("density", density[spec]))  // initial density
            {
                parseErrorMessage(specString + ".density");
            }
            if (!ppSpec.query("num_gaussians", num_gaussians[spec]))
            {
                parseErrorMessage(specString + ".num_gaussians");
            }
            for (int ng = 0; ng < num_gaussians[spec]; ng++)
            {
                amrex::Vector<amrex::Real> vm, vt;
                amrex::Real vw;
                std::string vmString = "vMean_g" + std::to_string(ng);
                if (!ppSpec.queryarr(vmString.c_str(), vm))
                {
                    parseErrorMessage(specString + "." + vmString);
                }
                vvm.push_back(vm);
                std::string vtString = "vThermal_g" + std::to_string(ng);
                if (!ppSpec.queryarr(vtString.c_str(), vt))
                {
                    parseErrorMessage(specString + "." + vmString);
                }
                vvt.push_back(vt);
                std::string vwString = "vWeight_g" + std::to_string(ng);
                if (!ppSpec.query(vwString.c_str(), vw))
                {
                    parseErrorMessage(specString + "." + vmString);
                }
                vvw.push_back(vw);
            }
            meanVelocity.push_back(vvm);
            vThermal.push_back(vvt);
            vWeight.push_back(vvw);
        }

        // initial fields
        if (!pp.query("Bx", Bx))
        {
            parseErrorMessage("Bx");
        }
        if (!pp.query("By", By))
        {
            parseErrorMessage("By");
        }
        if (!pp.query("Bz", Bz))
        {
            parseErrorMessage("Bz");
        }
        if (!pp.query("Ex", Ex))
        {
            parseErrorMessage("Ex");
        }
        if (!pp.query("Ey", Ey))
        {
            parseErrorMessage("Ey");
        }
        if (!pp.query("Ez", Ez))
        {
            parseErrorMessage("Ez");
        }
        if (!pp.query("phi", phi))
        {
            parseErrorMessage("phi");
        }
        if (!pp.query("rho", rho))
        {
            parseErrorMessage("rho");
        }

        // code data
        if (!pp.query("tolerance_particles", tolerance_particles))
        {
            parseErrorMessage("tolerance_particles");
        }
        if (!pp.query("max_coarsening_level", max_coarsening_level))
        {
            parseErrorMessage("max_coarsening_level");
        }
        if (!pp.query("restart", restart))
        {
            parseErrorMessage("restart");
        }
        if (!pp.query("checkpoint_file", checkpoint_file))
        {
            parseErrorMessage("checkpoint_file");
        }
        if (!pp.query("curr_step", curr_step))
        {
            parseErrorMessage("curr_step");
        }

        // diagnostics
        if (!pp.query("save_fields", save_fields))
        {
            parseErrorMessage("save_fields");
        }
        if (!pp.query("save_particles", save_particles))
        {
            parseErrorMessage("save_particles");
        }
        if (!pp.query("save_checkpoint", save_checkpoint))
        {
            parseErrorMessage("save_checkpoint");
        }

        // scaling parameters
        if (!pp.query("epref_ep0", epref_ep0))
        {
            parseErrorMessage("epref_ep0");
        }
        if (!pp.query("curlB_scaling", curlB_scaling))
        {
            parseErrorMessage("curlB_scaling");
        }
        if (!pp.query("energy_b_scaling", energy_b_scaling))
        {
            parseErrorMessage("energy_b_scaling");
        }

        amrex::Print() << "------------------------------------\n";
        amrex::Print() << "Input file parameters have been read\n";
        amrex::Print() << "------------------------------------\n";
        // set dependent parameters
        set_computed_params();
    }

    void set_params(
        std::string sm = "Weibel",
        amrex::IntVect n_cell_vector = amrex::IntVect{AMREX_D_DECL(
            24, 8,
            8)},  // std::array<int,GEMPIC_SPACEDIM> n_cell_vector = {AMREX_D_DECL(24, 8, 8)},
        amrex::GpuArray<int, numspec> n_part_per_cell_in = {100}, int NS = 10, int FX = 12,
        int FV = 12, int FS = 12, amrex::IntVect is_periodic_vector = amrex::IntVect{AMREX_D_DECL(1, 1, 1)},
        amrex::IntVect max_grid_size_vector = amrex::IntVect{AMREX_D_DECL(4, 4, 4)}, amrex::Real DT = 0.02,
        amrex::GpuArray<amrex::Real, numspec> C = {-1.0},
        amrex::GpuArray<amrex::Real, numspec> M = {1.0}, amrex::Real K = 1.25,
        amrex::GpuArray<std::string, numspec> wf = {"1.0"}, std::string BX = "0.0",
        std::string BY = "0.0", std::string BZ = "1e-3 * cos(kvarx * x)", std::string EX = "0.0",
        std::string EY = "0.0", std::string EZ = "1e-3 * cos(kvarx * x)",
        std::string PHI = "4 * 0.5 * cos(0.5 * x)",
        // std::vector<int> NG = {1},
        amrex::GpuArray<int, numspec> NG = {1},
        // int NG = 1,
        int PROP = 0, amrex::Real TOL_P = 1.e-10, int MAX_C = 30, int RST = 0, std::string CF = "",
        int CD = 0,
        amrex::Vector<amrex::Vector<amrex::Vector<amrex::Real>>> meanVelocity_in = {{{0.0, 0.0,
                                                                                      0.0}}},
        amrex::Vector<amrex::Vector<amrex::Vector<amrex::Real>>> vThermal_in =
            {{{0.014142135623730949, 0.04898979485566356, 0.04898979485566356}}},
        amrex::Vector<amrex::Vector<amrex::Real>> vWeight_in = {{1.0}})
    {
        sim_name = sm;
        n_cell =
            n_cell_vector;  // amrex::IntVect{AMREX_D_DECL(n_cell_vector[0],n_cell_vector[1],n_cell_vector[2])};
        for (int i = 0; i < numspec; i++) n_part_per_cell[i] = n_part_per_cell_in[i];
        n_steps = NS;
        save_fields = FX;
        save_particles = FV;
        save_checkpoint = FS;
        // is_periodic_vector = PER;
        is_periodic =
            is_periodic_vector;  // amrex::IntVect{AMREX_D_DECL(is_periodic_vector[0],is_periodic_vector[1],is_periodic_vector[2])};
        max_grid_size =
            max_grid_size_vector;  // amrex::IntVect{AMREX_D_DECL(max_grid_size_vector[0],max_grid_size_vector[1],max_grid_size_vector[2])};
        dt = DT;
        for (int i = 0; i < numspec; i++)
        {
            charge[i] = C[i];
            mass[i] = M[i];
            density[i] = wf[i];
        }
        k = {K, K, K};
        Bx = BX;
        By = BY;
        Bz = BZ;
        Ex = EX;
        Ey = EY;
        Ez = EZ;
        phi = PHI;
        for (int i = 0; i < numspec; i++) num_gaussians[i] = NG[i];
        propagator = PROP;
        tolerance_particles = TOL_P;
        restart = RST;
        checkpoint_file = CF;
        curr_step = CD;
        meanVelocity = meanVelocity_in;
        vThermal = vThermal_in;
        vWeight = vWeight_in;

        // set dependent parameters
        set_computed_params();
    }

    void set_params_Weibel(std::string sim_name_in = "Weibel", int propagatorIn = 3,
                           int nStepsIn = 10, amrex::IntVect nCellIn = amrex::IntVect{AMREX_D_DECL(12, 8, 8)},
                           amrex::Array<int, numspec> nPartPerCellIn = {100},
                           amrex::Real dtIn = 0.02, int freqSliceIn = 100)
    {
        sim_name = sim_name_in;
        propagator = propagatorIn;
        n_steps = nStepsIn;
        n_cell = nCellIn;
        dt = dtIn;
        save_checkpoint = freqSliceIn;
        assert(numspec == 1);  // Only electrons
        n_part_per_cell[0] = nPartPerCellIn[0]; 
        charge[0] = -1.0;
        mass[0] = 1.0;
        density[0] = "1.0";
        Bx = "0.0";
        By = "0.0";
        Bz = "1e-3 * cos(kvarx * x)";
        Ex = "0.0";
        Ey = "0.0";
        Ez = "1e-3 * cos(kvarx * x)";
        phi = "4 * 0.5 * cos(0.5 * x)";
        is_periodic = amrex::IntVect{AMREX_D_DECL(1, 1, 1)};
        max_grid_size = amrex::IntVect{AMREX_D_DECL(4, 4, 4)};
        k = {AMREX_D_DECL(1.25, 1.25, 1.25)};
        num_gaussians[0] = 1;
        meanVelocity = {{{0.0, 0.0, 0.0}}};
        vThermal = {{{0.014142135623730949, 0.04898979485566356, 0.04898979485566356}}};
        vWeight = {{1.0}};
        // set dependent parameters
        set_computed_params();
    }

    void set_computed_params()
    {
        // set time_stagered and Nghost depending on propagator
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
            case 4:
                time_staggered = false;
                Nghost += 3;
                break;
            default:
                break;
        }
        // set nPartCellTot
        nPartCellTot = 0;
        for (int spec = 0; spec < numspec; spec++)
        {
            nPartCellTot += n_part_per_cell[spec];
        }

        // In periodic directions domain is [0, 2pi / k]
        for (int i = 0; i < GEMPIC_SPACEDIM; i++)
        {
            if (is_periodic[i] == 1)
            {
                domain_lo[i] = 0.0;
                domain_hi[i] = 2 * M_PI / k[i];
            }
        }
        real_box = amrex::RealBox({AMREX_D_DECL(domain_lo[xDir], domain_lo[yDir], domain_lo[zDir])},
                                  {AMREX_D_DECL(domain_hi[xDir], domain_hi[yDir], domain_hi[zDir])});

        kx = k[xDir];
        ky = k[yDir];
        kz = k[zDir];

        // Parsers
        // --------
        for (size_t i = 0; i < numspec; i++)
        {
            densityParse[i] = amrex::Parser(density[i]);
            densityParse[i].setConstant("kvarx", k[xDir]);
            densityParse[i].setConstant("kvary", k[yDir]);
            densityParse[i].setConstant("kvarz", k[zDir]);
            densityParse[i].registerVariables({AMREX_D_DECL("x", "y", "z"), "t"});
            densityEval[i] =
                densityParse[i].template compile<GEMPIC_SPACEDIM + 1>();  // 4 because there are three variables.
        }

        // initial fields
        BxParse = amrex::Parser(Bx);
        BxParse.setConstant("kvarx", k[xDir]);
        BxParse.setConstant("kvary", k[yDir]);
        BxParse.setConstant("kvarz", k[zDir]);
        BxParse.registerVariables({AMREX_D_DECL("x", "y", "z"), "t"});
        BxEval = BxParse.compile<GEMPIC_SPACEDIM + 1>();

        ByParse = amrex::Parser(By);
        ByParse.setConstant("kvarx", k[xDir]);
        ByParse.setConstant("kvary", k[yDir]);
        ByParse.setConstant("kvarz", k[zDir]);
        ByParse.registerVariables({AMREX_D_DECL("x", "y", "z"), "t"});
        ByEval = ByParse.compile<GEMPIC_SPACEDIM + 1>();

        BzParse = amrex::Parser(Bz);
        BzParse.setConstant("kvarx", k[xDir]);
        BzParse.setConstant("kvary", k[yDir]);
        BzParse.setConstant("kvarz", k[zDir]);
        BzParse.registerVariables({AMREX_D_DECL("x", "y", "z"), "t"});
        BzEval = BzParse.compile<GEMPIC_SPACEDIM + 1>();

        ExParse = amrex::Parser(Ex);
        ExParse.setConstant("kvarx", k[xDir]);
        ExParse.setConstant("kvary", k[yDir]);
        ExParse.setConstant("kvarz", k[zDir]);
        ExParse.registerVariables({AMREX_D_DECL("x", "y", "z"), "t"});
        ExEval = ExParse.compile<GEMPIC_SPACEDIM + 1>();

        EyParse = amrex::Parser(Ey);
        EyParse.setConstant("kvarx", k[xDir]);
        EyParse.setConstant("kvary", k[yDir]);
        EyParse.setConstant("kvarz", k[zDir]);
        EyParse.registerVariables({AMREX_D_DECL("x", "y", "z"), "t"});
        EyEval = EyParse.compile<GEMPIC_SPACEDIM + 1>();

        EzParse = amrex::Parser(Ez);
        EzParse.setConstant("kvarx", k[xDir]);
        EzParse.setConstant("kvary", k[yDir]);
        EzParse.setConstant("kvarz", k[zDir]);
        EzParse.registerVariables({AMREX_D_DECL("x", "y", "z"), "t"});
        EzEval = EzParse.compile<GEMPIC_SPACEDIM + 1>();

        phiParse = amrex::Parser(phi);
        phiParse.setConstant("kvarx", k[xDir]);
        phiParse.setConstant("kvary", k[yDir]);
        phiParse.setConstant("kvarz", k[zDir]);
        phiParse.registerVariables({AMREX_D_DECL("x", "y", "z"), "t"});
        phiEval = phiParse.compile<GEMPIC_SPACEDIM + 1>();

        rhoParse = amrex::Parser(rho);
        rhoParse.setConstant("kvarx", k[xDir]);
        rhoParse.setConstant("kvary", k[yDir]);
        rhoParse.setConstant("kvarz", k[zDir]);
        rhoParse.registerVariables({AMREX_D_DECL("x", "y", "z"), "t"});
        rhoEval = rhoParse.compile<GEMPIC_SPACEDIM + 1>();
    }
};

#endif
