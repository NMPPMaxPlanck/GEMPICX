#include <AMReX.H>
#include <AMReX_Box.H>
#include <AMReX_BoxArray.H>
#include <AMReX_CoordSys.H>
#include <AMReX_DistributionMapping.H>
#include <AMReX_Geometry.H>
#include <AMReX_IntVect.H>
#include <AMReX_Print.H>
#include <AMReX_REAL.H>
#include <AMReX_RealBox.H>

#include <GEMPIC_Config.H>
#include <GEMPIC_infrastructure.H>

#ifndef INITIALIZER_H_
#define INITIALIZER_H_

namespace Gempic {
namespace Init {

/**
 * Class containing all initialization parameters. At the moment it is possible to initialize by passing the parameters. In the future it will also be possible to initialize by passing a file.
 */
template<int vdim, int numspec>
class initializer
{
public:
    // parameters for infrastructure:
    amrex::IntVect n_cell;
    int max_grid_size;
    amrex::IntVect is_periodic;
    int Nghost;

    // parameters for time evolution:
    amrex::Real dt;
    int n_steps;

    // parameters for particles:
    std::array<amrex::Real, numspec> charge;
    std::array<amrex::Real, numspec> mass;
    std::array<int, numspec> Np_cell; //number of particles per cell
    int NP_cell_tot;

    std::array<amrex::Real,GEMPIC_SPACEDIM> k; // for domain and weight function
    amrex::RealBox real_box;

    //parameters for velocity distribution
    std::array<std::vector<amrex::Real>, vdim> Vmean;
    std::array<std::vector<amrex::Real>, vdim> Vdev;
    std::array<std::vector<amrex::Real>, vdim> Vweight;

    // propagator
    int propagator; // 0 : Boris FD, 1 HS FEM

    //parameter for iterative solvers
    amrex::Real tolerance_particles = 1.e-10;

    // constructor
    /**
     * Constructor.
     */
    initializer (){}

    /**
     * Manually set the attributes of initializer. The attributes are the number
     * of cells @p NC, the maximal grid size @p MG, the periodicity @p PER, the number of ghost cells @p NG, the time step @p DT, the number of time steps @p NST, the particle charges @p CH,
     * the particle masses @p MS, the number of particles per cell @p NPC, the mean of the velocity distributions @p VM, the deviation of the velicity distributions @p VD, the weight of the velocity
     * distributions @p VW and the particle distribution function @p WF.
     */

    void initialize_from_parameters(amrex::IntVect NC, int MG, amrex::IntVect PER, int NG, amrex::Real DT, int NST,
                                    std::array<amrex::Real, numspec> CH,
                                    std::array<amrex::Real, numspec> MS,
                                    std::array<int, numspec> NPC,
                                    std::array<amrex::Real,GEMPIC_SPACEDIM> K,
                                    std::array<std::vector<amrex::Real>, vdim> VM,
                                    std::array<std::vector<amrex::Real>, vdim> VD,
                                    std::array<std::vector<amrex::Real>, vdim> VW,
                                    int PG,
                                    amrex::Real TP = 1.e-10){
        n_cell = NC;
        max_grid_size = MG;
        is_periodic = PER;
        Nghost = NG;
        dt = DT;
        n_steps = NST;
        charge = CH;
        mass = MS;
        NP_cell_tot = 0;
        for (int spec = 0; spec < numspec; spec++) {
            Np_cell[spec] = NPC[spec];
            NP_cell_tot += NPC[spec];
        }
        for (int d=0;d<GEMPIC_SPACEDIM;++d)
          k[d] = K[d];
        propagator = PG;
        tolerance_particles = TP;

        double twopi = 2.0*3.14159265359;

        amrex::Real boxLo[GEMPIC_SPACEDIM] = {AMREX_D_DECL(0,0,0)};
        amrex::Real boxHi[GEMPIC_SPACEDIM] = {AMREX_D_DECL(twopi/k[0],twopi/k[1],twopi/k[2])};

        real_box.setLo(boxLo);
        real_box.setHi(boxHi);

        Vmean = VM;
        Vdev = VD;
        Vweight = VW;


    }

    void initialize_infrastructure(Infra::infrastructure *ifr){

        // Initialize infrastructure:

        ifr->n_cell = n_cell;
        ifr->max_grid_size = max_grid_size;

        for(int cc=0;cc<GEMPIC_SPACEDIM;cc++){
            ifr->is_periodic[cc] = is_periodic[cc];
        }
        ifr->real_box = real_box;

        for(int cc=0;cc<GEMPIC_SPACEDIM;cc++){
            ifr->Length[cc] = (real_box.hi(cc)-real_box.lo(cc));
        }

        amrex::IntVect dom_lo(AMREX_D_DECL(0, 0, 0));
        amrex::IntVect dom_hi(AMREX_D_DECL(n_cell[0]-1, n_cell[1]-1, n_cell[2]-1));

        ifr->domain.setSmall(dom_lo);
        ifr->domain.setBig(dom_hi);
        ifr->grid.define(ifr->domain);
        ifr->grid.maxSize(max_grid_size);
        ifr->geom.define(ifr->domain,&ifr->real_box,amrex::CoordSys::cartesian,ifr->is_periodic);
        ifr->dx = ifr->geom.CellSize();

        ifr->distriMap.define(ifr->grid);

        ifr->dxi[GEMPIC_SPACEDIM] = 1.;
        for (int cc=0;cc<GEMPIC_SPACEDIM;cc++){
            ifr->plo[cc] = ifr->geom.ProbLo()[cc];
            ifr->dxi[cc] = 1.0/ifr->dx[cc];
            ifr->dxi[GEMPIC_SPACEDIM] *= ifr->dxi[cc];
        }

        ifr->tolerance_particle = tolerance_particles;

    }

};
}
}

#endif
