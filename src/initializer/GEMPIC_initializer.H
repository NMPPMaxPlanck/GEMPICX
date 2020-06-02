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

#ifndef INITIALIZER_H_
#define INITIALIZER_H_

namespace Gempic {
namespace Init {

/**
 * Class containing all initialization parameters. At the moment it is possible to initialize by passing the parameters. In the future it will also be possible to initialize by passing a file.
 */
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
    int n_species;
    std::array<amrex::Real, GEMPIC_NUMSPEC> charge;
    std::array<amrex::Real, GEMPIC_NUMSPEC> mass;
    int Np_cell; //number of particles per cell

    amrex::Real k; // for domain and weight function
    amrex::RealBox real_box;

    //parameters for velocity distribution
    std::array<std::vector<amrex::Real>, GEMPIC_VDIM> Vmean;
    std::array<std::vector<amrex::Real>, GEMPIC_VDIM> Vdev;
    std::array<std::vector<amrex::Real>, GEMPIC_VDIM> Vweight;

    // function for weights
    double (*weight_fun) (std::array<double,GEMPIC_SPACEDIM> x, std::array<double,GEMPIC_VDIM> v,double k);

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
                                    std::array<amrex::Real, GEMPIC_NUMSPEC> CH,
                                    std::array<amrex::Real, GEMPIC_NUMSPEC> MS,
                                    int NPC,
                                    amrex::Real K,
                                    std::array<std::vector<amrex::Real>, GEMPIC_VDIM> VM,
                                    std::array<std::vector<amrex::Real>, GEMPIC_VDIM> VD,
                                    std::array<std::vector<amrex::Real>, GEMPIC_VDIM> VW,
                                    double WF (std::array<double,GEMPIC_SPACEDIM> x, std::array<double,GEMPIC_VDIM> v,
                                               double k)){
        n_cell = NC;
        max_grid_size = MG;
        is_periodic = PER;
        Nghost = NG;
        dt = DT;
        n_steps = NST;
        n_species = GEMPIC_NUMSPEC;
        charge = CH;
        mass = MS;
        Np_cell = NPC;
        k = K;

        double twopi = 2.0*3.14159265359;

        amrex::Real boxLo[GEMPIC_SPACEDIM] = {AMREX_D_DECL(0,0,0)};
        amrex::Real boxHi[GEMPIC_SPACEDIM] = {AMREX_D_DECL(twopi/k,twopi/k,twopi/k)};

        real_box.setLo(boxLo);
        real_box.setHi(boxHi);

        Vmean = VM;
        Vdev = VD;
        Vweight = VW;

        weight_fun = *WF;
    }

    /**
     * TODO: initialize the parameters of class by reading in a file
     */
    void initialize_from_file(){

    }
};
}
}

#endif
