#include <AMReX.H>
#include <AMReX_Print.H>

#include <AMReX_RealBox.H>
#include <AMReX_REAL.H>
#include <AMReX_BoxArray.H>
#include <AMReX_Geometry.H>
#include <AMReX_IntVect.H>
#include <AMReX_Box.H>
#include <AMReX_DistributionMapping.H>
#include <AMReX_CoordSys.H>

#include <gempic_Config.H>
#ifndef INITIALIZER_H_
#define INITIALIZER_H_

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
    std::array<std::array<amrex::Real, GEMPIC_NUMGAUS>, GEMPIC_VDIM> Vmean;
    std::array<std::array<amrex::Real, GEMPIC_NUMGAUS>, GEMPIC_VDIM> Vdev;
    std::array<std::array<amrex::Real, GEMPIC_NUMGAUS>, GEMPIC_VDIM> Vweight;

    // function for weights
    double (*weight_fun) (std::array<double,GEMPIC_SPACEDIM> x, std::array<double,GEMPIC_VDIM> v,int Np,double k);

    // constructor
    initializer (){}

    void initialize_from_parameters(amrex::IntVect NC, int MG, amrex::IntVect PER, int NG, amrex::Real DT, int NST,
                                    std::array<amrex::Real, GEMPIC_NUMSPEC> CH,
                                    std::array<amrex::Real, GEMPIC_NUMSPEC> MS,
                                    int NPC,
                                    amrex::Real K,
                                    std::array<std::array<amrex::Real, GEMPIC_NUMGAUS>, GEMPIC_VDIM> VM,
                                    std::array<std::array<amrex::Real, GEMPIC_NUMGAUS>, GEMPIC_VDIM> VD,
                                    std::array<std::array<amrex::Real, GEMPIC_NUMGAUS>, GEMPIC_VDIM> VW,
                                    double WF (std::array<double,GEMPIC_SPACEDIM> x, std::array<double,GEMPIC_VDIM> v,
                                               int Np,double k)){
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

    void initialize_from_file(){

    }
};

#endif
