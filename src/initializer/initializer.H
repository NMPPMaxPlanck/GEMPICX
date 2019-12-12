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
#ifndef INITIALIZER_H_
#define INITIALIZER_H_

class initializer
{
  public:
  // parameters for infrastructure:
  int n_cell[3];
  int max_grid_size;
  int is_periodic[3];

  // parameters for time evolution:
  amrex::Real dt;
  int n_steps;

  // parameters for particles:
  int n_species;
  std::array<amrex::Real, 1> charge;
  std::array<amrex::Real, 1> mass;
  int Np_cell; //number of particles per cell

  amrex::Real k; // for domain, should be chosen same as in function
  amrex::RealBox real_box;

  //parameters for velocity distribution
  std::array<std::array<amrex::Real, 1>, 3> Vmean;
  std::array<std::array<amrex::Real, 1>, 3> Vdev;
  std::array<std::array<amrex::Real, 1>, 3> Vweight;

  // function for weights
  double (*weight_fun) (double x,double y,double z,double v_x,double v_y,double v_z,int Np);
  
  // constructor
  initializer (){};

  void initialize_from_parameters(int NC[3], int MG, int PER[3], amrex::Real DT, int NST, int NSP,
				  std::array<amrex::Real, 1> CH,
				  std::array<amrex::Real, 1> MS,
				  int NPC,
				  amrex::Real K,
                  std::array<std::array<amrex::Real, 1>, 3> VM,
                  std::array<std::array<amrex::Real, 1>, 3> VD,
                  std::array<std::array<amrex::Real, 1>, 3> VW,
				  double WF (double x,double y,double z,
					     double v_x,double v_y,double v_z,int Np)){
    n_cell[0] = NC[0];
    n_cell[1] = NC[1];
    n_cell[2] = NC[2];
    max_grid_size = MG;
    is_periodic[0] = PER[0];
    is_periodic[1] = PER[1];
    is_periodic[2] = PER[2];
    dt = DT;
    n_steps = NST;
    n_species = NSP;
    charge[0] = CH[0];
    mass[0] = MS[0];
    Np_cell = NPC;
    k = K;

    double twopi = 2.0*3.14159265359;
    amrex::Real boxLo[3] = {0.0,0.0,0.0};
    //amrex::Real boxHi[3] = {twopi/k,twopi/k,twopi/k};
    amrex::Real boxHi[3] = {twopi/k,twopi/k/((double)n_cell[0]/(double)n_cell[1]),twopi/k/((double)n_cell[0]/(double)n_cell[2])};
    real_box.setLo(boxLo);
    real_box.setHi(boxHi);

    Vmean[0][0] = VM[0][0];
    Vmean[1][0] = VM[1][0];
    Vmean[2][0] = VM[2][0];
    Vdev[0][0] = VD[0][0];
    Vdev[1][0] = VD[1][0];
    Vdev[2][0] = VD[2][0];
    Vweight[0][0] = VW[0][0];
    Vweight[1][0] = VW[1][0];
    Vweight[2][0] = VW[2][0];
    
    weight_fun = *WF;
  };

  void initialize_from_file(){
    
  };
};

#endif
