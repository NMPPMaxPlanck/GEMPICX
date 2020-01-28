
#include <AMReX.H>
#include <AMReX_Print.H>
#include <particle_groups.H>
#include <maxwell_yee.H>
#include <particle_mesh_coupling.H>
#include <gempic_Config.H>

#include <AMReX_PlotFileUtil.H>
#include <AMReX_ParmParse.H>
#include <AMReX_Print.H>
#include <AMReX_Particles.H>

using namespace std;
using namespace amrex;

double E_x(std::array<double,GEMPIC_SPACEDIM> x, double t){return(cos(x[0]+x[1]+x[2]-sqrt(3.0)*t));}
double E_y(std::array<double,GEMPIC_SPACEDIM> x, double t){return(-2*cos(x[0]+x[1]+x[2]-sqrt(3.0)*t));}
double E_z(std::array<double,GEMPIC_SPACEDIM> x, double t){return(cos(x[0]+x[1]+x[2]-sqrt(3.0)*t));}

double B_x(std::array<double,GEMPIC_SPACEDIM> x, double t){return(sqrt(3)*cos(x[0]+x[1]+x[2]-sqrt(3.0)*t));}
double B_y(std::array<double,GEMPIC_SPACEDIM> x, double t){return(0);}
double B_z(std::array<double,GEMPIC_SPACEDIM> x, double t){return(-sqrt(3)*cos(x[0]+x[1]+x[2]-sqrt(3.0)*t));}

double WF (std::array<double,GEMPIC_SPACEDIM> x, std::array<double,GEMPIC_VDIM> v,int Np,double k) {
  return(0.0);
}

void main_main ()
{
  double pi = 3.14159265359;

  // make pointer-array for functions
  double (*fields[GEMPIC_VDIM+GEMPIC_BDIM]) (std::array<double,GEMPIC_SPACEDIM> x, double t);
  fields[0] = E_x;
#if (GEMPIC_VDIM > 1)
  fields[1] = E_y;
#endif
#if (GEMPIC_VDIM > 2)
  fields[2] = E_z;
#endif

  fields[GEMPIC_VDIM] = B_x;
#if (GEMPIC_BDIM > 1)
  fields[GEMPIC_VDIM+1] = B_y;
#endif
#if (GEMPIC_BDIM > 2)
  fields[GEMPIC_VDIM+2] = B_z;
#endif

//------------------------------------------------------------------------------
  // build infrastructure
  initializer init;
  amrex::IntVect is_periodic(AMREX_D_DECL(1,1,1));
  amrex::IntVect n_cell(AMREX_D_DECL(64,64,64));
  init.initialize_from_parameters(n_cell,32,is_periodic,1,0.01,5,{1.0},{1.0},1000,1,
                  {0.0},{1.0},{1.0},WF);

  infrastructure infra(init);

//------------------------------------------------------------------------------
  //Initialize Particle Groups
  particle_groups part_gr(init, infra);

  //Add particles one by one
  int N_parts = 3;
  int species = 0;
  std::array<double,GEMPIC_SPACEDIM> position;
  std::array<double,GEMPIC_VDIM> velocity;
  Real weight;

  using ParticleType = amrex::Particle<GEMPIC_VDIM+1, 0>; // Particle template
  auto& particles = (*(part_gr).mypc[species]).GetParticles(0)[std::make_pair(0, 0)];

  for (int pp=0;pp<N_parts;pp++) {
    position = {AMREX_D_DECL((double)pp, (double)(pp%2)*pi, 0.0)};
    velocity = {AMREX_V_DECL(1.0, -1.0, 0.0)};
    weight = 1.0;
    part_gr.add_particle(species, position, velocity, weight, particles);
  }

//------------------------------------------------------------------------------
  //Initialize Maxwell Yee
  int n_steps = 5;
  maxwell_yee mw_yee(init, infra, init.Nghost);
  mw_yee.init_E_B(fields, infra);

//------------------------------------------------------------------------------
  //Test CIC
  std::ofstream ofs("test_particle_groups.output", std::ofstream::out);
  Print(ofs) << "field_interpolation" << endl;
  
  Real tol = 0.01;
  std::array<amrex::Real,GEMPIC_SPACEDIM> plo;
  std::array<amrex::Real,GEMPIC_SPACEDIM> dxi;
  plo[0] = infra.geom.ProbLo()[0];
  dxi[0] = 1.0/infra.dx[0];
#if (GEMPIC_SPACEDIM > 1)
  plo[1] = infra.geom.ProbLo()[1];
  dxi[1] = 1.0/infra.dx[1];
#endif
#if (GEMPIC_SPACEDIM > 1)
  plo[2] = infra.geom.ProbLo()[2];
  dxi[2] = 1.0/infra.dx[2];
#endif

  // fill ghost cells
  mw_yee.FillBD(infra);

  // deposit charge in rho one particle at a time
  for (int spec=0;spec<init.n_species;spec++) {
    (*part_gr.mypc[spec]).Redistribute(); // assign particles to the tile they are in
    for (ParIter<GEMPIC_VDIM+1,0,0,0> pti(*part_gr.mypc[spec], 0); pti.isValid(); ++pti) {
      auto& particles = pti.GetArrayOfStructs();
      const long np  = pti.numParticles();

      Array4<Real> const& rhoarr = mw_yee.rho[pti].array();
      for (int pp=0;pp<np;pp++) {
    gempic_deposit_cic(particles[pp], init.charge[spec], rhoarr, plo, dxi);
      }
    }
  }
  // deposit charge in J one component and particle at a time
  for (int spec=0;spec<init.n_species;spec++) {
    (*part_gr.mypc[spec]).Redistribute(); // assign particles to the tile they are in
    for (ParIter<GEMPIC_VDIM+1,0,0,0> pti(*part_gr.mypc[spec], 0); pti.isValid(); ++pti) {
      auto& particles = pti.GetArrayOfStructs();
      const long np  = pti.numParticles();
      
      for(int cc=0;cc<GEMPIC_VDIM;cc++){
	Array4<Real> const& jarr = (*mw_yee.J_Array[cc])[pti].array();
	for (int pp=0;pp<np;pp++) {
      gempic_deposit_J_cic(particles[pp], init.charge[spec], cc, jarr, infra.ploE[cc], dxi);
	}
      }
    }
  }
  // interpolate fields at particle positions one particle at a time (and push them)
  std::array<double,GEMPIC_VDIM> eres;
  std::array<double,GEMPIC_BDIM> bres;
  double esol, bsol;
  for (int spec=0;spec<init.n_species;spec++) {
    (*part_gr.mypc[spec]).Redistribute(); // assign particles to the tile they are in
    Real chargemass = init.charge[spec]/init.mass[spec];
    
    for (ParIter<GEMPIC_VDIM+1,0,0,0> pti(*part_gr.mypc[spec], 0); pti.isValid(); ++pti) {
      auto& particles = pti.GetArrayOfStructs();
      const long np  = pti.numParticles();

      for (int pp=0;pp<np;pp++){
          std::array<double,GEMPIC_SPACEDIM> x = {AMREX_D_DECL(particles[pp].pos(0),particles[pp].pos(1),particles[pp].pos(2))};
    for (int cc=0;cc<GEMPIC_VDIM;cc++){
	  Array4<Real> const& earr = (*mw_yee.E_Array[cc])[pti].array();
	  
	  //E-field
      eres[cc] = gempic_interpolate_cic(particles[pp], earr, infra.ploE[cc], dxi);
      esol = (*fields[cc])(x,0.0);
      if (std::abs(eres[cc]-esol) > tol) {
	    Print(ofs) << "check results at particle " << pp << ", E-component " << cc <<  endl;
	  }
    }
    for (int cc=0;cc<GEMPIC_BDIM;cc++){
      Array4<Real> const& barr = (*mw_yee.B_Array[cc])[pti].array();

	  //B-field
      bres[cc] = gempic_interpolate_cic(particles[pp], barr, infra.ploB[cc], dxi);
      bsol = (*fields[cc+GEMPIC_VDIM])(x,0.0);
      if (std::abs(bres[cc]-bsol) > tol) {
	    Print(ofs) << "check results at particle " << pp << ", B-component " << cc <<  endl;
	  }
	}
    array<Real,GEMPIC_SPACEDIM+GEMPIC_VDIM> newPos = push_particle(particles[pp], init.dt, chargemass, eres, bres, infra);
    copy(begin(newPos),end(newPos),&particles.data()[pp*(GEMPIC_SPACEDIM+GEMPIC_VDIM+2)]);
      }
    }
    ofs.close();
  }
}

int main(int argc, char* argv[])
{
  amrex::Initialize(argc,argv);

  main_main();

  amrex::Finalize();
}
