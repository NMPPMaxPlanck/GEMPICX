/*------------------------------------------------------------------------------
 Test 3D Maxwell Yee Solver (finite differences) on periodic grid

  For the Maxwell-equations we use the solution
  E(x,t) =  \begin{pmatrix} \cos(x_1+x_2+x_3 - \sqrt{3} t) \\
                          -2\cos(x_1+x_2+x_3 - \sqrt(3) t) \\
                            \cos(x_1+x_2+x_3 - \sqrt{3} t) \end{pmatrix}
  B(x,t) = \begin{pmatrix} \sqrt{3} \cos(x_1+x_2+x_3 - \sqrt{3} t) \\ 0 \\ -\sqrt{3} \cos(x_1+x_2+x_3 - \sqrt{3} t) \end{pmatrix}

 For the Poisson equation we use:
 E(x,t) = \begin{pmatrix} -\sin(x)\cos(y)\cos(z)-0.5\sin(2x)cos(2y)cos(2z)\\
                          -\cos(x)\sin(y)\cos(z)-0.5\cos(2x)sin(2y)cos(2z)\\
                          -\cos(x)\cos(y)\sin(z)-0.5\cos(2x)cos(2y)sin(2z) \end{pmatrix}
------------------------------------------------------------------------------*/

#include <tinyexpr.h>

#include <AMReX.H>
#include <AMReX_ParmParse.H>
#include <AMReX_PlotFileUtil.H>
#include <AMReX_Print.H>

#include <GEMPIC_Config.H>
#include <GEMPIC_maxwell_yee.H>
#include <GEMPIC_gempic_norm.H>

using namespace std;
using namespace amrex;
using namespace Gempic;

using namespace Field_solvers;

//------------------------------------------------------------------------------
// Solutions and RHS
double cosMult(std::array<double,GEMPIC_SPACEDIM> x, double coefficient=1., std::array<int,GEMPIC_SPACEDIM> sinInd = {AMREX_D_DECL(0,0,0)}) {
    double res = sinInd[0]?sin(coefficient*x[0]):cos(coefficient*x[0]);
#if (GEMPIC_SPACEDIM > 1)
    res *= sinInd[1]?sin(coefficient*x[1]):cos(coefficient*x[1]);
#endif
#if (GEMPIC_SPACEDIM > 2)
    res *= sinInd[2]?sin(coefficient*x[2]):cos(coefficient*x[2]);
#endif
    return(res);
}


#if (GEMPIC_SPACEDIM == 1)
template<int vdim>
double E_x(std::array<double,GEMPIC_SPACEDIM> x, double t)
{
    if (vdim == 1)
        return(cos(std::accumulate(x.begin(), x.end(), 0.)));
    else if (vdim == 2)
        return(cos(x[0]));
    else if (vdim == 3)
        return(0.);
}
template<int vdim>
double E_y(std::array<double,GEMPIC_SPACEDIM> x, double t)(std::array<double,GEMPIC_SPACEDIM> x, double t)
{
    if (vdim == 2)
        return(cos(x[0])*cos(t));
    else
        return(0.);
}
template<int vdim>
double B_x(std::array<double,GEMPIC_SPACEDIM> x, double t)
{
    if (vdim == 2)
        return(cos(x[0]);
    else
        return(0.);
}
#elif (GEMPIC_SPACEDIM == 2)
template<int vdim>
double E_x(std::array<double,GEMPIC_SPACEDIM> x, double t)
{
    if (vdim == 1)
        return(0.);
    else if (vdim == 2)
        return(cos(x[0])*sin(x[1])*sin(sqrt(2)*t)/sqrt(2));
    else if (vdim == 3)
        return(cos(std::accumulate(x.begin(), x.end(), 0.)-sqrt(2.0)*t));
}
template<int vdim>
double E_y(std::array<double,GEMPIC_SPACEDIM> x, double t)
{
    if (vdim == 1)
        return(0.);
    else if (vdim == 2)
        return(-sin(x[0])*cos(x[1])*sin(sqrt(2)*t)/sqrt(2));
    else if (vdim == 3)
        return(-cos(std::accumulate(x.begin(), x.end(), 0.)-sqrt(2.0)*t));
}
template<int vdim>
double E_z(std::array<double,GEMPIC_SPACEDIM> x, double t)
{
    if (vdim == 3)
        return(-sqrt(2.0)*cos(std::accumulate(x.begin(), x.end(), 0.)-sqrt(2.0)*t));
    else
        return(0.);
}

template<int vdim>
double B_x(std::array<double,GEMPIC_SPACEDIM> x, double t)
{
    if (vdim == 2)
        return(-cos(x[0])*cos(x[1])*cos(sqrt(2)*t));
    else if (vdim == 3)
        return(-cos(std::accumulate(x.begin(), x.end(), 0.)-sqrt(2.0)*t));
}

template<int vdim>
double B_y(std::array<double,GEMPIC_SPACEDIM> x, double t)
{
    if ( vdim == 3 )
        return(cos(std::accumulate(x.begin(), x.end(), 0.)-sqrt(2.0)*t));
    else
        return(0.);
}
template<int vdim>
double B_z(std::array<double,GEMPIC_SPACEDIM> x, double t)
{
    if (vdim ==  3)
        return(-sqrt(2)*cos(std::accumulate(x.begin(), x.end(), 0.)-sqrt(2.0)*t));
    else
        return(0.);
}

#elif (GEMPIC_SPACEDIM == 3)
template<int vdim>
double E_x(std::array<double,GEMPIC_SPACEDIM> x, double t){
    return(cos(std::accumulate(x.begin(), x.end(), 0.)-sqrt(3.0)*t));
}
template<int vdim>
double E_y(std::array<double,GEMPIC_SPACEDIM> x, double t){return(-2*cos(std::accumulate(x.begin(), x.end(), 0.)-sqrt(3.0)*t));}
template<int vdim>
double E_z(std::array<double,GEMPIC_SPACEDIM> x, double t){return(cos(std::accumulate(x.begin(), x.end(), 0.)-sqrt(3.0)*t));}

template<int vdim>
double B_x(std::array<double,GEMPIC_SPACEDIM> x, double t){return(sqrt(3)*cos(std::accumulate(x.begin(), x.end(), 0.)-sqrt(3.0)*t));}
template<int vdim>
double B_y(std::array<double,GEMPIC_SPACEDIM> x, double t){return(0);}
template<int vdim>
double B_z(std::array<double,GEMPIC_SPACEDIM> x, double t){return(-sqrt(3)*cos(std::accumulate(x.begin(), x.end(), 0.)-sqrt(3.0)*t));}


#endif

double Ep_x(std::array<double,GEMPIC_SPACEDIM> x,double t){return(-cosMult(x, 1., {AMREX_D_DECL(1,0,0)})-0.5*cosMult(x, 2., {AMREX_D_DECL(1,0,0)}));}

template<int vdim>
double Ep_y(std::array<double,GEMPIC_SPACEDIM> x, double t)
{
    if ((GEMPIC_SPACEDIM == 1) & (vdim == 2))
        return(0.);
    else
        return(-cosMult(x, 1., {AMREX_D_DECL(0,1,0)})-0.5*cosMult(x, 2., {AMREX_D_DECL(0,1,0)}));
}

template <int vdim>
double Ep_z(std::array<double,GEMPIC_SPACEDIM> x,double t)
{
    if (GEMPIC_SPACEDIM==2 && vdim == 3)
        return(0.);
    else
        return(-cosMult(x, 1., {AMREX_D_DECL(0,0,1)})-0.5*cosMult(x, 2., {AMREX_D_DECL(0,0,1)}));
}

template<int vdim, int numspec>
void main_main ()
{
    int bdim = int(floor(vdim/2.5)*2+1);
    std::cout << "x: " << GEMPIC_SPACEDIM << " | v,E: " << vdim << " | B: " << bdim << std::endl;
    // make pointer-array for functions
    double (*fields[vdim+bdim]) (std::array<double,GEMPIC_SPACEDIM> x, double t);
    fields[0] = E_x<vdim>;
    if (vdim > 1){
        fields[1] = E_y<vdim>;
    }
    if (vdim > 2) {
        fields[2] = E_z<vdim>;
    }

    fields[vdim] = B_x<vdim>;
if (bdim > 1) {
    fields[vdim+1] = B_y<vdim>;
}
if (bdim > 2) {
    fields[vdim+2] = B_z<vdim>;
}

    double (*fields_poisson[vdim]) (std::array<double,GEMPIC_SPACEDIM> x,double t);
    fields_poisson[0] = Ep_x;
    if (vdim > 1){
        fields_poisson[1] = Ep_y<vdim>;
    }
    if (vdim > 2) {
        fields_poisson[2] = Ep_z<vdim>;
    }

    //------------------------------------------------------------------------------
    array<Real,vdim+int(floor(vdim/2.5)*2+1)> E_B_error; //array for storing errors

    //------------------------------------------------------------------------------
    // Initialize Infrastructure
    initializer<vdim, numspec> init;
    amrex::IntVect is_periodic(AMREX_D_DECL(1,1,1));
    amrex::IntVect n_cell(AMREX_D_DECL(128,128,128));

    std::array<std::vector<amrex::Real>, vdim> VM{};
    std::array<std::vector<amrex::Real>, vdim> VD{};
    std::array<std::vector<amrex::Real>, vdim> VW{};

    VM[0].push_back(0.0);
    VD[0].push_back(1.0);
    VW[0].push_back(1.0);
    if (vdim > 1) {
        VM[1].push_back(0.0);
        VD[1].push_back(1.0);
        VW[1].push_back(1.0);
    }
    if (vdim > 2) {
        VM[2].push_back(0.0);
        VD[2].push_back(1.0);
        VW[2].push_back(1.0);
    }

    std::array<int, GEMPIC_SPACEDIM> degs = {AMREX_D_DECL(GEMPIC_DEG_X, GEMPIC_DEG_Y, GEMPIC_DEG_Z)};
    int maxdeg = *(std::max_element(degs.begin(), degs.end()));

    init.initialize_from_parameters(n_cell,32,is_periodic,maxdeg,0.01,5,{1.0},{1.0},1000,0.5,
                                    VM,VD,VW,0);
    //n_cell, max_grid_size, periodic, Nghost, dt, n_steps, charge, mass, n_part_per_cell, k, vel_mean, vel_dev, vel_weight, weight_fun

    Infra::infrastructure infra;
    init.initialize_infrastructure(&infra);
    //std::cout << "[" << infra.real_box.lo()[0] << ", " << infra.real_box.hi()[0] << "]x[" << infra.real_box.lo()[1] << "," << infra.real_box.hi()[1] << "]x[" << infra.real_box.lo()[2] << ", " << infra.real_box.hi()[2] << "]" << std::endl;
    //std::array<double,GEMPIC_SPACEDIM> x = {0.0,0.0,0.75};
    //double t = 2.0-0.25/sqrt(3);
    //std::cout << fields[0](x,t) << "|" << fields[1](x,t) << "|" << fields[2](x,t) << "|" << fields[3](x,t) << "|" << fields[4](x,t) << "|" << fields[5](x,t) << "|" << std::endl;

    //------------------------------------------------------------------------------
    // Solve
    maxwell_yee<vdim> mw_yee(init, infra, init.Nghost);


    for (int i=0; i<vdim; i++) {
        (*(mw_yee).J_Array[i]).setVal(0.0, 0); // value and component
        (*(mw_yee).J_Array[i]).FillBoundary(infra.geom.periodicity());
    }

    mw_yee.init_E_B(fields, infra);

    //AllPrintToFile("test_output_pre_rename.output") << endl;
    std::cout <<  "step: " << 0 << std::endl;
    E_B_error = mw_yee.computeError(fields, true, infra);
    AllPrintToFile("test_output_pre_rename.output") << endl;
    AllPrintToFile("test_output_pre_rename.output") << "Maxwell" << endl;
    AllPrintToFile("test_output_pre_rename.output") << "step " << 0 << endl;
    switch (vdim) {
    case 1:
        AllPrintToFile("test_output_pre_rename.output").SetPrecision(5) << "Ex error: " << E_B_error[0] << std::endl;
        break;
    case 2:
        AllPrintToFile("test_output_pre_rename.output").SetPrecision(5) << "Ex error: " << E_B_error[0] << " |Ey error: " << E_B_error[1] << std::endl;
        break;
    case 3:
        AllPrintToFile("test_output_pre_rename.output").SetPrecision(5) << "Ex error: " << E_B_error[0] << " |Ey error: " << E_B_error[1] << " |Ez error: " << E_B_error[2] << std::endl;
        break;
    }

    switch (bdim) {
    case 1:
        amrex::AllPrintToFile("test_output_pre_rename.output").SetPrecision(5) << "Bx error: " << E_B_error[vdim] << std::endl;
        break;
    case 3:
        amrex::AllPrintToFile("test_output_pre_rename.output").SetPrecision(5) << "Bx error: " << E_B_error[vdim] << " |By error: " << E_B_error[vdim+1] << " |Bz error: " << E_B_error[vdim+2] << std::endl;
        break;
    }

    for (int n=1;n<=mw_yee.nsteps;n++){
        std::cout << "step: " << n << std::endl;
        mw_yee.advance(infra, mw_yee.dt);
        E_B_error = mw_yee.computeError(fields, true, infra);

        AllPrintToFile("test_output_pre_rename.output") << "step " << n << endl;
        switch (vdim) {
        case 1:
            AllPrintToFile("test_output_pre_rename.output").SetPrecision(5) << "Ex error: " << E_B_error[0] << std::endl;
            break;
        case 2:
            AllPrintToFile("test_output_pre_rename.output").SetPrecision(5) << "Ex error: " << E_B_error[0] << " |Ey error: " << E_B_error[1] << std::endl;
            break;
        case 3:
            AllPrintToFile("test_output_pre_rename.output").SetPrecision(5) << "Ex error: " << E_B_error[0] << " |Ey error: " << E_B_error[1] << " |Ez error: " << E_B_error[2] << std::endl;
            break;
        }
        switch (bdim) {
        case 1:
            amrex::AllPrintToFile("test_output_pre_rename.output").SetPrecision(5) << "Bx error: " << E_B_error[vdim] << std::endl;
            break;
        case 3:
            amrex::AllPrintToFile("test_output_pre_rename.output").SetPrecision(5) << "Bx error: " << E_B_error[vdim] << " |By error: " << E_B_error[vdim+1] << " |Bz error: " << E_B_error[vdim+2] << std::endl;
            break;
        }

        std::ofstream ofsPHI("mw_yee_E" + std::to_string(n) + ".output", std::ofstream::out);
        for (amrex::MFIter mfi(*(mw_yee.E_Array[0])); mfi.isValid(); ++mfi ) {
            amrex::Print(ofsPHI) << (*(mw_yee.E_Array[0]))[mfi] << std::endl;
        }
        ofsPHI.close();
    }

    //------------------------------------------------------------------------------
    // Second maxwell test
    maxwell_yee<vdim> mw_yee_2(init, infra, init.Nghost);

    for (int i=0; i<vdim; i++) {
        (*(mw_yee_2).J_Array[i]).setVal(0.0, 0); // value and component
        (*(mw_yee_2).J_Array[i]).FillBoundary(infra.geom.periodicity());
    }

    mw_yee_2.init_E_B(fields, infra);

    std::cout <<  "step: " << 0 << std::endl;
    E_B_error = mw_yee_2.computeError(fields, true, infra);
    AllPrintToFile("test_output_pre_rename.output") << endl;
    AllPrintToFile("test_output_pre_rename.output") << "Maxwell" << endl;
    AllPrintToFile("test_output_pre_rename.output") << "step " << 0 << endl;
    switch (vdim) {
    case 1:
        AllPrintToFile("test_output_pre_rename.output").SetPrecision(5) << "Ex error: " << E_B_error[0] << std::endl;
        break;
    case 2:
        AllPrintToFile("test_output_pre_rename.output").SetPrecision(5) << "Ex error: " << E_B_error[0] << " |Ey error: " << E_B_error[1] << std::endl;
        break;
    case 3:
        AllPrintToFile("test_output_pre_rename.output").SetPrecision(5) << "Ex error: " << E_B_error[0] << " |Ey error: " << E_B_error[1] << " |Ez error: " << E_B_error[2] << std::endl;
        break;
    }
    switch (bdim) {
    case 1:
        amrex::AllPrintToFile("test_output_pre_rename.output").SetPrecision(5) << "Bx error: " << E_B_error[vdim] << std::endl;
        break;
    case 3:
        amrex::AllPrintToFile("test_output_pre_rename.output").SetPrecision(5) << "Bx error: " << E_B_error[vdim] << " |By error: " << E_B_error[vdim+1] << " |Bz error: " << E_B_error[vdim+2] << std::endl;
        break;
    }

    for (int n=1;n<=mw_yee_2.nsteps;n++){
        std::cout << "step: " << n << std::endl;
        mw_yee_2.advance_time();
        mw_yee_2.advance_E_from_B(infra, mw_yee_2.dt);
        mw_yee_2.advance_E_from_J(infra, mw_yee_2.dt);
        mw_yee_2.advance_B(infra, mw_yee_2.dt);
        E_B_error = mw_yee_2.computeError(fields, true, infra);

        AllPrintToFile("test_output_pre_rename.output") << "step " << n << endl;
        switch (vdim) {
        case 1:
            AllPrintToFile("test_output_pre_rename.output").SetPrecision(5) << "Ex error: " << E_B_error[0] << std::endl;
            break;
        case 2:
            AllPrintToFile("test_output_pre_rename.output").SetPrecision(5) << "Ex error: " << E_B_error[0] << " |Ey error: " << E_B_error[1] << std::endl;
            break;
        case 3:
            AllPrintToFile("test_output_pre_rename.output").SetPrecision(5) << "Ex error: " << E_B_error[0] << " |Ey error: " << E_B_error[1] << " |Ez error: " << E_B_error[2] << std::endl;
            break;
        }
        switch (bdim) {
        case 1:
            amrex::AllPrintToFile("test_output_pre_rename.output").SetPrecision(5) << "Bx error: " << E_B_error[vdim] << std::endl;
            break;
        case 3:
            amrex::AllPrintToFile("test_output_pre_rename.output").SetPrecision(5) << "Bx error: " << E_B_error[vdim] << " |By error: " << E_B_error[vdim+1] << " |Bz error: " << E_B_error[vdim+2] << std::endl;
            break;
        }
    }

    //------------------------------------------------------------------------------
    // Poisson

#if (GEMPIC_SPACEDIM == 1)
    std::string phi = "cos(x) + 1.0/4.0*cos(2*x))";
    std::string rho = "-cos(x)-cos(2*x)";
#elif (GEMPIC_SPACEDIM == 2)
    std::string phi = "cos(x)*cos(y) + 1.0/4.0*cos(2*x)*cos(2*y)";
    std::string rho = "-2*(cos(x)*cos(y)+cos(2*x)*cos(2*y))";
#else
    std::string phi = "-cos(x)*cos(y)*cos(z) - 1.0/4.0*cos(2*x)*cos(2*y)*cos(2*z)";
    std::string rho = "-3*(cos(x)*cos(y)*cos(z)+cos(2*x)*cos(2*y)*cos(2*z))";
#endif

    double x, y, z;
    int err;
    te_variable read_vars[] = {{"x", &x}, {"y", &y}, {"z", &z}};
    int varcount = 3;

    te_expr *rho_parse = te_compile(rho.c_str(), read_vars, varcount, &err);
    te_expr *phi_parse = te_compile(phi.c_str(), read_vars, varcount, &err);

    mw_yee.init_rho_phi(infra, phi_parse, rho_parse, &x, &y, &z);
    mw_yee.solve_poisson(infra);
    E_B_error = mw_yee.computeError(fields_poisson, false, infra);

    AllPrintToFile("test_output_pre_rename.output") << endl;
    AllPrintToFile("test_output_pre_rename.output") << "Poisson" << endl;
    switch (vdim) {
    case 1:
        AllPrintToFile("test_output_pre_rename.output").SetPrecision(5) << "Ex error: " << E_B_error[0] << std::endl;
        break;
    case 2:
        AllPrintToFile("test_output_pre_rename.output").SetPrecision(5) << "Ex error: " << E_B_error[0] << " |Ey error: " << E_B_error[1] << std::endl;
        break;
    case 3:
        AllPrintToFile("test_output_pre_rename.output").SetPrecision(5) << "Ex error: " << E_B_error[0] << " |Ey error: " << E_B_error[1] << " |Ez error: " << E_B_error[2] << std::endl;
        break;
    }


    //------------------------------------------------------------------------------
    // Rho from E

    AllPrintToFile("test_output_pre_rename.output") << endl;
    AllPrintToFile("test_output_pre_rename.output") << "rho_from_E" << endl;
    mw_yee.rho_from_E(infra); // fills rho_gauss_law
    mw_yee.rho_gauss_law.minus(mw_yee.rho, 0, 1, 0);
    AllPrintToFile("test_output_pre_rename.output").SetPrecision(5) << "rho Error: " << Utils::gempic_norm(&(mw_yee.rho_gauss_law), infra, 2) << std::endl;
    AllPrintToFile("test_output_pre_rename.output").SetPrecision(5) << "rho Norm: " << Utils::gempic_norm(&(mw_yee.rho), infra, 2) << std::endl;

    //ofs.close();
    if (ParallelDescriptor::MyProc()==0) std::rename("test_output_pre_rename.output.0", "test_maxwell_yee.output");
}

int main(int argc, char* argv[])
{
    amrex::Initialize(argc,argv);

#if (GEMPIC_SPACEDIM == 1)
    main_main<1>();
    main_main<2>();
#elif (GEMPIC_SPACEDIM == 2)
    main_main<2>();
    main_main<3>();
#elif (GEMPIC_SPACEDIM == 3)
    main_main<3, 1>();
#endif
    amrex::Finalize();
}



