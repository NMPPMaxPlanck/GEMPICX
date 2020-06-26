#include <cmath>
#include <iostream>

#include <AMReX.H>
#include <AMReX_Print.H>
#include <AMReX_PlotFileUtil.H>
#include <AMReX_ParmParse.H>

#include "tinyexpr.h"

using namespace std;

// The following function is an example to showcase how a compiled expression can be passed through a function
void test_parsed(te_expr *comp_fct, double *x, double *y, double *z, double *alpha){
    *x = 0.0;
    *alpha = 1.0;
    const double result = te_eval(comp_fct);
    std::cout << "Result parsed in outer function (x = " << *x << ", alpha = " << *alpha << "): " << result << std::endl;
}

int main(int argc, char *argv[])
{

    std::string readfct;
    readfct = "alpha * cos(x)"; // later need type const char * (get with .c_str() )
    std::cout << "function to evaluate: " << readfct << std::endl;

    double x, y, z, alpha;
    int err;
    te_variable read_vars[] = {{"x", &x}, {"y", &y}, {"z", &z}, {"alpha", &alpha}};
    int varcount = 4;
    te_expr *comp_fct = te_compile(readfct.c_str(), read_vars, varcount, &err);

    // Evaluate here:
    x = 3.14159265359/2.0;
    const double result = te_eval(comp_fct);
    std::cout << "Result parsed in main (x = " << x << ", alpha = " << alpha << "): " << result << std::endl;

    //Evaluate in other function
    test_parsed(comp_fct, &x, &y, &z, &alpha);


    return 0;

}


