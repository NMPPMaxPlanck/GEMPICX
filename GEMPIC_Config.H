#ifndef GEMPIC_CONFIG_H_
#define GEMPIC_CONFIG_H_

#define GEMPIC_SPACEDIM AMREX_SPACEDIM

#define GEMPIC_NUMSPEC 1
#define GEMPIC_ELECTROMAGNETIC true

#define MAXGAUSS 2 // maximum number of velocity gaussians

#define GEMPIC_CTEST_TOL 1e-4
#define GEMPIC_CTEST_RELTOL 1e-4

#if GEMPIC_SPACEDIM==1
/**
 * The GEMPIC_SPACEDIM(a,b,c) macro multiplies the elements of a
 * comma-separated list of 1, 2, or all 3 of the arguments of the call,
 * depending on the value of AMREX_SPACEDIM.
 */
#   define GEMPIC_D_MULT(a, b, c) (a)
#elif GEMPIC_SPACEDIM==2
#   define GEMPIC_D_MULT(a, b, c) (a) * (b)
#elif GEMPIC_SPACEDIM==3
#   define GEMPIC_D_MULT(a, b, c) (a) * (b) * (c)
#endif


#endif
