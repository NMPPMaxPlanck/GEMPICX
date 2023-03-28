#ifndef GEMPIC_CONFIG_H_
#define GEMPIC_CONFIG_H_

#define GEMPIC_SPACEDIM AMREX_SPACEDIM

#define GEMPIC_NUMSPEC 1
#define GEMPIC_ELECTROMAGNETIC true

#define MAXGAUSS 2 // maximum number of velocity gaussians

#define GEMPIC_CTEST_TOL 1e-4
#define GEMPIC_CTEST_RELTOL 1e-4

#if GEMPIC_SPACEDIM==1
#    define GEMPIC_D_DECL_0(a, b, c) a, 0, 0
#elif GEMPIC_SPACEDIM==2
#    define GEMPIC_D_DECL_0(a, b, c) a, b, 0
#elif GEMPIC_SPACEDIM==3
#    define GEMPIC_D_DECL_0(a, b, c) a, b, c
#endif

#if GEMPIC_SPACEDIM==1
#   define GEMPIC_D_COND(a, b, c) a
#elif GEMPIC_SPACEDIM==2
#   define GEMPIC_D_COND(a, b, c) a b
#elif GEMPIC_SPACEDIM==3
#   define GEMPIC_D_COND(a, b, c) a b c 
#endif

#endif
