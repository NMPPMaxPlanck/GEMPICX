#ifndef GEMPIC_CONFIG_H_
#define GEMPIC_CONFIG_H_

//#define GEMPIC_SPACEDIM 3
#define GEMPIC_SPACEDIM AMREX_SPACEDIM
#define GEMPIC_VDIM GEMPIC_SPACEDIM
#define GEMPIC_NUMSPEC 1
#define GEMPIC_NUMGAUS 1

#if (GEMPIC_VDIM == 1)
#  define AMREX_V_DECL(a,b,c) a
#elif (GEMPIC_VDIM == 2)
#  define AMREX_V_DECL(a,b,c) a,b
#elif (GEMPIC_VDIM == 3)
#  define AMREX_V_DECL(a,b,c) a,b,c
#endif

#endif
