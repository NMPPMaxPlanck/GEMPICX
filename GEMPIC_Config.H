#ifndef GEMPIC_CONFIG_H_
#define GEMPIC_CONFIG_H_

//#define GEMPIC_SPACEDIM 3
#define GEMPIC_SPACEDIM AMREX_SPACEDIM
#define GEMPIC_VDIM 3
#define GEMPIC_NUMSPEC 1

#define GEMPIC_DEG_X 1
#define GEMPIC_DEG_Y 1
#define GEMPIC_DEG_Z 1

#if (GEMPIC_VDIM < 3)
#  define GEMPIC_BDIM 1
#else
#  define GEMPIC_BDIM 3
#endif

#if (GEMPIC_VDIM == 1)
#  define AMREX_V_DECL(a,b,c) a
#elif (GEMPIC_VDIM == 2)
#  define AMREX_V_DECL(a,b,c) a,b
#elif (GEMPIC_VDIM == 3)
#  define AMREX_V_DECL(a,b,c) a,b,c
#endif

#if (GEMPIC_BDIM == 1)
#  define AMREX_B_DECL(a,b,c) a
#elif (GEMPIC_BDIM == 3)
#  define AMREX_B_DECL(a,b,c) a,b,c
#endif

#endif
