#include "MAG2dEB.H"
#include "MAG2dEB_K.H"

using namespace amrex;

void
MAG2dEB::initEBPoisson ()
{
  const auto prob_lo = geom.ProbLoArray();
  const auto dx = geom.CellSizeArray();
  
#ifdef _OPENMP
#pragma omp parallel if(Gpu::notInLaunchRegion())
#endif
  for (MFIter mfi(rhs, TilingIfNotGPU()); mfi.isValid(); ++mfi) {
    const Box& bx = mfi.tilebox();
    auto rhsfab = rhs.array(mfi);
    auto exactfab = exact_solution.array(mfi);
    auto acoeffab = acoef.array(mfi);
    auto bcoeffab = bcoef.array(mfi);    
    ParallelFor(bx,
		[=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept
		{
		  actual_init_ebpoisson(i,j,rhsfab,exactfab,
					acoeffab,bcoeffab,prob_lo,dx);
		});
  }
  solution.setVal(0.0);
}

void
MAG2dEB::initEBPolar()
{
  const auto prob_lo = geom.ProbLoArray();
  const auto dx = geom.CellSizeArray();
  
#ifdef _OPENMP
#pragma omp parallel if(Gpu::notInLaunchRegion())
#endif
  for (MFIter mfi(rhs, TilingIfNotGPU()); mfi.isValid(); ++mfi) {
    const Box& bx = mfi.tilebox();
    auto rhsfab = rhs.array(mfi);
    auto exactfab = exact_solution.array(mfi);
    auto acoeffab = acoef.array(mfi);
    auto bcoeffab = bcoef.array(mfi);

    ParallelFor(bx,
		[=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept
		{
		   actual_init_ebpolar(i,j,rhsfab,exactfab,
					acoeffab,bcoeffab,prob_lo,dx);
		});
  }
  solution.setVal(0.0);
}

