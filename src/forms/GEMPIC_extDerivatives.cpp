#include <GEMPIC_Fields.H>


using namespace GEMPIC_Fields;

/** 
 * @brief Implementation of the differential operators of a de Rham Complex: grad, curl and div
 *
 * 
 */

void DeRhamComplex::curl(const DeRhamField<Grid::primal, Space::edge>& oneForm,
                               DeRhamField<Grid::primal, Space::face>& twoForm)
{

   // Component-0 of curl
   for (amrex::MFIter mfi(twoForm.data[0]); mfi.isValid(); ++mfi)
   {
       const amrex::Box &bx = mfi.validbox();

       amrex::Array4<amrex::Real> const &twoForm0 = (twoForm.data[0])[mfi].array();
#if (GEMPIC_SPACEDIM > 2)
       amrex::Array4<amrex::Real const> const &oneForm1 = (oneForm.data[1])[mfi].const_array();
#endif
#if (GEMPIC_SPACEDIM > 1)
       amrex::Array4<amrex::Real const> const &oneForm2 = (oneForm.data[2])[mfi].const_array();
#endif
       ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k)
       {
          twoForm0(i, j, k) = 0
#if (GEMPIC_SPACEDIM > 1)
                            + oneForm2(i, j + 1, k) - oneForm2(i, j, k)
#endif
#if (GEMPIC_SPACEDIM > 2)
                            - oneForm1(i, j, k + 1) + oneForm1(i, j, k)
#endif
                                                                        ;
       });
   }
   twoForm.data[0].AverageSync(geom.periodicity());
   (twoForm.data[0]).FillBoundary_nowait(geom.periodicity());


   // Component-1 of curl
   for (amrex::MFIter mfi(twoForm.data[1]); mfi.isValid(); ++mfi)
   {
       const amrex::Box &bx = mfi.validbox();

       amrex::Array4<amrex::Real> const &twoForm1 = (twoForm.data[1])[mfi].array();
       amrex::Array4<amrex::Real const> const &oneForm2 = (oneForm.data[2])[mfi].const_array();
#if (GEMPIC_SPACEDIM > 2)
       amrex::Array4<amrex::Real const> const &oneForm0 = (oneForm.data[0])[mfi].const_array();
#endif
       ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k)
       {
          twoForm1(i, j, k) =
#if (GEMPIC_SPACEDIM > 2)
                              oneForm0(i, j, k + 1) - oneForm0(i, j, k)
#endif
                            - oneForm2(i + 1, j, k) + oneForm2(i, j, k); 
          
       });
   }
   twoForm.data[1].AverageSync(geom.periodicity());
   (twoForm.data[1]).FillBoundary_nowait(geom.periodicity());


   // Component-2 of curl
   for (amrex::MFIter mfi(twoForm.data[2]); mfi.isValid(); ++mfi)
   {
       const amrex::Box &bx = mfi.validbox();

       amrex::Array4<amrex::Real> const &twoForm2 = (twoForm.data[2])[mfi].array();
#if (GEMPIC_SPACEDIM > 1)
       amrex::Array4<amrex::Real const> const &oneForm0 = (oneForm.data[0])[mfi].const_array();
#endif
       amrex::Array4<amrex::Real const> const &oneForm1 = (oneForm.data[1])[mfi].const_array();
       ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k)
       {
          twoForm2(i, j, k) = oneForm1(i + 1, j, k) - oneForm1(i, j, k) 
#if (GEMPIC_SPACEDIM > 1)
                            - oneForm0(i, j + 1, k) + oneForm0(i, j, k)
#endif
                                                                        ;
       });
   }
   twoForm.data[2].AverageSync(geom.periodicity());
   (twoForm.data[2]).FillBoundary_nowait(geom.periodicity());

   // Wait for completed communication of guard data
   (twoForm.data[0]).FillBoundary_finish();
   (twoForm.data[1]).FillBoundary_finish();
   (twoForm.data[2]).FillBoundary_finish();
}


void DeRhamComplex::curl(const DeRhamField<Grid::dual, Space::edge>& oneForm,
                               DeRhamField<Grid::dual, Space::face>& twoForm)
{
   // Component-0 of curl
   for (amrex::MFIter mfi(twoForm.data[0]); mfi.isValid(); ++mfi)
   {
       const amrex::Box &bx = mfi.validbox();

       amrex::Array4<amrex::Real> const &twoForm0 = (twoForm.data[0])[mfi].array();
#if (GEMPIC_SPACEDIM > 2)
       amrex::Array4<amrex::Real const> const &oneForm1 = (oneForm.data[1])[mfi].const_array();
#endif
#if (GEMPIC_SPACEDIM > 1)
       amrex::Array4<amrex::Real const> const &oneForm2 = (oneForm.data[2])[mfi].const_array();
#endif
       ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k)
       {
          twoForm0(i, j, k) = 0
#if (GEMPIC_SPACEDIM > 1)
                            + oneForm2(i, j, k) - oneForm2(i, j - 1, k)
#endif
#if (GEMPIC_SPACEDIM > 2)
                            - oneForm1(i, j, k) + oneForm1(i, j, k - 1);
#endif
       });
   }

   twoForm.data[0].AverageSync(geom.periodicity());
   (twoForm.data[0]).FillBoundary_nowait(geom.periodicity());


   // Component-1 of curl
   for (amrex::MFIter mfi(twoForm.data[1]); mfi.isValid(); ++mfi)
   {
       const amrex::Box &bx = mfi.validbox();

       amrex::Array4<amrex::Real> const &twoForm1 = (twoForm.data[1])[mfi].array();
       amrex::Array4<amrex::Real const> const &oneForm2 = (oneForm.data[2])[mfi].const_array();
#if (GEMPIC_SPACEDIM > 2)
       amrex::Array4<amrex::Real const> const &oneForm0 = (oneForm.data[0])[mfi].const_array();
#endif
       ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k)
       {
          twoForm1(i, j, k) =
#if (GEMPIC_SPACEDIM > 2)
                              oneForm0(i, j, k) - oneForm0(i, j, k - 1)
#endif
                            - oneForm2(i, j, k) + oneForm2(i - 1, j, k); 
          
       });
   }

   twoForm.data[1].AverageSync(geom.periodicity());
   (twoForm.data[1]).FillBoundary_nowait(geom.periodicity());


   // Component-2 of curl
   for (amrex::MFIter mfi(twoForm.data[2]); mfi.isValid(); ++mfi)
   {
       const amrex::Box &bx = mfi.validbox();

       amrex::Array4<amrex::Real> const &twoForm2 = (twoForm.data[2])[mfi].array();
#if (GEMPIC_SPACEDIM > 1)
       amrex::Array4<amrex::Real const> const &oneForm0 = (oneForm.data[0])[mfi].const_array();
#endif
       amrex::Array4<amrex::Real const> const &oneForm1 = (oneForm.data[1])[mfi].const_array();

       ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k)
       {
          twoForm2(i, j, k) = oneForm1(i, j, k) - oneForm1(i - 1, j, k)
#if (GEMPIC_SPACEDIM > 1)
                            - oneForm0(i, j, k) + oneForm0(i, j - 1, k)
#endif
                                                                        ;
       });
   }

   twoForm.data[2].AverageSync(geom.periodicity());
   (twoForm.data[2]).FillBoundary_nowait(geom.periodicity());

   // Wait for completed communication of guard data
   (twoForm.data[0]).FillBoundary_finish();
   (twoForm.data[1]).FillBoundary_finish();
   (twoForm.data[2]).FillBoundary_finish();
}


void DeRhamComplex::grad(const DeRhamField<Grid::primal, Space::node>& zeroForm,
                               DeRhamField<Grid::primal, Space::edge>& oneForm)
{
    for (int comp = 0; comp < 3; ++comp)
    {
        for (amrex::MFIter mfi(oneForm.data[comp]); mfi.isValid(); ++mfi)
        {
            const amrex::Box &bx = mfi.validbox();

            amrex::Array4<amrex::Real const> const &zeroFormMF = (zeroForm.data)[mfi].const_array();
            amrex::Array4<amrex::Real> const &oneFormMF = (oneForm.data[comp])[mfi].array();

            // Calculate gradient component by component
            if (comp == 0)
            {
                ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k)
                {
                    oneFormMF(i, j, k) = zeroFormMF(i + 1, j, k) - zeroFormMF(i, j, k);
                });
            }

            if (comp == 1)
            {
                ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k)
                {
                    oneFormMF(i, j, k) = 0
#if (GEMPIC_SPACEDIM > 1)
                                        + zeroFormMF(i, j + 1, k) - zeroFormMF(i, j, k);
#endif
                });
            }
            
            if (comp == 2)
            {
                ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k)
                {
                    oneFormMF(i, j, k) = 0
#if (GEMPIC_SPACEDIM > 2)
                                        + zeroFormMF(i, j, k + 1) - zeroFormMF(i, j, k);
#endif
                });
            }
        }
        oneForm.data[comp].AverageSync(geom.periodicity());
        (oneForm.data[comp]).FillBoundary_nowait(geom.periodicity());
    }
    (oneForm.data[0]).FillBoundary_finish();
    (oneForm.data[1]).FillBoundary_finish();
    (oneForm.data[2]).FillBoundary_finish();
}


void DeRhamComplex::grad(const DeRhamField<Grid::dual, Space::node>& zeroForm,
                               DeRhamField<Grid::dual, Space::edge>& oneForm)
{
    for (int comp = 0; comp < 3; ++comp)
    {
        for (amrex::MFIter mfi(oneForm.data[comp]); mfi.isValid(); ++mfi)
        {
            const amrex::Box &bx = mfi.validbox();

            amrex::Array4<amrex::Real const> const &zeroFormMF = (zeroForm.data)[mfi].const_array();
            amrex::Array4<amrex::Real> const &oneFormMF = (oneForm.data[comp])[mfi].array();

            // Calculate gradient component by component
            if (comp == 0)
            {
                ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k)
                {
                    oneFormMF(i, j, k) = zeroFormMF(i, j, k) - zeroFormMF(i - 1, j, k);
                });
            }

            if (comp == 1)
            {
                ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k)
                {
                    oneFormMF(i, j, k) = 0
#if (GEMPIC_SPACEDIM > 1)
                                        + zeroFormMF(i, j, k) - zeroFormMF(i, j - 1, k);
#endif
                });
            }
            
            if (comp == 2)
            {
                ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k)
                {
                    oneFormMF(i, j, k) = 0
#if (GEMPIC_SPACEDIM > 2)
                                        + zeroFormMF(i, j, k) - zeroFormMF(i, j, k - 1);
#endif
                });
            }
        }
        oneForm.data[comp].AverageSync(geom.periodicity());
        (oneForm.data[comp]).FillBoundary_nowait(geom.periodicity());
    }
    
    (oneForm.data[0]).FillBoundary_finish();
    (oneForm.data[1]).FillBoundary_finish();
    (oneForm.data[2]).FillBoundary_finish();
}


void DeRhamComplex::div(const DeRhamField<primal, Space::face>& twoForm,
                              DeRhamField<primal, Space::cell>& threeForm)
{
    for (amrex::MFIter mfi(threeForm.data); mfi.isValid(); ++mfi)
    {
        const amrex::Box &bx = mfi.validbox();

        amrex::Array4<amrex::Real const> const &twoFormMF0 = (twoForm.data[0])[mfi].const_array();
#if (GEMPIC_SPACEDIM > 1)
        amrex::Array4<amrex::Real const> const &twoFormMF1 = (twoForm.data[1])[mfi].const_array();
#endif
#if (GEMPIC_SPACEDIM > 2)
        amrex::Array4<amrex::Real const> const &twoFormMF2 = (twoForm.data[2])[mfi].const_array();
#endif
        amrex::Array4<amrex::Real> const &threeFormMF = (threeForm.data)[mfi].array();
        
        ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k)
        {
            threeFormMF(i, j, k) = (twoFormMF0(i + 1, j, k) - twoFormMF0(i, j, k))
#if (GEMPIC_SPACEDIM > 1)
                                 + (twoFormMF1(i, j + 1, k) - twoFormMF1(i, j, k))
#endif
#if (GEMPIC_SPACEDIM > 2)
                                 + (twoFormMF2(i, j, k + 1) - twoFormMF2(i, j, k))
#endif
                                                                                    ;
        });
    }
    threeForm.averageSync();
    threeForm.fillBoundary();
}


void DeRhamComplex::div(const DeRhamField<dual, Space::face>& twoForm,
                              DeRhamField<dual, Space::cell>& threeForm)
{
    for (amrex::MFIter mfi(threeForm.data); mfi.isValid(); ++mfi)
    {
        const amrex::Box &bx = mfi.validbox();

        amrex::Array4<amrex::Real const> const &twoFormMF0 = (twoForm.data[0])[mfi].const_array();
#if (GEMPIC_SPACEDIM > 1)
        amrex::Array4<amrex::Real const> const &twoFormMF1 = (twoForm.data[1])[mfi].const_array();
#endif
#if (GEMPIC_SPACEDIM > 2)
        amrex::Array4<amrex::Real const> const &twoFormMF2 = (twoForm.data[2])[mfi].const_array();
#endif
        amrex::Array4<amrex::Real> const &threeFormMF = (threeForm.data)[mfi].array();
        
        ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k)
        {
            threeFormMF(i, j, k) = (twoFormMF0(i, j, k) - twoFormMF0(i - 1, j, k))
#if (GEMPIC_SPACEDIM > 1)
                                 + (twoFormMF1(i, j, k) - twoFormMF1(i, j - 1, k))
#endif
#if (GEMPIC_SPACEDIM > 2)
                                 + (twoFormMF2(i, j, k) - twoFormMF2(i, j, k - 1))
#endif
                                                                                    ;

        });
    }
    threeForm.averageSync();
    threeForm.fillBoundary();
}
