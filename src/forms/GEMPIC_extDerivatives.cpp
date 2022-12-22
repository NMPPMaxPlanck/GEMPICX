#include <GEMPIC_Fields.H>

/* Implementation of the differential operators of a de Rham Complex: grad, curl and div
 *
 * This is a 3D3V demonstration with 3D B field
 */

using namespace GEMPIC_Fields;

// Beware of the indexing on Amrex. For a derivative that goes to the ghost region
// df = f[i+1] - f[i], going to -1 breaks centeredness of the differential form
void DeRhamComplex::curl(const DeRhamField<Grid::primal, Space::edge>& oneForm,
                               DeRhamField<Grid::primal, Space::face>& twoForm)
{

   // Component-0 of curl
   for (amrex::MFIter mfi(twoForm.data[0]); mfi.isValid(); ++mfi)
   {
       const amrex::Box &bx = mfi.validbox();

       amrex::Array4<amrex::Real> const &twoForm0 = (twoForm.data[0])[mfi].array();
       amrex::Array4<amrex::Real const> const &oneForm1 = (oneForm.data[1])[mfi].const_array();
       amrex::Array4<amrex::Real const> const &oneForm2 = (oneForm.data[2])[mfi].const_array();
       ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k)
       {
          twoForm0(i, j, k) = oneForm2(i, j + 1, k) - oneForm2(i, j, k) 
                            - oneForm1(i, j, k + 1) + oneForm1(i, j, k); 
       });
   }
   (twoForm.data[0]).FillBoundary_nowait(geom.periodicity());


   // Component-1 of curl
   for (amrex::MFIter mfi(twoForm.data[1]); mfi.isValid(); ++mfi)
   {
       const amrex::Box &bx = mfi.validbox();

       amrex::Array4<amrex::Real> const &twoForm1 = (twoForm.data[1])[mfi].array();
       amrex::Array4<amrex::Real const> const &oneForm2 = (oneForm.data[2])[mfi].const_array();
       amrex::Array4<amrex::Real const> const &oneForm0 = (oneForm.data[0])[mfi].const_array();
       ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k)
       {
          twoForm1(i, j, k) = oneForm0(i, j, k + 1) - oneForm0(i, j, k)
                            - oneForm2(i + 1, j, k) + oneForm2(i, j, k); 
          
       });
   }
   (twoForm.data[1]).FillBoundary_nowait(geom.periodicity());


   // Component-2 of curl
   for (amrex::MFIter mfi(twoForm.data[2]); mfi.isValid(); ++mfi)
   {
       const amrex::Box &bx = mfi.validbox();

       amrex::Array4<amrex::Real> const &twoForm2 = (twoForm.data[2])[mfi].array();
       amrex::Array4<amrex::Real const> const &oneForm0 = (oneForm.data[0])[mfi].const_array();
       amrex::Array4<amrex::Real const> const &oneForm1 = (oneForm.data[1])[mfi].const_array();
       ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k)
       {
          twoForm2(i, j, k) = oneForm1(i + 1, j, k) - oneForm1(i, j, k) 
                            - oneForm0(i, j + 1, k) + oneForm0(i, j, k); 
       });
   }
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
       amrex::Array4<amrex::Real const> const &oneForm1 = (oneForm.data[1])[mfi].const_array();
       amrex::Array4<amrex::Real const> const &oneForm2 = (oneForm.data[2])[mfi].const_array();
       ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k)
       {
          twoForm0(i, j, k) = oneForm2(i, j + 1, k) - oneForm2(i, j, k) 
                            - oneForm1(i, j, k + 1) + oneForm1(i, j, k); 
       });
   }
   (twoForm.data[0]).FillBoundary_nowait(geom.periodicity());


   // Component-1 of curl
   for (amrex::MFIter mfi(twoForm.data[1]); mfi.isValid(); ++mfi)
   {
       const amrex::Box &bx = mfi.validbox();

       amrex::Array4<amrex::Real> const &twoForm1 = (twoForm.data[1])[mfi].array();
       amrex::Array4<amrex::Real const> const &oneForm2 = (oneForm.data[2])[mfi].const_array();
       amrex::Array4<amrex::Real const> const &oneForm0 = (oneForm.data[0])[mfi].const_array();
       ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k)
       {
          twoForm1(i, j, k) = oneForm0(i, j, k + 1) - oneForm0(i, j, k)
                            - oneForm2(i + 1, j, k) + oneForm2(i, j, k); 
          
       });
   }
   (twoForm.data[1]).FillBoundary_nowait(geom.periodicity());


   // Component-2 of curl
   for (amrex::MFIter mfi(twoForm.data[2]); mfi.isValid(); ++mfi)
   {
       const amrex::Box &bx = mfi.validbox();

       amrex::Array4<amrex::Real> const &twoForm2 = (twoForm.data[2])[mfi].array();
       amrex::Array4<amrex::Real const> const &oneForm0 = (oneForm.data[0])[mfi].const_array();
       amrex::Array4<amrex::Real const> const &oneForm1 = (oneForm.data[1])[mfi].const_array();
       ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k)
       {
          twoForm2(i, j, k) = oneForm1(i + 1, j, k) - oneForm1(i, j, k) 
                            - oneForm0(i, j + 1, k) + oneForm0(i, j, k); 
       });
   }
   (twoForm.data[2]).FillBoundary_nowait(geom.periodicity());

   // Wait for completed communication of guard data
   (twoForm.data[0]).FillBoundary_finish();
   (twoForm.data[1]).FillBoundary_finish();
   (twoForm.data[2]).FillBoundary_finish();
}


/*
void DeRhamComplex::curl(const DeRhamField<dual, Space::face>& twoForm,
                               DeRhamField<dual, Space::edge>& oneForm)
{
   // Component-0 of curl
   for (amrex::MFIter mfi(oneForm.data[0]); mfi.isValid(); ++mfi)
   {
       const amrex::Box &bx = mfi.validbox();

       amrex::Array4<amrex::Real> const &oneForm0 = (oneForm.data[0])[mfi].array();
       amrex::Array4<amrex::Real const> const &twoForm1 = (twoForm.data[1])[mfi].const_array();
       amrex::Array4<amrex::Real const> const &twoForm2 = (twoForm.data[2])[mfi].const_array();
       ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k)
       {
          oneForm0(i, j, k) = twoForm2(i, j + 1, k) - twoForm2(i, j, k) 
                            - twoForm1(i, j, k + 1) + twoForm1(i, j, k); 
       });
   }
   (oneForm.data[0]).FillBoundary_nowait(geom.periodicity());


   // Component-1 of curl
   for (amrex::MFIter mfi(oneForm.data[1]); mfi.isValid(); ++mfi)
   {
       const amrex::Box &bx = mfi.validbox();

       amrex::Array4<amrex::Real> const &oneForm1 = (oneForm.data[1])[mfi].array();
       amrex::Array4<amrex::Real const> const &twoForm2 = (twoForm.data[2])[mfi].const_array();
       amrex::Array4<amrex::Real const> const &twoForm0 = (twoForm.data[0])[mfi].const_array();
       ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k)
       {
          oneForm1(i, j, k) = twoForm0(i, j, k + 1) - twoForm0(i, j, k)
                            - twoForm2(i + 1, j, k) + twoForm2(i, j, k); 
          
       });
   }
   (oneForm.data[1]).FillBoundary_nowait(geom.periodicity());


   // Component-2 of curl
   for (amrex::MFIter mfi(oneForm.data[2]); mfi.isValid(); ++mfi)
   {
       const amrex::Box &bx = mfi.validbox();

       amrex::Array4<amrex::Real> const &oneForm2 = (oneForm.data[2])[mfi].array();
       amrex::Array4<amrex::Real const> const &twoForm0 = (twoForm.data[0])[mfi].const_array();
       amrex::Array4<amrex::Real const> const &twoForm1 = (twoForm.data[1])[mfi].const_array();
       ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k)
       {
          oneForm2(i, j, k) = twoForm1(i + 1, j, k) - twoForm1(i, j, k) 
                            - twoForm0(i, j + 1, k) + twoForm0(i, j, k); 
       });
   }
   (oneForm.data[2]).FillBoundary_nowait(geom.periodicity());

   // Wait for completed communication of guard data
   (oneForm.data[0]).FillBoundary_finish();
   (oneForm.data[1]).FillBoundary_finish();
   (oneForm.data[2]).FillBoundary_finish();
}
*/
