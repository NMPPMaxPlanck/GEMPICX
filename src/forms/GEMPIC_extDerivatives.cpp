#include <GEMPIC_Fields.H>


using namespace GEMPIC_Fields;

/**
* @brief Computes the discrete curl \f$\mathbb{C}\f$ on the primal grid
* 
* Description:
* Using the geometric degrees of freedom. No approximations involved.
* 
* For directions 1 and 2, the k-forms are taken from the "stacked" de Rham complex
* 
* @param oneForm : DeRhamField<Grid::primal, Space::edge>, 1-form \f$u^1\f$ holding the edge integrals
* @param twoForm : DeRhamField<Grid::primal, Space::face>, 2-form \f$\mathbb{C} u^1\f$ holding the resulting face integrals
* 
* @return void
*/
void DeRhamComplex::curl(const DeRhamField<Grid::primal, Space::edge>& oneForm,
                               DeRhamField<Grid::primal, Space::face>& twoForm)
{
   // Component-0 of curl
   for (amrex::MFIter mfi(twoForm.data[xDir]); mfi.isValid(); ++mfi)
   {
       const amrex::Box &bx = mfi.validbox();

       amrex::Array4<amrex::Real> const &twoForm0 = (twoForm.data[xDir])[mfi].array();
#if (GEMPIC_SPACEDIM > 2)
       amrex::Array4<amrex::Real const> const &oneForm1 = (oneForm.data[yDir])[mfi].const_array();
#endif
#if (GEMPIC_SPACEDIM > 1)
       amrex::Array4<amrex::Real const> const &oneForm2 = (oneForm.data[zDir])[mfi].const_array();
#endif
       ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k)
       {
          twoForm0(i, j, k) = GEMPIC_D_ADD(0., oneForm2(i, j + 1, k) - oneForm2(i, j, k), - oneForm1(i, j, k + 1) + oneForm1(i, j, k));
       });
   }
   twoForm.data[xDir].AverageSync(geom.periodicity());
   twoForm.data[xDir].FillBoundary_nowait(geom.periodicity());

   // Component-1 of curl
   for (amrex::MFIter mfi(twoForm.data[yDir]); mfi.isValid(); ++mfi)
   {
       const amrex::Box &bx = mfi.validbox();

       amrex::Array4<amrex::Real> const &twoForm1 = (twoForm.data[yDir])[mfi].array();
       amrex::Array4<amrex::Real const> const &oneForm2 = (oneForm.data[zDir])[mfi].const_array();
#if (GEMPIC_SPACEDIM > 2)
       amrex::Array4<amrex::Real const> const &oneForm0 = (oneForm.data[xDir])[mfi].const_array();
#endif
       ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k)
       {
          twoForm1(i, j, k) = GEMPIC_D_ADD(- oneForm2(i + 1, j, k) + oneForm2(i, j, k), 0., oneForm0(i, j, k + 1) - oneForm0(i, j, k));
       });
   }
   twoForm.data[yDir].AverageSync(geom.periodicity());
   twoForm.data[yDir].FillBoundary_nowait(geom.periodicity());


   // Component-2 of curl
   for (amrex::MFIter mfi(twoForm.data[zDir]); mfi.isValid(); ++mfi)
   {
       const amrex::Box &bx = mfi.validbox();

       amrex::Array4<amrex::Real> const &twoForm2 = (twoForm.data[zDir])[mfi].array();
#if (GEMPIC_SPACEDIM > 1)
       amrex::Array4<amrex::Real const> const &oneForm0 = (oneForm.data[xDir])[mfi].const_array();
#endif
       amrex::Array4<amrex::Real const> const &oneForm1 = (oneForm.data[yDir])[mfi].const_array();
       ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k)
       {
          twoForm2(i, j, k) = GEMPIC_D_ADD(oneForm1(i + 1, j, k) - oneForm1(i, j, k), - oneForm0(i, j + 1, k) + oneForm0(i, j, k), 0.);                                                                        ;
       });
   }
   twoForm.data[zDir].AverageSync(geom.periodicity());
   twoForm.data[zDir].FillBoundary_nowait(geom.periodicity());

   // Wait for completed communication of guard data
   twoForm.data[xDir].FillBoundary_finish();
   twoForm.data[yDir].FillBoundary_finish();
   twoForm.data[zDir].FillBoundary_finish();
}

/**
* @brief Computes the discrete curl \f$\mathbb{C}\f$ on the dual grid
* 
* Description:
* Using the geometric degrees of freedom. No approximations involved.
* 
* For directions 1 and 2, the k-forms are taken from the "stacked" de Rham complex
* 
* @param oneForm : DeRhamField<Grid::dual, Space::edge>, 1-form \f$\tilde{u}^1\f$ holding the edge integrals
* @param twoForm : DeRhamField<Grid::dual, Space::face>, 2-form \f$\tilde{C} \tilde{u}^1\f$ holding the resulting face integrals
* 
* @return void
*/
void DeRhamComplex::curl(const DeRhamField<Grid::dual, Space::edge>& oneForm,
                               DeRhamField<Grid::dual, Space::face>& twoForm)
{
   // Component-0 of curl
   for (amrex::MFIter mfi(twoForm.data[xDir]); mfi.isValid(); ++mfi)
   {
       const amrex::Box &bx = mfi.validbox();

       amrex::Array4<amrex::Real> const &twoForm0 = (twoForm.data[xDir])[mfi].array();
#if (GEMPIC_SPACEDIM > 2)
       amrex::Array4<amrex::Real const> const &oneForm1 = (oneForm.data[yDir])[mfi].const_array();
#endif
#if (GEMPIC_SPACEDIM > 1)
       amrex::Array4<amrex::Real const> const &oneForm2 = (oneForm.data[zDir])[mfi].const_array();
#endif
       ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k)
       {
          twoForm0(i, j, k) = GEMPIC_D_ADD(0., oneForm2(i, j, k) - oneForm2(i, j - 1, k), - oneForm1(i, j, k) + oneForm1(i, j, k - 1));
       });
   }

   twoForm.data[xDir].AverageSync(geom.periodicity());
   twoForm.data[xDir].FillBoundary_nowait(geom.periodicity());


   // Component-1 of curl
   for (amrex::MFIter mfi(twoForm.data[yDir]); mfi.isValid(); ++mfi)
   {
       const amrex::Box &bx = mfi.validbox();

       amrex::Array4<amrex::Real> const &twoForm1 = (twoForm.data[yDir])[mfi].array();
       amrex::Array4<amrex::Real const> const &oneForm2 = (oneForm.data[zDir])[mfi].const_array();
#if (GEMPIC_SPACEDIM > 2)
       amrex::Array4<amrex::Real const> const &oneForm0 = (oneForm.data[xDir])[mfi].const_array();
#endif
       ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k)
       {
          twoForm1(i, j, k) = GEMPIC_D_ADD(- oneForm2(i, j, k) + oneForm2(i - 1, j, k), 0., oneForm0(i, j, k) - oneForm0(i, j, k - 1));          
       });
   }

   twoForm.data[yDir].AverageSync(geom.periodicity());
   twoForm.data[yDir].FillBoundary_nowait(geom.periodicity());


   // Component-2 of curl
   for (amrex::MFIter mfi(twoForm.data[zDir]); mfi.isValid(); ++mfi)
   {
       const amrex::Box &bx = mfi.validbox();

       amrex::Array4<amrex::Real> const &twoForm2 = (twoForm.data[zDir])[mfi].array();
#if (GEMPIC_SPACEDIM > 1)
       amrex::Array4<amrex::Real const> const &oneForm0 = (oneForm.data[xDir])[mfi].const_array();
#endif
       amrex::Array4<amrex::Real const> const &oneForm1 = (oneForm.data[yDir])[mfi].const_array();

       ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k)
       {
          twoForm2(i, j, k) = GEMPIC_D_ADD(oneForm1(i, j, k) - oneForm1(i - 1, j, k), - oneForm0(i, j, k) + oneForm0(i, j - 1, k), 0.);
       });
   }

   twoForm.data[zDir].AverageSync(geom.periodicity());
   twoForm.data[zDir].FillBoundary_nowait(geom.periodicity());

   // Wait for completed communication of guard data
   twoForm.data[xDir].FillBoundary_finish();
   twoForm.data[yDir].FillBoundary_finish();
   twoForm.data[zDir].FillBoundary_finish();
}

/**
* @brief Computes the discrete gradient \f$\mathbb{G}\f$ on the primal grid
* 
* Description:
* Using the geometric degrees of freedom. No approximations involved.
* 
* For directions 1 and 2, the k-forms are taken from the "stacked" de Rham complex
* 
* @param zeroForm : DeRhamField<Grid::primal, Space::node>, 0-form \f$u^0\f$ holding the node values
* @param oneForm : DeRhamField<Grid::primal, Space::edge>, 1-form \f$\mathbb{G} u^0\f$ holding the resulting edge integrals
* 
* @return void
*/
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
            if (comp == xDir)
            {
                ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k)
                {
                    oneFormMF(i, j, k) = zeroFormMF(i + 1, j, k) - zeroFormMF(i, j, k);
                });
            }

            if (comp == yDir)
            {
                ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k)
                {
                    oneFormMF(i, j, k) = GEMPIC_D_ADD(0., zeroFormMF(i, j + 1, k) - zeroFormMF(i, j, k), 0.);
                });
            }

            if (comp == zDir)
            {
                ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k)
                {
                    oneFormMF(i, j, k) = GEMPIC_D_ADD(0., 0., zeroFormMF(i, j, k + 1) - zeroFormMF(i, j, k));
                });
            }
        }
        oneForm.data[comp].AverageSync(geom.periodicity());
        oneForm.data[comp].FillBoundary_nowait(geom.periodicity());
    }
    oneForm.data[xDir].FillBoundary_finish();
    oneForm.data[yDir].FillBoundary_finish();
    oneForm.data[zDir].FillBoundary_finish();
}

/**
* @brief Computes the discrete gradient \f$\mathbb{G}\f$ on the dual grid
* 
* Description:
* Using the geometric degrees of freedom. No approximations involved.
* 
* For directions 1 and 2, the k-forms are taken from the "stacked" de Rham complex
* 
* @param zeroForm : DeRhamField<Grid::dual, Space::node>, 0-form \f$\tilde{u}^0\f$ holding the node values
* @param oneForm : DeRhamField<Grid::dual, Space::edge>, 1-form \f$\tilde{G} \tilde{u}^0\f$ holding the resulting edge integrals
* 
* @return void
*/
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
            if (comp == xDir)
            {
                ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k)
                {
                    oneFormMF(i, j, k) = zeroFormMF(i, j, k) - zeroFormMF(i - 1, j, k);
                });
            }

            if (comp == yDir)
            {
                ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k)
                {
                    oneFormMF(i, j, k) = GEMPIC_D_ADD(0., zeroFormMF(i, j, k) - zeroFormMF(i, j - 1, k), 0.);
                });
            }
            
            if (comp == zDir)
            {
                ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k)
                {
                    oneFormMF(i, j, k) = GEMPIC_D_ADD(0., 0., zeroFormMF(i, j, k) - zeroFormMF(i, j, k - 1));
                });
            }
        }
        oneForm.data[comp].AverageSync(geom.periodicity());
        oneForm.data[comp].FillBoundary_nowait(geom.periodicity());
    }
    
    oneForm.data[xDir].FillBoundary_finish();
    oneForm.data[yDir].FillBoundary_finish();
    oneForm.data[zDir].FillBoundary_finish();
}

/**
* @brief Computes the discrete divergence \f$\mathbb{D}\f$ on the primal grid
* 
* Description:
* Using the geometric degrees of freedom. No approximations involved.
* 
* For directions 1 and 2, the k-forms are taken from the "stacked" de Rham complex
* 
* @param twoForm : DeRhamField<Grid::primal, Space::face>, 2-form \f$u^2\f$ holding the face integrals
* @param threeForm : DeRhamField<Grid::primal, Space::cell>, 3-form \f$\mathbb{D} u^2\f$ holding the resulting cell integrals
* 
* @return void
*/
void DeRhamComplex::div(const DeRhamField<Grid::primal, Space::face>& twoForm,
                              DeRhamField<Grid::primal, Space::cell>& threeForm)
{
    for (amrex::MFIter mfi(threeForm.data); mfi.isValid(); ++mfi)
    {
        const amrex::Box &bx = mfi.validbox();

        amrex::Array4<amrex::Real const> const &twoFormMF0 = (twoForm.data[xDir])[mfi].const_array();
#if (GEMPIC_SPACEDIM > 1)
        amrex::Array4<amrex::Real const> const &twoFormMF1 = (twoForm.data[yDir])[mfi].const_array();
#endif
#if (GEMPIC_SPACEDIM > 2)
        amrex::Array4<amrex::Real const> const &twoFormMF2 = (twoForm.data[zDir])[mfi].const_array();
#endif
        amrex::Array4<amrex::Real> const &threeFormMF = (threeForm.data)[mfi].array();
        
        ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k)
        {
            threeFormMF(i, j, k) = GEMPIC_D_ADD(twoFormMF0(i + 1, j, k) - twoFormMF0(i, j, k), twoFormMF1(i, j + 1, k) - twoFormMF1(i, j, k), twoFormMF2(i, j, k + 1) - twoFormMF2(i, j, k));
        });
    }
    threeForm.averageSync();
    threeForm.fillBoundary();
}

/**
* @brief Computes the discrete divergence \f$\mathbb{D}\f$ on the dual grid
* 
* Description:
* Using the geometric degrees of freedom. No approximations involved.
* 
* For directions 1 and 2, the k-forms are taken from the "stacked" de Rham complex
* 
* @param twoForm : DeRhamField<Grid::primal, Space::dual>, 2-form \f$\tilde{u}^2\f$ holding the face integrals
* @param threeForm : DeRhamField<Grid::primal, Space::dual>, 3-form \f$\tilde{D} \tilde{u}^2\f$ holding the resulting cell integrals
* 
* @return void
*/
void DeRhamComplex::div(const DeRhamField<Grid::dual, Space::face>& twoForm,
                              DeRhamField<Grid::dual, Space::cell>& threeForm)
{
    

    for (amrex::MFIter mfi(threeForm.data); mfi.isValid(); ++mfi)
    {
        const amrex::Box &bx = mfi.validbox();

        amrex::Array4<amrex::Real const> const &twoFormMF0 = (twoForm.data[xDir])[mfi].const_array();
#if (GEMPIC_SPACEDIM > 1)
        amrex::Array4<amrex::Real const> const &twoFormMF1 = (twoForm.data[yDir])[mfi].const_array();
#endif
#if (GEMPIC_SPACEDIM > 2)
        amrex::Array4<amrex::Real const> const &twoFormMF2 = (twoForm.data[zDir])[mfi].const_array();
#endif
        amrex::Array4<amrex::Real> const &threeFormMF = (threeForm.data)[mfi].array();
        
        ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k)
        {
            threeFormMF(i, j, k) = GEMPIC_D_ADD(twoFormMF0(i, j, k) - twoFormMF0(i - 1, j, k), twoFormMF1(i, j, k) - twoFormMF1(i, j - 1, k), twoFormMF2(i, j, k) - twoFormMF2(i, j, k - 1));
        });
    }
    threeForm.averageSync();
    threeForm.fillBoundary();
}
