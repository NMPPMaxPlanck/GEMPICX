#include <GEMPIC_Fields.H>
#include <GEMPIC_Params.H>
#include <GEMPIC_FDDeRhamComplex.H>

using namespace GEMPIC_Fields;
using namespace GEMPIC_FDDeRhamComplex;

FDDeRhamComplex::FDDeRhamComplex(Parameters params) : DeRhamComplex::DeRhamComplex(params)
{
    // Parameters used in the projection and hodge
    m_dr = params.dr();
    m_geom = params.geometry();
    m_grid = params.grid();
    m_distriMap = params.distriMap();
    m_nGhost = params.degree()/2 - 1;

    // There is only one components in each MultiFab as the different components of the forms are centered differently
    m_tempPrimalZeroForm.define(amrex::convert(m_grid, amrex::IndexType(amrex::IntVect{AMREX_D_DECL(1, 1, 1)})), m_distriMap, 1, m_nGhost);
    m_tempDualZeroForm.define(amrex::convert(m_grid, amrex::IndexType(amrex::IntVect{AMREX_D_DECL(0, 0, 0)})), m_distriMap, 1, m_nGhost);

    m_tempPrimalOneForm[xDir].define(amrex::convert(m_grid, amrex::IndexType(amrex::IntVect{AMREX_D_DECL(0, 1, 1)})), m_distriMap, 1, m_nGhost);
    m_tempPrimalOneForm[yDir].define(amrex::convert(m_grid, amrex::IndexType(amrex::IntVect{AMREX_D_DECL(1, 0, 1)})), m_distriMap, 1, m_nGhost);
    m_tempPrimalOneForm[zDir].define(amrex::convert(m_grid, amrex::IndexType(amrex::IntVect{AMREX_D_DECL(1, 1, 0)})), m_distriMap, 1, m_nGhost);

    m_tempDualOneForm[xDir].define(amrex::convert(m_grid, amrex::IndexType(amrex::IntVect{AMREX_D_DECL(1, 0, 0)})), m_distriMap, 1, m_nGhost);
    m_tempDualOneForm[yDir].define(amrex::convert(m_grid, amrex::IndexType(amrex::IntVect{AMREX_D_DECL(0, 1, 0)})), m_distriMap, 1, m_nGhost);
    m_tempDualOneForm[zDir].define(amrex::convert(m_grid, amrex::IndexType(amrex::IntVect{AMREX_D_DECL(0, 0, 1)})), m_distriMap, 1, m_nGhost);

    m_tempPrimalTwoForm[xDir].define(amrex::convert(m_grid, amrex::IndexType(amrex::IntVect{AMREX_D_DECL(1, 0, 0)})), m_distriMap, 1, m_nGhost);
    m_tempPrimalTwoForm[yDir].define(amrex::convert(m_grid, amrex::IndexType(amrex::IntVect{AMREX_D_DECL(0, 1, 0)})), m_distriMap, 1, m_nGhost);
    m_tempPrimalTwoForm[zDir].define(amrex::convert(m_grid, amrex::IndexType(amrex::IntVect{AMREX_D_DECL(0, 0, 1)})), m_distriMap, 1, m_nGhost);

    m_tempDualTwoForm[xDir].define(amrex::convert(m_grid, amrex::IndexType(amrex::IntVect{AMREX_D_DECL(0, 1, 1)})), m_distriMap, 1, m_nGhost);
    m_tempDualTwoForm[yDir].define(amrex::convert(m_grid, amrex::IndexType(amrex::IntVect{AMREX_D_DECL(1, 0, 1)})), m_distriMap, 1, m_nGhost);
    m_tempDualTwoForm[zDir].define(amrex::convert(m_grid, amrex::IndexType(amrex::IntVect{AMREX_D_DECL(1, 1, 0)})), m_distriMap, 1, m_nGhost);


    m_tempPrimalThreeForm.define(amrex::convert(m_grid, amrex::IndexType(amrex::IntVect{AMREX_D_DECL(0, 0, 0)})), m_distriMap, 1, m_nGhost);
    m_tempDualThreeForm.define(amrex::convert(m_grid, amrex::IndexType(amrex::IntVect{AMREX_D_DECL(1, 1, 1)})), m_distriMap, 1, m_nGhost);


    m_quadPoints[0] = amrex::GpuArray<amrex::Real, m_maxGaussNodes>({0.});
    m_quadPoints[1] = amrex::GpuArray<amrex::Real, m_maxGaussNodes>({-0.5773502691896257, 0.5773502691896257});
    m_quadPoints[2] = amrex::GpuArray<amrex::Real, m_maxGaussNodes>({-0.7745966692414834, 0., 0.7745966692414834});
    m_quadPoints[3] = amrex::GpuArray<amrex::Real, m_maxGaussNodes>({-0.8611363115940526, -0.3399810435848563, 0.3399810435848563,
                                                                    0.8611363115940526});
    m_quadPoints[4] = amrex::GpuArray<amrex::Real, m_maxGaussNodes>({-0.9061798459386640, -0.5384693101056831, 0., 0.5384693101056831,
                                                                    0.9061798459386640});
    m_quadPoints[5] = amrex::GpuArray<amrex::Real, m_maxGaussNodes>({-0.9324695142031521, -0.6612093864662645, -0.2386191860831969,
                                                                    0.2386191860831969, 0.6612093864662645, 0.9324695142031521});
    m_quadPoints[6] = amrex::GpuArray<amrex::Real, m_maxGaussNodes>({-0.9491079123427585, -0.7415311855993945, -0.4058451513773972,
                                                                    0., 0.4058451513773972, 0.7415311855993945, 0.9491079123427585});
    m_quadPoints[7] = amrex::GpuArray<amrex::Real, m_maxGaussNodes>({-0.9602898564975363, -0.7966664774136267, -0.5255324099163290,
                                                                    -0.1834346424956498, 0.1834346424956498, 0.5255324099163290,
                                                                    0.7966664774136267, 0.9602898564975363});
    m_quadPoints[8] = amrex::GpuArray<amrex::Real, m_maxGaussNodes>({-0.9681602395076261, -0.8360311073266358, -0.6133714327005904,
                                                                    -0.3242534234038089, 0., 0.3242534234038089, 0.6133714327005904,
                                                                    0.8360311073266358, 0.9681602395076261});
    m_quadPoints[9] = amrex::GpuArray<amrex::Real, m_maxGaussNodes>({-0.9739065285171717, -0.8650633666889845, -0.6794095682990244,
                                                                    -0.4333953941292472, -0.1488743389816312, 0.1488743389816312,
                                                                    0.4333953941292472, 0.6794095682990244, 0.8650633666889845,
                                                                    0.9739065285171717});

    m_quadWeights[0] = amrex::GpuArray<amrex::Real, m_maxGaussNodes>({2.});
    m_quadWeights[1] = amrex::GpuArray<amrex::Real, m_maxGaussNodes>({1., 1.});
    m_quadWeights[2] = amrex::GpuArray<amrex::Real, m_maxGaussNodes>({0.5555555555555556, 0.8888888888888888, 0.5555555555555556});
    m_quadWeights[3] = amrex::GpuArray<amrex::Real, m_maxGaussNodes>({0.3478548451374538, 0.6521451548625461, 0.6521451548625461,
                                                                    0.3478548451374538});
    m_quadWeights[4] = amrex::GpuArray<amrex::Real, m_maxGaussNodes>({0.2369268850561891, 0.4786286704993665, 0.5688888888888889,
                                                                    0.4786286704993665, 0.2369268850561891});
    m_quadWeights[5] = amrex::GpuArray<amrex::Real, m_maxGaussNodes>({0.1713244923791704, 0.3607615730481386, 0.4679139345726910,
                                                                    0.4679139345726910, 0.3607615730481386, 0.1713244923791704});
    m_quadWeights[6] = amrex::GpuArray<amrex::Real, m_maxGaussNodes>({0.1294849661688697, 0.2797053914892766, 0.3818300505051189,
                                                                    0.4179591836734694, 0.3818300505051189, 0.2797053914892766,
                                                                    0.1294849661688697});
    m_quadWeights[7] = amrex::GpuArray<amrex::Real, m_maxGaussNodes>({0.1012285362903763, 0.2223810344533745, 0.3137066458778873,
                                                                    0.3626837833783620, 0.3626837833783620, 0.3137066458778873,
                                                                    0.2223810344533745, 0.1012285362903763});
    m_quadWeights[8] = amrex::GpuArray<amrex::Real, m_maxGaussNodes>({0.0812743883615744, 0.1806481606948574, 0.2606106964029354,
                                                                    0.3123470770400029, 0.3302393550012598, 0.3123470770400029,
                                                                    0.2606106964029354, 0.1806481606948574, 0.0812743883615744});
    m_quadWeights[9] = amrex::GpuArray<amrex::Real, m_maxGaussNodes>({0.0666713443086881, 0.1494513491505806, 0.2190863625159820,
                                                                    0.2692667193099963, 0.2955242247147529, 0.2955242247147529,
                                                                    0.2692667193099963, 0.2190863625159820, 0.1494513491505806,
                                                                    0.0666713443086881});
}

FDDeRhamComplex::~FDDeRhamComplex() {}

/**
* @brief Computes the Restriction operator \f$R_0\f$
* 
* Description:
* Using the geometric degrees of freedeom:
* Evaluates func at time t on the nodes of the primal grid
* 
* For dimensions 1 and 2, the 0-form is taken from the "stacked" de Rham complex
* 
* @param func : ParserExecutor, function to be projected
* @param t : Real, time at which func is to be evaluated
* @param field : DeRhamField<primal, Space::node>, 0-form \f$u^0\f$ holding the node values
* 
* @return void
*/
void FDDeRhamComplex::projection (amrex::ParserExecutor<GEMPIC_SPACEDIM + 1> func, amrex::Real t,
                                      DeRhamField<Grid::primal, Space::node>& field)
{
    for (amrex::MFIter mfi(field.data, true); mfi.isValid(); ++mfi)
    {
        const amrex::Box &bx = mfi.tilebox();
        amrex::Array4<amrex::Real> const &zeroForm = (field.data)[mfi].array();

        const amrex::RealVect dr = m_dr;
        const amrex::GpuArray<amrex::Real, GEMPIC_SPACEDIM> r0 = m_geom.ProbLoArray();

        ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k)
        {

            // Compute the position of the point i, j, k
            amrex::GpuArray<amrex::Real, GEMPIC_SPACEDIM> r =
            {
                AMREX_D_DECL(r0[xDir] + i * dr[xDir], r0[yDir] + j * dr[yDir], r0[zDir] + k * dr[zDir])
            };

            // Assign point values to zeroForm 
            zeroForm(i, j, k) = func(AMREX_D_DECL(r[xDir], r[yDir], r[zDir]), t);
        });

    }

    field.fillBoundary();
    field.averageSync();

}

/**
* @brief Computes the Restriction operator \f$R_3\f$
* 
* Description:
* Using the geometric degrees of freedeom:
* Integrates func at time t over the cells of the primal grid by using a Gauss quadrature rule
* 
* For dimensions 1 and 2, the 3-form is taken from the "stacked" de Rham complex
* 
* @param func : ParserExecutor, function to be projected
* @param t : Real, time at which func is to be evaluated
* @param field : DeRhamField<primal, Space::cell>, 3-form \f$u^3\f$ holding the cell integrals
* @param gaussNodes: int, number of Gauss nodes to be used for quadratue
*
* @return void
*/
void FDDeRhamComplex::projection (amrex::ParserExecutor<GEMPIC_SPACEDIM + 1> func, amrex::Real t,
                                      DeRhamField<Grid::primal, Space::cell>& field, int gaussNodes)
{
    const int nQuad = (gaussNodes <= m_maxGaussNodes) ? (gaussNodes > 0 ? gaussNodes : 1) : m_maxGaussNodes;
    if (nQuad != gaussNodes)
        amrex::Print() << "Gauss formula with " << gaussNodes << " nodes not available, using " << nQuad << " nodes instead!" << std::endl;
    const amrex::GpuArray<amrex::Real, m_maxGaussNodes> quadPoints = m_quadPoints[nQuad-1];
    const amrex::GpuArray<amrex::Real, m_maxGaussNodes> quadWeights = m_quadWeights[nQuad-1];

    // Volume integral 
    for (amrex::MFIter mfi(field.data, true); mfi.isValid(); ++mfi)
    {
        const amrex::Box &bx = mfi.tilebox();
        amrex::Array4<amrex::Real> const &threeForm = (field.data)[mfi].array();

        const amrex::RealVect dr = m_dr;
        const amrex::GpuArray<amrex::Real, GEMPIC_SPACEDIM> r0 = m_geom.ProbLoArray();

        const amrex::GpuArray<amrex::Real, GEMPIC_SPACEDIM> drHalf =
        {
            AMREX_D_DECL(dr[xDir] / 2, dr[yDir] / 2, dr[zDir] / 2)
        };

        ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k)
        {

            // Compute the position of the point i, j, k
            amrex::GpuArray<amrex::Real, GEMPIC_SPACEDIM> r =
            {
                AMREX_D_DECL(r0[xDir] + i * dr[xDir], r0[yDir] + j * dr[yDir], r0[zDir] + k * dr[zDir])
            };

            // Midpoint for the quadrature rule
            amrex::GpuArray<amrex::Real, GEMPIC_SPACEDIM> midpoint =
            {
                AMREX_D_DECL(r[xDir] + drHalf[xDir], r[yDir] + drHalf[yDir], r[zDir] + drHalf[zDir])
            };

            amrex::Real integral = 0.0;

            // Integral over the xy plane and z direction
            GEMPIC_D_LOOP_BEGIN(
            for (int qx = 0; qx < nQuad; ++qx),
                for (int qy = 0; qy < nQuad; ++qy),
                    for (int qz = 0; qz < nQuad; ++qz))
                         // Update location of quadrature points
                        r = {AMREX_D_DECL(midpoint[xDir] + quadPoints[qx] * drHalf[xDir], midpoint[yDir] + quadPoints[qy] * drHalf[yDir], midpoint[zDir] + quadPoints[qz] * drHalf[zDir])};
            
                        // Increment integral according to quadrature rule in the z direction with dx and dy
                        integral += GEMPIC_D_MULT(quadWeights[qx], quadWeights[qy], quadWeights[qz]) * func(AMREX_D_DECL(r[xDir], r[yDir], r[zDir]), t);
            GEMPIC_D_LOOP_END
        
            // Rescale the integral and assign it to degrees of freedom
            threeForm(i, j, k) = integral * GEMPIC_D_MULT(drHalf[xDir], drHalf[yDir], drHalf[zDir]);
        });

    }

    field.fillBoundary();
}

/**
* @brief Computes the Restriction operator \f$\tilde{R}_0\f$
* 
* Description:
* Using the geometric degrees of freedeom:
* Evaluates func at time t on the nodes of the dual grid
* 
* For dimensions 1 and 2, the 0-form is taken from the "stacked" de Rham complex
* 

* @param func : ParserExecutor, function to be projected
* @param t : Real, time at which func is to be evaluated
* @param field : DeRhamField<dual, Space::node>, 0-form \f$\tilde{u}^0\f$ holding the node values
* 
* @return void
*/
void FDDeRhamComplex::projection (amrex::ParserExecutor<GEMPIC_SPACEDIM + 1> func, amrex::Real t,
                                      DeRhamField<Grid::dual, Space::node>& field)
{
    for (amrex::MFIter mfi(field.data, true); mfi.isValid(); ++mfi)
    {
        const amrex::Box &bx = mfi.tilebox();
        amrex::Array4<amrex::Real> const &zeroForm = (field.data)[mfi].array();

        const amrex::RealVect dr = m_dr;
        const amrex::GpuArray<amrex::Real, GEMPIC_SPACEDIM> r0 = m_geom.ProbLoArray();

        ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k)
        {

            // Compute the position of the point i + 1/2, j + 1/2, k + 1/2
            amrex::GpuArray<amrex::Real, GEMPIC_SPACEDIM> r =
            {
                AMREX_D_DECL(r0[xDir] + 0.5*dr[xDir] + i * dr[xDir], r0[yDir] + 0.5*dr[yDir] + j * dr[yDir], r0[zDir] + 0.5*dr[zDir] + k * dr[zDir])
            };


            // Assign point values to zeroForm 
            zeroForm(i, j, k) = func(AMREX_D_DECL(r[xDir], r[yDir], r[zDir]), t);
        });

    }

    field.fillBoundary();
    field.averageSync();

}

/**
* @brief Computes the Restriction operator \f$\tilde{R}_3\f$
* 
* Description:
* Using the geometric degrees of freedeom:
* Integrates func at time t over the cells of the dual grid by using a Gauss quadrature rule
* 
* For dimensions 1 and 2, the 3-form is taken from the "stacked" de Rham complex
* 
* @param func : ParserExecutor, function to be projected
* @param t : Real, time at which func is to be evaluated
* @param field : DeRhamField<dual, Space::cell>, 3-form \f$\tilde{u}^3\f$ holding the cell integrals
* @param gaussNodes: int, number of Gauss nodes to be used for quadratue
* 
* @return void
*/
void FDDeRhamComplex::projection (amrex::ParserExecutor<GEMPIC_SPACEDIM + 1> func, amrex::Real t,
                                      DeRhamField<Grid::dual, Space::cell>& field, int gaussNodes)
{
    const int nQuad = (gaussNodes <= m_maxGaussNodes) ? (gaussNodes > 0 ? gaussNodes : 1) : m_maxGaussNodes;
    if (nQuad != gaussNodes)
        amrex::Print() << "Gauss formula with " << gaussNodes << " nodes not available, using " << nQuad << " nodes instead!" << std::endl;
    const amrex::GpuArray<amrex::Real, m_maxGaussNodes> quadPoints = m_quadPoints[nQuad-1];
    const amrex::GpuArray<amrex::Real, m_maxGaussNodes> quadWeights = m_quadWeights[nQuad-1];

    // Volume integral 
    for (amrex::MFIter mfi(field.data, true); mfi.isValid(); ++mfi)
    {
        const amrex::Box &bx = mfi.tilebox();
        amrex::Array4<amrex::Real> const &threeForm = (field.data)[mfi].array();

        const amrex::RealVect dr = m_dr;
        const amrex::GpuArray<amrex::Real, GEMPIC_SPACEDIM> r0 = m_geom.ProbLoArray();


        const amrex::GpuArray<amrex::Real, GEMPIC_SPACEDIM> drHalf =
        {
            AMREX_D_DECL(dr[xDir] / 2, dr[yDir] / 2, dr[zDir] / 2)
        };

        ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k)
        {

            // Compute the position of the point i, j, k
            amrex::GpuArray<amrex::Real, GEMPIC_SPACEDIM> r =
            {
                AMREX_D_DECL(r0[xDir] + 0.5*dr[xDir] + i * dr[xDir], r0[yDir] + 0.5*dr[yDir] + j * dr[yDir], r0[zDir] + 0.5*dr[zDir] + k * dr[zDir])
            };

            // Midpoint for the quadrature rule
            amrex::GpuArray<amrex::Real, GEMPIC_SPACEDIM> midpoint =
            {
                AMREX_D_DECL(r[xDir] - drHalf[xDir], r[yDir] - drHalf[yDir], r[zDir] - drHalf[zDir])
            };

            amrex::Real integral = 0.0;

            // Integral over the xy plane and z direction
            GEMPIC_D_LOOP_BEGIN(
            for (int qx = 0; qx < nQuad; ++qx),
                for (int qy = 0; qy < nQuad; ++qy),
                    for (int qz = 0; qz < nQuad; ++qz))
                         // Update location of quadrature points
                        r = {AMREX_D_DECL(midpoint[xDir] + quadPoints[qx] * drHalf[xDir], midpoint[yDir] + quadPoints[qy] * drHalf[yDir], midpoint[zDir] + quadPoints[qz] * drHalf[zDir])};
            
                        // Increment integral according to quadrature rule in the z direction with dx and dy
                        integral += GEMPIC_D_MULT(quadWeights[qx], quadWeights[qy], quadWeights[qz]) * func(AMREX_D_DECL(r[xDir], r[yDir], r[zDir]), t);
            GEMPIC_D_LOOP_END
        
            // Rescale the integral and assign it to degrees of freedom
            threeForm(i, j, k) = integral * GEMPIC_D_MULT(drHalf[xDir], drHalf[yDir], drHalf[zDir]);
        });
    }

    field.fillBoundary();
    field.averageSync();

}

/**
* @brief Computes the Restriction operator \f$\tilde{R}_1\f$
* 
* Description:
* Using the geometric degrees of freedeom:
* Integrates func at time t over the edges of the dual grid by using a Gauss quadrature rule
* 
* For dimensions 1 and 2, the 1-form is taken from the "stacked" de Rham complex
* 
* @param func : ParserExecutor, function to be projected
* @param t : Real, time at which func is to be evaluated
* @param field : DeRhamField<dual, Space::edge>, 1-form \f$\tilde{u}^1\f$ holding the edge integrals
* @param gaussNodes: int, number of Gauss nodes to be used for quadratue
* 
* @return void
*/
void FDDeRhamComplex::projection (amrex::Array<amrex::ParserExecutor<GEMPIC_SPACEDIM + 1>, 3> func, amrex::Real t,
                                      DeRhamField<Grid::dual, Space::edge>& field, int gaussNodes)
{
    const int nQuad = (gaussNodes <= m_maxGaussNodes) ? (gaussNodes > 0 ? gaussNodes : 1) : m_maxGaussNodes;
    if (nQuad != gaussNodes)
        amrex::Print() << "Gauss formula with " << gaussNodes << " nodes not available, using " << nQuad << " nodes instead!" << std::endl;
    const amrex::GpuArray<amrex::Real, m_maxGaussNodes> quadPoints = m_quadPoints[nQuad-1];
    const amrex::GpuArray<amrex::Real, m_maxGaussNodes> quadWeights = m_quadWeights[nQuad-1];

    // Do the loop over comp-direction
    for (int comp = 0; comp < 3; ++comp)
    {
        for (amrex::MFIter mfi(field.data[comp], true); mfi.isValid(); ++mfi)
        {
            const amrex::Box &bx = mfi.tilebox();
            amrex::Array4<amrex::Real> const &oneForm = (field.data[comp])[mfi].array();

            const amrex::RealVect dr = m_dr;
            const amrex::GpuArray<amrex::Real, GEMPIC_SPACEDIM> r0 = m_geom.ProbLoArray();

            ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k)
            {

                // Compute the position of the point i + 1/2, j + 1/2, k + 1/2
                amrex::GpuArray<amrex::Real, GEMPIC_SPACEDIM> r =
                {
                    AMREX_D_DECL(r0[xDir] + 0.5*dr[xDir] + i * dr[xDir], r0[yDir] + 0.5*dr[yDir] + j * dr[yDir], r0[zDir] + 0.5*dr[zDir] + k * dr[zDir])
                };
                
#if (GEMPIC_SPACEDIM < 3)
                if (comp < GEMPIC_SPACEDIM) // 1-forms in 1D, 2D are node centered in the last 1, 2 components respectively
#endif
                {
                    const amrex::Real drHalf = dr[comp] / 2;
                    // Midpoint for the quadrature rule
                    amrex::Real midpoint = r[comp] - drHalf;
                    amrex::Real integral = 0.0;

                    // Integral over the edge along the comp direction
                    for (int q = 0; q < nQuad; ++q)
                    {
                        // Update location of quadrature point
                        r[comp] = midpoint + quadPoints[q] * drHalf;
                    
                        // Increment integral according to quadrature rule
                        integral += quadWeights[q] * func[comp](AMREX_D_DECL(r[xDir], r[yDir], r[zDir]), t);
                    } 
                
                    // Rescale the integral and assign it to array of degrees of freedom
                    oneForm(i, j, k) = integral * drHalf;
                }
#if (GEMPIC_SPACEDIM < 3)
                else
                {
                    oneForm(i, j, k) = func[comp](AMREX_D_DECL(r[xDir], r[yDir], r[zDir]), t);
                }
#endif          
            });

        }

    }

    field.fillBoundary();
    field.averageSync();
}

/**
* @brief Computes the Restriction operator \f$R_2\f$
* 
* Description:
* Using the geometric degrees of freedeom:
* Integrates func at time t over the faces of the primal grid by using a Gauss quadrature rule
* 
* For dimensions 1 and 2, the 2-form is taken from the "stacked" de Rham complex
* 
* @param func : ParserExecutor, function to be projected
* @param t : Real, time at which func is to be evaluated
* @param field : DeRhamField<primal, Space::face>, 2-form \f$u^2\f$ holding the face integrals
* @param gaussNodes: int, number of Gauss nodes to be used for quadratue
* 
* @return void
*/
void FDDeRhamComplex::projection (amrex::Array<amrex::ParserExecutor<GEMPIC_SPACEDIM + 1>, 3> func, amrex::Real t,
                                      DeRhamField<Grid::primal, Space::face>& field, int gaussNodes)
{

    const int nQuad = (gaussNodes <= m_maxGaussNodes) ? (gaussNodes > 0 ? gaussNodes : 1) : m_maxGaussNodes;
    if (nQuad != gaussNodes)
        amrex::Print() << "Gauss formula with " << gaussNodes << " nodes not available, using " << nQuad << " nodes instead!" << std::endl;
    const amrex::GpuArray<amrex::Real, m_maxGaussNodes> quadPoints = m_quadPoints[nQuad-1];
    const amrex::GpuArray<amrex::Real, m_maxGaussNodes> quadWeights = m_quadWeights[nQuad-1];

    // x-direction. Plane YZ
    for (amrex::MFIter mfi(field.data[xDir], true); mfi.isValid(); ++mfi)
    {
        const amrex::Box &bx = mfi.tilebox();
        amrex::Array4<amrex::Real> const &twoForm = (field.data[xDir])[mfi].array();

        const amrex::RealVect dr = m_dr;
        const amrex::GpuArray<amrex::Real, GEMPIC_SPACEDIM> r0 = m_geom.ProbLoArray();

        const amrex::GpuArray<amrex::Real, GEMPIC_SPACEDIM> drHalf =
        {
            AMREX_D_DECL(dr[xDir] / 2, dr[yDir] / 2, dr[zDir] / 2)
        };

        ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k)
        {
            // Compute the position of the point i, j, k
            amrex::GpuArray<amrex::Real, GEMPIC_SPACEDIM> r =
            {
                AMREX_D_DECL(r0[xDir] + i * dr[xDir], r0[yDir] + j * dr[yDir], r0[zDir] + k * dr[zDir])
            };

            // Midpoint for the quadrature rule
            amrex::GpuArray<amrex::Real, GEMPIC_SPACEDIM> midpoint =
            {
                AMREX_D_DECL(r[xDir] + drHalf[xDir], r[yDir] + drHalf[yDir], r[zDir] + drHalf[zDir])
            };

            amrex::Real integral = 0.0;

            // Integral over the xy plane and z direction
            GEMPIC_D_LOOP_BEGIN(,
            for (int qy = 0; qy < nQuad; ++qy),
                for (int qz = 0; qz < nQuad; ++qz))
                    // Update location of quadrature points
                    r = {AMREX_D_DECL(r[xDir], midpoint[yDir] + quadPoints[qy] * drHalf[yDir], midpoint[zDir] + quadPoints[qz] * drHalf[zDir])};

                    integral += GEMPIC_D_MULT(1., quadWeights[qy], quadWeights[qz]) * func[xDir](AMREX_D_DECL(r[xDir], r[yDir], r[zDir]), t); // in 1D this is just evalutation
            GEMPIC_D_LOOP_END
        
            // Rescale the integral and assign it to array of degrees of freedom
            twoForm(i, j, k) = GEMPIC_D_MULT(integral, drHalf[yDir], drHalf[zDir]);
        });

    }

    // y-direction. Plane XZ
    for (amrex::MFIter mfi(field.data[yDir], true); mfi.isValid(); ++mfi)
    {
        const amrex::Box &bx = mfi.tilebox();

        amrex::Array4<amrex::Real> const &twoForm = (field.data[yDir])[mfi].array();

        const amrex::RealVect dr = m_dr;
        const amrex::GpuArray<amrex::Real, GEMPIC_SPACEDIM> r0 = m_geom.ProbLoArray();


        const amrex::GpuArray<amrex::Real, GEMPIC_SPACEDIM> drHalf =
        {
            AMREX_D_DECL(dr[xDir] / 2, dr[yDir] / 2, dr[zDir] / 2)
        };
        ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k)
        {

            // Compute the position of the point i, j, k
            amrex::GpuArray<amrex::Real, GEMPIC_SPACEDIM> r =
            {
                AMREX_D_DECL(r0[xDir] + i * dr[xDir], r0[yDir] + j * dr[yDir], r0[zDir] + k * dr[zDir])
            };

            // Midpoint for the quadrature rule
            amrex::GpuArray<amrex::Real, GEMPIC_SPACEDIM> midpoint =
            {
                AMREX_D_DECL(r[xDir] + drHalf[xDir], r[yDir] + drHalf[yDir], r[zDir] + drHalf[zDir])
            };

            amrex::Real integral = 0.0;

            // Integral over the xy plane and z direction
            GEMPIC_D_LOOP_BEGIN(
            for (int qx = 0; qx < nQuad; ++qx),,
                for (int qz = 0; qz < nQuad; ++qz))
                    // Update location of quadrature points
                    r = {AMREX_D_DECL(midpoint[xDir] + quadPoints[qx] * drHalf[xDir], r[yDir], midpoint[zDir] + quadPoints[qz] * drHalf[zDir])};

                    integral += GEMPIC_D_MULT(quadWeights[qx], 1., quadWeights[qz]) * func[yDir](AMREX_D_DECL(r[xDir], r[yDir], r[zDir]), t);
            GEMPIC_D_LOOP_END
            
            // Rescale the integral and assign it to array of degrees of freedom
            twoForm(i, j, k) = GEMPIC_D_MULT(integral * drHalf[xDir], 1, drHalf[zDir]);
        });

    }

    // z-direction. Plane XY
    for (amrex::MFIter mfi(field.data[zDir], true); mfi.isValid(); ++mfi)
    {
        const amrex::Box &bx = mfi.tilebox();
        amrex::Array4<amrex::Real> const &twoForm = (field.data[zDir])[mfi].array();

        const amrex::RealVect dr = m_dr;
        const amrex::GpuArray<amrex::Real, GEMPIC_SPACEDIM> r0 = m_geom.ProbLoArray();

        const amrex::GpuArray<amrex::Real, GEMPIC_SPACEDIM> drHalf =
        {
            AMREX_D_DECL(dr[xDir] / 2, dr[yDir] / 2, dr[zDir] / 2)
        };
        ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k)
        {

            // Compute the position of the point i, j, k
            amrex::GpuArray<amrex::Real, GEMPIC_SPACEDIM> r =
            {
                AMREX_D_DECL(r0[xDir] + i * dr[xDir], r0[yDir] + j * dr[yDir], r0[zDir] + k * dr[zDir])
            };

            // Midpoint for the quadrature rule
            amrex::GpuArray<amrex::Real, GEMPIC_SPACEDIM> midpoint =
            {
                AMREX_D_DECL(r[xDir] + drHalf[xDir], r[yDir] + drHalf[yDir], r[zDir] + drHalf[zDir])
            };

            amrex::Real integral = 0.0;

            // Integral over the xy plane and z direction
            GEMPIC_D_LOOP_BEGIN(
            for (int qx = 0; qx < nQuad; ++qx),
                for (int qy = 0; qy < nQuad; ++qy),)
                    // Update location of quadrature points
                    r = {AMREX_D_DECL(midpoint[xDir] + quadPoints[qx] * drHalf[xDir], midpoint[yDir] + quadPoints[qy] * drHalf[yDir], r[zDir])};

                    integral += GEMPIC_D_MULT(quadWeights[qx], quadWeights[qy], 1.) * func[zDir](AMREX_D_DECL(r[xDir], r[yDir], r[zDir]), t);
            GEMPIC_D_LOOP_END
            
            // Rescale the integral and assign it to array of degrees of freedom
            twoForm(i, j, k) = GEMPIC_D_MULT(integral * drHalf[xDir], drHalf[yDir], 1);
        });

    }

    field.fillBoundary();
    field.averageSync();

}

/**
* @brief Computes the Restriction operator \f$R_1\f$
* 
* Description:
* Using the geometric degrees of freedeom:
* Integrates func at time t over the edges of the primal grid by using a Gauss quadrature rule
* 
* For dimensions 1 and 2, the 1-form is taken from the "stacked" de Rham complex
* 
* @param func : ParserExecutor, function to be projected
* @param t : Real, time at which func is to be evaluated
* @param field : DeRhamField<primal, Space::edge>, 1-form \f$u^1\f$ holding the edge integrals
* @param gaussNodes: int, number of Gauss nodes to be used for quadratue
* 
* @return void
*/
void FDDeRhamComplex::projection (amrex::Array<amrex::ParserExecutor<GEMPIC_SPACEDIM + 1>, 3> func, amrex::Real t,
                                      DeRhamField<Grid::primal, Space::edge>& field, int gaussNodes)

{
    

    const int nQuad = (gaussNodes <= m_maxGaussNodes) ? (gaussNodes > 0 ? gaussNodes : 1) : m_maxGaussNodes;
    if (nQuad != gaussNodes)
        amrex::Print() << "Gauss formula with " << gaussNodes << " nodes not available, using " << nQuad << " nodes instead!" << std::endl;
    const amrex::GpuArray<amrex::Real, m_maxGaussNodes> quadPoints = m_quadPoints[nQuad-1];
    const amrex::GpuArray<amrex::Real, m_maxGaussNodes> quadWeights = m_quadWeights[nQuad-1];

    // Do the loop over comp-direction
    for (int comp = 0; comp < 3; ++comp)
    {
        for (amrex::MFIter mfi(field.data[comp], true); mfi.isValid(); ++mfi)
        {
            const amrex::Box &bx = mfi.tilebox();
            amrex::Array4<amrex::Real> const &oneForm = (field.data[comp])[mfi].array();

            const amrex::RealVect dr = m_dr;
            const amrex::GpuArray<amrex::Real, GEMPIC_SPACEDIM> r0 = m_geom.ProbLoArray();

            ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k)
            {

                // Compute the position of the point i, j, k
                amrex::GpuArray<amrex::Real, GEMPIC_SPACEDIM> r =
                {
                    AMREX_D_DECL(r0[xDir] + i * dr[xDir], r0[yDir] + j * dr[yDir], r0[zDir] + k * dr[zDir])
                };

#if (GEMPIC_SPACEDIM < 3)
                if (comp < GEMPIC_SPACEDIM) // 1-forms in 1D, 2D are node centered in the last 1, 2 components respectively
#endif
                {
                    const amrex::Real drHalf = dr[comp] / 2;
                    // Midpoint for the quadrature rule
                    amrex::Real midpoint = r[comp] + drHalf;
                    amrex::Real integral = 0.0;

                    // Integral over the edge along the comp direction
                    for (int q = 0; q < nQuad; ++q)
                    {
                        // Update location of quadrature point
                        r[comp] = midpoint + quadPoints[q] * drHalf;
                    
                        // Increment integral according to quadrature rule
                        integral += quadWeights[q] * func[comp](AMREX_D_DECL(r[xDir], r[yDir], r[zDir]), t);
                    } 
                
                    // Rescale the integral and assign it to array of degrees of freedom
                    oneForm(i, j, k) = integral* drHalf;
                }
#if (GEMPIC_SPACEDIM < 3)
                else
                {
                    oneForm(i, j, k) = func[comp](AMREX_D_DECL(r[xDir], r[yDir], r[zDir]), t);
                }
#endif          
                
            });

        }

    }

    field.fillBoundary();
    field.averageSync();
}

/**
* @brief Computes the Restriction operator \f$\tilde{R}_2\f$
* 
* Description:
* Using the geometric degrees of freedeom:
* Integrates func at time t over the faces of the dual grid by using a Gauss quadrature rule
* 
* For dimensions 1 and 2, the 2-form is taken from the "stacked" de Rham complex
* 
* @param func : ParserExecutor, function to be projected
* @param t : Real, time at which func is to be evaluated
* @param field : DeRhamField<dual, Space::face>, 2-form \f$\tilde{u}^2\f$ holding the face integrals
* @param gaussNodes: int, number of Gauss nodes to be used for quadratue
* 
* @return void
*/
void FDDeRhamComplex::projection (amrex::Array<amrex::ParserExecutor<GEMPIC_SPACEDIM + 1>, 3> func, amrex::Real t,
                                      DeRhamField<Grid::dual, Space::face>& field, int gaussNodes)
{
    const int nQuad = (gaussNodes <= m_maxGaussNodes) ? (gaussNodes > 0 ? gaussNodes : 1) : m_maxGaussNodes;
    if (nQuad != gaussNodes)
        amrex::Print() << "Gauss formula with " << gaussNodes << " nodes not available, using " << nQuad << " nodes instead!" << std::endl;
    const amrex::GpuArray<amrex::Real, m_maxGaussNodes> quadPoints = m_quadPoints[nQuad-1];
    const amrex::GpuArray<amrex::Real, m_maxGaussNodes> quadWeights = m_quadWeights[nQuad-1];

    // x-direction. Plane YZ
    for (amrex::MFIter mfi(field.data[xDir], true); mfi.isValid(); ++mfi)
    {
        const amrex::Box &bx = mfi.tilebox();
        amrex::Array4<amrex::Real> const &twoForm = (field.data[xDir])[mfi].array();

        const amrex::RealVect dr = m_dr;
        const amrex::GpuArray<amrex::Real, GEMPIC_SPACEDIM> r0 = m_geom.ProbLoArray();


        const amrex::GpuArray<amrex::Real, GEMPIC_SPACEDIM> drHalf =
        {
            AMREX_D_DECL(dr[xDir] / 2, dr[yDir] / 2, dr[zDir] / 2)
        };

        ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k)
        {
            // Compute the position of the point i, j, k
            amrex::GpuArray<amrex::Real, GEMPIC_SPACEDIM> r =
            {
                AMREX_D_DECL(r0[xDir] + 0.5*dr[xDir] + i * dr[xDir], r0[yDir] + 0.5*dr[yDir] + j * dr[yDir], r0[zDir] + 0.5*dr[zDir] + k * dr[zDir])
            };

            // Midpoint for the quadrature rule. The minus for dual means that it integrates from j-1/2 to j+1/2
            amrex::GpuArray<amrex::Real, GEMPIC_SPACEDIM> midpoint =
            {
                AMREX_D_DECL(r[xDir] - drHalf[xDir], r[yDir] - drHalf[yDir], r[zDir] - drHalf[zDir])
            };

            amrex::Real integral = 0.0;

            // Integral over the xy plane and z direction
            GEMPIC_D_LOOP_BEGIN(,
            for (int qy = 0; qy < nQuad; ++qy),
                for (int qz = 0; qz < nQuad; ++qz))
                    // Update location of quadrature points
                    r = {AMREX_D_DECL(r[xDir], midpoint[yDir] + quadPoints[qy] * drHalf[yDir], midpoint[zDir] + quadPoints[qz] * drHalf[zDir])};

                    integral += GEMPIC_D_MULT(1., quadWeights[qy], quadWeights[qz]) * func[xDir](AMREX_D_DECL(r[xDir], r[yDir], r[zDir]), t); // in 1D this is just evalutation
            GEMPIC_D_LOOP_END
        
            // Rescale the integral and assign it to array of degrees of freedom
            twoForm(i, j, k) = GEMPIC_D_MULT(integral, drHalf[yDir], drHalf[zDir]);
        });

    }

    // y-direction. Plane XZ
    for (amrex::MFIter mfi(field.data[yDir], true); mfi.isValid(); ++mfi)
    {
        const amrex::Box &bx = mfi.tilebox();

        amrex::Array4<amrex::Real> const &twoForm = (field.data[yDir])[mfi].array();

        const amrex::RealVect dr = m_dr;
        const amrex::GpuArray<amrex::Real, GEMPIC_SPACEDIM> r0 = m_geom.ProbLoArray();


        const amrex::GpuArray<amrex::Real, GEMPIC_SPACEDIM> drHalf =
        {
            AMREX_D_DECL(dr[xDir] / 2, dr[yDir] / 2, dr[zDir] / 2)
        };

        ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k)
        {

            // Compute the position of the point i, j, k
            amrex::GpuArray<amrex::Real, GEMPIC_SPACEDIM> r =
            {
                AMREX_D_DECL(r0[xDir] + 0.5*dr[xDir] + i * dr[xDir], r0[yDir] + 0.5*dr[yDir] + j * dr[yDir], r0[zDir] + 0.5*dr[zDir] + k * dr[zDir])
            };


            // Midpoint for the quadrature rule
            amrex::GpuArray<amrex::Real, GEMPIC_SPACEDIM> midpoint =
            {
                AMREX_D_DECL(r[xDir] - drHalf[xDir], r[yDir] - drHalf[yDir], r[zDir] - drHalf[zDir])
            };

            amrex::Real integral = 0.0;

            // Integral over the xy plane and z direction
            GEMPIC_D_LOOP_BEGIN(
            for (int qx = 0; qx < nQuad; ++qx),,
                for (int qz = 0; qz < nQuad; ++qz))
                    // Update location of quadrature points
                    r = {AMREX_D_DECL(midpoint[xDir] + quadPoints[qx] * drHalf[xDir], r[yDir], midpoint[zDir] + quadPoints[qz] * drHalf[zDir])};

                    integral += GEMPIC_D_MULT(quadWeights[qx], 1., quadWeights[qz]) * func[yDir](AMREX_D_DECL(r[xDir], r[yDir], r[zDir]), t);
            GEMPIC_D_LOOP_END

            // Rescale the integral and assign it to array of degrees of freedom
            twoForm(i, j, k) = GEMPIC_D_MULT(integral* drHalf[xDir], 1, drHalf[zDir]);
        });

    }

    // z-direction. Plane XY
    for (amrex::MFIter mfi(field.data[zDir], true); mfi.isValid(); ++mfi)
    {
        const amrex::Box &bx = mfi.tilebox();
        amrex::Array4<amrex::Real> const &twoForm = (field.data[zDir])[mfi].array();

        const amrex::RealVect dr = m_dr;
        const amrex::GpuArray<amrex::Real, GEMPIC_SPACEDIM> r0 = m_geom.ProbLoArray();


        const amrex::GpuArray<amrex::Real, GEMPIC_SPACEDIM> drHalf =
        {
            AMREX_D_DECL(dr[xDir] / 2, dr[yDir] / 2, dr[zDir] / 2)
        };
        
        ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k)
        {

            // Compute the position of the point i, j, k
            amrex::GpuArray<amrex::Real, GEMPIC_SPACEDIM> r =
            {
                AMREX_D_DECL(r0[xDir] + 0.5*dr[xDir] + i * dr[xDir], r0[yDir] + 0.5*dr[yDir] + j * dr[yDir], r0[zDir] + 0.5*dr[zDir] + k * dr[zDir])
            };


            // Midpoint for the quadrature rule
            amrex::GpuArray<amrex::Real, GEMPIC_SPACEDIM> midpoint =
            {
                AMREX_D_DECL(r[xDir] - drHalf[xDir], r[yDir] - drHalf[yDir], r[zDir] - drHalf[zDir])
            };

            amrex::Real integral = 0.0;

            // Integral over the xy plane and z direction
            GEMPIC_D_LOOP_BEGIN(
            for (int qx = 0; qx < nQuad; ++qx),
                for (int qy = 0; qy < nQuad; ++qy),)
                    // Update location of quadrature points
                    r = {AMREX_D_DECL(midpoint[xDir] + quadPoints[qx] * drHalf[xDir], midpoint[yDir] + quadPoints[qy] * drHalf[yDir], r[zDir])};

                    integral += GEMPIC_D_MULT(quadWeights[qx], quadWeights[qy], 1.) * func[zDir](AMREX_D_DECL(r[xDir], r[yDir], r[zDir]), t);
            GEMPIC_D_LOOP_END
        
            // Rescale the integral and assign it to array of degrees of freedom
            twoForm(i, j, k) = GEMPIC_D_MULT(integral * drHalf[xDir], drHalf[yDir], 1);
        });

    }
    field.fillBoundary();
    field.averageSync();
}
