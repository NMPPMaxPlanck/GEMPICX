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

    m_tempPrimalOneForm[0].define(amrex::convert(m_grid, amrex::IndexType(amrex::IntVect{AMREX_D_DECL(0, 1, 1)})), m_distriMap, 1, m_nGhost);
    m_tempPrimalOneForm[1].define(amrex::convert(m_grid, amrex::IndexType(amrex::IntVect{AMREX_D_DECL(1, 0, 1)})), m_distriMap, 1, m_nGhost);
    m_tempPrimalOneForm[2].define(amrex::convert(m_grid, amrex::IndexType(amrex::IntVect{AMREX_D_DECL(1, 1, 0)})), m_distriMap, 1, m_nGhost);

    m_tempDualOneForm[0].define(amrex::convert(m_grid, amrex::IndexType(amrex::IntVect{AMREX_D_DECL(1, 0, 0)})), m_distriMap, 1, m_nGhost);
    m_tempDualOneForm[1].define(amrex::convert(m_grid, amrex::IndexType(amrex::IntVect{AMREX_D_DECL(0, 1, 0)})), m_distriMap, 1, m_nGhost);
    m_tempDualOneForm[2].define(amrex::convert(m_grid, amrex::IndexType(amrex::IntVect{AMREX_D_DECL(0, 0, 1)})), m_distriMap, 1, m_nGhost);

    m_tempPrimalTwoForm[0].define(amrex::convert(m_grid, amrex::IndexType(amrex::IntVect{AMREX_D_DECL(1, 0, 0)})), m_distriMap, 1, m_nGhost);
    m_tempPrimalTwoForm[1].define(amrex::convert(m_grid, amrex::IndexType(amrex::IntVect{AMREX_D_DECL(0, 1, 0)})), m_distriMap, 1, m_nGhost);
    m_tempPrimalTwoForm[2].define(amrex::convert(m_grid, amrex::IndexType(amrex::IntVect{AMREX_D_DECL(0, 0, 1)})), m_distriMap, 1, m_nGhost);

    m_tempDualTwoForm[0].define(amrex::convert(m_grid, amrex::IndexType(amrex::IntVect{AMREX_D_DECL(0, 1, 1)})), m_distriMap, 1, m_nGhost);
    m_tempDualTwoForm[1].define(amrex::convert(m_grid, amrex::IndexType(amrex::IntVect{AMREX_D_DECL(1, 0, 1)})), m_distriMap, 1, m_nGhost);
    m_tempDualTwoForm[2].define(amrex::convert(m_grid, amrex::IndexType(amrex::IntVect{AMREX_D_DECL(1, 1, 0)})), m_distriMap, 1, m_nGhost);


    m_tempPrimalThreeForm.define(amrex::convert(m_grid, amrex::IndexType(amrex::IntVect{AMREX_D_DECL(0, 0, 0)})), m_distriMap, 1, m_nGhost);
    m_tempDualThreeForm.define(amrex::convert(m_grid, amrex::IndexType(amrex::IntVect{AMREX_D_DECL(1, 1, 1)})), m_distriMap, 1, m_nGhost);

}

FDDeRhamComplex::~FDDeRhamComplex() {}


void FDDeRhamComplex::projection (amrex::ParserExecutor<GEMPIC_SPACEDIM + 1> func, amrex::Real t,
                                      DeRhamField<Grid::primal, Space::node>& field)
{
    // Point values on the primal grid.
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
                AMREX_D_DECL(r0[0] + i * dr[0], r0[1] + j * dr[1], r0[2] + k * dr[2])
            };


            // Assign point values to zeroForm 
            zeroForm(i, j, k) = func(AMREX_D_DECL(r[0], r[1], r[2]), t);
        });

    }

    field.fillBoundary();
    field.averageSync();

}

void FDDeRhamComplex::projection (amrex::ParserExecutor<GEMPIC_SPACEDIM + 1> func, amrex::Real t,
                                      DeRhamField<Grid::primal, Space::cell>& field)
{
    const int nQuad = 3;
    const amrex::GpuArray<amrex::Real, nQuad> quadPoints = {-0.7745966692414834, 0.0, 0.7745966692414834};
    const amrex::GpuArray<amrex::Real, nQuad> quadWeights = {5./9., 8./9., 5./9.}; // Can be initialized somewhere else

    // Volume integral 
    for (amrex::MFIter mfi(field.data, true); mfi.isValid(); ++mfi)
    {
        const amrex::Box &bx = mfi.tilebox();
        amrex::Array4<amrex::Real> const &threeForm = (field.data)[mfi].array();

        const amrex::RealVect dr = m_dr;
        const amrex::GpuArray<amrex::Real, GEMPIC_SPACEDIM> r0 = m_geom.ProbLoArray();

        const amrex::GpuArray<amrex::Real, GEMPIC_SPACEDIM> drHalf =
        {
            AMREX_D_DECL(dr[0] / 2, dr[1] / 2, dr[2] / 2)
        };

        ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k)
        {

            // Compute the position of the point i, j, k
            amrex::GpuArray<amrex::Real, GEMPIC_SPACEDIM> r =
            {
                AMREX_D_DECL(r0[0] + i * dr[0], r0[1] + j * dr[1], r0[2] + k * dr[2])
            };

            // Midpoint for the quadrature rule
            amrex::GpuArray<amrex::Real, GEMPIC_SPACEDIM> midpoint =
            {
                AMREX_D_DECL(r[0] + drHalf[0], r[1] + drHalf[1], r[2] + drHalf[2])
            };

            amrex::Real integral = 0.0;

            // Integral over the xy plane and z direction
            GEMPIC_D_LOOP_BEGIN(
            for (int qx = 0; qx < nQuad; ++qx),
                for (int qy = 0; qy < nQuad; ++qy),
                    for (int qz = 0; qz < nQuad; ++qz))
                         // Update location of quadrature points
                        r = {AMREX_D_DECL(midpoint[0] + quadPoints[qx] * drHalf[0], midpoint[1] + quadPoints[qy] * drHalf[1], midpoint[2] + quadPoints[qz] * drHalf[2])};
            
                        // Increment integral according to quadrature rule in the z direction with dx and dy
                        integral += GEMPIC_D_MULT(quadWeights[qx], quadWeights[qy], quadWeights[qz]) * func(AMREX_D_DECL(r[0], r[1], r[2]), t);
            GEMPIC_D_LOOP_END
        
            // Rescale the integral and assign it to degrees of freedom
            threeForm(i, j, k) = integral * GEMPIC_D_MULT(drHalf[0], drHalf[1], drHalf[2]);
        });

    }

    field.fillBoundary();
}

void FDDeRhamComplex::projection (amrex::ParserExecutor<GEMPIC_SPACEDIM + 1> func, amrex::Real t,
                                      DeRhamField<Grid::dual, Space::node>& field)
{
    // Point values on the dual grid.
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
                AMREX_D_DECL(r0[0] + 0.5*dr[0] + i * dr[0], r0[1] + 0.5*dr[1] + j * dr[1], r0[2] + 0.5*dr[2] + k * dr[2])
            };


            // Assign point values to zeroForm 
            zeroForm(i, j, k) = func(AMREX_D_DECL(r[0], r[1], r[2]), t);
        });

    }

    field.fillBoundary();
    field.averageSync();

}


void FDDeRhamComplex::projection (amrex::ParserExecutor<GEMPIC_SPACEDIM + 1> func, amrex::Real t,
                                      DeRhamField<Grid::dual, Space::cell>& field)
{
    const int nQuad = 3;
    const amrex::GpuArray<amrex::Real, nQuad> quadPoints = {-0.7745966692414834, 0.0, 0.7745966692414834};
    const amrex::GpuArray<amrex::Real, nQuad> quadWeights = {5./9., 8./9., 5./9.}; // Can be initialized somewhere else

    // Volume integral 
    for (amrex::MFIter mfi(field.data, true); mfi.isValid(); ++mfi)
    {
        const amrex::Box &bx = mfi.tilebox();
        amrex::Array4<amrex::Real> const &threeForm = (field.data)[mfi].array();

        const amrex::RealVect dr = m_dr;
        const amrex::GpuArray<amrex::Real, GEMPIC_SPACEDIM> r0 = m_geom.ProbLoArray();


        const amrex::GpuArray<amrex::Real, GEMPIC_SPACEDIM> drHalf =
        {
            AMREX_D_DECL(dr[0] / 2, dr[1] / 2, dr[2] / 2)
        };

        ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k)
        {

            // Compute the position of the point i, j, k
            amrex::GpuArray<amrex::Real, GEMPIC_SPACEDIM> r =
            {
                AMREX_D_DECL(r0[0] + 0.5*dr[0] + i * dr[0], r0[1] + 0.5*dr[1] + j * dr[1], r0[2] + 0.5*dr[2] + k * dr[2])
            };

            // Midpoint for the quadrature rule
            amrex::GpuArray<amrex::Real, GEMPIC_SPACEDIM> midpoint =
            {
                AMREX_D_DECL(r[0] - drHalf[0], r[1] - drHalf[1], r[2] - drHalf[2])
            };

            amrex::Real integral = 0.0;

            // Integral over the xy plane and z direction
            GEMPIC_D_LOOP_BEGIN(
            for (int qx = 0; qx < nQuad; ++qx),
                for (int qy = 0; qy < nQuad; ++qy),
                    for (int qz = 0; qz < nQuad; ++qz))
                         // Update location of quadrature points
                        r = {AMREX_D_DECL(midpoint[0] + quadPoints[qx] * drHalf[0], midpoint[1] + quadPoints[qy] * drHalf[1], midpoint[2] + quadPoints[qz] * drHalf[2])};
            
                        // Increment integral according to quadrature rule in the z direction with dx and dy
                        integral += GEMPIC_D_MULT(quadWeights[qx], quadWeights[qy], quadWeights[qz]) * func(AMREX_D_DECL(r[0], r[1], r[2]), t);
            GEMPIC_D_LOOP_END
        
            // Rescale the integral and assign it to degrees of freedom
            threeForm(i, j, k) = integral * GEMPIC_D_MULT(drHalf[0], drHalf[1], drHalf[2]);
        });
    }

    field.fillBoundary();
    field.averageSync();

}


void FDDeRhamComplex::projection (amrex::Array<amrex::ParserExecutor<GEMPIC_SPACEDIM + 1>, 3> func, amrex::Real t,
                                      DeRhamField<Grid::dual, Space::edge>& field) 
{
    // Gauss quadrature
    const int nQuad = 3;
    const amrex::GpuArray<amrex::Real, nQuad> quadPoints = {-0.7745966692414834, 0.0, 0.7745966692414834};
    const amrex::GpuArray<amrex::Real, nQuad> quadWeights = {5./9., 8./9., 5./9.}; // Can be initialized somewhere else

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
                    AMREX_D_DECL(r0[0] + 0.5*dr[0] + i * dr[0], r0[1] + 0.5*dr[1] + j * dr[1], r0[2] + 0.5*dr[2] + k * dr[2])
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
                        integral += quadWeights[q] * func[comp](AMREX_D_DECL(r[0], r[1], r[2]), t);
                    } 
                
                    // Rescale the integral and assign it to array of degrees of freedom
                    oneForm(i, j, k) = integral * drHalf;
                }
#if (GEMPIC_SPACEDIM < 3)
                else
                {
                    oneForm(i, j, k) = func[comp](AMREX_D_DECL(r[0], r[1], r[2]), t);
                }
#endif          
            });

        }

    }

    field.fillBoundary();
    field.averageSync();
}

void FDDeRhamComplex::projection (amrex::Array<amrex::ParserExecutor<GEMPIC_SPACEDIM + 1>, 3> func, amrex::Real t,
                                      DeRhamField<Grid::primal, Space::face>& field)
{
    // Gauss quadrature
    const int nQuad = 3;
    const amrex::GpuArray<amrex::Real, nQuad> quadPoints = {-0.7745966692414834, 0.0, 0.7745966692414834};
    const amrex::GpuArray<amrex::Real, nQuad> quadWeights = {5./9., 8./9., 5./9.}; // Can be initialized somewhere else

    // x-direction. Plane YZ
    for (amrex::MFIter mfi(field.data[0], true); mfi.isValid(); ++mfi)
    {
        const amrex::Box &bx = mfi.tilebox();
        amrex::Array4<amrex::Real> const &twoForm = (field.data[0])[mfi].array();

        const amrex::RealVect dr = m_dr;
        const amrex::GpuArray<amrex::Real, GEMPIC_SPACEDIM> r0 = m_geom.ProbLoArray();

        const amrex::GpuArray<amrex::Real, GEMPIC_SPACEDIM> drHalf =
        {
            AMREX_D_DECL(dr[0] / 2, dr[1] / 2, dr[2] / 2)
        };

        ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k)
        {
            // Compute the position of the point i, j, k
            amrex::GpuArray<amrex::Real, GEMPIC_SPACEDIM> r =
            {
                AMREX_D_DECL(r0[0] + i * dr[0], r0[1] + j * dr[1], r0[2] + k * dr[2])
            };

            // Midpoint for the quadrature rule
            amrex::GpuArray<amrex::Real, GEMPIC_SPACEDIM> midpoint =
            {
                AMREX_D_DECL(r[0] + drHalf[0], r[1] + drHalf[1], r[2] + drHalf[2])
            };

            amrex::Real integral = 0.0;

            // Integral over the xy plane and z direction
            GEMPIC_D_LOOP_BEGIN(,
            for (int qy = 0; qy < nQuad; ++qy),
                for (int qz = 0; qz < nQuad; ++qz))
                    // Update location of quadrature points
                    r = {AMREX_D_DECL(r[0], midpoint[1] + quadPoints[qy] * drHalf[1], midpoint[2] + quadPoints[qz] * drHalf[2])};

                    integral += GEMPIC_D_MULT(1., quadWeights[qy], quadWeights[qz]) * func[0](AMREX_D_DECL(r[0], r[1], r[2]), t); // in 1D this is just evalutation
            GEMPIC_D_LOOP_END
        
            // Rescale the integral and assign it to array of degrees of freedom
            twoForm(i, j, k) = GEMPIC_D_MULT(integral, drHalf[1], drHalf[2]);
        });

    }

    // y-direction. Plane XZ
    for (amrex::MFIter mfi(field.data[1], true); mfi.isValid(); ++mfi)
    {
        const amrex::Box &bx = mfi.tilebox();

        amrex::Array4<amrex::Real> const &twoForm = (field.data[1])[mfi].array();

        const amrex::RealVect dr = m_dr;
        const amrex::GpuArray<amrex::Real, GEMPIC_SPACEDIM> r0 = m_geom.ProbLoArray();


        const amrex::GpuArray<amrex::Real, GEMPIC_SPACEDIM> drHalf =
        {
            AMREX_D_DECL(dr[0] / 2, dr[1] / 2, dr[2] / 2)
        };
        ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k)
        {

            // Compute the position of the point i, j, k
            amrex::GpuArray<amrex::Real, GEMPIC_SPACEDIM> r =
            {
                AMREX_D_DECL(r0[0] + i * dr[0], r0[1] + j * dr[1], r0[2] + k * dr[2])
            };

            // Midpoint for the quadrature rule
            amrex::GpuArray<amrex::Real, GEMPIC_SPACEDIM> midpoint =
            {
                AMREX_D_DECL(r[0] + drHalf[0], r[1] + drHalf[1], r[2] + drHalf[2])
            };

            amrex::Real integral = 0.0;

            // Integral over the xy plane and z direction
            GEMPIC_D_LOOP_BEGIN(
            for (int qx = 0; qx < nQuad; ++qx),,
                for (int qz = 0; qz < nQuad; ++qz))
                    // Update location of quadrature points
                    r = {AMREX_D_DECL(midpoint[0] + quadPoints[qx] * drHalf[0], r[1], midpoint[2] + quadPoints[qz] * drHalf[2])};

                    integral += GEMPIC_D_MULT(quadWeights[qx], 1., quadWeights[qz]) * func[1](AMREX_D_DECL(r[0], r[1], r[2]), t);
            GEMPIC_D_LOOP_END
            
            // Rescale the integral and assign it to array of degrees of freedom
            twoForm(i, j, k) = GEMPIC_D_MULT(integral * drHalf[0], 1, drHalf[2]);
        });

    }

    // z-direction. Plane XY
    for (amrex::MFIter mfi(field.data[2], true); mfi.isValid(); ++mfi)
    {
        const amrex::Box &bx = mfi.tilebox();
        amrex::Array4<amrex::Real> const &twoForm = (field.data[2])[mfi].array();

        const amrex::RealVect dr = m_dr;
        const amrex::GpuArray<amrex::Real, GEMPIC_SPACEDIM> r0 = m_geom.ProbLoArray();


        const amrex::GpuArray<amrex::Real, GEMPIC_SPACEDIM> drHalf =
        {
            AMREX_D_DECL(dr[0] / 2, dr[1] / 2, dr[2] / 2)
        };
        ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k)
        {

            // Compute the position of the point i, j, k
            amrex::GpuArray<amrex::Real, GEMPIC_SPACEDIM> r =
            {
                AMREX_D_DECL(r0[0] + i * dr[0], r0[1] + j * dr[1], r0[2] + k * dr[2])
            };

            // Midpoint for the quadrature rule
            amrex::GpuArray<amrex::Real, GEMPIC_SPACEDIM> midpoint =
            {
                AMREX_D_DECL(r[0] + drHalf[0], r[1] + drHalf[1], r[2] + drHalf[2])
            };

            amrex::Real integral = 0.0;

            // Integral over the xy plane and z direction
            GEMPIC_D_LOOP_BEGIN(
            for (int qx = 0; qx < nQuad; ++qx),
                for (int qy = 0; qy < nQuad; ++qy),)
                    // Update location of quadrature points
                    r = {AMREX_D_DECL(midpoint[0] + quadPoints[qx] * drHalf[0], midpoint[1] + quadPoints[qy] * drHalf[1], r[2])};

                    integral += GEMPIC_D_MULT(quadWeights[qx], quadWeights[qy], 1.) * func[2](AMREX_D_DECL(r[0], r[1], r[2]), t);
            GEMPIC_D_LOOP_END
            
            // Rescale the integral and assign it to array of degrees of freedom
            twoForm(i, j, k) = GEMPIC_D_MULT(integral * drHalf[0], drHalf[1], 1);
        });

    }

    field.fillBoundary();
    field.averageSync();

}

void FDDeRhamComplex::projection (amrex::Array<amrex::ParserExecutor<GEMPIC_SPACEDIM + 1>, 3> func, amrex::Real t,
                                      DeRhamField<Grid::primal, Space::edge>& field)

{
    // Gauss quadrature
    const int nQuad = 3;
    const amrex::GpuArray<amrex::Real, nQuad> quadPoints = {-0.7745966692414834, 0.0, 0.7745966692414834};
    const amrex::GpuArray<amrex::Real, nQuad> quadWeights = {5./9., 8./9., 5./9.}; // Can be initialized somewhere else

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
                    AMREX_D_DECL(r0[0] + i * dr[0], r0[1] + j * dr[1], r0[2] + k * dr[2])
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
                        integral += quadWeights[q] * func[comp](AMREX_D_DECL(r[0], r[1], r[2]), t);
                    } 
                
                    // Rescale the integral and assign it to array of degrees of freedom
                    oneForm(i, j, k) = integral* drHalf;
                }
#if (GEMPIC_SPACEDIM < 3)
                else
                {
                    oneForm(i, j, k) = func[comp](AMREX_D_DECL(r[0], r[1], r[2]), t);
                }
#endif          
                
            });

        }

    }

    field.fillBoundary();
    field.averageSync();
}


void FDDeRhamComplex::projection (amrex::Array<amrex::ParserExecutor<GEMPIC_SPACEDIM + 1>, 3> func, amrex::Real t,
                                      DeRhamField<Grid::dual, Space::face>& field)
{
    // Gauss quadrature 
    const int nQuad = 3;
    const amrex::GpuArray<amrex::Real, nQuad> quadPoints = {-0.7745966692414834, 0.0, 0.7745966692414834};
    const amrex::GpuArray<amrex::Real, nQuad> quadWeights = {5./9., 8./9., 5./9.}; // Can be initialized somewhere else

    // x-direction. Plane YZ
    for (amrex::MFIter mfi(field.data[0], true); mfi.isValid(); ++mfi)
    {
        const amrex::Box &bx = mfi.tilebox();
        amrex::Array4<amrex::Real> const &twoForm = (field.data[0])[mfi].array();

        const amrex::RealVect dr = m_dr;
        const amrex::GpuArray<amrex::Real, GEMPIC_SPACEDIM> r0 = m_geom.ProbLoArray();


        const amrex::GpuArray<amrex::Real, GEMPIC_SPACEDIM> drHalf =
        {
            AMREX_D_DECL(dr[0] / 2, dr[1] / 2, dr[2] / 2)
        };

        ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k)
        {
            // Compute the position of the point i, j, k
            amrex::GpuArray<amrex::Real, GEMPIC_SPACEDIM> r =
            {
                AMREX_D_DECL(r0[0] + 0.5*dr[0] + i * dr[0], r0[1] + 0.5*dr[1] + j * dr[1], r0[2] + 0.5*dr[2] + k * dr[2])
            };

            // Midpoint for the quadrature rule. The minus for dual means that it integrates from j-1/2 to j+1/2
            amrex::GpuArray<amrex::Real, GEMPIC_SPACEDIM> midpoint =
            {
                AMREX_D_DECL(r[0] - drHalf[0], r[1] - drHalf[1], r[2] - drHalf[2])
            };

            amrex::Real integral = 0.0;

            // Integral over the xy plane and z direction
            GEMPIC_D_LOOP_BEGIN(,
            for (int qy = 0; qy < nQuad; ++qy),
                for (int qz = 0; qz < nQuad; ++qz))
                    // Update location of quadrature points
                    r = {AMREX_D_DECL(r[0], midpoint[1] + quadPoints[qy] * drHalf[1], midpoint[2] + quadPoints[qz] * drHalf[2])};

                    integral += GEMPIC_D_MULT(1., quadWeights[qy], quadWeights[qz]) * func[0](AMREX_D_DECL(r[0], r[1], r[2]), t); // in 1D this is just evalutation
            GEMPIC_D_LOOP_END
        
            // Rescale the integral and assign it to array of degrees of freedom
            twoForm(i, j, k) = GEMPIC_D_MULT(integral, drHalf[1], drHalf[2]);
        });

    }

    // y-direction. Plane XZ
    for (amrex::MFIter mfi(field.data[1], true); mfi.isValid(); ++mfi)
    {
        const amrex::Box &bx = mfi.tilebox();

        amrex::Array4<amrex::Real> const &twoForm = (field.data[1])[mfi].array();

        const amrex::RealVect dr = m_dr;
        const amrex::GpuArray<amrex::Real, GEMPIC_SPACEDIM> r0 = m_geom.ProbLoArray();


        const amrex::GpuArray<amrex::Real, GEMPIC_SPACEDIM> drHalf =
        {
            AMREX_D_DECL(dr[0] / 2, dr[1] / 2, dr[2] / 2)
        };

        ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k)
        {

            // Compute the position of the point i, j, k
            amrex::GpuArray<amrex::Real, GEMPIC_SPACEDIM> r =
            {
                AMREX_D_DECL(r0[0] + 0.5*dr[0] + i * dr[0], r0[1] + 0.5*dr[1] + j * dr[1], r0[2] + 0.5*dr[2] + k * dr[2])
            };


            // Midpoint for the quadrature rule
            amrex::GpuArray<amrex::Real, GEMPIC_SPACEDIM> midpoint =
            {
                AMREX_D_DECL(r[0] - drHalf[0], r[1] - drHalf[1], r[2] - drHalf[2])
            };

            amrex::Real integral = 0.0;

            // Integral over the xy plane and z direction
            GEMPIC_D_LOOP_BEGIN(
            for (int qx = 0; qx < nQuad; ++qx),,
                for (int qz = 0; qz < nQuad; ++qz))
                    // Update location of quadrature points
                    r = {AMREX_D_DECL(midpoint[0] + quadPoints[qx] * drHalf[0], r[1], midpoint[2] + quadPoints[qz] * drHalf[2])};

                    integral += GEMPIC_D_MULT(quadWeights[qx], 1., quadWeights[qz]) * func[1](AMREX_D_DECL(r[0], r[1], r[2]), t);
            GEMPIC_D_LOOP_END

            // Rescale the integral and assign it to array of degrees of freedom
            twoForm(i, j, k) = GEMPIC_D_MULT(integral* drHalf[0], 1, drHalf[2]);
        });

    }

    // z-direction. Plane XY
    for (amrex::MFIter mfi(field.data[2], true); mfi.isValid(); ++mfi)
    {
        const amrex::Box &bx = mfi.tilebox();
        amrex::Array4<amrex::Real> const &twoForm = (field.data[2])[mfi].array();

        const amrex::RealVect dr = m_dr;
        const amrex::GpuArray<amrex::Real, GEMPIC_SPACEDIM> r0 = m_geom.ProbLoArray();


        const amrex::GpuArray<amrex::Real, GEMPIC_SPACEDIM> drHalf =
        {
            AMREX_D_DECL(dr[0] / 2, dr[1] / 2, dr[2] / 2)
        };
        
        ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k)
        {

            // Compute the position of the point i, j, k
            amrex::GpuArray<amrex::Real, GEMPIC_SPACEDIM> r =
            {
                AMREX_D_DECL(r0[0] + 0.5*dr[0] + i * dr[0], r0[1] + 0.5*dr[1] + j * dr[1], r0[2] + 0.5*dr[2] + k * dr[2])
            };


            // Midpoint for the quadrature rule
            amrex::GpuArray<amrex::Real, GEMPIC_SPACEDIM> midpoint =
            {
                AMREX_D_DECL(r[0] - drHalf[0], r[1] - drHalf[1], r[2] - drHalf[2])
            };

            amrex::Real integral = 0.0;

            // Integral over the xy plane and z direction
            GEMPIC_D_LOOP_BEGIN(
            for (int qx = 0; qx < nQuad; ++qx),
                for (int qy = 0; qy < nQuad; ++qy),)
                    // Update location of quadrature points
                    r = {AMREX_D_DECL(midpoint[0] + quadPoints[qx] * drHalf[0], midpoint[1] + quadPoints[qy] * drHalf[1], r[2])};

                    integral += GEMPIC_D_MULT(quadWeights[qx], quadWeights[qy], 1.) * func[2](AMREX_D_DECL(r[0], r[1], r[2]), t);
            GEMPIC_D_LOOP_END
        
            // Rescale the integral and assign it to array of degrees of freedom
            twoForm(i, j, k) = GEMPIC_D_MULT(integral * drHalf[0], drHalf[1], 1);
        });

    }
    field.fillBoundary();
    field.averageSync();
}
