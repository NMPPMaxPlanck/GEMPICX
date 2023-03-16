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
    m_tempPrimalZeroForm.define(amrex::convert(m_grid, amrex::IndexType(amrex::IntVect(1, 1, 1))) , m_distriMap, 1, m_nGhost);
    m_tempDualZeroForm.define(amrex::convert(m_grid, amrex::IndexType(amrex::IntVect(0, 0, 0))) , m_distriMap, 1, m_nGhost);

    m_tempPrimalOneForm[0].define(amrex::convert(m_grid, amrex::IndexType(amrex::IntVect(0, 1, 1))) , m_distriMap, 1, m_nGhost);
    m_tempPrimalOneForm[1].define(amrex::convert(m_grid, amrex::IndexType(amrex::IntVect(1, 0, 1))) , m_distriMap, 1, m_nGhost);
    m_tempPrimalOneForm[2].define(amrex::convert(m_grid, amrex::IndexType(amrex::IntVect(1, 1, 0))) , m_distriMap, 1, m_nGhost);

    m_tempDualOneForm[0].define(amrex::convert(m_grid, amrex::IndexType(amrex::IntVect(1, 0, 0))) , m_distriMap, 1, m_nGhost);
    m_tempDualOneForm[1].define(amrex::convert(m_grid, amrex::IndexType(amrex::IntVect(0, 1, 0))) , m_distriMap, 1, m_nGhost);
    m_tempDualOneForm[2].define(amrex::convert(m_grid, amrex::IndexType(amrex::IntVect(0, 0, 1))) , m_distriMap, 1, m_nGhost);

    m_tempPrimalTwoForm[0].define(amrex::convert(m_grid, amrex::IndexType(amrex::IntVect(1, 0, 0))) , m_distriMap, 1, m_nGhost);
    m_tempPrimalTwoForm[1].define(amrex::convert(m_grid, amrex::IndexType(amrex::IntVect(0, 1, 0))) , m_distriMap, 1, m_nGhost);
    m_tempPrimalTwoForm[2].define(amrex::convert(m_grid, amrex::IndexType(amrex::IntVect(0, 0, 1))) , m_distriMap, 1, m_nGhost);

    m_tempDualTwoForm[0].define(amrex::convert(m_grid, amrex::IndexType(amrex::IntVect(0, 1, 1))) , m_distriMap, 1, m_nGhost);
    m_tempDualTwoForm[1].define(amrex::convert(m_grid, amrex::IndexType(amrex::IntVect(1, 0, 1))) , m_distriMap, 1, m_nGhost);
    m_tempDualTwoForm[2].define(amrex::convert(m_grid, amrex::IndexType(amrex::IntVect(1, 1, 0))) , m_distriMap, 1, m_nGhost);


    m_tempPrimalThreeForm.define(amrex::convert(m_grid, amrex::IndexType(amrex::IntVect(0, 0, 0))) , m_distriMap, 1, m_nGhost);
    m_tempDualThreeForm.define(amrex::convert(m_grid, amrex::IndexType(amrex::IntVect(1, 1, 1))) , m_distriMap, 1, m_nGhost);

}

FDDeRhamComplex::~FDDeRhamComplex() {}


void FDDeRhamComplex::projection (amrex::ParserExecutor<4> func, amrex::Real t,
                                      DeRhamField<Grid::primal, Space::node>& field)
{
    // Point values on the primal grid.
    for (amrex::MFIter mfi(field.data, true); mfi.isValid(); ++mfi)
    {
        const amrex::Box &bx = mfi.tilebox();
        const amrex::IntVect lbound = bx.smallEnd();
        const amrex::IntVect ubound = bx.bigEnd();
        amrex::Array4<amrex::Real> const &zeroForm = (field.data)[mfi].array();

        const amrex::RealVect dr = m_dr;
        const amrex::GpuArray<amrex::Real, 3> r0 = m_geom.ProbLoArray(); // Put ProbLo in parameters as a getter. Domainbounds...

        ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k)
        {

            // Compute the position of the point i, j, k
            amrex::GpuArray<amrex::Real, 3> r =
            {
             r0[0] + i*dr[0],
             r0[1] + j*dr[1],
             r0[2] + k*dr[2]
            };


            // Assign point values to zeroForm 
            zeroForm(i, j, k) = func(r[0], r[1], r[2], t);
        });

    }

    field.fillBoundary();
    field.averageSync();

}

void FDDeRhamComplex::projection (amrex::ParserExecutor<4> func, amrex::Real t,
                                      DeRhamField<Grid::primal, Space::cell>& field)
{
    const int nQuad = 3;
    const amrex::GpuArray<amrex::Real, nQuad> quadPoints = {-std::sqrt(3./5.), 0.0, std::sqrt(3./5.)}; // Does the compiler compute this once or everytime function is called ?
    const amrex::GpuArray<amrex::Real, nQuad> quadWeights = {5./9. , 8./9., 5./9.}; // Can be initialized somewhere else

    // Volume integral 
    for (amrex::MFIter mfi(field.data, true); mfi.isValid(); ++mfi)
    {
        const amrex::Box &bx = mfi.tilebox();
        const amrex::IntVect lbound = bx.smallEnd();
        const amrex::IntVect ubound = bx.bigEnd();
        amrex::Array4<amrex::Real> const &threeForm = (field.data)[mfi].array();

        const amrex::RealVect dr = m_dr;
        const amrex::GpuArray<amrex::Real, 3> r0 = m_geom.ProbLoArray(); // Put ProbLo in parameters as a getter. Domainbounds...


        const amrex::Real drHalfX = dr[0]/2; 
        const amrex::Real drHalfY = dr[1]/2; 
        const amrex::Real drHalfZ = dr[2]/2; 

        ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k)
        {

            // Compute the position of the point i, j, k
            amrex::GpuArray<amrex::Real, 3> r =
            {
             r0[0] + i*dr[0],
             r0[1] + j*dr[1],
             r0[2] + k*dr[2]
            };


            // Midpoint for the quadrature rule
            amrex::Real midpointX = r[0] + drHalfX;
            amrex::Real midpointY = r[1] + drHalfY;
            amrex::Real midpointZ = r[2] + drHalfZ;
            amrex::Real integral = 0.0;

            // Integral over the xy plane and z direction
            for (int qx=0; qx<nQuad; ++qx)
            {
                // Update location of quadrature point
                r[0] = midpointX + quadPoints[qx] * drHalfX;

                for (int qy=0; qy<nQuad; ++qy)
                {
                    r[1] = midpointY + quadPoints[qy] * drHalfY;

                    for (int qz=0; qz<nQuad; ++qz)
                    {
                        r[2] = midpointZ + quadPoints[qz] *drHalfZ;
                
                        // Increment integral according to quadrature rule in the z direction with dx and dy
                        integral += quadWeights[qx] * quadWeights[qy] * quadWeights[qz] * func(r[0], r[1], r[2], t);
                    }
                } 
            }
        
            // Rescale the integral and assign it to degrees of freedom
            threeForm(i, j, k) = integral*drHalfX*drHalfY*drHalfZ;
        });

    }

    field.fillBoundary();
    field.averageSync();
}

void FDDeRhamComplex::projection (amrex::ParserExecutor<4> func, amrex::Real t,
                                      DeRhamField<Grid::dual, Space::node>& field)
{
    // Point values on the dual grid.
    for (amrex::MFIter mfi(field.data, true); mfi.isValid(); ++mfi)
    {
        const amrex::Box &bx = mfi.tilebox();
        const amrex::IntVect lbound = bx.smallEnd();
        const amrex::IntVect ubound = bx.bigEnd();
        amrex::Array4<amrex::Real> const &zeroForm = (field.data)[mfi].array();

        const amrex::RealVect dr = m_dr;
        const amrex::GpuArray<amrex::Real, 3> r0 = m_geom.ProbLoArray(); // Put ProbLo in parameters as a getter. Domainbounds...

        ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k)
        {

            // Compute the position of the point i + 1/2, j + 1/2, k + 1/2
            amrex::GpuArray<amrex::Real, 3> r =
            {
             (r0[0] + 0.5*dr[0]) + i*dr[0],
             (r0[1] + 0.5*dr[1]) + j*dr[1],
             (r0[2] + 0.5*dr[2]) + k*dr[2]
            };


            // Assign point values to zeroForm 
            zeroForm(i, j, k) = func(r[0], r[1], r[2], t);
        });

    }

    field.fillBoundary();
    field.averageSync();

}


void FDDeRhamComplex::projection (amrex::ParserExecutor<4> func, amrex::Real t,
                                      DeRhamField<Grid::dual, Space::cell>& field)
{
    const int nQuad = 3;
    const amrex::GpuArray<amrex::Real, nQuad> quadPoints = {-std::sqrt(3./5.), 0.0, std::sqrt(3./5.)}; // Does the compiler compute this once or everytime function is called ?
    const amrex::GpuArray<amrex::Real, nQuad> quadWeights = {5./9. , 8./9., 5./9.}; // Can be initialized somewhere else

    // Volume integral 
    for (amrex::MFIter mfi(field.data, true); mfi.isValid(); ++mfi)
    {
        const amrex::Box &bx = mfi.tilebox();
        const amrex::IntVect lbound = bx.smallEnd();
        const amrex::IntVect ubound = bx.bigEnd();
        amrex::Array4<amrex::Real> const &threeForm = (field.data)[mfi].array();

        const amrex::RealVect dr = m_dr;
        const amrex::GpuArray<amrex::Real, 3> r0 = m_geom.ProbLoArray(); // Put ProbLo in parameters as a getter. Domainbounds...


        const amrex::Real drHalfX = dr[0]/2; 
        const amrex::Real drHalfY = dr[1]/2; 
        const amrex::Real drHalfZ = dr[2]/2; 

        ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k)
        {

            // Compute the position of the point i + 1/2, j + 1/2, k + 1/2
            amrex::GpuArray<amrex::Real, 3> r =
            {
             (r0[0] + 0.5*dr[0]) + i*dr[0],
             (r0[1] + 0.5*dr[1]) + j*dr[1],
             (r0[2] + 0.5*dr[2]) + k*dr[2]
            };



            // Midpoint for the quadrature rule
            amrex::Real midpointX = r[0] - drHalfX;
            amrex::Real midpointY = r[1] - drHalfY;
            amrex::Real midpointZ = r[2] - drHalfZ;
            amrex::Real integral = 0.0;

            // Integral over the xy plane and z direction
            for (int qx=0; qx<nQuad; ++qx)
            {
                // Update location of quadrature point
                r[0] = midpointX + quadPoints[qx] * drHalfX;

                for (int qy=0; qy<nQuad; ++qy)
                {
                    r[1] = midpointY + quadPoints[qy] * drHalfY;

                    for (int qz=0; qz<nQuad; ++qz)
                    {
                        r[2] = midpointZ + quadPoints[qz] *drHalfZ;
                
                        // Increment integral according to quadrature rule in the z direction with dx and dy
                        integral += quadWeights[qx] * quadWeights[qy] * quadWeights[qz] * func(r[0], r[1], r[2], t);
                    }
                } 
            }
        
            // Rescale the integral and assign it to degrees of freedom
            threeForm(i, j, k) = integral*drHalfX*drHalfY*drHalfZ;
        });
    }

    field.fillBoundary();
    field.averageSync();

}


#if GEMPIC_SPACEDIM > 1
void FDDeRhamComplex::projection (amrex::Array<amrex::ParserExecutor<4>, GEMPIC_SPACEDIM> func, amrex::Real t,
                                      DeRhamField<Grid::dual, Space::edge>& field) 
{
    // Rescale for Gauss Quadrature. 1D 
    const int nQuad = 3;
    const amrex::GpuArray<amrex::Real, nQuad> quadPoints = {-std::sqrt(3./5.), 0.0, std::sqrt(3./5.)}; // Does the compiler compute this once or everytime function is called ?
    const amrex::GpuArray<amrex::Real, nQuad> quadWeights = {5./9. , 8./9., 5./9.}; // Can be initialized somewhere else

    // Do the loop over comp-direction
    for (int comp = 0; comp < GEMPIC_SPACEDIM; ++comp)
    {
        for (amrex::MFIter mfi(field.data[comp], true); mfi.isValid(); ++mfi)
        {
            const amrex::Box &bx = mfi.tilebox();
            const amrex::IntVect lbound = bx.smallEnd();
            const amrex::IntVect ubound = bx.bigEnd();
            amrex::Array4<amrex::Real> const &oneForm = (field.data[comp])[mfi].array();

            const amrex::RealVect dr = m_dr;
            const amrex::GpuArray<amrex::Real, 3> r0 = m_geom.ProbLoArray(); // Put ProbLo in parameters as a getter. Domainbounds...


            const amrex::Real drHalf = dr[comp]/2; 
            ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k)
            {

                // Compute the position of the point i + 1/2, j + 1/2, k + 1/2
                amrex::GpuArray<amrex::Real, 3> r =
                {
                 (r0[0] + 0.5*dr[0]) + i*dr[0],
                 (r0[1] + 0.5*dr[1]) + j*dr[1],
                 (r0[2] + 0.5*dr[2]) + k*dr[2]
                };
                
                //amrex::Print() << "r: (" << r[0] << "," << r[1] << "," << r[2] << ")" << std::endl;

                // Midpoint for the quadrature rule
                amrex::Real midpoint = r[comp] - drHalf;
                amrex::Real integral = 0.0;

                // Integral over the edge along the comp direction
                for (int q=0; q<nQuad; ++q)
                {
                    // Update location of quadrature point
                    r[comp] = midpoint + quadPoints[q] * drHalf;
                
                    // Increment integral according to quadrature rule
                    integral += quadWeights[q] * func[comp](r[0], r[1], r[2], t);
                } 
            
                // Rescale the integral and assign it to array of degrees of freedom
                oneForm(i, j, k) = integral*drHalf;
            });

        }

    }

    field.fillBoundary();
    field.averageSync();
}

void FDDeRhamComplex::projection (amrex::Array<amrex::ParserExecutor<4>, GEMPIC_SPACEDIM> func, amrex::Real t,
                                      DeRhamField<Grid::primal, Space::face>& field)
{
    // Rescale for Gauss Quadrature. 2D. Quadrature points and weights are the same 
    const int nQuad = 3;
    const amrex::GpuArray<amrex::Real, nQuad> quadPoints = {-std::sqrt(3./5.), 0.0, std::sqrt(3./5.)}; // Does the compiler compute this once or everytime function is called ?
    const amrex::GpuArray<amrex::Real, nQuad> quadWeights = {5./9. , 8./9., 5./9.}; // Can be initialized somewhere else

    // x-direction. Plane YZ
    for (amrex::MFIter mfi(field.data[0], true); mfi.isValid(); ++mfi)
    {
        const amrex::Box &bx = mfi.tilebox();
        const amrex::IntVect lbound = bx.smallEnd();
        const amrex::IntVect ubound = bx.bigEnd();
        amrex::Array4<amrex::Real> const &twoForm = (field.data[0])[mfi].array();

        const amrex::RealVect dr = m_dr;
        const amrex::GpuArray<amrex::Real, 3> r0 = m_geom.ProbLoArray(); // Put ProbLo in parameters as a getter. Domainbounds...


        const amrex::Real drHalfY = dr[1]/2; 
        const amrex::Real drHalfZ = dr[2]/2; 
        ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k)
        {

            // Compute the position of the point i, j, k
            amrex::GpuArray<amrex::Real, 3> r =
            {
             r0[0] + i*dr[0],
             r0[1] + j*dr[1],
             r0[2] + k*dr[2]
            };


            // Midpoint for the quadrature rule
            amrex::Real midpointY = r[1] + drHalfY;
            amrex::Real midpointZ = r[2] + drHalfZ;
            amrex::Real integral = 0.0;

            // Integral over the xy plane and z direction
            for (int qy=0; qy<nQuad; ++qy)
            {
                // Update location of quadrature point
                r[1] = midpointY + quadPoints[qy] * drHalfY;

                for (int qz=0; qz<nQuad; ++qz)
                {
                    r[2] = midpointZ + quadPoints[qz] *drHalfZ;
            
                    // Increment integral according to quadrature rule in the z direction with dx and dy
                    integral += quadWeights[qy] * quadWeights[qz] * func[0](r[0], r[1], r[2], t);
                }
            } 
        
            // Rescale the integral and assign it to array of degrees of freedom
            twoForm(i, j, k) = integral*drHalfY*drHalfZ;
        });

    }

    // y-direction. Plane XZ
    for (amrex::MFIter mfi(field.data[1], true); mfi.isValid(); ++mfi)
    {
        const amrex::Box &bx = mfi.tilebox();

        const amrex::IntVect lbound = bx.smallEnd();
        const amrex::IntVect ubound = bx.bigEnd();
        amrex::Array4<amrex::Real> const &twoForm = (field.data[1])[mfi].array();

        const amrex::RealVect dr = m_dr;
        const amrex::GpuArray<amrex::Real, 3> r0 = m_geom.ProbLoArray(); // Put ProbLo in parameters as a getter. Domainbounds...


        const amrex::Real drHalfX = dr[0]/2; 
        const amrex::Real drHalfZ = dr[2]/2; 
        ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k)
        {

            // Compute the position of the point i, j, k
            amrex::GpuArray<amrex::Real, 3> r =
            {
             r0[0] + i*dr[0],
             r0[1] + j*dr[1],
             r0[2] + k*dr[2]
            };


            // Midpoint for the quadrature rule
            amrex::Real midpointX = r[0] + drHalfX;
            amrex::Real midpointZ = r[2] + drHalfZ;
            amrex::Real integral = 0.0;

            // Integral over the xy plane and z direction
            for (int qx=0; qx<nQuad; ++qx)
            {
                // Update location of quadrature point
                r[0] = midpointX + quadPoints[qx] * drHalfX;

                for (int qz=0; qz<nQuad; ++qz)
                {
                    r[2] = midpointZ + quadPoints[qz] *drHalfZ;
            
                    // Increment integral according to quadrature rule in the z direction with dx and dy
                    integral += quadWeights[qx] * quadWeights[qz] * func[1](r[0], r[1], r[2], t);
                }
            } 

            // Rescale the integral and assign it to array of degrees of freedom
            twoForm(i, j, k) = integral*drHalfX*drHalfZ;
        });

    }

    // z-direction. Plane XY
    for (amrex::MFIter mfi(field.data[2], true); mfi.isValid(); ++mfi)
    {
        const amrex::Box &bx = mfi.tilebox();
        const amrex::IntVect lbound = bx.smallEnd();
        const amrex::IntVect ubound = bx.bigEnd();
        amrex::Array4<amrex::Real> const &twoForm = (field.data[2])[mfi].array();

        const amrex::RealVect dr = m_dr;
        const amrex::GpuArray<amrex::Real, 3> r0 = m_geom.ProbLoArray(); // Put ProbLo in parameters as a getter. Domainbounds...


        const amrex::Real drHalfX = dr[0]/2; 
        const amrex::Real drHalfY = dr[1]/2; 
        ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k)
        {

            // Compute the position of the point i, j, k
            amrex::GpuArray<amrex::Real, 3> r =
            {
             r0[0] + i*dr[0],
             r0[1] + j*dr[1],
             r0[2] + k*dr[2]
            };


            // Midpoint for the quadrature rule
            amrex::Real midpointX = r[0] + drHalfX;
            amrex::Real midpointY = r[1] + drHalfY;
            amrex::Real integral = 0.0;

            // Integral over the xy plane and z direction
            for (int qx=0; qx<nQuad; ++qx)
            {
                // Update location of quadrature point
                r[0] = midpointX + quadPoints[qx] * drHalfX;

                for (int qy=0; qy<nQuad; ++qy)
                {
                    r[1] = midpointY + quadPoints[qy] *drHalfY;
            
                    // Increment integral according to quadrature rule in the z direction with dx and dy
                    integral += quadWeights[qx] * quadWeights[qy] * func[2](r[0], r[1], r[2], t);
                }
            } 
        
            // Rescale the integral and assign it to array of degrees of freedom
            twoForm(i, j, k) = integral*drHalfX*drHalfY;
        });

    }

    field.fillBoundary();
    field.averageSync();

}

void FDDeRhamComplex::projection (amrex::Array<amrex::ParserExecutor<4>, GEMPIC_SPACEDIM> func, amrex::Real t,
                                      DeRhamField<Grid::primal, Space::edge>& field)

{
    // Rescale for Gauss Quadrature. 1D 
    const int nQuad = 3;
    const amrex::GpuArray<amrex::Real, nQuad> quadPoints = {-std::sqrt(3./5.), 0.0, std::sqrt(3./5.)}; // Does the compiler compute this once or everytime function is called ?
    const amrex::GpuArray<amrex::Real, nQuad> quadWeights = {5./9. , 8./9., 5./9.}; // Can be initialized somewhere else

    // Do the loop over comp-direction
    for (int comp = 0; comp < GEMPIC_SPACEDIM; ++comp)
    {
        for (amrex::MFIter mfi(field.data[comp], true); mfi.isValid(); ++mfi)
        {
            const amrex::Box &bx = mfi.tilebox();
            const amrex::IntVect lbound = bx.smallEnd();
            const amrex::IntVect ubound = bx.bigEnd();
            amrex::Array4<amrex::Real> const &oneForm = (field.data[comp])[mfi].array();

            const amrex::RealVect dr = m_dr;
            const amrex::GpuArray<amrex::Real, 3> r0 = m_geom.ProbLoArray(); // Put ProbLo in parameters as a getter. Domainbounds...


            const amrex::Real drHalf = dr[comp]/2; 
            ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k)
            {

                // Compute the position of the point i, j, k
                amrex::GpuArray<amrex::Real, 3> r =
                {
                 r0[0] + i*dr[0],
                 r0[1] + j*dr[1],
                 r0[2] + k*dr[2]
                };


                // Midpoint for the quadrature rule
                amrex::Real midpoint = r[comp] + drHalf;
                amrex::Real integral = 0.0;

                // Integral over the edge along the comp direction
                for (int q=0; q<nQuad; ++q)
                {
                    // Update location of quadrature point
                    r[comp] = midpoint + quadPoints[q] * drHalf;
                
                    // Increment integral according to quadrature rule
                    integral += quadWeights[q] * func[comp](r[0], r[1], r[2], t);
                } 
            
                // Rescale the integral and assign it to array of degrees of freedom
                oneForm(i, j, k) = integral*drHalf;
            });

        }

    }

    field.fillBoundary();
    field.averageSync();
}


void FDDeRhamComplex::projection (amrex::Array<amrex::ParserExecutor<4>, GEMPIC_SPACEDIM> func, amrex::Real t,
                                      DeRhamField<Grid::dual, Space::face>& field)
{
    // Rescale for Gauss Quadrature. 2D. Quadrature points and weights are the same 
    const int nQuad = 3;
    const amrex::GpuArray<amrex::Real, nQuad> quadPoints = {-std::sqrt(3./5.), 0.0, std::sqrt(3./5.)}; // Does the compiler compute this once or everytime function is called ?
    const amrex::GpuArray<amrex::Real, nQuad> quadWeights = {5./9. , 8./9., 5./9.}; // Can be initialized somewhere else

    // x-direction. Plane YZ
    for (amrex::MFIter mfi(field.data[0], true); mfi.isValid(); ++mfi)
    {
        const amrex::Box &bx = mfi.tilebox();
        const amrex::IntVect lbound = bx.smallEnd();
        const amrex::IntVect ubound = bx.bigEnd();
        amrex::Array4<amrex::Real> const &twoForm = (field.data[0])[mfi].array();

        const amrex::RealVect dr = m_dr;
        const amrex::GpuArray<amrex::Real, 3> r0 = m_geom.ProbLoArray(); // Put ProbLo in parameters as a getter. Domainbounds...


        const amrex::Real drHalfY = dr[1]/2; 
        const amrex::Real drHalfZ = dr[2]/2; 
        ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k)
        {

            // Compute the position of the point i, j, k
            amrex::GpuArray<amrex::Real, 3> r =
            {
                (r0[0] + 0.5*dr[0]) + i*dr[0],
                (r0[1] + 0.5*dr[1]) + j*dr[1],
                (r0[2] + 0.5*dr[2]) + k*dr[2]
            };

            // Midpoint for the quadrature rule. The minus for dual means that it integrates from j-1/2 to j+1/2
            amrex::Real midpointY = r[1] - drHalfY;
            amrex::Real midpointZ = r[2] - drHalfZ;
            amrex::Real integral = 0.0;

            // Integral over the xy plane and z direction
            for (int qy=0; qy<nQuad; ++qy)
            {
                // Update location of quadrature point
                r[1] = midpointY + quadPoints[qy] * drHalfY;

                for (int qz=0; qz<nQuad; ++qz)
                {
                    r[2] = midpointZ + quadPoints[qz] *drHalfZ;
            
                    // Increment integral according to quadrature rule in the z direction with dz and dy
                    integral += quadWeights[qy] * quadWeights[qz] * func[0](r[0], r[1], r[2], t);
                }
            } 
        
            // Rescale the integral and assign it to array of degrees of freedom
            twoForm(i, j, k) = integral*drHalfY*drHalfZ;
        });

    }

    // y-direction. Plane XZ
    for (amrex::MFIter mfi(field.data[1], true); mfi.isValid(); ++mfi)
    {
        const amrex::Box &bx = mfi.tilebox();

        const amrex::IntVect lbound = bx.smallEnd();
        const amrex::IntVect ubound = bx.bigEnd();
        amrex::Array4<amrex::Real> const &twoForm = (field.data[1])[mfi].array();

        const amrex::RealVect dr = m_dr;
        const amrex::GpuArray<amrex::Real, 3> r0 = m_geom.ProbLoArray(); // Put ProbLo in parameters as a getter. Domainbounds...


        const amrex::Real drHalfX = dr[0]/2; 
        const amrex::Real drHalfZ = dr[2]/2; 
        ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k)
        {

            // Compute the position of the point i, j, k
            amrex::GpuArray<amrex::Real, 3> r =
            {
                (r0[0] + 0.5*dr[0]) + i*dr[0],
                (r0[1] + 0.5*dr[1]) + j*dr[1],
                (r0[2] + 0.5*dr[2]) + k*dr[2]
            };


            // Midpoint for the quadrature rule
            amrex::Real midpointX = r[0] - drHalfX;
            amrex::Real midpointZ = r[2] - drHalfZ;
            amrex::Real integral = 0.0;

            // Integral over the xy plane and z direction
            for (int qx=0; qx<nQuad; ++qx)
            {
                // Update location of quadrature point
                r[0] = midpointX + quadPoints[qx] * drHalfX;

                for (int qz=0; qz<nQuad; ++qz)
                {
                    r[2] = midpointZ + quadPoints[qz] *drHalfZ;
            
                    // Increment integral according to quadrature rule in the z direction with dx and dy
                    integral += quadWeights[qx] * quadWeights[qz] * func[1](r[0], r[1], r[2], t);
                }
            } 

            // Rescale the integral and assign it to array of degrees of freedom
            twoForm(i, j, k) = integral*drHalfX*drHalfZ;
        });

    }

    // z-direction. Plane XY
    for (amrex::MFIter mfi(field.data[2], true); mfi.isValid(); ++mfi)
    {
        const amrex::Box &bx = mfi.tilebox();
        const amrex::IntVect lbound = bx.smallEnd();
        const amrex::IntVect ubound = bx.bigEnd();
        amrex::Array4<amrex::Real> const &twoForm = (field.data[2])[mfi].array();

        const amrex::RealVect dr = m_dr;
        const amrex::GpuArray<amrex::Real, 3> r0 = m_geom.ProbLoArray(); // Put ProbLo in parameters as a getter. Domainbounds...


        const amrex::Real drHalfX = dr[0]/2; 
        const amrex::Real drHalfY = dr[1]/2; 
        ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k)
        {

            // Compute the position of the point i, j, k
            amrex::GpuArray<amrex::Real, 3> r =
            {
                (r0[0] + 0.5*dr[0]) + i*dr[0],
                (r0[1] + 0.5*dr[1]) + j*dr[1],
                (r0[2] + 0.5*dr[2]) + k*dr[2]
            };


            // Midpoint for the quadrature rule
            amrex::Real midpointX = r[0] - drHalfX;
            amrex::Real midpointY = r[1] - drHalfY;
            amrex::Real integral = 0.0;

            // Integral over the xy plane and z direction
            for (int qx=0; qx<nQuad; ++qx)
            {
                // Update location of quadrature point
                r[0] = midpointX + quadPoints[qx] * drHalfX;

                for (int qy=0; qy<nQuad; ++qy)
                {
                    r[1] = midpointY + quadPoints[qy] *drHalfY;
            
                    // Increment integral according to quadrature rule in the z direction with dx and dy
                    integral += quadWeights[qx] * quadWeights[qy] * func[2](r[0], r[1], r[2], t);
                }
            } 
        
            // Rescale the integral and assign it to array of degrees of freedom
            twoForm(i, j, k) = integral*drHalfX*drHalfY;
        });

    }
    field.fillBoundary();
    field.averageSync();
}


#endif
