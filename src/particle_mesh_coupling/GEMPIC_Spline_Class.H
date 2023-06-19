#ifndef SPLINE_CLASS_
#define SPLINE_CLASS_

namespace Spline {

/**
 * @brief This object will contain the value of all the non-vanishing local spline basis functions at particle positions
 * 
 * @tparam degX Degree of the spline in x-direction
 * @tparam degY Degree of the spline in y-direction
 * @tparam degZ Degree of the spline in z-direction
*/
template<int degX, int degY, int degZ>
class SplineBase {

public:
    /**
     * @brief Construct a new SplineBase object
     * 
     * @details Initialize the basis by calling the update1DBasis method for each direction x, y and z
     * 
     * @param position Current position of the particle
     * @param plo (rename this one)
     * @param dxInverse Inverse of the cell size in all three directions x, y and z (It's an array)
     * @details Additional details about xInverse
     *  - dxInverse[0] == 1/dx
     *  - dxInverse[1] == 1/dy
     *  - dxInverse[2] == 1/dz
     *  - (Potentially dim < 3)
     *  - dxInverse[dim] == 1/(dx * dy * dz)
    */
    SplineBase(const amrex::GpuArray<amrex::Real, GEMPIC_SPACEDIM> &position,
               const amrex::GpuArray<amrex::Real, GEMPIC_SPACEDIM> &plo,
               const amrex::GpuArray<amrex::Real, GEMPIC_SPACEDIM + 1> &dxInverse)
    {
        AMREX_D_TERM(this->span[0] = 0;,
                     this->span[1] = 0;,
                     this->span[2] = 0;);
                     
        AMREX_D_TERM((update1DSplines<0, degX>(position[0], plo[0], dxInverse[0]));,
                     (update1DSplines<1, degY>(position[1], plo[1], dxInverse[1]));,
                     (update1DSplines<2, degZ>(position[2], plo[2], dxInverse[2])););
    }

    /**
     * @brief An array of indices
     * 
     * @details At the end of the function initBSplinesAtPositions, span is the index of the first field-coefficient corresponding to a non-zero spline at the particle location.
    */
    amrex::GpuArray<int, GEMPIC_SPACEDIM> span;

    /**
     * @brief Cell centered spline
    */
    amrex::GpuArray<amrex::GpuArray<amrex::Real, std::max(std::max(degX, degY), degZ)>, GEMPIC_SPACEDIM> splineCell;

    /**
     * @brief Node centered spline
    */
    amrex::GpuArray<amrex::GpuArray<amrex::Real, std::max(std::max(degX, degY), degZ) + 1>, GEMPIC_SPACEDIM> splineNode;

    /**
     * @brief Compute new particle position
     * 
     * @tparam dim The direction in which to compute: x = 0, y = 1, z = 2
     * @tparam deg The degree of the spline
     * 
     * @param position Current position of the particle
     * @param plo (rename this one)
     * @param dxInverse Inverse of the cell size in all three directions x, y and z (It's an array)
     * 
     * @return xint
     * @details At the end of the function, xint is the distance from the endpoint of the first non-zero spline at the particle location. It is normalized by dx and lies between 0 and 1.
    */
    template<int dim, int deg>
    amrex::Real initBSplinesAtPositions(const amrex::Real position, const amrex::Real plo, const amrex::Real dxInverse)
    {
        // Compute normalized position xint and span
        amrex::Real xint = (position - plo) * dxInverse;
        span[dim] = std::floor(xint);
        xint = xint - span[dim];

        if (deg % 2 == 0)
        {
            if (xint < 0.5)
            {
                span[dim] -= deg / 2;
                xint = 0.5 - xint;
            }
            else
            {
                span[dim] += -deg / 2 + 1;
                xint = 1.5 - xint;
            }
        }
        else
        {
            span[dim] -= (deg - 1) / 2;
            xint = 1. - xint;
        }
        return xint;
    }

    /**
     * @brief Update the basis by evaluating the splines at the new particle position
     * 
     * @tparam dim The direction in which to compute: x = 0, y = 1, z = 2
     * @tparam deg The degree of the spline
     * 
     * @param position Current position of the particle
     * @param plo (rename this one)
     * @param dxInverse Inverse of the cell size in all three directions x, y and z (It's an array)
    */
    template<int dim, int deg>
    void update1DSplines(const amrex::Real position, const amrex::Real plo, const amrex::Real dxInverse)
    {
        const amrex::Real xint = initBSplinesAtPositions<dim, deg> (position, plo, dxInverse);

        splineEval<deg - 1, std::max(std::max(degX, degY), degZ)>(xint, splineCell[dim]);
        splineEval<deg, std::max(std::max(degX, degY), degZ) + 1>(xint, splineNode[dim]);
    }

    /**
     * @brief Compute the basis coefficient for one- or two-forms
     * 
     * @tparam dim The direction in which to compute: x = 0, y = 1, z = 2
     * @tparam form Is it a 1-form or a 2-form?
     * 
     * @param i The index of the x-dim basis function
     * @param j The index of the y-dim basis function
     * @param k The index of the z-dim basis function
     * 
     * @return base The value of the calculated coefficient
    */
    template<int dim, int form>
    amrex::Real evalBSpline(int i, int j, int k) const
    {
        if (form == 0) {
                  double base = GEMPIC_D_MULT(splineNode[0][i],
                                              splineNode[1][j],
                                              splineNode[2][k]);
                  return base;
        }
        else if (form == 1) {
            if (dim == 0) {
                double base = GEMPIC_D_MULT(splineCell[0][i],
                                            splineNode[1][j],
                                            splineNode[2][k]);
                return base;
            }
            else if (dim == 1) {
                double base = GEMPIC_D_MULT(splineNode[0][i],
                                            splineCell[1][j],
                                            splineNode[2][k]);
                return base;
            }
            else if (dim == 2) {
                double base = GEMPIC_D_MULT(splineNode[0][i],
                                            splineNode[1][j],
                                            splineCell[2][k]);
                return base;
            }
        }
        else if (form == 2) {
            if (dim == 0)
            {
                double base = GEMPIC_D_MULT(splineNode[0][i],
                                            splineCell[1][j],
                                            splineCell[2][k]);
                return base;
            }
            else if (dim == 1)
            {
                double base = GEMPIC_D_MULT(splineCell[0][i],
                                            splineNode[1][j],
                                            splineCell[2][k]);
                return base;
            }
            else if (dim == 2)
            {
                double base = GEMPIC_D_MULT(splineCell[0][i],
                                            splineCell[1][j],
                                            splineNode[2][k]);
                return base;
            }
        }
        else if (form == 3){
                  double base = GEMPIC_D_MULT(splineCell[0][i],
                                              splineCell[1][j],
                                              splineCell[2][k]);
                  return base;
        }
        else {
            throw std::runtime_error("The template parameter form must be 1 or 2.");
        }
    }

    /**
     * @brief Compute the basis coefficient for one- or two-forms
     * 
     * @tparam vDim The number of dimensions for the velocity
     * @tparam form Is it a 1-form or a 2-form?
     * 
     * @param fieldArray The array of field coefficients of the field to be treated like an electrical field or a magnetic field
     * 
     * @return fields The newly computed field values
    */
    template<int vDim, int form>
    AMREX_GPU_HOST_DEVICE amrex::GpuArray<amrex::Real, vDim> evalField (const amrex::GpuArray<amrex::Array4<amrex::Real>, vDim> fieldArray) const
    {
        amrex::GpuArray<amrex::Real, vDim> fields;
        for (int comp = 0; comp < vDim; comp++)
        {
            fields[comp] = 0.;
        }

GEMPIC_D_LOOP_BEGIN(for (int ii = 0; ii <= degX; ++ii),
                    for (int jj = 0; jj <= degY; ++jj),
                    for (int kk = 0; kk <= degZ; ++kk));

                const amrex::GpuArray<int, GEMPIC_SPACEDIM> cell_index = {
                    AMREX_D_DECL(this->span[0] + ii, this->span[1] + jj, this->span[2] + kk)};
                if (ii < degX)
                {
                    const amrex::Real sCoeff = evalBSpline<0, form>(ii, jj, kk);
                    fields[0] +=
                        sCoeff *
                        (fieldArray[0])(AMREX_D_DECL(cell_index[0], cell_index[1], cell_index[2]), 0);
                }
                if (jj < degY)
                {
                    const amrex::Real sCoeff = evalBSpline<1, form>(ii, jj, kk);
                    fields[1] +=
                        sCoeff *
                        (fieldArray[1])(AMREX_D_DECL(cell_index[0], cell_index[1], cell_index[2]), 0);
                }
                if (kk < degZ)
                {
                    const amrex::Real sCoeff = evalBSpline<2, form>(ii, jj, kk);
                    fields[2] +=
                        sCoeff *
                        (fieldArray[2])(AMREX_D_DECL(cell_index[0], cell_index[1], cell_index[2]), 0);
                }
GEMPIC_D_LOOP_END;

    return (fields);
    }

    /**
     * @brief Evaluate the spline for a given degree at a given point x
     * 
     * @tparam degree Which degree spline should be evaluated
     * @tparam length The length of the array of results
     * 
     * @param x The position at which the spline shall be evaluated
     * @param s The array in which the results will be stored
    */
    template <int degree, int length>
    AMREX_GPU_HOST_DEVICE constexpr void splineEval (const amrex::Real x,
                                                     amrex::GpuArray<amrex::Real, length> &s) const
    {
        amrex::Real factor;
        switch (degree)
        {
            case 0:
                s[0] = 1.;
                break;
            case 1:
                s[0] = x;
                s[1] = 1. - x;
                break;
            case 2:
                s[0] = (x * x) * 0.5;
                s[1] = (1. - x) * x + 0.5;
                s[2] = ((x-2.)*x + 1.) * 0.5;
                break;
            case 3:
                factor = 1./6.;
                s[0] = (x * x * x) * factor;
                s[1] = (((-x + 1) * x + 1.) * 3. * x + 1.) * factor;
                s[2] = ((3. * x - 6.) * x * x + 4.) * factor;
                // s[3] = pow((1. - x), 3) * factor;
                s[3] = (((-x + 3.) * x - 3.) * x + 1.) * factor;
                break;
            case 4:
                factor = 1./24.;
                s[0] = (x * x * x * x) * factor;
                s[1] = ((((-x + 1.) * 4. * x + 6.) * x + 4.) * x + 1.) * factor;
                s[2] = ((((6. * x - 12.) * x - 6.) * x + 12.) * x + 11.) * factor;
                s[3] = ((((-4. * x + 12.) * x - 6.) * x - 12.) * x + 11.) * factor;
                // s[4] = pow(1. - x, 4) * factor;
                s[4] = ((((x - 4.) * x + 6.) * x - 4.) * x + 1.) * factor;
                break;
            case 5:
                factor = 1./120.;
                
                s[0] = (x * x * x * x * x) * factor;
                s[1] = (((((-x + 1.) * x + 2.) * x + 2.) * x + 1.) * 5. * x + 1.) * factor;
                s[2] = (((((x - 2.) * x - 2.) * x + 2.) * x + 5.) * 10. * x + 26.) * factor;
                s[3] = (((-x + 3.) * x * x - 6.) * x * 10. * x + 66.) * factor;
                s[4] = (((((x - 4.) * x + 4.) * x + 4.) * x - 10.) * 5. * x + 26.) * factor;
                s[5] = (((((-x + 5.) * x - 10.) * x + 10.) * x - 5.) * x + 1.) * factor;
                break;
        }
    }
};

/**
 * @brief This object will contain the value of all the non-vanishing local spline basis functions at particle positions including a primitive
 *
 * @details This class inherits from the SplineBase class and adds a data member and methods for handling a primitive
 * 
 * @tparam degX Degree of the spline in x-direction
 * @tparam degY Degree of the spline in y-direction
 * @tparam degZ Degree of the spline in z-direction
*/
template<int degX, int degY, int degZ>
class SplineWithPrimitive : public SplineBase<degX, degY, degZ> {

public:
    /**
     * @brief Construct a new SplineWithPrimitive object
     * 
     * @details Initialize the primitive by calling the update1DPrimitive method for each direction x, y and z
     * 
     * @param position Current position of the particle
     * @param plo (rename this one)
     * @param dxInverse Inverse of the cell size in all three directions x, y and z (It's an array)
     * @details Additional details about xInverse
     *  - dxInverse[0] == 1/dx
     *  - dxInverse[1] == 1/dy
     *  - dxInverse[2] == 1/dz
     *  - (Potentially dim < 3)
     *  - dxInverse[dim] == 1/(dx * dy * dz)
    */
    SplineWithPrimitive(const amrex::GpuArray<amrex::Real, GEMPIC_SPACEDIM> &position,
                        const amrex::GpuArray<amrex::Real, GEMPIC_SPACEDIM> &plo,
                        const amrex::GpuArray<amrex::Real, GEMPIC_SPACEDIM + 1> &dxInverse
                        ) : SplineBase<degX, degY, degZ>(position, plo, dxInverse)
    {
        AMREX_D_TERM((update1DPrimitive<0, degX>(position[0], plo[0], dxInverse[0]));,
                     (update1DPrimitive<1, degY>(position[1], plo[1], dxInverse[1]));,
                     (update1DPrimitive<2, degZ>(position[2], plo[2], dxInverse[2])););
    }

    /**
     * @brief An array of indices
     * 
     * @details At the end of the function initBSplinesAtPositions, span is the index of the first field-coefficient corresponding to a non-zero spline at the particle location.
    */
    amrex::GpuArray<int, GEMPIC_SPACEDIM> spanOld;

    /**
     * @brief primitive of the new spline
    */
    amrex::GpuArray<amrex::GpuArray<amrex::Real, std::max(std::max(degX, degY), degZ)>, GEMPIC_SPACEDIM> primitiveNew;

    /**
     * @brief primitive of the old spline
    */
    amrex::GpuArray<amrex::GpuArray<amrex::Real, std::max(std::max(degX, degY), degZ)>, GEMPIC_SPACEDIM> primitiveOld;

    /**
     * @brief Update the basis by evaluating the splines at the new particle position
     * 
     * @tparam dim The direction in which to compute: x = 0, y = 1, z = 2
     * @tparam deg The degree of the spline
     * 
     * @param position Current position of the particle
     * @param plo (rename this one)
     * @param dxInverse Inverse of the cell size in all three directions x, y and z (It's an array)
    */
    template<int dim, int deg>
    void update1DSplines(const amrex::Real position, const amrex::Real plo, const amrex::Real dxInverse)
    {
        spanOld[dim] = std::move(this->span[dim]);
        const amrex::Real xint = this->template initBSplinesAtPositions<dim, deg> (position, plo, dxInverse);

        this->template splineEval<deg - 1, std::max(std::max(degX, degY), degZ)>(xint, this->splineCell[dim]);
        this->template splineEval<deg, std::max(std::max(degX, degY), degZ) + 1>(xint, this->splineNode[dim]);
    }

    /**
     * @brief Update the primitive by evaluating splines at the new particle position
     * 
     * @tparam dim The direction in which to compute: x = 0, y = 1, z = 2
     * @tparam deg The degree of the spline
     * 
     * @param position Current position of the particle
     * @param plo (rename this one)
     * @param dxInverse Inverse of the cell size in all three directions x, y and z (It's an array)
    */
    template<int dim, int deg>
    void update1DPrimitive(const amrex::Real position, const amrex::Real plo, const amrex::Real dxInverse)
    {
        const amrex::Real xint = this->template initBSplinesAtPositions<dim, deg>(position, plo, dxInverse);

        primitiveOld[dim] = std::move(primitiveNew[dim]);
        primitiveEval<deg - 1, std::max(std::max(degX, degY), degZ)>(xint, primitiveNew[dim]);
    }

    /**
     * @brief Compute the difference between primitives of the old spline and the new spline
     * 
     * @tparam dim The direction in which to compute: x = 0, y = 1, z = 2
     * @tparam deg The degree of the spline
     * 
     * @param dx Cell size in all three directions x, y and z (It's an array)
     * @param index Index at which to evaluate the primitive difference
     * 
     * @return primitiveDifference The difference between primitives of the old spline and the new spline
    */
    template<int dim, int deg>
    amrex::Real computePrimitiveDifference(const amrex::GpuArray<amrex::Real, GEMPIC_SPACEDIM> dx, const int index) const
    {
        // Size of support of difference of primitives
        const int local_size = std::abs(this->span[dim] - this->spanOld[dim]) + deg;

        amrex::Real primitiveDifference = 0;

        if (this->spanOld[dim] < this->span[dim])
        {   
            if (index < deg) {
                primitiveDifference = this->primitiveOld[dim][index] * dx[dim];
            }
            if (index >= deg && index < local_size) {
                primitiveDifference = dx[dim];
            }
            if (index >= local_size - deg && index <= local_size - 1) {
                primitiveDifference -= this->primitiveNew[dim][index + deg - local_size] * dx[dim];
            }
        }
        else
        {
            if (index < deg) { 
                primitiveDifference = -this->primitiveNew[dim][index] * dx[dim];
            }
            if (index >= deg && index < local_size) {
                primitiveDifference = -dx[dim];
            }
            if (index >= local_size - deg && index <= local_size - 1) {
                primitiveDifference += this->primitiveOld[dim][index + deg - local_size] * dx[dim];
            }
        }

        return primitiveDifference;
    }

    /**
     * @brief Evaluate the primitive for a given degree at a given point x
     * 
     * @tparam degree Which degree primitive should be evaluated
     * @tparam length The length of the array of results
     * 
     * @param x The position at which the primitive shall be evaluated
     * @param s The array in which the results will be stored
    */
    template <int degree, int length>
    AMREX_GPU_HOST_DEVICE constexpr void primitiveEval(const amrex::Real x,
                                                       amrex::GpuArray<amrex::Real, length> &s) const
    {
        amrex::Real factor;
        switch (degree)
        {
            case -1:
                s[0] = 1;
                break;
            case 0:
                s[0] = x;
                break;
            case 1:
                s[0] = (x*x) * 0.5;
                s[1] = ((-x + 2.) * x + 1) * 0.5;
                break;
            case 2:
                factor = (1./6.);
                s[0] = (x*x*x) * factor;
                s[1] = (((-2. * x + 3) * x + 3.) * x + 1.) * factor;
                s[2] = (((-x + 3.) * x - 1.) * x - 1.) * factor + 1.;
                break;
            case 3:
                factor = (1./24.);
                s[0] = (x*x * x*x) * factor;
                s[1] = -pow(x, 4) / 8. + pow(x, 3) / 6. + pow(x, 2) / 4. + x / 6. + 1. / 24.;
                s[2] = pow(x, 4) / 8. - pow(x, 3) / 3. + x * 2. / 3. + 0.5;
                // s[3] = -pow((1. - x), 4) * factor + 1.;
                s[3] = -((((x - 4.) * x + 6) * x - 4.) * x + 1) * factor + 1.;
                break;
            case 4:
                factor = (1./120.);
                s[0] = (x*x * x * x*x) * factor;
                s[1] = (((((-4. * x + 5.) * x + 10.) * x + 10.) * x + 5.) * x + 1.) * factor;
                s[2] = (((((6. * x - 15.) * x - 10.) * x + 30.) * x + 55.) * x + 27.) * factor;
                s[3] = (((((-4. * x + 15.) * x - 10.) * x - 30.) * x + 55.) * x + 93.) * factor;
                //s[4] = ((((x - 5.) * x + 8) * x*x - 5.) * x + 1.) * factor + 1.;
                s[4] = -(((((-x + 5.) * x - 10.) * x + 10.) * x - 5.) * x + 1.) * factor + 1.;
                break;
            case 5:
                factor = (1./720.);
                s[0] = pow(x, 6) * factor;
                s[1] =
                    ((((((-5. * x + 6.) * x + 15.) * x + 20.) * x + 15.) * x + 6.) * x + 1.) * factor;
                s[2] =
                    ((((((10. * x - 24.) * x - 30.) * x + 40.) * x + 150.) * x + 156.) * x + 58.) * factor;
                s[3] = ((((-10. * x + 36.) * x * x - 120.) * x * x + 396.) * x + 360.) * factor;
                s[4] =
                    ((((((5. * x - 24.) * x + 30.) * x + 40.) * x - 150.) * x + 156.) * x + 662.) * factor;
                s[5] = ((((((-x + 6.) * x - 15.) * x + 20.) * x - 15.) * x + 6.) * x + 719.) * factor;
                break;
        }
    }
};

}
#endif