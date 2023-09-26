#ifndef SPLINE_CLASS_
#define SPLINE_CLASS_

namespace Spline {

/**
 * @brief Objects of this class will carry the information of all non-vanishing grid B-splines on a given particle position
 * 
 * Description:
 * Two types of univariate B-splines are involved:
 * 
 *  - "node splines" (centered on integer nodes) defined as
 * 
 *    nodeS_i( x ) := ref_S_{degX}( (x - nx_i)/dx )  
 *
 *  - "cell splines" (centered on cell mid-points) defined as
 * 
 *    cellS_i( x ) := (1/dx) * ref_S_{degX-1}( (x - nx_{i+1/2})/dx )
 *
 * where: 
 *  
 *  - ref_S_{deg} is the (symmetric, centered) cardinal B-spline of degree deg with support [-(deg+1)/2, (deg+1)/2]
 *  - dx is the cell size (in the relevant direction)
 *  - nx_i = (plo + i * dx) is the i-th grid node (in the relevant dimension) 
 *  
 * These splines obtained by applying the dual node- and cell- reduction operators 
 * (corresponding to tR0 and tR1 in 1D) to the "particle spline"
 * 
 *    partS( x - x_p ) := (1/dx) * ref_S_{degX-1}( (x - x_p)/dx )
 * 
 * (centered at the particle, with support [x_p-degX/2, x_p+degX/2] and unit integral), indeed:
 * 
 *  - the reduction on a dual node yields a "cell spline" evaluated at the particle:
 * 
 *      tR0_i( partS(x-x_p) ) = partS( nx_{i+1/2} - x_p ) = cellS_i( x_p )
 * 
 *  - the reduction on a dual cell yields a "node spline" evaluated at the particle: 
 * 
 *      tR1_i( partS(x-x_p) ) = \f$ \int_{nx_{i-1/2}}^{nx_{i+1/2}} \f$ partS( x  - x_p ) = nodeS_i( x_p )
 * 
 * @tparam degX : degree of the node-spline in x-direction
 * @tparam degY : degree of the node-spline in y-direction
 * @tparam degZ : degree of the node-spline in z-direction
*/

template<int degX, int degY, int degZ>
class SplineBase {

public:
    /**
     * @brief Construct a new SplineBase object
     * 
     * Description: 
     * for a given (particle) position xx_p, this constructor computes the values of 
     *  - firstIndex[dim] for each dimension (the index of the first splines contributing to xx_p) 
     *  - cellSplineVals[dim] the array containing the non-zero values of cell splines at xx_p
     *  - nodeSplineVals[dim] the array containing the non-zero values of node splines at xx_p
     * 
     * @param position : position of the particle
     * @param plo : low end of the problem domain
     * @param dxInverse : inverse of the cell size in all three directions x, y and z
     
     * @details Additional details about dxInverse
     *  - dxInverse[0] == 1/dx
     *  - dxInverse[1] == 1/dy
     *  - dxInverse[2] == 1/dz
     *  - (Potentially dim < 3)
     *  - dxInverse[GEMPIC_SPACEDIM] == dxInverse[0] * .. * dxInverse[GEMPIC_SPACEDIM-1]
    */

    AMREX_GPU_HOST_DEVICE SplineBase (const amrex::GpuArray<amrex::Real, GEMPIC_SPACEDIM> &position,
               const amrex::GpuArray<amrex::Real, GEMPIC_SPACEDIM> &plo,
               const amrex::GpuArray<amrex::Real, GEMPIC_SPACEDIM + 1> &dxInverse)
    {
        AMREX_D_TERM((update1DSplines<0>(position[0], plo[0], dxInverse[0]));,
                     (update1DSplines<1>(position[1], plo[1], dxInverse[1]));,
                     (update1DSplines<2>(position[2], plo[2], dxInverse[2])););
    }

    /**
     * @brief An array of spline degrees stored for convenience
    */
    static constexpr int splineDegrees[GEMPIC_SPACEDIM] = {AMREX_D_DECL(degX, degY, degZ)};

    /**
     * @brief Max spline degree stored fo convenience
    */
    static constexpr int maxDeg{AMREX_D_PICK(
        degX,
        std::max(degX, degY),
        std::max(std::max(degX, degY), degZ)
    )};
    /**
     * @brief firstIndex[dim] is the index of the first spline(s) contributing at the particle position, along dimension dim
     * 
     * Description: 
     *  
     * this array is computed in the function initBSplinesAtPositions: its values are such that we have 
     * 
     *   S_i(x_p) > 0  only for  i = firstIndex[0] .. firstIndex[0] + degX
     *   S_j(y_p) > 0  only for  j = firstIndex[1] .. firstIndex[1] + degY
     *   S_k(z_p) > 0  only for  k = firstIndex[2] .. firstIndex[2] + degZ
     * 
     * for S = nodeS and
     * 
     *   S_i(x_p) > 0  only for  i = firstIndex[0] .. firstIndex[0] + degX - 1
     *   S_j(y_p) > 0  only for  j = firstIndex[1] .. firstIndex[1] + degY - 1
     *   S_k(z_p) > 0  only for  k = firstIndex[2] .. firstIndex[2] + degZ - 1
     * 
     * for S = cellS
    */
    amrex::GpuArray<int, GEMPIC_SPACEDIM> firstIndex;

    /**
     * @brief Values of cell-splines at the particle position
     * 
     * Description: 
     *  
     * this array is computed in the function update1DSplines,
     * it will store the values of the univariate cell-splines at the particle position:
     * 
     *   cellSplineVals[0][i] = cellS_{firstIndex[0]+i}(x_p)   for  i = 0 .. degX - 1
     *   cellSplineVals[1][j] = cellS_{firstIndex[1]+j}(y_p)   for  j = 0 .. degY - 1
     *   cellSplineVals[2][k] = cellS_{firstIndex[2]+k}(z_p)   for  k = 0 .. degZ - 1
     * 
     * where cellS is the univariate cell spline defined in the Class docstring
    */
    amrex::GpuArray<amrex::GpuArray<amrex::Real, maxDeg>, GEMPIC_SPACEDIM> cellSplineVals;

    /**
     * @brief Values of node-splines at the particle position
     * 
     * Description: 
     *  
     * this array is computed in the function update1DSplines,
     * it will store the values of the univariate cell-splines at the particle position:
     * 
     *   nodeSplineVals[0][i] = nodeS_{firstIndex[0]+i}(x_p)   for  i = 0 .. degX
     *   nodeSplineVals[1][j] = nodeS_{firstIndex[1]+j}(y_p)   for  j = 0 .. degY
     *   nodeSplineVals[2][k] = nodeS_{firstIndex[2]+k}(z_p)   for  k = 0 .. degZ
     * 
     * where nodeS is the univariate cell spline defined in the Class docstring
    */
    amrex::GpuArray<amrex::GpuArray<amrex::Real, maxDeg + 1>, GEMPIC_SPACEDIM> nodeSplineVals;

    /**
     * @brief Compute the first spline index and the offset for a given particle position
     * 
     * Description: 
     * Given a particle position x_p in the given dimension dim, this function computes: 
     *  - firstIndex[dim] : the index of the first contributing spline (see above):
     * and
     *  - xint = firstIndex[dim] + (deg+1)/2 - (x_p - plo)/dx 
     *    where deg is the degree of the node spline in the given dimension
     * 
     * xint is the normalized (1/dx) distance from the endpoint of the first non-zero node Spline at the particle location. 
     * It will be used in the function splineEval to compute the values of the node- and cell-splines at x_p
     * 
     * @tparam dim : the direction in which to compute: x = 0, y = 1, z = 2
     * 
     * @param position : position of the particle
     * @param plo : low end of the problem domain
     * @param dxInverse_dim : inverse of the cell size in the given direction
     * 
     * @return xint
    */
    template<int dim>
    AMREX_GPU_HOST_DEVICE amrex::Real initBSplinesAtPositions (const amrex::Real position, const amrex::Real plo, const amrex::Real dxInverse_dim)
    {
        amrex::Real radius = ( splineDegrees[dim] + 1. ) / 2.;
        amrex::Real normalized_pos = (position - plo)*dxInverse_dim;
        firstIndex[dim] = 1. + floor( normalized_pos - radius );
        amrex::Real xint = firstIndex[dim] + radius - normalized_pos;

        return xint;
    }

    /**
     * @brief sets the values of the contributing cell- and node-splines at the given particle position
     * 
     * Description: 
     * after this function call, the values of cellSplineVals[dim][i] and nodeSplineVals[dim][i] 
     * are set for i = 0 .. deg, according to their documentation above
     * 
     * @tparam dim : the direction in which to compute: x = 0, y = 1, z = 2
     * 
     * @param position Current position of the particle
     * @param plo : low end of the problem domain
     * @param dxInverse_dim : inverse of the cell size in the given direction
    */
    template<int dim>
    AMREX_GPU_HOST_DEVICE void update1DSplines (const amrex::Real position, const amrex::Real plo, const amrex::Real dxInverse_dim)
    {
        const amrex::Real xint = initBSplinesAtPositions<dim>(position, plo, dxInverse_dim);

        constexpr int deg{splineDegrees[dim]};
        splineEval<deg - 1, maxDeg>(xint, cellSplineVals[dim]);
        // Scaling according to documentation
        for (size_t i{0}; i < deg; i++) {
            cellSplineVals[dim][i] *= dxInverse_dim;
        }
        splineEval<deg, maxDeg + 1>(xint, nodeSplineVals[dim]);
    }

    /**
     * @brief Compute a dual dof (defined by the corresponding reduction operator tR{3-l}) of a vectorized particle spline
     * 
     * Description: 
     * This function computes a dual dof of the tensor-product particle spline
     * centered at the particle position encoded in this spline object, with a vector orientation for l = 1 or 2
     * 
     * Specifically, it computes 
     *  - tR{3-l}_ijk(partS(xx-xx_p))  if l = 0 or 3
     *  - tR{3-l}_ijk(e_dim * partS(xx-xx_p))  if l = 1 or 2
     * where 
     *   partS is the (tensor-product) particle spline defined above, centered at the particle position xx_p
     * 
     * According to the univariate formulas above:
     *   tR0_i(partS(x-x_p)) = cellS_i( x_p )
     * and  
     *   tR1_i(partS(x-x_p)) = nodeS_i( x_p )
     * the multivariate dofs are obtained as products of univariate node- and cell-splines evaluated at the particle position
     * 
     * @tparam form : index l of the associated primal form
     * @tparam dim : for l = 1 or 2, the vectorized direction of the particle slpine: x = 0, y = 1, z = 2
     * 
     * @param i The index of the x-dim basis function
     * @param j The index of the y-dim basis function
     * @param k The index of the z-dim basis function
     * 
     * @return base The value of the calculated coefficient
    */
    template<int form, int dim = -1>
    AMREX_GPU_HOST_DEVICE amrex::Real evalBSpline (int i, int j, int k) const
    {
        if constexpr(form == 0) {
                  double base = GEMPIC_D_MULT(nodeSplineVals[0][i],
                                              nodeSplineVals[1][j],
                                              nodeSplineVals[2][k]);
                  return base;
        }
        else if constexpr(form == 1) {
            if constexpr(dim == 0) {
                double base = GEMPIC_D_MULT(cellSplineVals[0][i],
                                            nodeSplineVals[1][j],
                                            nodeSplineVals[2][k]);
                return base;
            }
            else if constexpr(dim == 1) {
                double base = GEMPIC_D_MULT(nodeSplineVals[0][i],
                                            cellSplineVals[1][j],
                                            nodeSplineVals[2][k]);
                return base;
            }
            else if constexpr(dim == 2) {
                double base = GEMPIC_D_MULT(nodeSplineVals[0][i],
                                            nodeSplineVals[1][j],
                                            cellSplineVals[2][k]);
                return base;
            }
        }
        else if constexpr(form == 2) {
            if constexpr(dim == 0)
            {
                double base = GEMPIC_D_MULT(nodeSplineVals[0][i],
                                            cellSplineVals[1][j],
                                            cellSplineVals[2][k]);
                return base;
            }
            else if constexpr(dim == 1)
            {
                double base = GEMPIC_D_MULT(cellSplineVals[0][i],
                                            nodeSplineVals[1][j],
                                            cellSplineVals[2][k]);
                return base;
            }
            else if constexpr(dim == 2)
            {
                double base = GEMPIC_D_MULT(cellSplineVals[0][i],
                                            cellSplineVals[1][j],
                                            nodeSplineVals[2][k]);
                return base;
            }
        }
        else if constexpr(form == 3){
                  double base = GEMPIC_D_MULT(cellSplineVals[0][i],
                                              cellSplineVals[1][j],
                                              cellSplineVals[2][k]);
                  return base;
        }
        
        return 0;

        // error: Device code does not support exception handling
//        else {
//            throw std::runtime_error("The template parameter form must be 0, 1, 2 or 3.");
//        }
    }

    /**
     * @brief Compute the value of the spline-smoothed field at a particle position, from its l-form coefficients
     * 
     * Description: 
     * 
     * Given a primal l-form (l = 1 or 2) field with coefficients (edge or face dofs) F_ijk
     * this function computes the value of the spline-smoothed field at xx_p:
     * 
     * F^S(xx_p) = sum_d (e_d * sum_{ijk} F_ijk * tR{3-l}_ijk(e_d * partS(xx-xx_p)))
     * 
     * where:
     * - xx_p = (x_p, y_p, z_p) is the particle position encoded in the spline object
     * - e_d is the canonical basis vector along grid dimension d 
     * - tR{3-l} is the dual reduction operator for (3-l)-forms (here in 3D)
     * - partS is the tensor-product particle spline (approximating the Dirac mass), ie
     *     partS( xx-xx_p ) = prod_d (1/dx_d) * ref_S_{deg_d-1}((xx - xx_p)_d/dx_d)
     *   (see also the class documentation)
     * 
     * Examples: 
     * 
     * - for a (primal) 1-form electric field with (primal edge) dofs E_i, the function computes
     *   
     *   E^S(xx_p) = sum_{d = components} e_d ( sum_{i = primal edges / dual faces} E_i tR2_i(e_d * partS_p(xx)) )
     *   
     *   where tR2_i(.) is the reduction corresponding to a flux on the dual face i
     * 
     * - for a (primal) 2-form magnetic field with (primal face) dofs B_i, the function computes
     * 
     *   B^S(xx_p) = sum_{d = components} e_d ( sum_{i = primal faces / dual edges} B_i tR1_i(e_d * partS_p(xx)) )
     *   
     *   where tR1_i(.) is the reduction corresponding to a circulation on the dual edge i
     * 
     * for details about the algorithm see the documentation
     * https://gempic.pages.mpcdf.de/gempic/latex/particleMesh.html#smoothed-electromagnetic-fields
     *
     * @tparam vDim : number of field components (coincides with the velocity dimension)
     * @tparam form : index of the l form (1 or 2 for now)
     * 
     * @param fieldArray : array of field coefficients
     * 
     * @return fields : value of the spline-smoothed field at particle position
    */
    template<int vDim, int form>
    AMREX_GPU_HOST_DEVICE amrex::GpuArray<amrex::Real, vDim> evalSplineField (const amrex::GpuArray<amrex::Array4<amrex::Real>, vDim> fieldArray) const
    {
        bool xCond{false};
        bool yCond{false};
        bool zCond{false};

        amrex::GpuArray<amrex::Real, vDim> fields;
        for (int comp = 0; comp < vDim; comp++)
        {
            fields[comp] = 0.;
        }
        
        AMREX_D_PICK(
        int jj{0}; int kk{0};,
        int kk{0};,)
        GEMPIC_D_LOOP_BEGIN(for (int ii = 0; ii <= degX; ++ii),
                            for (int jj = 0; jj <= degY; ++jj),
                            for (int kk = 0; kk <= degZ; ++kk))
            const amrex::Dim3 cell_index =
                {AMREX_D_DECL(this->firstIndex[0] + ii, this->firstIndex[1] + jj, this->firstIndex[2] + kk)};

            // Ensure that we only access existing elements of cellSplineVals
            if constexpr (form == 1) {
                xCond = (ii < degX);
            } else if constexpr (form == 2) {
                xCond = AMREX_D_TERM(true, && (jj < degY), && (kk < degZ));
            }
            if (xCond)
            {
                const amrex::Real sCoeff = evalBSpline<form, 0>(ii, jj, kk);
                fields[0] +=
                    sCoeff *
                    (fieldArray[0])(cell_index.x, cell_index.y, cell_index.z, 0);
            }

            if constexpr(vDim > 1) {
                // Ensure that we only access existing elements of cellSplineVals
                if constexpr (form == 1) {
                    yCond = AMREX_D_TERM(true, && (jj < degY),);
                } else if constexpr (form == 2) {
                    yCond = AMREX_D_TERM((ii < degX),, && (kk < degZ));
                }
                if (yCond)
                {
                    const amrex::Real sCoeff = evalBSpline<form, 1>(ii, jj, kk);
                    fields[1] +=
                        sCoeff *
                        (fieldArray[1])(cell_index.x, cell_index.y, cell_index.z, 0);
                }
            }
            if constexpr(vDim == 3) {
                // Ensure that we only access existing elements of cellSplineVals
                if constexpr (form == 1) {
                    zCond = AMREX_D_TERM(true,, && (kk < degZ));
                } else if constexpr (form == 2) {
                    zCond = AMREX_D_TERM((ii < degX), && (jj < degY),);
                }
                if (zCond)
                {
                    const amrex::Real sCoeff = evalBSpline<form, 2>(ii, jj, kk);
                    fields[2] +=
                        sCoeff *
                        (fieldArray[2])(cell_index.x, cell_index.y, cell_index.z, 0);
                }
            }
        GEMPIC_D_LOOP_END

    return (fields);
    }

    /**
     * @brief Compute and store the values of the cardinal spline at regular values x+i
     * 
     * Description: 
     *   
     * given 0 <= x < 1, compute and store the values
     *   s[i] = ref_S_{degree}(x+i-(deg+1)/2) for i = 0 .. degree
     * where ref_S_{degree} is the centered cardinal B-spline of given degree with support [-(degree+1)/2, (degree+1)/2]
     * 
     * when called with 
     *  - degree = degX-1 
     *  - x = xint = firstIndex[dim] + (degX+1)/2 - (x_p - plo)/dx 
     * this function computes 
     *  s[i] = ref_S_{degX-1}(firstIndex[dim] + 1/2 - (x_p - plo)/dx + i) = dx * cellS_i(x_p)
     * 
     * when called with 
     *  - degree = degX 
     *  - x = xint = firstIndex[dim] + (degX+1)/2 - (x_p - plo)/dx 
     * this function computes 
     *  s[i] = ref_S_{degX}(firstIndex[dim] - (x_p - plo)/dx + i) = nodeS_i(x_p)
     *  
     * @tparam degree : degree of the cardinal spline
     * @tparam length : length of the stored array
     * 
     * @param x : the offset to integer positions at which the spline shall be evaluated
     * @param s : the array in which the results will be stored
    */
    template <int degree, int length>
    AMREX_GPU_HOST_DEVICE constexpr void splineEval (const amrex::Real x,
                                                     amrex::GpuArray<amrex::Real, length> &s) const
    {
        amrex::Real factor = 0;
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
    AMREX_GPU_HOST_DEVICE SplineWithPrimitive (const amrex::GpuArray<amrex::Real, GEMPIC_SPACEDIM> &position,
                        const amrex::GpuArray<amrex::Real, GEMPIC_SPACEDIM> &plo,
                        const amrex::GpuArray<amrex::Real, GEMPIC_SPACEDIM + 1> &dxInverse
                        ) : SplineBase<degX, degY, degZ>(position, plo, dxInverse)
    {
        firstIndexOld = {};

        primitiveNew = {};

        AMREX_D_TERM((update1DPrimitive<0>(position[0], plo[0], dxInverse[0]));,
                     (update1DPrimitive<1>(position[1], plo[1], dxInverse[1]));,
                     (update1DPrimitive<2>(position[2], plo[2], dxInverse[2])););
    }

    /**
     * @brief An array of indices
     * 
     * @details At the end of the function initBSplinesAtPositions, firstIndex is the index of the first field-coefficient corresponding to a non-zero spline at the particle location.
    */
    amrex::GpuArray<int, GEMPIC_SPACEDIM> firstIndexOld;

    using primitiveType = amrex::GpuArray<amrex::GpuArray<amrex::Real, SplineBase<degX, degY, degZ>::maxDeg>, GEMPIC_SPACEDIM>;
    /**
     * @brief primitive of the new spline
    */
    primitiveType primitiveNew;

    /**
     * @brief primitive of the old spline
    */
    primitiveType primitiveOld;

    /**
     * @brief sets the values of the contributing cell- and node-splines at the given particle position
     * 
     * Description: 
     * after this function call, the values of cellSplineVals[dim][i] and nodeSplineVals[dim][i] 
     * are set for i = 0 .. deg, according to their documentation above
     * 
     * @tparam dim : the direction in which to compute: x = 0, y = 1, z = 2
     * 
     * @param position Current position of the particle
     * @param plo : low end of the problem domain
     * @param dxInverse_dim : inverse of the cell size in the given direction
    */
    template<int dim>
    AMREX_GPU_HOST_DEVICE void update1DSplines (const amrex::Real position, const amrex::Real plo, const amrex::Real dxInverse_dim)
    {
        this->firstIndexOld[dim] = std::move(this->firstIndex[dim]);
        const amrex::Real xint = this->template initBSplinesAtPositions<dim> (position, plo, dxInverse_dim);

        constexpr int deg{SplineBase<degX, degY, degZ>::splineDegrees[dim]};
        this->template splineEval<deg - 1, SplineBase<degX, degY, degZ>::maxDeg>(xint, this->cellSplineVals[dim]);
        for (size_t i{0}; i < deg; i++) {
            this->cellSplineVals[dim][i] *= dxInverse_dim;
        }
        this->template splineEval<deg, SplineBase<degX, degY, degZ>::maxDeg + 1>(xint, this->nodeSplineVals[dim]);
    }

    /**
     * @brief Update the primitive by evaluating splines at the new particle position
     * 
     * Description: 
     * after this function call, the values of primitiveNew are moved to primitiveOld and new 
     * values of primitiveNew are set for i = 0 .. deg - 1
     * 
     * @tparam dim The direction in which to compute: x = 0, y = 1, z = 2
     * 
     * @param position Current position of the particle
     * @param plo (rename this one)
     * @param dxInverse_dim Inverse of the cell size in the given direction
    */
    template<int dim>
    AMREX_GPU_HOST_DEVICE void update1DPrimitive (const amrex::Real position, const amrex::Real plo, const amrex::Real dxInverse_dim)
    {
        const amrex::Real xint = this->template initBSplinesAtPositions<dim>(position, plo, dxInverse_dim);

        primitiveOld[dim] = std::move(primitiveNew[dim]);
        constexpr int deg{SplineBase<degX, degY, degZ>::splineDegrees[dim]};
        primitiveEval<deg - 1, SplineBase<degX, degY, degZ>::maxDeg>(xint, primitiveNew[dim]);
    }

    /**
     * @brief Compute the difference between primitives of the old spline and the new spline
     * 
     * @tparam dim The direction in which to compute: x = 0, y = 1, z = 2
     * 
     * @param dx Cell size in all three directions x, y and z (It's an array)
     * @param index Index at which to evaluate the primitive difference
     * 
     * @return primitiveDifference The difference between primitives of the old spline and the new spline
    */
    template<int dim>
    AMREX_GPU_HOST_DEVICE amrex::Real computePrimitiveDifference (const amrex::GpuArray<amrex::Real, GEMPIC_SPACEDIM> dx, const int index) const
    {
        constexpr int deg{SplineBase<degX, degY, degZ>::splineDegrees[dim]};
        // Size of support of difference of primitives
        const int local_size = std::abs(SplineBase<degX, degY, degZ>::firstIndex[dim] - this->firstIndexOld[dim]) + deg;

        amrex::Real primitiveDifference = 0;

        if (this->firstIndexOld[dim] < this->firstIndex[dim])
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
    AMREX_GPU_HOST_DEVICE constexpr void primitiveEval (const amrex::Real x,
                                                       amrex::GpuArray<amrex::Real, length> &s) const
    {
        amrex::Real factor = 0;
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
