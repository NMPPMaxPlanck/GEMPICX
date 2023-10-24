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
 *  - nx_i = (plo + i * dx) is the i-th grid node (in the relevant direction) 
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
     *  - firstIndex[dir] for each direction (the index of the first splines contributing to xx_p) 
     *  - cellSplineVals[dir] the array containing the non-zero values of cell splines at xx_p
     *  - nodeSplineVals[dir] the array containing the non-zero values of node splines at xx_p
     * 
     * @param position : position of the particle
     * @param plo : low end of the problem domain
     * @param dxInverse : inverse of the cell size in all three directions x, y and z
     
     * @details Additional details about dxInverse
     *  - dxInverse[xDir] == 1/dx
     *  - dxInverse[yDir] == 1/dy
     *  - dxInverse[zDir] == 1/dz
     *  - (Potentially dim < 3)
     *  - dxInverse[GEMPIC_SPACEDIM] == dxInverse[xDir] * .. * dxInverse[GEMPIC_SPACEDIM-1]
    */

    AMREX_GPU_HOST_DEVICE SplineBase (const amrex::GpuArray<amrex::Real, GEMPIC_SPACEDIM> &position,
               const amrex::GpuArray<amrex::Real, GEMPIC_SPACEDIM> &plo,
               const amrex::GpuArray<amrex::Real, GEMPIC_SPACEDIM + 1> &dxInverse)
    {
        AMREX_D_TERM((update1DSplines<xDir>(position[xDir], plo[xDir], dxInverse[xDir]));,
                     (update1DSplines<yDir>(position[yDir], plo[yDir], dxInverse[yDir]));,
                     (update1DSplines<zDir>(position[zDir], plo[zDir], dxInverse[zDir])););
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
     * @brief firstIndex[dir] is the index of the first spline(s) contributing at the particle position, along direction dir
     * 
     * Description: 
     *  
     * this array is computed in the function initBSplinesAtPositions: its values are such that we have 
     * 
     *   S_i(x_p) > 0  only for  i = firstIndex[xDir] .. firstIndex[xDir] + degX
     *   S_j(y_p) > 0  only for  j = firstIndex[yDir] .. firstIndex[yDir] + degY
     *   S_k(z_p) > 0  only for  k = firstIndex[zDir] .. firstIndex[zDir] + degZ
     * 
     * for S = nodeS and
     * 
     *   S_i(x_p) > 0  only for  i = firstIndex[xDir] .. firstIndex[xDir] + degX - 1
     *   S_j(y_p) > 0  only for  j = firstIndex[yDir] .. firstIndex[yDir] + degY - 1
     *   S_k(z_p) > 0  only for  k = firstIndex[zDir] .. firstIndex[zDir] + degZ - 1
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
     *   cellSplineVals[xDir][i] = cellS_{firstIndex[xDir]+i}(x_p)   for  i = 0 .. degX - 1
     *   cellSplineVals[yDir][j] = cellS_{firstIndex[yDir]+j}(y_p)   for  j = 0 .. degY - 1
     *   cellSplineVals[zDir][k] = cellS_{firstIndex[zDir]+k}(z_p)   for  k = 0 .. degZ - 1
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
     *   nodeSplineVals[xDir][i] = nodeS_{firstIndex[xDir]+i}(x_p)   for  i = 0 .. degX
     *   nodeSplineVals[yDir][j] = nodeS_{firstIndex[yDir]+j}(y_p)   for  j = 0 .. degY
     *   nodeSplineVals[zDir][k] = nodeS_{firstIndex[zDir]+k}(z_p)   for  k = 0 .. degZ
     * 
     * where nodeS is the univariate cell spline defined in the Class docstring
    */
    amrex::GpuArray<amrex::GpuArray<amrex::Real, maxDeg + 1>, GEMPIC_SPACEDIM> nodeSplineVals;

    /**
     * @brief Compute the first spline index and the offset for a given particle position
     * 
     * Description: 
     * Given a particle position x_p in the given direction dir, this function computes: 
     *  - firstIndex[dir] : the index of the first contributing spline (see above):
     * and
     *  - refOffset = firstIndex[dir] + (deg+1)/2 - (x_p - plo)/dx 
     *    where deg is the degree of the node spline in the given direction
     * 
     * 0 < refOffset ≤ 1  is the normalized (1/dx) distance between the particle position and the last point in the support of the first contributing spline (of both node and cell type).
     * It will be used in the function splineEval to compute the values of the node- and cell-splines at x_p
     * 
     * @tparam dir : the direction in which to compute: x = 0, y = 1, z = 2
     * 
     * @param position : position of the particle
     * @param plo : low end of the problem domain
     * @param dxInverse_dir : inverse of the cell size in the given direction
     * 
     * @return refOffset
    */
    template<Direction dir>
    AMREX_GPU_HOST_DEVICE amrex::Real initBSplinesAtPositions (const amrex::Real position, const amrex::Real plo, const amrex::Real dxInverse_dir)
    {
        amrex::Real radius = ( splineDegrees[dir] + 1. ) / 2.;
        amrex::Real normalized_pos = (position - plo)*dxInverse_dir;
        firstIndex[dir] = 1. + floor( normalized_pos - radius );
        amrex::Real refOffset = firstIndex[dir] + radius - normalized_pos;

        return refOffset;
    }

    /**
     * @brief sets the values of the contributing cell- and node-splines at the given particle position
     * 
     * Description: 
     * after this function call, the values of cellSplineVals[dir][i] and nodeSplineVals[dir][i] 
     * are set for i = 0 .. deg, according to their documentation above
     * 
     * @tparam dir : the direction in which to compute: xDir, yDir, zDir
     * 
     * @param position Current position of the particle
     * @param plo : low end of the problem domain
     * @param dxInverse_dir : inverse of the cell size in the given direction
    */
    template<Direction dir>
    AMREX_GPU_HOST_DEVICE void update1DSplines (const amrex::Real position, const amrex::Real plo, const amrex::Real dxInverse_dir)
    {
        const amrex::Real refOffset = initBSplinesAtPositions<dir>(position, plo, dxInverse_dir);

        constexpr int deg{splineDegrees[dir]};
        splineEval<deg - 1, maxDeg>(refOffset, cellSplineVals[dir]);
        // Scaling according to documentation
        for (size_t i{0}; i < deg; i++) {
            cellSplineVals[dir][i] *= dxInverse_dir;
        }
        splineEval<deg, maxDeg + 1>(refOffset, nodeSplineVals[dir]);
    }

    /**
     * @brief Compute a dual dof (defined by the corresponding reduction operator tR{3-l}) of a vectorized particle spline
     * 
     * Description: 
     * This function computes a dual dof of the tensor-product particle spline
     * centered at the particle position encoded in this spline object, with a vector orientation for l = 1 or 2
     *Field::DualOneForm 
     * Specifically, it computes 
     *  - tR{3-l}_ijk(partS(xx-xx_p))  if l = 0 or 3
     *  - tR{3-l}_ijk(e_dir * partS(xx-xx_p))  if l = 1 or 2
     * where 
     *   partS is the (tensor-product) particle spline defined above, centered at the particle position xx_p
     * 
     * According to the univariate formulas above:
     *   tR0_i(partS(x-x_p)) = cellS_i( x_p )
     * and  
     *   tR1_i(partS(x-x_p)) = nodeS_i( x_p )
     * the multivariate dofs are obtained as products of univariate node- and cell-splines evaluated at the particle position
     * 
     * @tparam form : index l of the associated dual form
     * @tparam dir : for l = 1 or 2, the vectorized direction of the particle slpine: x = 0, y = 1, z = 2
     * 
     * @param i The index of the x-dir basis function
     * @param j The index of the y-dir basis function
     * @param k The index of the z-dir basis function
     * 
     * @return base The value of the calculated coefficient
    */
    template<Field form, Direction dir>
    AMREX_GPU_HOST_DEVICE amrex::Real evalBSpline (int i, int j, int k) const
    {
        static_assert(form == Field::DualOneForm || form == Field::DualTwoForm,
                      "Function called with primal form, or superfluous direction supplied.");
        if constexpr(form == Field::DualTwoForm) {
            if constexpr(dir == xDir) {
                return GEMPIC_D_MULT(cellSplineVals[xDir][i],
                                     nodeSplineVals[yDir][j],
                                     nodeSplineVals[zDir][k]);
            }
            else if constexpr(dir == yDir) {
                return GEMPIC_D_MULT(nodeSplineVals[xDir][i],
                                     cellSplineVals[yDir][j],
                                     nodeSplineVals[zDir][k]);
            }
            else if constexpr(dir == zDir) {
                return GEMPIC_D_MULT(nodeSplineVals[xDir][i],
                                     nodeSplineVals[yDir][j],
                                     cellSplineVals[zDir][k]);
            }
        }
        else if constexpr(form == Field::DualOneForm) {
            if constexpr(dir == xDir)
            {
                return GEMPIC_D_MULT(nodeSplineVals[xDir][i],
                                     cellSplineVals[yDir][j],
                                     cellSplineVals[zDir][k]);
            }
            else if constexpr(dir == yDir)
            {
                return GEMPIC_D_MULT(cellSplineVals[xDir][i],
                                     nodeSplineVals[yDir][j],
                                     cellSplineVals[zDir][k]);
            }
            else if constexpr(dir == zDir)
            {
                return GEMPIC_D_MULT(cellSplineVals[xDir][i],
                                     cellSplineVals[yDir][j],
                                     nodeSplineVals[zDir][k]);
            }
        }
    }

    template<Field form>
    AMREX_GPU_HOST_DEVICE amrex::Real evalBSpline (int i, int j, int k) const
    {
        static_assert(form == Field::DualZeroForm || form == Field::DualThreeForm, 
                      "Function called with primal form, or no direction supplied.");
        if constexpr(form == Field::DualThreeForm) {
            return GEMPIC_D_MULT(nodeSplineVals[xDir][i],
                                 nodeSplineVals[yDir][j],
                                 nodeSplineVals[zDir][k]);
        }
        else if constexpr(form == Field::DualZeroForm){
            return GEMPIC_D_MULT(cellSplineVals[xDir][i],
                                 cellSplineVals[yDir][j],
                                 cellSplineVals[zDir][k]);
        }
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
     * - e_d is the canonical basis vector along grid direction d 
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
    template<Field form, unsigned int vDim>
    AMREX_GPU_HOST_DEVICE amrex::GpuArray<amrex::Real, vDim> evalSplineField (const amrex::GpuArray<amrex::Array4<amrex::Real>, vDim> fieldArray) const
    {
        static_assert(form == Field::PrimalOneForm || form == Field::PrimalTwoForm, "Attempting smoothing of incorrect field type.");
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
                {AMREX_D_DECL(this->firstIndex[xDir] + ii, this->firstIndex[yDir] + jj, this->firstIndex[zDir] + kk)};

            // Ensure that we only access existing elements of cellSplineVals
            if constexpr (form == Field::PrimalOneForm) {
                xCond = (ii < degX);
            } else if constexpr (form == Field::PrimalTwoForm) {
                xCond = AMREX_D_TERM(true, && (jj < degY), && (kk < degZ));
            }
            if (xCond)
            {
                const amrex::Real sCoeff = evalBSpline<complementForm<form>(), xDir>(ii, jj, kk);
                fields[xDir] +=
                    sCoeff *
                    (fieldArray[xDir])(cell_index.x, cell_index.y, cell_index.z, 0);
            }

            if constexpr(vDim > 1) {
                // Ensure that we only access existing elements of cellSplineVals
                if constexpr (form == Field::PrimalOneForm) {
                    yCond = AMREX_D_TERM(true, && (jj < degY),);
                } else if constexpr (form == Field::PrimalTwoForm) {
                    yCond = AMREX_D_TERM((ii < degX),, && (kk < degZ));
                }
                if (yCond)
                {
                    const amrex::Real sCoeff = evalBSpline<complementForm<form>(), yDir>(ii, jj, kk);
                    fields[yDir] +=
                        sCoeff *
                        (fieldArray[yDir])(cell_index.x, cell_index.y, cell_index.z, 0);
                }
            }
            if constexpr(vDim == 3) {
                // Ensure that we only access existing elements of cellSplineVals
                if constexpr (form == Field::PrimalOneForm) {
                    zCond = AMREX_D_TERM(true,, && (kk < degZ));
                } else if constexpr (form == Field::PrimalTwoForm) {
                    zCond = AMREX_D_TERM((ii < degX), && (jj < degY),);
                }
                if (zCond)
                {
                    const amrex::Real sCoeff = evalBSpline<complementForm<form>(), zDir>(ii, jj, kk);
                    fields[zDir] +=
                        sCoeff *
                        (fieldArray[zDir])(cell_index.x, cell_index.y, cell_index.z, 0);
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
     * given 0 <= xRef < 1, compute and store the values
     *   s[i] = ref_S_{degree}(x+i-(deg+1)/2) for i = 0 .. degree
     * where ref_S_{degree} is the centered cardinal B-spline of given degree with support [-(degree+1)/2, (degree+1)/2]
     * 
     * when called with 
     *  - degree = degX-1 
     *  - xRef = refOffset = firstIndex[dir] + (degX+1)/2 - (x_p - plo)/dx 
     * this function computes 
     *  s[i] = ref_S_{degX-1}(firstIndex[dir] + 1/2 - (x_p - plo)/dx + i) = dx * cellS_i(x_p)
     * 
     * when called with 
     *  - degree = degX 
     *  - xRef = refOffset = firstIndex[dir] + (degX+1)/2 - (x_p - plo)/dx 
     * this function computes 
     *  s[i] = ref_S_{degX}(firstIndex[dir] - (x_p - plo)/dx + i) = nodeS_i(x_p)
     *  
     * @tparam degree : degree of the cardinal spline
     * @tparam length : length of the stored array
     * 
     * @param xRef : the offset to integer positions at which the spline shall be evaluated
     * @param s : the array in which the results will be stored
    */
    template <int degree, unsigned int length>
    AMREX_GPU_HOST_DEVICE constexpr void splineEval (const amrex::Real xRef,
                                                     amrex::GpuArray<amrex::Real, length> &s) const
    {
        amrex::Real factor = 0;
        switch (degree)
        {
            case 0:
                s[0] = 1.;
                break;
            case 1:
                s[0] = xRef;
                s[1] = 1. - xRef;
                break;
            case 2:
                s[0] = (xRef * xRef) * 0.5;
                s[1] = (1. - xRef) * xRef + 0.5;
                s[2] = ((xRef -2.) * xRef + 1.) * 0.5;
                break;
            case 3:
                factor = 1./6.;
                s[0] = (xRef * xRef * xRef) * factor;
                s[1] = (((-xRef + 1) * xRef + 1.) * 3. * xRef + 1.) * factor;
                s[2] = ((3. * xRef - 6.) * xRef * xRef + 4.) * factor;
                // s[3] = pow((1. - xRef), 3) * factor;
                s[3] = (((-xRef + 3.) * xRef - 3.) * xRef + 1.) * factor;
                break;
            case 4:
                factor = 1./24.;
                s[0] = (xRef * xRef * xRef * xRef) * factor;
                s[1] = ((((-xRef + 1.) * 4. * xRef + 6.) * xRef + 4.) * xRef + 1.) * factor;
                s[2] = ((((6. * xRef - 12.) * xRef - 6.) * xRef + 12.) * xRef + 11.) * factor;
                s[3] = ((((-4. * xRef + 12.) * xRef - 6.) * xRef - 12.) * xRef + 11.) * factor;
                // s[4] = pow(1. - xRef, 4) * factor;
                s[4] = ((((xRef - 4.) * xRef + 6.) * xRef - 4.) * xRef + 1.) * factor;
                break;
            case 5:
                factor = 1./120.;
                
                s[0] = (xRef * xRef * xRef * xRef * xRef) * factor;
                s[1] = (((((-xRef + 1.) * xRef + 2.) * xRef + 2.) * xRef + 1.) * 5. * xRef + 1.) * factor;
                s[2] = (((((xRef - 2.) * xRef - 2.) * xRef + 2.) * xRef + 5.) * 10. * xRef + 26.) * factor;
                s[3] = (((-xRef + 3.) * xRef * xRef - 6.) * xRef * 10. * xRef + 66.) * factor;
                s[4] = (((((xRef - 4.) * xRef + 4.) * xRef + 4.) * xRef - 10.) * 5. * xRef + 26.) * factor;
                s[5] = (((((-xRef + 5.) * xRef - 10.) * xRef + 10.) * xRef - 5.) * xRef + 1.) * factor;
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
     *  - dxInverse[xDir] == 1/dx
     *  - dxInverse[yDir] == 1/dy
     *  - dxInverse[zDir] == 1/dz
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

        AMREX_D_TERM((update1DPrimitive<xDir>(position[xDir], plo[xDir], dxInverse[xDir]));,
                     (update1DPrimitive<yDir>(position[yDir], plo[yDir], dxInverse[yDir]));,
                     (update1DPrimitive<zDir>(position[zDir], plo[zDir], dxInverse[zDir])););
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
     * after this function call, the values of cellSplineVals[dir][i] and nodeSplineVals[dir][i] 
     * are set for i = 0 .. deg, according to their documentation above
     * 
     * @tparam dir : the direction in which to compute: x = 0, y = 1, z = 2
     * 
     * @param position Current position of the particle
     * @param plo : low end of the problem domain
     * @param dxInverse_dir : inverse of the cell size in the given direction
    */
    template<Direction dir>
    AMREX_GPU_HOST_DEVICE void update1DSplines (const amrex::Real position, const amrex::Real plo, const amrex::Real dxInverse_dir)
    {
        this->firstIndexOld[dir] = std::move(this->firstIndex[dir]);
        const amrex::Real refOffset = this->template initBSplinesAtPositions<dir> (position, plo, dxInverse_dir);

        constexpr int deg{SplineBase<degX, degY, degZ>::splineDegrees[dir]};
        this->template splineEval<deg - 1, SplineBase<degX, degY, degZ>::maxDeg>(refOffset, this->cellSplineVals[dir]);
        for (size_t i{0}; i < deg; i++) {
            this->cellSplineVals[dir][i] *= dxInverse_dir;
        }
        this->template splineEval<deg, SplineBase<degX, degY, degZ>::maxDeg + 1>(refOffset, this->nodeSplineVals[dir]);
    }

    /**
     * @brief Update the primitive by evaluating splines at the new particle position
     * 
     * Description: 
     * after this function call, the values of primitiveNew are moved to primitiveOld and new 
     * values of primitiveNew are set for i = 0 .. deg - 1
     * 
     * @tparam dir The direction in which to compute: x = 0, y = 1, z = 2
     * 
     * @param position Current position of the particle
     * @param plo (rename this one)
     * @param dxInverse_dir Inverse of the cell size in the given direction
    */
    template<Direction dir>
    AMREX_GPU_HOST_DEVICE void update1DPrimitive (const amrex::Real position, const amrex::Real plo, const amrex::Real dxInverse_dir)
    {
        const amrex::Real refOffset = this->template initBSplinesAtPositions<dir>(position, plo, dxInverse_dir);

        primitiveOld[dir] = std::move(primitiveNew[dir]);
        constexpr int deg{SplineBase<degX, degY, degZ>::splineDegrees[dir]};
        primitiveEval<deg - 1>(refOffset, primitiveNew[dir]);
    }

    /**
     * @brief Compute the difference between primitives of the old spline and the new spline
     * 
     * @tparam dir The direction in which to compute: x = 0, y = 1, z = 2
     * 
     * @param dx Cell size in all three directions x, y and z (It's an array)
     * @param index Index at which to evaluate the primitive difference
     * 
     * @return primitiveDifference The difference between primitives of the old spline and the new spline
    */
    template<Direction dir>
    AMREX_GPU_HOST_DEVICE amrex::Real computePrimitiveDifference (const amrex::GpuArray<amrex::Real, GEMPIC_SPACEDIM> dx, const int index) const
    {
        constexpr int deg{SplineBase<degX, degY, degZ>::splineDegrees[dir]};
        // Size of support of difference of primitives
        const int local_size = std::abs(SplineBase<degX, degY, degZ>::firstIndex[dir] - this->firstIndexOld[dir]) + deg;

        amrex::Real primitiveDifference = 0;

        if (this->firstIndexOld[dir] < this->firstIndex[dir])
        {   
            if (index < deg) {
                primitiveDifference = this->primitiveOld[dir][index] * dx[dir];
            }
            if (index >= deg && index < local_size) {
                primitiveDifference = dx[dir];
            }
            if (index >= local_size - deg && index <= local_size - 1) {
                primitiveDifference -= this->primitiveNew[dir][index + deg - local_size] * dx[dir];
            }
        }
        else
        {
            if (index < deg) { 
                primitiveDifference = -this->primitiveNew[dir][index] * dx[dir];
            }
            if (index >= deg && index < local_size) {
                primitiveDifference = -dx[dir];
            }
            if (index >= local_size - deg && index <= local_size - 1) {
                primitiveDifference += this->primitiveOld[dir][index + deg - local_size] * dx[dir];
            }
        }

        return primitiveDifference;
    }

    /**
     * @brief Compute and store the values of the primitive at regular values x+i
     * 
     * @tparam degree : degree of the primitive
     * @tparam length : length of the stored array
     * 
     * @param xRef : the offset to integer positions at which the spline shall be evaluated
     * @param s : the array in which the results will be stored
    */
    template <int degree, unsigned int length>
    AMREX_GPU_HOST_DEVICE constexpr void primitiveEval (const amrex::Real xRef,
                                                       amrex::GpuArray<amrex::Real, length> &s) const
    {
        amrex::Real factor = 0;
        switch (degree)
        {
            case -1:
                s[0] = 1;
                break;
            case 0:
                s[0] = xRef;
                break;
            case 1:
                s[0] = (xRef * xRef) * 0.5;
                s[1] = ((-xRef + 2.) * xRef + 1) * 0.5;
                break;
            case 2:
                factor = (1./6.);
                s[0] = (xRef * xRef * xRef) * factor;
                s[1] = (((-2. * xRef + 3) * xRef + 3.) * xRef + 1.) * factor;
                s[2] = (((-xRef + 3.) * xRef - 1.) * xRef - 1.) * factor + 1.;
                break;
            case 3:
                factor = (1./24.);
                s[0] = (xRef * xRef * xRef * xRef) * factor;
                s[1] = -pow(xRef, 4) / 8. + pow(xRef, 3) / 6. + pow(xRef, 2) / 4. + xRef / 6. + 1. / 24.;
                s[2] = pow(xRef, 4) / 8. - pow(xRef, 3) / 3. + xRef * 2. / 3. + 0.5;
                // s[3] = -pow((1. - xRef), 4) * factor + 1.;
                s[3] = -((((xRef - 4.) * xRef + 6) * xRef - 4.) * xRef + 1) * factor + 1.;
                break;
            case 4:
                factor = (1./120.);
                s[0] = (xRef * xRef * xRef * xRef * xRef) * factor;
                s[1] = (((((-4. * xRef + 5.) * xRef + 10.) * xRef + 10.) * xRef + 5.) * xRef + 1.) * factor;
                s[2] = (((((6. * xRef - 15.) * xRef - 10.) * xRef + 30.) * xRef + 55.) * xRef + 27.) * factor;
                s[3] = (((((-4. * xRef + 15.) * xRef - 10.) * xRef - 30.) * xRef + 55.) * xRef + 93.) * factor;
                //s[4] = ((((xRef - 5.) * xRef + 8) * x*x - 5.) * xRef + 1.) * factor + 1.;
                s[4] = -(((((-xRef + 5.) * xRef - 10.) * xRef + 10.) * xRef - 5.) * xRef + 1.) * factor + 1.;
                break;
            case 5:
                factor = (1./720.);
                s[0] = pow(xRef, 6) * factor;
                s[1] =
                    ((((((-5. * xRef + 6.) * xRef + 15.) * xRef + 20.) * xRef + 15.) * xRef + 6.) * xRef + 1.) * factor;
                s[2] =
                    ((((((10. * xRef - 24.) * xRef - 30.) * xRef + 40.) * xRef + 150.) * xRef + 156.) * xRef + 58.) * factor;
                s[3] = ((((-10. * xRef + 36.) * xRef * xRef - 120.) * xRef * xRef + 396.) * xRef + 360.) * factor;
                s[4] =
                    ((((((5. * xRef - 24.) * xRef + 30.) * xRef + 40.) * xRef - 150.) * xRef + 156.) * xRef + 662.) * factor;
                s[5] = ((((((-xRef + 6.) * xRef - 15.) * xRef + 20.) * xRef - 15.) * xRef + 6.) * xRef + 719.) * factor;
                break;
        }
    }
};

}
#endif
