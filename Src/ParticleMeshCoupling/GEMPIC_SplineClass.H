#ifndef SPLINE_CLASS_
#define SPLINE_CLASS_

#include "GEMPIC_BSpline.H"

namespace Gempic::ParticleMeshCoupling
{
namespace Impl
{
enum class PrimitiveDifferenceRegion
{
    firstSpline,
    noSplines,
    bothSplines,
    lastSpline,
};
} //namespace Impl

/**
 * @brief Implementation of the B-Splines based on the description given
 *   in @verbatim embed:rst:inline :ref:`Particle-Mesh Coupling` @endverbatim
 *
 *
 * @details Calculate and store "node", "cell" spline values for every
 *          degree of freedom which has non vanishing support for a given degree at
 *          \f$(x_p,y_p,z_p)\f$, which is an arbitrary reference position
 *          (e.g. particle coordinate).
 *          The implemented tensor product spline function is
 *          @f[
 *          S^{degX}(x - x_p)S^{degY}(y - y_p)S^{degY}(z - z_p).
 *          @f]
 *          which has a symmetric support around the reference position.
 *
 *          For every Direction which GEMPIC was compiled for an object of the class
 *          stores the values:
 *
 *          - \f$ \{\mathrm{S}^{\mathrm{cell}}_{i}(x_p)\} \f$ in @link m_cellSplineVals @endlink
 *          - \f$ \{\mathrm{S}^{\mathrm{node}}_{i}(x_p)\} \f$ in @link m_nodeSplineVals @endlink
 *
 *          where \f$ \{i\} \f$ are the sets of the degrees of which contribute to the
 *          particle position \f$x_p\f$ with a non vanishing support. More details on the
 *          support of "cell" and "node" splines can be found in the documentation page
 *          @verbatim embed:rst:inline :ref:`Particle-Mesh Coupling` @endverbatim.
 *
 *         Additionally, this class implements the reduction operators \f$\tilde{\mathcal{R}} \f$
 *         defined in @verbatim embed:rst:inline :ref:`Particle-Mesh Coupling` @endverbatim
 *
 *
 * ToDo: Extract the mappings between different spaces in the
 * deRahm complex. Those are bloating the class.
 *
 * @tparam degX : degree of the node-spline in x-direction
 * @tparam degY : degree of the node-spline in y-direction
 * @tparam degZ : degree of the node-spline in z-direction
 */
template <int degX, int degY, int degZ>
class SplineBase
{
public:
    /**
     * @brief Constructor calculates the contribution of spline values at the degrees of freedom
     *
     * @param position : position of the particle
     * @param plo : low end of the problem domain
     * @param dxInverse : inverse of the cell size in all three directions x, y and z

     * @details Additional details about dxInverse
     *  - dxInverse[Direction::xDir] == 1/dx
     *  - dxInverse[Direction::yDir] == 1/dy
     *  - dxInverse[Direction::zDir] == 1/dz
    */
    AMREX_GPU_HOST_DEVICE SplineBase (const amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> &position,
                                      const amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> &plo,
                                      const amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> &dxInverse)
    {
        //BL_PROFILE("Gempic::ParticleMeshCoupling::SplineBase::SplineBase()");
        AMREX_D_TERM(
            (update_1d_splines<Direction::xDir>(position[xDir], plo[xDir], dxInverse[xDir]));
            , (update_1d_splines<Direction::yDir>(position[yDir], plo[yDir], dxInverse[yDir]));
            , (update_1d_splines<Direction::zDir>(position[zDir], plo[zDir], dxInverse[zDir])););
    }

    /**
     * @brief An array of spline degrees stored for convenience. Some may be unused in 1D or 2D
     */
    static constexpr int s_splineDegrees[3] = {degX, degY, degZ};

    /**
     * @brief Max spline degree stored for convenience
     */
    static constexpr int s_maxDeg{
        AMREX_D_PICK(degX, std::max(degX, degY), std::max(std::max(degX, degY), degZ))};
    /**
     * @brief firstIndex[dir] is the index of the first spline(s) contributing at the particle
     * position, along direction dir
     *
     * @details
     *
     * - Contribution for node splines:
     *
     *   \f$ S_i(x) > 0 \f$ for i = firstIndex[Direction] .. firstIndex[Direction] + deg
     *
     * - Contribution for cell splines
     *
     *   \f$ S_i(x) > 0 \f$ for i = firstIndex[Direction] .. firstIndex[Direction] + deg - 1
     */
    amrex::GpuArray<int, AMREX_SPACEDIM> m_firstIndex;

    /**
     * @brief Values of a cell spline at the degrees of freedom to a particle position \f$ x_p \f$
     *
     * @details m_cellSplineVals[Direction][i] = \f$
     * \mathrm{S}^{\mathrm{cell}}_{\mathrm{firstIndex[Direction]}+i}(x_p) \f$ for  i = 0 .. deg - 1
     * when called with
     *  - degree = degX
     *  - xRef = refOffset = firstIndex[dir] + (degX+1)/2 - (\f$ x_p \f$ - plo)/dx
     * this class computes
     *  s[i] = \f$ \mathrm{S}^{degX}\f$(firstIndex[dir] - (\f$ x_p \f$ - plo)/dx + i) = \f$
     * \mathrm{S}^{\mathrm{node}}_i(x_p) \f$
     */
    amrex::GpuArray<amrex::GpuArray<amrex::Real, s_maxDeg>, AMREX_SPACEDIM> m_cellSplineVals;

    /**
     * @brief Values of a node spline at the degrees of freedom to a particle position \f$ x_p \f$
     *
     * @details nodeSplineVals[Direction][i] = \f$
     * \mathrm{S}^{node}_{\mathrm{firstIndex[Direction]}+i}(x_p) \f$ for  i = 0 .. deg
     * when called with
     *  - degree = degX-1
     *  - xRef = refOffset = firstIndex[dir] + (degX+1)/2 - (\f$ x_p \f$ - plo)/dx
     * this class computes
     *  s[i] = \f$ S^{degX-1}\f$(firstIndex[dir] + 1/2 - (\f$ x_p \f$ - plo)/dx + i)
     * = dx * \f$ \mathrm{S}^{\mathrm{cell}}_{i}(x_p) \f$
     */
    amrex::GpuArray<amrex::GpuArray<amrex::Real, s_maxDeg + 1>, AMREX_SPACEDIM> m_nodeSplineVals;

    /**
     * //ToDo either change members and do not return anything or return something and be const.
     * @brief Compute the first spline index and the offset for a given particle position
     *
     * @details
     * Given a particle position \f$ x_p \f$ in the given direction dir, this function computes:
     *  - firstIndex[dir] : the index of the first contributing spline (see above):
     * and
     *  - refOffset = firstIndex[dir] + (deg+1)/2 - (\f$ x_p \f$ - plo)/dx
     *    where deg is the degree of the node spline in the given direction
     *
     * 0 < refOffset ≤ 1  is the normalized (1/dx) distance between the particle position and the
     * last point in the support of the first contributing spline (of both node and cell type). It
     * will be used in the function splineEval to compute the values of the node- and cell-splines
     * at \f$ x_p \f$
     *
     * @tparam dir : the direction in which to compute
     *
     * @param position : position of the particle
     * @param plo : low end of the problem domain
     * @param dxInverseDir : inverse of the cell size in the given direction
     *
     * @return refOffset
     */
    template <Direction dir>
    AMREX_GPU_HOST_DEVICE amrex::Real init_b_splines_at_positions (const amrex::Real position,
                                                                   const amrex::Real plo,
                                                                   const amrex::Real dxInverseDir)
    {
        amrex::Real radius = (s_splineDegrees[dir] + 1.) / 2.;
        amrex::Real normalizedPos = (position - plo) * dxInverseDir;
        m_firstIndex[dir] = 1. + floor(normalizedPos - radius);
        amrex::Real refOffset = m_firstIndex[dir] + radius - normalizedPos;

        return refOffset;
    }

    /**
     * @brief Set the values of the contributing cell- and node-splines for the position \f$ x_p \f$
     *
     * @details Set cellSplineVals and nodeSplineVals for the direction according to their
     * documentation
     *
     * @tparam dir : the direction in which to compute: Direction::xDir, Direction::yDir,
     * Direction::zDir
     *
     * @param position Current position of the particle
     * @param plo : low end of the problem domain
     * @param dxInverseDir : inverse of the cell size in the given direction
     */
    template <Direction dir>
    AMREX_GPU_HOST_DEVICE void update_1d_splines (const amrex::Real position,
                                                  const amrex::Real plo,
                                                  const amrex::Real dxInverseDir)
    {
        const amrex::Real refOffset = init_b_splines_at_positions<dir>(position, plo, dxInverseDir);

        constexpr int deg{s_splineDegrees[dir]};
        spline_eval<deg - 1, s_maxDeg>(refOffset, m_cellSplineVals[dir]);
        // Scaling according to documentation
        for (size_t i{0}; i < deg; i++)
        {
            m_cellSplineVals[dir][i] *= dxInverseDir;
        }
        spline_eval<deg, s_maxDeg + 1>(refOffset, m_nodeSplineVals[dir]);
    }

    /**
     * @brief Compute a dual dof (defined by the corresponding reduction operator \f$ t_{R\{3-l\}})
     *\f$ of a vectorized particle spline
     *
     * Description:
     * This function computes a dual dof of the tensor-product particle spline
     * centered at the particle position encoded in this spline object, with a vector orientation
     *for l = 1 or 2 Field::DualOneForm Specifically, it computes
     *  - \f$ t_{R\{3-l\}_{ijk}}(\mathrm{partS}(xx-xx_p)) \f$ if l = 0 or 3
     *  - \f$ t_{R\{3-l\}_{ijk}}(e_{dir} * \mathrm{partS}(xx-xx_p)) \f$ if l = 1 or 2
     * where
     *   partS is the (tensor-product) particle spline defined above, centered at the particle
     *position \f$ xx_p \f$
     *
     * According to the univariate formulas above:
     *   \f$ t_{R0_i}(\mathrm{partS}(x-x_p)) = \mathrm{cellS}_i( x_p ) \f$
     * and
     *   \f$ t_{R1_i}(\mathrm{partS}(x-x_p)) = \mathrm{nodeS}_i( x_p ) \f$
     * the multivariate dofs are obtained as products of univariate node- and cell-splines evaluated
     *at the particle position
     *
     * @tparam form : index l of the associated dual form
     * @tparam dir : for l = 1 or 2, the vectorized direction of the particle slpine: x = 0, y = 1,
     *z = 2
     *
     * @param i The index of the x-dir basis function
     * @param j The index of the y-dir basis function
     * @param k The index of the z-dir basis function
     *
     * @return base The value of the calculated coefficient
     */
    template <Field form, Direction dir>
    AMREX_GPU_HOST_DEVICE amrex::Real eval_b_spline (int i, int j, int k) const
    {
        static_assert(form == Field::DualOneForm || form == Field::DualTwoForm,
                      "Function called with primal form, or superfluous direction supplied.");
        if constexpr (form == Field::DualTwoForm)
        {
            if constexpr (dir == Direction::xDir)
            {
                return GEMPIC_D_MULT(m_cellSplineVals[Direction::xDir][i],
                                     m_nodeSplineVals[Direction::yDir][j],
                                     m_nodeSplineVals[Direction::zDir][k]);
            }
            else if constexpr (dir == Direction::yDir)
            {
                return GEMPIC_D_MULT(m_nodeSplineVals[Direction::xDir][i],
                                     m_cellSplineVals[Direction::yDir][j],
                                     m_nodeSplineVals[Direction::zDir][k]);
            }
            else if constexpr (dir == Direction::zDir)
            {
                return GEMPIC_D_MULT(m_nodeSplineVals[Direction::xDir][i],
                                     m_nodeSplineVals[Direction::yDir][j],
                                     m_cellSplineVals[Direction::zDir][k]);
            }
        }
        else if constexpr (form == Field::DualOneForm)
        {
            if constexpr (dir == Direction::xDir)
            {
                return GEMPIC_D_MULT(m_nodeSplineVals[Direction::xDir][i],
                                     m_cellSplineVals[Direction::yDir][j],
                                     m_cellSplineVals[Direction::zDir][k]);
            }
            else if constexpr (dir == Direction::yDir)
            {
                return GEMPIC_D_MULT(m_cellSplineVals[Direction::xDir][i],
                                     m_nodeSplineVals[Direction::yDir][j],
                                     m_cellSplineVals[Direction::zDir][k]);
            }
            else if constexpr (dir == Direction::zDir)
            {
                return GEMPIC_D_MULT(m_cellSplineVals[Direction::xDir][i],
                                     m_cellSplineVals[Direction::yDir][j],
                                     m_nodeSplineVals[Direction::zDir][k]);
            }
        }
    }

    template <Field form>
    AMREX_GPU_HOST_DEVICE amrex::Real eval_b_spline (int i, int j, int k) const
    {
        static_assert(form == Field::DualZeroForm || form == Field::DualThreeForm,
                      "Function called with primal form, or no direction supplied.");
        if constexpr (form == Field::DualThreeForm)
        {
            return GEMPIC_D_MULT(m_nodeSplineVals[Direction::xDir][i],
                                 m_nodeSplineVals[Direction::yDir][j],
                                 m_nodeSplineVals[Direction::zDir][k]);
        }
        else if constexpr (form == Field::DualZeroForm)
        {
            return GEMPIC_D_MULT(m_cellSplineVals[Direction::xDir][i],
                                 m_cellSplineVals[Direction::yDir][j],
                                 m_cellSplineVals[Direction::zDir][k]);
        }
    }

    /**
     * @brief Compute the value of the spline-smoothed field at a particle position, from its l-form
     * coefficients
     *
     * @details Given a primal l-form (l = 1 or 2) field with coefficients (edge or face dofs) \f$
     * F_{ijk} \f$ this function computes the value of the spline-smoothed field at \f$ xx_p: \f$
     *
     * \f$ F^S(xx_p) = \sum_d (e_d * \sum_{ijk} F_{ijk} * t_{R\{3-l\}_{ijk}}(e_d *
     * \mathrm{partS}(xx-xx_p))) \f$
     *
     * where:
     * - \f$ xx_p \f$ = (\f$ x_p \f$, \f$ y_p \f$, \f$ z_p \f$) is the particle position encoded in
     * the spline object
     * - \f$ e_d \f$ is the canonical basis vector along grid direction d
     * - \f$ t_{R\{3-l\}} \f$ is the dual reduction operator for (3-l)-forms (here in 3D)
     * - partS is the tensor-product particle spline (approximating the Dirac mass), ie
     *     \f$ \mathrm{partS}( xx-xx_p ) = \prod_d (1/dx_d) * \mathrm{ref}_{S_{deg_{d-1}}}((xx -
     * xx_{p})_{d}/dx_{d}) \f$ (see also the class documentation)
     *
     * Examples:
     *
     * - for a (primal) 1-form electric field with (primal edge) dofs E_i, the function computes
     *
     *   \f$ E^S(xx_p) = \sum_{d = \mathrm{components}} e_d ( \sum_{i = \mathrm{primal\ edges /
     * dual\ faces}} E_i t_{R2_i}(e_d * \mathrm{partS}_p(xx)) ) \f$
     *
     *   where \f$ t_{R2_i}(.) \f$ is the reduction corresponding to a flux on the dual face i
     *
     * - for a (primal) 2-form magnetic field with (primal face) dofs \f$ B_i \f$, the function
     * computes
     *
     *   \f$ B^S(xx_p) = \sum_{d = \mathrm{components}} e_d ( \sum_{i = \mathrm{primal\ faces /
     * dual\ edges}} B_i t_{R1_i}(e_d * \mathrm{partS}_p(xx)) ) \f$
     *
     *   where \f$ t_{R1_i}(.) \f$ is the reduction corresponding to a circulation on the dual edge
     * i
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
    template <Field form, unsigned int vDim>
    AMREX_GPU_HOST_DEVICE amrex::GpuArray<amrex::Real, vDim> eval_spline_field (
        const amrex::GpuArray<amrex::Array4<amrex::Real>, vDim> fieldArray) const
    {
        static_assert(form == Field::PrimalOneForm || form == Field::PrimalTwoForm,
                      "Attempting smoothing of incorrect field type.");
        bool xCond{false};
        bool yCond{false};
        bool zCond{false};

        amrex::GpuArray<amrex::Real, vDim> fields;
        for (int comp = 0; comp < vDim; comp++)
        {
            fields[comp] = 0.;
        }

        GEMPIC_D_EXCL(int jj{0};, int kk{0};, )
        GEMPIC_D_LOOP_BEGIN(for (int ii = 0; ii <= degX; ++ii), for (int jj = 0; jj <= degY; ++jj),
                            for (int kk = 0; kk <= degZ; ++kk))
            const amrex::Dim3 cellIndex =
                (amrex::IntVect{AMREX_D_DECL(this->m_firstIndex[Direction::xDir] + ii,
                                             this->m_firstIndex[Direction::yDir] + jj,
                                             this->m_firstIndex[Direction::zDir] + kk)})
                    .dim3();

            // Ensure that we only access existing elements of cellSplineVals
            if constexpr (form == Field::PrimalOneForm)
            {
                xCond = (ii < degX);
            }
            else if constexpr (form == Field::PrimalTwoForm)
            {
                xCond = AMREX_D_TERM(true, &&(jj < degY), &&(kk < degZ));
            }
            if (xCond)
            {
                const amrex::Real sCoeff =
                    eval_b_spline<complement_form<form>(), Direction::xDir>(ii, jj, kk);
                fields[Direction::xDir] += sCoeff * (fieldArray[Direction::xDir])(
                                                        cellIndex.x, cellIndex.y, cellIndex.z, 0);
            }

            if constexpr (vDim > 1)
            {
                // Ensure that we only access existing elements of cellSplineVals
                if constexpr (form == Field::PrimalOneForm)
                {
                    yCond = AMREX_D_TERM(true, &&(jj < degY), );
                }
                else if constexpr (form == Field::PrimalTwoForm)
                {
                    yCond = AMREX_D_TERM((ii < degX), , &&(kk < degZ));
                }
                if (yCond)
                {
                    const amrex::Real sCoeff =
                        eval_b_spline<complement_form<form>(), Direction::yDir>(ii, jj, kk);
                    fields[Direction::yDir] +=
                        sCoeff *
                        (fieldArray[Direction::yDir])(cellIndex.x, cellIndex.y, cellIndex.z, 0);
                }
            }
            if constexpr (vDim == 3)
            {
                // Ensure that we only access existing elements of cellSplineVals
                if constexpr (form == Field::PrimalOneForm)
                {
                    zCond = AMREX_D_TERM(true, , &&(kk < degZ));
                }
                else if constexpr (form == Field::PrimalTwoForm)
                {
                    zCond = AMREX_D_TERM((ii < degX), &&(jj < degY), );
                }
                if (zCond)
                {
                    const amrex::Real sCoeff =
                        eval_b_spline<complement_form<form>(), Direction::zDir>(ii, jj, kk);
                    fields[Direction::zDir] +=
                        sCoeff *
                        (fieldArray[Direction::zDir])(cellIndex.x, cellIndex.y, cellIndex.z, 0);
                }
            }
        GEMPIC_D_LOOP_END

        return (fields);
    }

    template <unsigned int nWtsSum>
    AMREX_GPU_HOST_DEVICE void eval_e_field_spline_weights (
        amrex::GpuArray<amrex::Real, nWtsSum> &eWts,
        amrex::GpuArray<amrex::IntVect, nWtsSum> &eCells) const
    {
        constexpr int nWtsX[3] = {degX, (degX + 1),
                                  (degX + 1)}; // number of x,y,z-field weights in X-direction
        constexpr int nWtsY[3] = {(degY + 1), degY,
                                  (degY + 1)}; // number of x,y,z-field weights in Y-direction
        constexpr int nWtsZ[3] = {(degZ + 1), (degZ + 1),
                                  degZ}; // number of x,y,z-field weights in Z-direction

        constexpr int nWts[3] = {
            GEMPIC_D_MULT(nWtsX[xDir], nWtsY[xDir], nWtsZ[xDir]),  // number of x-field weights
            GEMPIC_D_MULT(nWtsX[yDir], nWtsY[yDir], nWtsZ[yDir]),  // number of y-field weights
            GEMPIC_D_MULT(nWtsX[zDir], nWtsY[zDir], nWtsZ[zDir])}; // number of z-field weights

        constexpr int nWtsStart[3] = {0, nWts[xDir], (nWts[xDir] + nWts[yDir])};

        GEMPIC_D_EXCL(int jj{0};, int kk{0};, )

        // Weights for E at particle location
        for (int ndir = 0; ndir < 3; ++ndir)
        {
            GEMPIC_D_LOOP_BEGIN(for (int ii = 0; ii < nWtsX[ndir]; ++ii),
                                for (int jj = 0; jj < nWtsY[ndir]; ++jj),
                                for (int kk = 0; kk < nWtsZ[ndir]; ++kk))

                const amrex::Real sCoeff =
                    ((ndir == 0)
                         ? eval_b_spline<complement_form<Field::PrimalOneForm>(), Direction::xDir>(
                               ii, jj, kk)
                         : ((ndir == 1) ? eval_b_spline<complement_form<Field::PrimalOneForm>(),
                                                        Direction::yDir>(ii, jj, kk)
                                        : eval_b_spline<complement_form<Field::PrimalOneForm>(),
                                                        Direction::zDir>(ii, jj, kk)));

                eWts[nWtsStart[ndir] + kk * nWtsY[ndir] * nWtsX[ndir] + jj * nWtsX[ndir] + ii] =
                    sCoeff;
                eCells[nWtsStart[ndir] + kk * nWtsY[ndir] * nWtsX[ndir] + jj * nWtsX[ndir] + ii] =
                    amrex::IntVect{AMREX_D_DECL(this->m_firstIndex[Direction::xDir] + ii,
                                                this->m_firstIndex[Direction::yDir] + jj,
                                                this->m_firstIndex[Direction::zDir] + kk)};
            GEMPIC_D_LOOP_END
        }
    }

    /**
     * @brief Compute and store the values of the cardinal spline at regular values x+i
     *
     * @details Given 0 <= xRef < 1, compute and store the values \f$ s[i] =
     * S^{degree}(x+i-(deg+1)/2) \f$ for i = 0 .. degree. Here \f$ S^{degree} \f$ is the centered
     * cardinal B-spline of given degree with support [-(degree+1)/2, (degree+1)/2]
     *
     * @tparam degree : degree of the cardinal spline
     * @tparam length : length of the stored array
     *
     * @param xRef : the offset to integer positions at which the spline shall be evaluated
     * @param s : the array in which the results will be stored
     */
    // NOLINTBEGIN(readability-magic-numbers)
    template <int degree, unsigned int length>
    AMREX_GPU_HOST_DEVICE constexpr void spline_eval (const amrex::Real xRef,
                                                      amrex::GpuArray<amrex::Real, length> &s) const
    {
        s = cardinal_b_splines<degree, length>(xRef);
    }
    // NOLINTEND(readability-magic-numbers)
};

/**
 * @brief This object will contain the value of all the non-vanishing local spline basis functions
 * at particle positions including a primitive
 *
 * @details This class inherits from the SplineBase class and adds a data member and methods for
 * handling a primitive
 *
 * @tparam degX Degree of the spline in x-direction
 * @tparam degY Degree of the spline in y-direction
 * @tparam degZ Degree of the spline in z-direction
 */
template <int degX, int degY, int degZ>
class SplineWithPrimitive : public SplineBase<degX, degY, degZ>
{
public:
    /**
     * @brief Construct a new SplineWithPrimitive object
     *
     * @details Initialize the primitive by calling the update1DPrimitive method for each direction
     * x, y and z
     *
     * @param position Current position of the particle
     * @param plo (rename this one)
     * @param dxInverse Inverse of the cell size in all three directions x, y and z (It's an array)
     * @details Additional details about xInverse
     *  - dxInverse[Direction::xDir] == 1/dx
     *  - dxInverse[Direction::yDir] == 1/dy
     *  - dxInverse[Direction::zDir] == 1/dz
     */
    AMREX_GPU_HOST_DEVICE SplineWithPrimitive (
        const amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> &position,
        const amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> &plo,
        const amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> &dxInverse) :
        SplineBase<degX, degY, degZ>(position, plo, dxInverse)
    {
        m_firstIndexOld = {};

        m_primitiveNew = {};

        const amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> refOffsets = {
            AMREX_D_DECL(this->template init_b_splines_at_positions<xDir>(position[xDir], plo[xDir],
                                                                          dxInverse[xDir]),
                         this->template init_b_splines_at_positions<yDir>(position[yDir], plo[yDir],
                                                                          dxInverse[yDir]),
                         this->template init_b_splines_at_positions<zDir>(position[zDir], plo[zDir],
                                                                          dxInverse[zDir]))};
        AMREX_D_TERM((update1_d_primitive<xDir>(refOffsets[xDir]));
                     , (update1_d_primitive<yDir>(refOffsets[yDir]));
                     , (update1_d_primitive<zDir>(refOffsets[zDir])););
    }

    /**
     * @brief An array of indices
     *
     * @details At the end of the function initBSplinesAtPositions, firstIndex is the index of the
     * first field-coefficient corresponding to a non-zero spline at the particle location.
     */
    amrex::GpuArray<int, AMREX_SPACEDIM> m_firstIndexOld;
    // The actually first index of the two primitives
    amrex::GpuArray<int, AMREX_SPACEDIM> m_firstIndexStart;

    using primitiveType =
        amrex::GpuArray<amrex::GpuArray<amrex::Real, SplineBase<degX, degY, degZ>::s_maxDeg>,
                        AMREX_SPACEDIM>;
    /**
     * @brief primitive of the new spline
     */
    primitiveType m_primitiveNew;

    /**
     * @brief primitive of the old spline
     */
    primitiveType m_primitiveOld;

    /**
     * @brief order of primitives
     */
    enum PrimitiveOrder
    {
        oldFirst,
        newFirst,
    };
    amrex::GpuArray<PrimitiveOrder, AMREX_SPACEDIM> m_primitiveOrders;

    /**
     * @brief sets the values of the contributing cell- and node-splines at the given particle
     * position
     *
     * Description:
     * after this function call, the values of cellSplineVals[dir][i] and nodeSplineVals[dir][i]
     * are set for i = 0 .. deg, according to their documentation above
     *
     * @tparam dir : the direction in which to compute: x = 0, y = 1, z = 2
     *
     * @param position Current position of the particle
     * @param plo : low end of the problem domain
     * @param dxInverseDir : inverse of the cell size in the given direction
     */
    template <Direction dir>
    AMREX_GPU_HOST_DEVICE void update_1d_splines (const amrex::Real position,
                                                  const amrex::Real plo,
                                                  const amrex::Real dxInverseDir)
    {
        this->m_firstIndexOld[dir] = std::move(this->m_firstIndex[dir]);

        const amrex::Real refOffset =
            this->template init_b_splines_at_positions<dir>(position, plo, dxInverseDir);

        constexpr int deg{SplineBase<degX, degY, degZ>::s_splineDegrees[dir]};
        this->template spline_eval<deg - 1, SplineBase<degX, degY, degZ>::s_maxDeg>(
            refOffset, this->m_cellSplineVals[dir]);
        for (size_t i{0}; i < deg; i++)
        {
            this->m_cellSplineVals[dir][i] *= dxInverseDir;
        }
        this->template spline_eval<deg, SplineBase<degX, degY, degZ>::s_maxDeg + 1>(
            refOffset, this->m_nodeSplineVals[dir]);

        update1_d_primitive<dir>(refOffset);
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
     * @param refOffset Normalized offset position of the particle
     */
    template <Direction dir>
    AMREX_GPU_HOST_DEVICE void update1_d_primitive (const amrex::Real refOffset)
    {
        if (this->m_firstIndex[dir] < this->m_firstIndexOld[dir])
        {
            this->m_firstIndexStart[dir] = this->m_firstIndex[dir];
            this->m_primitiveOrders[dir] = newFirst;
        }
        else
        {
            this->m_firstIndexStart[dir] = this->m_firstIndexOld[dir];
            this->m_primitiveOrders[dir] = oldFirst;
        }

        m_primitiveOld[dir] = std::move(m_primitiveNew[dir]);
        constexpr int deg{SplineBase<degX, degY, degZ>::s_splineDegrees[dir]};
        primitive_eval<deg - 1>(refOffset, m_primitiveNew[dir]);
    }

    /**
     * @brief Utility function for selecting a cellIndex corresponding to the primitive difference
     * calculated in compute_primitive_difference
     *
     * @param i, j, k The indices for the loop, with i the index in compute_primitive_difference
     */
    template <Direction pDir>
    AMREX_GPU_HOST_DEVICE inline amrex::Dim3 select_primitive_cellindex (const int i,
                                                                         const int j,
                                                                         const int k)
    {
        if constexpr (pDir == xDir)
        {
            return {GEMPIC_D_PAD(this->m_firstIndexStart[xDir] + i, this->m_firstIndex[yDir] + j,
                                 this->m_firstIndex[zDir] + k)};
        }
        else if constexpr (pDir == yDir)
        {
            return {GEMPIC_D_PAD(this->m_firstIndex[xDir] + k, this->m_firstIndexStart[yDir] + i,
                                 this->m_firstIndex[zDir] + j)};
        }
        else if constexpr (pDir == zDir)
        {
            return {GEMPIC_D_PAD(this->m_firstIndex[xDir] + j, this->m_firstIndex[yDir] + k,
                                 this->m_firstIndexStart[zDir] + i)};
        }
    }
    /**
     * @brief Compute the difference between primitives of the old spline and the new spline
     *
     * @tparam dir The direction in which to compute
     * @tparam region The region of the calculation. Can be one of four:
     *                firstSpline: Only the lowest index spline needs to be taken into account
     *                lastSpline: Only the highest index spline needs to be taken into account
     *                noSplines: Region between splines
     *                bothSplines: Both splines need to be taken into account
     *
     * @param dx Cell size in all three directions x, y and z (It's an array)
     * @param index Index at which to evaluate the primitive difference
     *
     * @return primitiveDifference The difference between primitives of the old spline and the new
     * spline
     */
    template <Direction dir, Impl::PrimitiveDifferenceRegion region>
    AMREX_GPU_HOST_DEVICE amrex::Real compute_primitive_difference (
        const amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> dx, const int index) const
    {
        //BL_PROFILE(
        //    "Gempic::ParticleMeshCoupling::SplineWithPrimitive::compute_primitive_difference()");
        using namespace Impl; // utility functions for this function
        constexpr int deg{SplineBase<degX, degY, degZ>::s_splineDegrees[dir]};
        // Size of support of difference of primitives
        const int localSize = std::abs(this->m_firstIndex[dir] - this->m_firstIndexOld[dir]) + deg;

        if (this->m_primitiveOrders[dir] == oldFirst)
        {
            if constexpr (region == PrimitiveDifferenceRegion::firstSpline)
            {
                return this->m_primitiveOld[dir][index];
            }
            else if constexpr (region == PrimitiveDifferenceRegion::noSplines)
            {
                return 1.0;
            }
            else if constexpr (region == PrimitiveDifferenceRegion::bothSplines)
            {
                return (this->m_primitiveOld[dir][index] -
                        this->m_primitiveNew[dir][index + deg - localSize]);
            }
            else //if constexpr (region == PrimitiveDifferenceRegion::lastSpline)
            {
                return (1 - this->m_primitiveNew[dir][index + deg - localSize]);
            }
        }
        // switching all primitives and all signs
        else //if (this->m_primitiveOrders[dir] == newFirst)
        {
            if constexpr (region == PrimitiveDifferenceRegion::firstSpline)
            {
                return -this->m_primitiveNew[dir][index];
            }
            else if constexpr (region == PrimitiveDifferenceRegion::noSplines)
            {
                return -1.0;
            }
            else if constexpr (region == PrimitiveDifferenceRegion::bothSplines)
            {
                return -(this->m_primitiveNew[dir][index] -
                         this->m_primitiveOld[dir][index + deg - localSize]);
            }
            else //if constexpr (region == PrimitiveDifferenceRegion::lastSpline)
            {
                return -(1 - this->m_primitiveOld[dir][index + deg - localSize]);
            }
        }
    }

    /**
     * @brief Compute the difference between primitives of the old spline and the new spline. Slower
     * version for when the regime is unknown
     *
     * @tparam dir The direction in which to compute: x = 0, y = 1, z = 2
     *
     * @param dx Cell size in all three directions x, y and z (It's an array)
     * @param index Index at which to evaluate the primitive difference
     *
     * @return primitiveDifference The difference between primitives of the old spline and the new
     * spline
     */
    template <Direction dir>
    AMREX_GPU_HOST_DEVICE inline amrex::Real compute_primitive_difference (
        const amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> dx, const int index) const
    {
        constexpr int deg{SplineBase<degX, degY, degZ>::s_splineDegrees[dir]};
        // Size of support of difference of primitives
        const int localSize = std::abs(this->m_firstIndex[dir] - this->m_firstIndexOld[dir]) + deg;
        if (index >= localSize || index < 0)
        {
            return 0;
        }
        else if (index < std::min(deg, localSize - deg))
        {
            constexpr auto region{Impl::PrimitiveDifferenceRegion::firstSpline};
            return this->compute_primitive_difference<dir, region>(dx, index);
        }
        else if (index >= std::max(deg, localSize - deg))
        {
            constexpr auto region{Impl::PrimitiveDifferenceRegion::lastSpline};
            return this->compute_primitive_difference<dir, region>(dx, index);
        }
        else if (index >= deg && index < localSize - deg)
        {
            constexpr auto region{Impl::PrimitiveDifferenceRegion::noSplines};
            return this->compute_primitive_difference<dir, region>(dx, index);
        }
        else // (index < deg && index >= localSize - deg)
        {
            constexpr auto region{Impl::PrimitiveDifferenceRegion::bothSplines};
            return this->compute_primitive_difference<dir, region>(dx, index);
        }
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
    // NOLINTBEGIN(readability-magic-numbers)
    template <int degree, unsigned int length>
    AMREX_GPU_HOST_DEVICE inline constexpr void primitive_eval (
        const amrex::Real xRef, amrex::GpuArray<amrex::Real, length> &s) const
    {
        s = cardinal_b_splines_primitive<degree, length>(xRef);
    }
    // NOLINTEND(readability-magic-numbers)
};

/**
 * @brief This object will contain the value of all the non-vanishing local spline basis functions
 * at particle positions including first derivatives
 *
 * @details This class inherits from the SplineBase class and adds a derivative and methods for
 * handling it
 *
 * @tparam degX Degree of the spline in x-direction
 * @tparam degY Degree of the spline in y-direction
 * @tparam degZ Degree of the spline in z-direction
 */
template <int degX, int degY, int degZ>
class SplineWithFirstDerivative : public SplineBase<degX, degY, degZ>
{
public:
    /**
     * @brief Construct a new SplineWithFirstDerivative object
     *
     * @details Initialize the primitive by calling the update1DPrimitive method for each direction
     * x, y and z
     *
     * @param position Current position of the particle
     * @param plo (rename this one)
     * @param dxInverse Inverse of the cell size in all three directions x, y and z (It's an array)
     * @details Additional details about xInverse
     *  - dxInverse[Direction::xDir] == 1/dx
     *  - dxInverse[Direction::yDir] == 1/dy
     *  - dxInverse[Direction::zDir] == 1/dz
     *  - (Potentially dim < 3)
     *  - dxInverse[dim] == 1/(dx * dy * dz)
     */
    AMREX_GPU_HOST_DEVICE SplineWithFirstDerivative (
        const amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> &position,
        const amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> &plo,
        const amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> &dxInverse,
        const amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> &dx) :
        SplineBase<degX, degY, degZ>(position, plo, dxInverse)
    {
        m_cellSplineFirstDerivs = {};

        m_nodeSplineFirstDerivs = {};

        AMREX_D_TERM((update_1d_first_derivative<Direction::xDir>(position[xDir], plo[xDir],
                                                                  dxInverse[xDir], dx[xDir]));
                     , (update_1d_first_derivative<Direction::yDir>(position[yDir], plo[yDir],
                                                                    dxInverse[yDir], dx[yDir]));
                     , (update_1d_first_derivative<Direction::zDir>(position[zDir], plo[zDir],
                                                                    dxInverse[zDir], dx[zDir])););
    }

    /**
     * @brief first derivative of the cell spline
     */
    amrex::GpuArray<amrex::GpuArray<amrex::Real, SplineBase<degX, degY, degZ>::s_maxDeg>,
                    AMREX_SPACEDIM>
        m_cellSplineFirstDerivs;

    /**
     * @brief first derivative of the node spline
     */
    amrex::GpuArray<amrex::GpuArray<amrex::Real, SplineBase<degX, degY, degZ>::s_maxDeg + 1>,
                    AMREX_SPACEDIM>
        m_nodeSplineFirstDerivs;

    /**
     * @brief Set the values of the contributing cell- and node-spline derivatives for the position
     * \f$ x_p \f$
     *
     * @details Set cellSplineVals and nodeSplineVals for the direction according to their
     * documentation
     *
     * @tparam dir : the direction in which to compute: Direction::xDir, Direction::yDir,
     * Direction::zDir
     *
     * @param position Current position of the particle
     * @param plo : low end of the problem domain
     * @param dxInverseDir : inverse of the cell size in the given direction
     */
    template <Direction dir>
    AMREX_GPU_HOST_DEVICE void update_1d_first_derivative (const amrex::Real position,
                                                           const amrex::Real plo,
                                                           const amrex::Real dxInverseDir,
                                                           const amrex::Real dxDir)
    {
        const amrex::Real refOffset =
            this->template init_b_splines_at_positions<dir>(position, plo, dxInverseDir);

        constexpr int deg{SplineBase<degX, degY, degZ>::s_splineDegrees[dir]};
        first_derivative_eval<deg - 1, SplineBase<degX, degY, degZ>::s_maxDeg>(
            refOffset, m_cellSplineFirstDerivs[dir]);
        // Scaling according to documentation
        for (size_t i{0}; i < deg; i++)
        {
            m_cellSplineFirstDerivs[dir][i] *= dxInverseDir * dxInverseDir;
        }
        first_derivative_eval<deg, SplineBase<degX, degY, degZ>::s_maxDeg + 1>(
            refOffset, m_nodeSplineFirstDerivs[dir]);
        for (size_t i{0}; i < deg; i++)
        {
            m_nodeSplineFirstDerivs[dir][i] *= dxInverseDir;
        }
    }

    /**
     * @brief Compute and store the values of the derivative at regular values x+i
     *
     * @tparam degree : degree of the derivative
     * @tparam length : length of the stored array
     *
     * @param xRef : the offset to integer positions at which the spline shall be evaluated
     * @param s : the array in which the results will be stored
     */
    // NOLINTBEGIN(readability-magic-numbers)
    template <int degree, unsigned int length>
    AMREX_GPU_HOST_DEVICE inline constexpr void first_derivative_eval (
        const amrex::Real xRef, amrex::GpuArray<amrex::Real, length> &s) const
    {
        s = cardinal_b_splines_first_derivative<degree>(xRef);
    }
    // NOLINTEND(readability-magic-numbers)

    /**
     * @brief Compute a dual dof derivative (defined by the corresponding reduction operator \f$
     *t_{R\{3-l\}}) \f$ of a vectorized particle spline
     *
     * Description:
     * This function computes a dual dof of the tensor-product particle spline
     * centered at the particle position encoded in this spline object, with a vector orientation
     *for l = 1 or 2 Field::DualOneForm Specifically, it computes
     *  - \f$ t_{R\{3-l\}_{ijk}}(\mathrm{partS}(xx-xx_p)) \f$ if l = 0 or 3
     *  - \f$ t_{R\{3-l\}_{ijk}}(e_{dir} * \mathrm{partS}(xx-xx_p)) \f$ if l = 1 or 2
     * where
     *   partS is the (tensor-product) particle spline defined above, centered at the particle
     *position \f$ xx_p \f$
     *
     * According to the univariate formulas above:
     *   \f$ t_{R0_i}(\mathrm{partS}(x-x_p)) = \mathrm{cellS}_i( x_p ) \f$
     * and
     *   \f$ t_{R1_i}(\mathrm{partS}(x-x_p)) = \mathrm{nodeS}_i( x_p ) \f$
     * the multivariate dofs are obtained as products of univariate node- and cell-splines evaluated
     *at the particle position
     *
     * @tparam form : index l of the associated dual form
     * @tparam dir : for l = 1 or 2, the vectorized direction of the particle slpine: x = 0, y = 1,
     *z = 2
     *
     * @param i The index of the x-dir basis function
     * @param j The index of the y-dir basis function
     * @param k The index of the z-dir basis function
     *
     * @return base The value of the calculated coefficient
     */
    template <Field form, Direction dir, Direction dirDeriv>
    AMREX_GPU_HOST_DEVICE amrex::Real eval_b_spline_derivative (int i, int j, int k) const
    {
        static_assert(form == Field::DualOneForm || form == Field::DualTwoForm,
                      "Function called with primal form, or superfluous direction supplied.");
        if constexpr (form == Field::DualTwoForm)
        {
            if constexpr (dir == Direction::xDir)
            {
                return GEMPIC_D_MULT(
                    ((dirDeriv == Direction::xDir)
                         ? m_cellSplineFirstDerivs[Direction::xDir][i]
                         : SplineBase<degX, degY, degZ>::m_cellSplineVals[Direction::xDir][i]),
                    ((dirDeriv == Direction::yDir)
                         ? m_nodeSplineFirstDerivs[Direction::yDir][j]
                         : SplineBase<degX, degY, degZ>::m_nodeSplineVals[Direction::yDir][j]),
                    ((dirDeriv == Direction::zDir)
                         ? m_nodeSplineFirstDerivs[Direction::zDir][k]
                         : SplineBase<degX, degY, degZ>::m_nodeSplineVals[Direction::zDir][k]));
            }
            else if constexpr (dir == Direction::yDir)
            {
                return GEMPIC_D_MULT(
                    ((dirDeriv == Direction::xDir)
                         ? m_nodeSplineFirstDerivs[Direction::xDir][i]
                         : SplineBase<degX, degY, degZ>::m_nodeSplineVals[Direction::xDir][i]),
                    ((dirDeriv == Direction::yDir)
                         ? m_cellSplineFirstDerivs[Direction::yDir][j]
                         : SplineBase<degX, degY, degZ>::m_cellSplineVals[Direction::yDir][j]),
                    ((dirDeriv == Direction::zDir)
                         ? m_nodeSplineFirstDerivs[Direction::zDir][k]
                         : SplineBase<degX, degY, degZ>::m_nodeSplineVals[Direction::zDir][k]));
            }
            else if constexpr (dir == Direction::zDir)
            {
                return GEMPIC_D_MULT(
                    ((dirDeriv == Direction::xDir)
                         ? m_nodeSplineFirstDerivs[Direction::xDir][i]
                         : SplineBase<degX, degY, degZ>::m_nodeSplineVals[Direction::xDir][i]),
                    ((dirDeriv == Direction::yDir)
                         ? m_nodeSplineFirstDerivs[Direction::yDir][j]
                         : SplineBase<degX, degY, degZ>::m_nodeSplineVals[Direction::yDir][j]),
                    ((dirDeriv == Direction::zDir)
                         ? m_cellSplineFirstDerivs[Direction::zDir][k]
                         : SplineBase<degX, degY, degZ>::m_cellSplineVals[Direction::zDir][k]));
            }
        }
        else if constexpr (form == Field::DualOneForm)
        {
            if constexpr (dir == Direction::xDir)
            {
                return GEMPIC_D_MULT(
                    ((dirDeriv == Direction::xDir)
                         ? m_nodeSplineFirstDerivs[Direction::xDir][i]
                         : SplineBase<degX, degY, degZ>::m_nodeSplineVals[Direction::xDir][i]),
                    ((dirDeriv == Direction::yDir)
                         ? m_cellSplineFirstDerivs[Direction::yDir][j]
                         : SplineBase<degX, degY, degZ>::m_cellSplineVals[Direction::yDir][j]),
                    ((dirDeriv == Direction::zDir)
                         ? m_cellSplineFirstDerivs[Direction::zDir][k]
                         : SplineBase<degX, degY, degZ>::m_cellSplineVals[Direction::zDir][k]));
            }
            else if constexpr (dir == Direction::yDir)
            {
                return GEMPIC_D_MULT(
                    ((dirDeriv == Direction::xDir)
                         ? m_cellSplineFirstDerivs[Direction::xDir][i]
                         : SplineBase<degX, degY, degZ>::m_cellSplineVals[Direction::xDir][i]),
                    ((dirDeriv == Direction::yDir)
                         ? m_nodeSplineFirstDerivs[Direction::yDir][j]
                         : SplineBase<degX, degY, degZ>::m_nodeSplineVals[Direction::yDir][j]),
                    ((dirDeriv == Direction::zDir)
                         ? m_cellSplineFirstDerivs[Direction::zDir][k]
                         : SplineBase<degX, degY, degZ>::m_cellSplineVals[Direction::zDir][k]));
            }
            else if constexpr (dir == Direction::zDir)
            {
                return GEMPIC_D_MULT(
                    ((dirDeriv == Direction::xDir)
                         ? m_cellSplineFirstDerivs[Direction::xDir][i]
                         : SplineBase<degX, degY, degZ>::m_cellSplineVals[Direction::xDir][i]),
                    ((dirDeriv == Direction::yDir)
                         ? m_cellSplineFirstDerivs[Direction::yDir][j]
                         : SplineBase<degX, degY, degZ>::m_cellSplineVals[Direction::yDir][j]),
                    ((dirDeriv == Direction::zDir)
                         ? m_nodeSplineFirstDerivs[Direction::zDir][k]
                         : SplineBase<degX, degY, degZ>::m_nodeSplineVals[Direction::zDir][k]));
            }
        }
    }
};

} // namespace Gempic::ParticleMeshCoupling
#endif
