#include "GEMPIC_ComputationalDomain.H"
#include "GEMPIC_Config.H"
#include "GEMPIC_FDDeRhamComplex.H"
#include "GEMPIC_Fields.H"
#include "GEMPIC_Interpolation.H"
#include "GEMPIC_NumericalIntegrationDifferentiation.H"
#include "GEMPIC_Parameters.H"

using namespace Gempic::Forms;

FDDeRhamComplex::FDDeRhamComplex(const ComputationalDomain& infra,
                                 const int hodgeDegree,
                                 const int maxSplineDegree,
                                 HodgeScheme hodgeScheme) :
    DeRhamComplex::DeRhamComplex{infra, hodgeDegree, maxSplineDegree}
{
    BL_PROFILE("Gempic::Forms::FDDeRhamComplex::FDDeRhamComplex()");
    // Parameters used in the projection and hodge
    for (size_t dir = 0; dir < AMREX_SPACEDIM; dir++)
    {
        m_dr[dir] = infra.geometry().CellSize(dir);
    }
    m_nGhost = DeRhamComplex::m_nGhost[xDir];
    m_hodgeScheme = hodgeScheme;

    // Read the scaling factors from input file and compute value for the Hodge operator
    Gempic::Io::Parameters parameters{"FDDeRhamComplex"};
    m_sV = 1.0;
    parameters.get_or_set("sV", m_sV);
    m_sOmega = 1.0;
    parameters.get_or_set("sOmega", m_sOmega);

    // There is only one components in each MultiFab as the different components of the forms are
    // centered differently
    m_tempPrimalZeroForm.define(
        amrex::convert(m_grid, amrex::IndexType(amrex::IntVect{AMREX_D_DECL(1, 1, 1)})),
        m_distriMap, 1, m_nGhost);
    m_tempDualZeroForm.define(
        amrex::convert(m_grid, amrex::IndexType(amrex::IntVect{AMREX_D_DECL(0, 0, 0)})),
        m_distriMap, 1, m_nGhost);

    m_tempPrimalOneForm[xDir].define(
        amrex::convert(m_grid, amrex::IndexType(amrex::IntVect{AMREX_D_DECL(0, 1, 1)})),
        m_distriMap, 1, m_nGhost);
    m_tempPrimalOneForm[yDir].define(
        amrex::convert(m_grid, amrex::IndexType(amrex::IntVect{AMREX_D_DECL(1, 0, 1)})),
        m_distriMap, 1, m_nGhost);
    m_tempPrimalOneForm[zDir].define(
        amrex::convert(m_grid, amrex::IndexType(amrex::IntVect{AMREX_D_DECL(1, 1, 0)})),
        m_distriMap, 1, m_nGhost);

    m_tempDualOneForm[xDir].define(
        amrex::convert(m_grid, amrex::IndexType(amrex::IntVect{AMREX_D_DECL(1, 0, 0)})),
        m_distriMap, 1, m_nGhost);
    m_tempDualOneForm[yDir].define(
        amrex::convert(m_grid, amrex::IndexType(amrex::IntVect{AMREX_D_DECL(0, 1, 0)})),
        m_distriMap, 1, m_nGhost);
    m_tempDualOneForm[zDir].define(
        amrex::convert(m_grid, amrex::IndexType(amrex::IntVect{AMREX_D_DECL(0, 0, 1)})),
        m_distriMap, 1, m_nGhost);

    m_tempPrimalTwoForm[xDir].define(
        amrex::convert(m_grid, amrex::IndexType(amrex::IntVect{AMREX_D_DECL(1, 0, 0)})),
        m_distriMap, 1, m_nGhost);
    m_tempPrimalTwoForm[yDir].define(
        amrex::convert(m_grid, amrex::IndexType(amrex::IntVect{AMREX_D_DECL(0, 1, 0)})),
        m_distriMap, 1, m_nGhost);
    m_tempPrimalTwoForm[zDir].define(
        amrex::convert(m_grid, amrex::IndexType(amrex::IntVect{AMREX_D_DECL(0, 0, 1)})),
        m_distriMap, 1, m_nGhost);

    m_tempDualTwoForm[xDir].define(
        amrex::convert(m_grid, amrex::IndexType(amrex::IntVect{AMREX_D_DECL(0, 1, 1)})),
        m_distriMap, 1, m_nGhost);
    m_tempDualTwoForm[yDir].define(
        amrex::convert(m_grid, amrex::IndexType(amrex::IntVect{AMREX_D_DECL(1, 0, 1)})),
        m_distriMap, 1, m_nGhost);
    m_tempDualTwoForm[zDir].define(
        amrex::convert(m_grid, amrex::IndexType(amrex::IntVect{AMREX_D_DECL(1, 1, 0)})),
        m_distriMap, 1, m_nGhost);

    m_tempPrimalThreeForm.define(
        amrex::convert(m_grid, amrex::IndexType(amrex::IntVect{AMREX_D_DECL(0, 0, 0)})),
        m_distriMap, 1, m_nGhost);
    m_tempDualThreeForm.define(
        amrex::convert(m_grid, amrex::IndexType(amrex::IntVect{AMREX_D_DECL(1, 1, 1)})),
        m_distriMap, 1, m_nGhost);
}

FDDeRhamComplex::~FDDeRhamComplex() {}

/**
 * @brief Computes the Restriction operator \f$R_0\f$
 *
 * @details
 * Using the geometric degrees of freedeom:
 * Evaluates func at time t on the nodes of the primal grid
 *
 * For dimensions 1 and 2, the 0-form is taken from the "stacked" de Rham complex
 *
 * @param funcs ParserExecutor, functions to be projected
 * @param t Real, time at which func is to be evaluated
 * @param field DeRhamField<primal, Space::node>, 0-form \f$u^0\f$ holding the node values
 */
void FDDeRhamComplex::projection (
    const amrex::Vector<amrex::ParserExecutor<AMREX_SPACEDIM + 1>>& funcs,
    amrex::Real t,
    DeRhamField<Grid::primal, Space::node>& field)
{
    BL_PROFILE("Gempic::Forms::FDDeRhamComplex::projection(<primal, node>)");
    size_t nComps{static_cast<size_t>(funcs.size())};
    int nCompField{field.m_data.nComp()};
    std::string msg = "The number of functions (" + std::to_string(nComps) +
                      ") doesn't correspond to the number of DeRhamField components (" +
                      std::to_string(nCompField) + ").";
    if (field.m_data.nComp() != funcs.size())
    {
        amrex::Assert(msg.c_str(), __FILE__, __LINE__);
    }

    const amrex::RealVect dr = m_dr;
    const amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> r0 = m_geom.ProbLoArray();

    for (amrex::MFIter mfi(field.m_data, true); mfi.isValid(); ++mfi)
    {
        const amrex::Box& bx = mfi.growntilebox(); //mfi.tilebox();
        amrex::Array4<amrex::Real> const& zeroForm = (field.m_data)[mfi].array();

        amrex::AsyncArray<amrex::ParserExecutor<AMREX_SPACEDIM + 1>> funcsGpu(&funcs[0], nComps);
        auto* const funcsGpuPtr = funcsGpu.data();

        ParallelFor(
            bx, nComps,
            [=] AMREX_GPU_DEVICE(int i, int j, int k, int n)
            {
                // Compute the position of the point i, j, k
                std::array<amrex::Real, AMREX_SPACEDIM> r = {AMREX_D_DECL(
                    r0[xDir] + i * dr[xDir], r0[yDir] + j * dr[yDir], r0[zDir] + k * dr[zDir])};

                // Assign point values to zeroForm
                zeroForm(i, j, k, n) = funcsGpuPtr[n](AMREX_D_DECL(r[xDir], r[yDir], r[zDir]), t);
            });
    }

    field.fill_boundary();
}

inline void FDDeRhamComplex::projection(amrex::ParserExecutor<AMREX_SPACEDIM + 1>& func,
                                        amrex::Real t,
                                        DeRhamField<Grid::primal, Space::node>& field)
{
    amrex::Vector<amrex::ParserExecutor<AMREX_SPACEDIM + 1>> funcs{func};
    projection(funcs, t, field);
}

/**
 * @brief Computes the Restriction operator \f$R_3\f$
 *
 * @details
 * Using the geometric degrees of freedeom:
 * Integrates func at time t over the cells of the primal grid by using a Gauss quadrature rule
 *
 * For dimensions 1 and 2, the 3-form is taken from the "stacked" de Rham complex
 *
 * @param funcs ParserExecutor, functions to be projected
 * @param t Real, time at which func is to be evaluated
 * @param field DeRhamField<primal, Space::cell>, 3-form \f$u^3\f$ holding the cell integrals
 * @param gaussNodes int, number of Gauss nodes to be used for quadrature
 */
void FDDeRhamComplex::projection (
    const amrex::Vector<amrex::ParserExecutor<AMREX_SPACEDIM + 1>>& funcs,
    amrex::Real t,
    DeRhamField<Grid::primal, Space::cell>& field,
    int gaussNodes)
{
    BL_PROFILE("Gempic::Forms::FDDeRhamComplex::projection(<primal, cell>)");
    amrex::Long nComps{funcs.size()};
    int nCompField{field.m_data.nComp()};
    std::string msg = "The number of functions (" + std::to_string(nComps) +
                      ") doesn't correspond to the number of DeRhamField components (" +
                      std::to_string(nCompField) + ").";
    if (field.m_data.nComp() != funcs.size())
    {
        amrex::Assert(msg.c_str(), __FILE__, __LINE__);
    }

    const GaussQuadrature integrate{gaussNodes};

    // Volume integral
    for (amrex::MFIter mfi(field.m_data, true); mfi.isValid(); ++mfi)
    {
        const amrex::Box& bx = mfi.growntilebox(); //projections are done also on ghost cells
        amrex::Array4<amrex::Real> const& threeForm = (field.m_data)[mfi].array();

        const amrex::RealVect dr = m_dr;
        const amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> r0 = m_geom.ProbLoArray();

        std::array<amrex::Real, 3> drHalf = {
            GEMPIC_D_PAD_ONE(dr[xDir] / 2, dr[yDir] / 2, dr[zDir] / 2)};
        for (int i{AMREX_SPACEDIM}; i < 3; i++) drHalf[i] = drHalf[i] / 2.0;

        amrex::AsyncArray<amrex::ParserExecutor<AMREX_SPACEDIM + 1>> funcsGpu(&funcs[0], nComps);
        auto* const funcsGpuPtr = funcsGpu.data();

        ParallelFor(
            bx, nComps,
            [=] AMREX_GPU_DEVICE(int i, int j, int k, int n)
            {
                // Compute the position of the point i, j, k
                std::array<amrex::Real, 3> r = {GEMPIC_D_PAD(
                    r0[xDir] + i * dr[xDir], r0[yDir] + j * dr[yDir], r0[zDir] + k * dr[zDir])};

                // Midpoint for the quadrature rule
                std::array<amrex::Real, 3> midpoint = {GEMPIC_D_PAD(
                    r[xDir] + drHalf[xDir], r[yDir] + drHalf[yDir], r[zDir] + drHalf[zDir])};

                auto f = [&] (amrex::Real x, amrex::Real y, amrex::Real z)
                { return funcsGpuPtr[n](AMREX_D_DECL(x, y, z), t); };

                // Rescale the integral and assign it to degrees of freedom
                threeForm(i, j, k, n) = integrate.volume(midpoint, drHalf, f);
            });
    }

    field.fill_boundary();
}

inline void FDDeRhamComplex::projection(amrex::ParserExecutor<AMREX_SPACEDIM + 1>& func,
                                        amrex::Real t,
                                        DeRhamField<Grid::primal, Space::cell>& field,
                                        int gaussNodes)
{
    amrex::Vector<amrex::ParserExecutor<AMREX_SPACEDIM + 1>> funcs{func};
    this->projection(funcs, t, field, gaussNodes);
}

/**
* @brief Computes the Restriction operator \f$\tilde{R}_0\f$
*
* @details
* Using the geometric degrees of freedom:
* Evaluates func at time t on the nodes of the dual grid
*
* For dimensions 1 and 2, the 0-form is taken from the "stacked" de Rham complex
*

* @param funcs ParserExecutor, functions to be projected
* @param t Real, time at which func is to be evaluated
* @param field DeRhamField<dual, Space::node>, 0-form \f$\tilde{u}^0\f$ holding the node values
*/
void FDDeRhamComplex::projection (
    const amrex::Vector<amrex::ParserExecutor<AMREX_SPACEDIM + 1>>& funcs,
    amrex::Real t,
    DeRhamField<Grid::dual, Space::node>& field)
{
    BL_PROFILE("Gempic::Forms::FDDeRhamComplex::projection(<dual, node>)");
    amrex::Long nComps{funcs.size()};
    int nCompField{field.m_data.nComp()};
    std::string msg = "The number of functions (" + std::to_string(nComps) +
                      ") doesn't correspond to the number of DeRhamField components (" +
                      std::to_string(nCompField) + ").";
    if (field.m_data.nComp() != funcs.size())
    {
        amrex::Assert(msg.c_str(), __FILE__, __LINE__);
    }

    for (amrex::MFIter mfi(field.m_data, true); mfi.isValid(); ++mfi)
    {
        const amrex::Box& bx = mfi.growntilebox(); //.tilebox();
        amrex::Array4<amrex::Real> const& zeroForm = (field.m_data)[mfi].array();

        const amrex::RealVect dr = m_dr;
        const amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> r0 = m_geom.ProbLoArray();

        amrex::AsyncArray<amrex::ParserExecutor<AMREX_SPACEDIM + 1>> funcsGpu(&funcs[0], nComps);
        auto* const funcsGpuPtr = funcsGpu.data();

        ParallelFor(bx, nComps,
                    [=] AMREX_GPU_DEVICE(int i, int j, int k, int n)
                    {
                        // Compute the position of the point i + 1/2, j + 1/2, k + 1/2
                        std::array<amrex::Real, AMREX_SPACEDIM> r = {
                            AMREX_D_DECL(r0[xDir] + 0.5 * dr[xDir] + i * dr[xDir],
                                         r0[yDir] + 0.5 * dr[yDir] + j * dr[yDir],
                                         r0[zDir] + 0.5 * dr[zDir] + k * dr[zDir])};

                        // Assign point values to zeroForm
                        zeroForm(i, j, k, n) =
                            funcsGpuPtr[n](AMREX_D_DECL(r[xDir], r[yDir], r[zDir]), t);
                    });
    }

    field.fill_boundary();
}

inline void FDDeRhamComplex::projection(amrex::ParserExecutor<AMREX_SPACEDIM + 1>& func,
                                        amrex::Real t,
                                        DeRhamField<Grid::dual, Space::node>& field)
{
    amrex::Vector<amrex::ParserExecutor<AMREX_SPACEDIM + 1>> funcs{func};
    this->projection(funcs, t, field);
}

/**
 * @brief Computes the Restriction operator \f$\tilde{R}_3\f$
 *
 * @details
 * Using the geometric degrees of freedom:
 * Integrates func at time t over the cells of the dual grid by using a Gauss quadrature rule
 *
 * For dimensions 1 and 2, the 3-form is taken from the "stacked" de Rham complex
 *
 * @param funcs ParserExecutor, functions to be projected
 * @param t Real, time at which func is to be evaluated
 * @param field DeRhamField<dual, Space::cell>, 3-form \f$\tilde{u}^3\f$ holding the cell
 * integrals
 * @param gaussNodes int, number of Gauss nodes to be used for quadrature
 */
void FDDeRhamComplex::projection (
    const amrex::Vector<amrex::ParserExecutor<AMREX_SPACEDIM + 1>>& funcs,
    amrex::Real t,
    DeRhamField<Grid::dual, Space::cell>& field,
    int gaussNodes)
{
    BL_PROFILE("Gempic::Forms::FDDeRhamComplex::projection(<dual, cell>)");
    amrex::Long nComps{funcs.size()};
    int nCompField{field.m_data.nComp()};
    std::string msg = "The number of functions (" + std::to_string(nComps) +
                      ") doesn't correspond to the number of DeRhamField components (" +
                      std::to_string(nCompField) + ").";
    if (field.m_data.nComp() != funcs.size())
    {
        amrex::Assert(msg.c_str(), __FILE__, __LINE__);
    }

    GaussQuadrature integrate{gaussNodes};

    // Volume integral
    for (amrex::MFIter mfi(field.m_data, true); mfi.isValid(); ++mfi)
    {
        const amrex::Box& bx = mfi.growntilebox(); //projections are done also on ghost cells
        amrex::Array4<amrex::Real> const& threeForm = (field.m_data)[mfi].array();

        const amrex::RealVect dr = m_dr;
        const amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> r0 = m_geom.ProbLoArray();

        std::array<amrex::Real, 3> drHalf = {
            GEMPIC_D_PAD_ONE(dr[xDir] / 2, dr[yDir] / 2, dr[zDir] / 2)};
        for (int i{AMREX_SPACEDIM}; i < 3; i++) drHalf[i] = drHalf[i] / 2.0;

        amrex::AsyncArray<amrex::ParserExecutor<AMREX_SPACEDIM + 1>> funcsGpu(&funcs[0], nComps);
        auto* const funcsGpuPtr = funcsGpu.data();

        ParallelFor(
            bx, nComps,
            [=] AMREX_GPU_DEVICE(int i, int j, int k, int n)
            {
                // Compute the position of the point i, j, k
                std::array<amrex::Real, 3> r = {GEMPIC_D_PAD(
                    r0[xDir] + drHalf[xDir] + i * dr[xDir], r0[yDir] + drHalf[yDir] + j * dr[yDir],
                    r0[zDir] + drHalf[zDir] + k * dr[zDir])};

                // Midpoint for the quadrature rule
                std::array<amrex::Real, 3> midpoint = {GEMPIC_D_PAD(
                    r[xDir] - drHalf[xDir], r[yDir] - drHalf[yDir], r[zDir] - drHalf[zDir])};

                auto f = [&] (amrex::Real x, amrex::Real y, amrex::Real z)
                { return funcsGpuPtr[n](AMREX_D_DECL(x, y, z), t); };

                // Rescale the integral and assign it to degrees of freedom
                threeForm(i, j, k, n) = integrate.volume(midpoint, drHalf, f);
            });
    }

    field.fill_boundary();
}

inline void FDDeRhamComplex::projection(amrex::ParserExecutor<AMREX_SPACEDIM + 1>& func,
                                        amrex::Real t,
                                        DeRhamField<Grid::dual, Space::cell>& field,
                                        int gaussNodes)
{
    amrex::Vector<amrex::ParserExecutor<AMREX_SPACEDIM + 1>> funcs{func};
    this->projection(funcs, t, field, gaussNodes);
}

/**
 * @brief Computes the Restriction operator \f$\tilde{R}_1\f$
 *
 * @details
 * Using the geometric degrees of freedeom:
 * Integrates func at time t over the edges of the dual grid by using a Gauss quadrature rule
 *
 * For dimensions 1 and 2, the 1-form is taken from the "stacked" de Rham complex
 *
 * @param funcs ParserExecutor, functions to be projected
 * @param t Real, time at which func is to be evaluated
 * @param field DeRhamField<dual, Space::edge>, 1-form \f$\tilde{u}^1\f$ holding the edge
 * integrals
 * @param gaussNodes int, number of Gauss nodes to be used for quadrature
 */
void FDDeRhamComplex::projection (
    const amrex::Vector<amrex::Array<amrex::ParserExecutor<AMREX_SPACEDIM + 1>, 3>>& funcs,
    amrex::Real t,
    DeRhamField<Grid::dual, Space::edge>& field,
    int gaussNodes)
{
    BL_PROFILE("Gempic::Forms::FDDeRhamComplex::projection(<dual, edge>)");
    auto nComps{static_cast<size_t>(funcs.size())};
    int nCompField{field.m_data[xDir].nComp()};
    std::string msg = "The number of functions (" + std::to_string(nComps) +
                      ") doesn't correspond to the number of DeRhamField components (" +
                      std::to_string(nCompField) + ").";
    if (field.m_data[xDir].nComp() != funcs.size())
    {
        amrex::Assert(msg.c_str(), __FILE__, __LINE__);
    }

    amrex::Gpu::DeviceVector<amrex::Array<amrex::ParserExecutor<AMREX_SPACEDIM + 1>, 3>> funcsGpu{
        nComps};
    amrex::Gpu::copyAsync(amrex::Gpu::hostToDevice, funcs.begin(), funcs.end(), funcsGpu.begin());
    auto* const funcsGpuPtr = funcsGpu.dataPtr();

    GaussQuadrature integrate{gaussNodes};

    // Do the loop over comp-direction
    for (int dir{0}; dir < 3; dir++)
    {
        for (amrex::MFIter mfi(field.m_data[dir], true); mfi.isValid(); ++mfi)
        {
            const amrex::Box& bx = mfi.growntilebox(); //projections are done also on ghost cells
            amrex::Array4<amrex::Real> const& oneForm = (field.m_data[dir])[mfi].array();

            const amrex::RealVect dr = m_dr;
            const amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> r0 = m_geom.ProbLoArray();

            ParallelFor(bx, nComps,
                        [=] AMREX_GPU_DEVICE(int i, int j, int k, int n)
                        {
                            // Compute the position of the point i + 1/2, j + 1/2, k + 1/2
                            std::array<amrex::Real, 3> r = {
                                GEMPIC_D_PAD(r0[xDir] + 0.5 * dr[xDir] + i * dr[xDir],
                                             r0[yDir] + 0.5 * dr[yDir] + j * dr[yDir],
                                             r0[zDir] + 0.5 * dr[zDir] + k * dr[zDir])};

#if (AMREX_SPACEDIM < 3)
                            if (dir < AMREX_SPACEDIM) // 1-forms in 1D, 2D are node centered in
                                                      // the last 1, 2 components respectively
#endif
                            {
                                auto f = [&] (amrex::Real val)
                                {
                                    switch (dir)
                                    {
                                        case Direction::xDir:
                                            return funcsGpuPtr[n][dir](
                                                AMREX_D_DECL(val, r[yDir], r[zDir]), t);
                                        case Direction::yDir:
                                            return funcsGpuPtr[n][dir](
                                                AMREX_D_DECL(r[xDir], val, r[zDir]), t);
                                        case Direction::zDir:
                                            return funcsGpuPtr[n][dir](
                                                AMREX_D_DECL(r[xDir], r[yDir], val), t);
                                        default:
                                            return 0.0;
                                    };
                                };
                                const amrex::Real drHalf = dr[dir] / 2;
                                // Midpoint for the quadrature rule
                                amrex::Real midpoint = r[dir] - drHalf;

                                // Rescale the integral and assign it to array of degrees of freedom
                                oneForm(i, j, k, n) = integrate.line(midpoint, drHalf, f);
                            }
#if (AMREX_SPACEDIM < 3)
                            else
                            {
                                oneForm(i, j, k, n) =
                                    funcsGpuPtr[n][dir](AMREX_D_DECL(r[xDir], r[yDir], r[zDir]), t);
                            }
#endif
                        });
        }
    }

    field.fill_boundary();
}

inline void FDDeRhamComplex::projection(
    amrex::Array<amrex::ParserExecutor<AMREX_SPACEDIM + 1>, 3>& func,
    amrex::Real t,
    DeRhamField<Grid::dual, Space::edge>& field,
    int gaussNodes)
{
    amrex::Vector<amrex::Array<amrex::ParserExecutor<AMREX_SPACEDIM + 1>, 3>> funcs{func};
    this->projection(funcs, t, field, gaussNodes);
}

/**
 * @brief Computes the Restriction operator \f$R_2\f$
 *
 * @details
 * Using the geometric degrees of freedeom:
 * Integrates func at time t over the faces of the primal grid by using a Gauss quadrature rule
 *
 * For dimensions 1 and 2, the 2-form is taken from the "stacked" de Rham complex
 *
 * @param funcs ParserExecutor, functions to be projected
 * @param t Real, time at which func is to be evaluated
 * @param field DeRhamField<primal, Space::face>, 2-form \f$u^2\f$ holding the face integrals
 * @param gaussNodes int, number of Gauss nodes to be used for quadrature
 */
void FDDeRhamComplex::projection (
    const amrex::Vector<amrex::Array<amrex::ParserExecutor<AMREX_SPACEDIM + 1>, 3>>& funcs,
    amrex::Real t,
    DeRhamField<Grid::primal, Space::face>& field,
    int gaussNodes)
{
    BL_PROFILE("Gempic::Forms::FDDeRhamComplex::projection(<primal, face>)");
    auto nComps{static_cast<size_t>(funcs.size())};
    int nCompField{field.m_data[xDir].nComp()};
    std::string msg = "The number of functions (" + std::to_string(nComps) +
                      ") doesn't correspond to the number of DeRhamField components (" +
                      std::to_string(nCompField) + ").";
    if (field.m_data[xDir].nComp() != funcs.size())
    {
        amrex::Assert(msg.c_str(), __FILE__, __LINE__);
    }

    amrex::Gpu::DeviceVector<amrex::Array<amrex::ParserExecutor<AMREX_SPACEDIM + 1>, 3>> funcsGpu{
        nComps};
    amrex::Gpu::copyAsync(amrex::Gpu::hostToDevice, funcs.begin(), funcs.end(), funcsGpu.begin());
    auto* const funcsGpuPtr = funcsGpu.dataPtr();

    GaussQuadrature integrate{gaussNodes};

    const amrex::RealVect dr = m_dr;
    const amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> r0 = m_geom.ProbLoArray();

    std::array<amrex::Real, 3> drHalf = {
        GEMPIC_D_PAD_ONE(dr[xDir] / 2, dr[yDir] / 2, dr[zDir] / 2)};
    for (int i{AMREX_SPACEDIM}; i < 3; i++) drHalf[i] = drHalf[i] / 2.0;

    // x-direction. Plane YZ
    // Can we guarantee the same distribution of boxes for in different directions for the same
    // field?
    // -> this would allow to use the same Iterator in all directions!
    for (amrex::MFIter mfi(field.m_data[xDir], true); mfi.isValid(); ++mfi)
    {
        const amrex::Box& bx = mfi.growntilebox();
        amrex::Array4<amrex::Real> const& twoForm = (field.m_data[xDir])[mfi].array();

        ParallelFor(
            bx, nComps,
            [=] AMREX_GPU_DEVICE(int i, int j, int k, int n)
            {
                // Compute the position of the point i, j, k
                std::array<amrex::Real, 3> r = {GEMPIC_D_PAD(
                    r0[xDir] + i * dr[xDir], r0[yDir] + j * dr[yDir], r0[zDir] + k * dr[zDir])};

                std::array<amrex::Real, 2> midpoint{r[yDir] + drHalf[yDir], r[zDir] + drHalf[zDir]};
                std::array<amrex::Real, 2> drHalfTmp{drHalf[yDir], drHalf[zDir]};

                auto f = [&] (amrex::Real y, amrex::Real z)
                { return funcsGpuPtr[n][xDir](AMREX_D_DECL(r[xDir], y, z), t); };

                twoForm(i, j, k, n) = integrate.surface(midpoint, drHalfTmp, f);
            });
    }

    // y-direction. Plane XZ
    for (amrex::MFIter mfi(field.m_data[yDir], true); mfi.isValid(); ++mfi)
    {
        const amrex::Box& bx = mfi.growntilebox();

        amrex::Array4<amrex::Real> const& twoForm = (field.m_data[yDir])[mfi].array();

        ParallelFor(
            bx, nComps,
            [=] AMREX_GPU_DEVICE(int i, int j, int k, int n)
            {
                // Compute the position of the point i, j, k
                std::array<amrex::Real, 3> r = {GEMPIC_D_PAD(
                    r0[xDir] + i * dr[xDir], r0[yDir] + j * dr[yDir], r0[zDir] + k * dr[zDir])};

                // Midpoint for the quadrature rule
                std::array<amrex::Real, 2> midpoint = {r[xDir] + drHalf[xDir],
                                                       r[zDir] + drHalf[zDir]};
                std::array<amrex::Real, 2> drHalfTmp{drHalf[xDir], drHalf[zDir]};

                auto f = [&] (amrex::Real x, amrex::Real z)
                { return funcsGpuPtr[n][yDir](AMREX_D_DECL(x, r[yDir], z), t); };

                twoForm(i, j, k, n) = integrate.surface(midpoint, drHalfTmp, f);
            });
    }

    // z-direction. Plane XY
    for (amrex::MFIter mfi(field.m_data[zDir], true); mfi.isValid(); ++mfi)
    {
        const amrex::Box& bx = mfi.growntilebox();
        amrex::Array4<amrex::Real> const& twoForm = (field.m_data[zDir])[mfi].array();

        ParallelFor(
            bx, nComps,
            [=] AMREX_GPU_DEVICE(int i, int j, int k, int n)
            {
                // Compute the position of the point i, j, k
                std::array<amrex::Real, 3> r{GEMPIC_D_PAD(
                    r0[xDir] + i * dr[xDir], r0[yDir] + j * dr[yDir], r0[zDir] + k * dr[zDir])};

                // Midpoint for the quadrature rule
                std::array<amrex::Real, 2> midpoint{r[xDir] + drHalf[xDir], r[yDir] + drHalf[yDir]};
                std::array<amrex::Real, 2> drHalfTmp{drHalf[xDir], drHalf[yDir]};
                auto f = [&] (amrex::Real x, amrex::Real y)
                { return funcsGpuPtr[n][zDir](AMREX_D_DECL(x, y, r[zDir]), t); };

                // Rescale the integral and assign it to array of degrees of freedom
                twoForm(i, j, k, n) = integrate.surface(midpoint, drHalfTmp, f);
            });
    }

    field.fill_boundary();
}

inline void FDDeRhamComplex::projection(
    amrex::Array<amrex::ParserExecutor<AMREX_SPACEDIM + 1>, 3>& func,
    amrex::Real t,
    DeRhamField<Grid::primal, Space::face>& field,
    int gaussNodes)
{
    amrex::Vector<amrex::Array<amrex::ParserExecutor<AMREX_SPACEDIM + 1>, 3>> funcs{func};
    this->projection(funcs, t, field, gaussNodes);
}

/**
 * @brief Computes the Restriction operator \f$R_1\f$
 *
 * @details
 * Using the geometric degrees of freedeom:
 * Integrates func at time t over the edges of the primal grid by using a Gauss quadrature rule
 *
 * For dimensions 1 and 2, the 1-form is taken from the "stacked" de Rham complex
 *
 * @param funcs ParserExecutor, functions to be projected
 * @param t Real, time at which func is to be evaluated
 * @param field DeRhamField<primal, Space::edge>, 1-form \f$u^1\f$ holding the edge integrals
 * @param gaussNodes int, number of Gauss nodes to be used for quadrature
 */
void FDDeRhamComplex::projection (
    const amrex::Vector<amrex::Array<amrex::ParserExecutor<AMREX_SPACEDIM + 1>, 3>>& funcs,
    amrex::Real t,
    DeRhamField<Grid::primal, Space::edge>& field,
    int gaussNodes)
{
    BL_PROFILE("Gempic::Forms::FDDeRhamComplex::projection(<primal, edge>)");
    auto nComps{static_cast<size_t>(funcs.size())};
    int nCompField{field.m_data[xDir].nComp()};
    std::string msg = "The number of functions (" + std::to_string(nComps) +
                      ") doesn't correspond to the number of DeRhamField components (" +
                      std::to_string(nCompField) + ").";
    if (field.m_data[xDir].nComp() != funcs.size())
    {
        amrex::Assert(msg.c_str(), __FILE__, __LINE__);
    }

    amrex::Gpu::DeviceVector<amrex::Array<amrex::ParserExecutor<AMREX_SPACEDIM + 1>, 3>> funcsGpu{
        nComps};
    amrex::Gpu::copyAsync(amrex::Gpu::hostToDevice, funcs.begin(), funcs.end(), funcsGpu.begin());
    auto* const funcsGpuPtr = funcsGpu.dataPtr();

    GaussQuadrature integrate{gaussNodes};

    // Do the loop over comp-direction
    for (int dir = 0; dir < 3; ++dir)
    {
        for (amrex::MFIter mfi(field.m_data[dir], true); mfi.isValid(); ++mfi)
        {
            const amrex::Box& bx = mfi.growntilebox(); //projections are done also on ghost cells
            amrex::Array4<amrex::Real> const& oneForm = (field.m_data[dir])[mfi].array();

            const amrex::RealVect dr = m_dr;
            const amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> r0 = m_geom.ProbLoArray();

            ParallelFor(
                bx, nComps,
                [=] AMREX_GPU_DEVICE(int i, int j, int k, int n)
                {
                    // compute the position of the point i, j, k
                    std::array<amrex::Real, AMREX_SPACEDIM> r = {AMREX_D_DECL(
                        r0[xDir] + i * dr[xDir], r0[yDir] + j * dr[yDir], r0[zDir] + k * dr[zDir])};

#if (AMREX_SPACEDIM < 3)
                    if (dir < AMREX_SPACEDIM) // 1-forms in 1D, 2D are node centered in the last
                                              // 1, 2 components respectively
#endif
                    {
                        auto f = [&] (amrex::Real val)
                        {
                            switch (dir)
                            {
                                case Direction::xDir:
                                    return funcsGpuPtr[n][dir](AMREX_D_DECL(val, r[yDir], r[zDir]),
                                                               t);
                                case Direction::yDir:
                                    return funcsGpuPtr[n][dir](AMREX_D_DECL(r[xDir], val, r[zDir]),
                                                               t);
                                case Direction::zDir:
                                    return funcsGpuPtr[n][dir](AMREX_D_DECL(r[xDir], r[yDir], val),
                                                               t);
                                default:
                                    return 0.0;
                            };
                        };
                        const amrex::Real drHalf = dr[dir] / 2;
                        // Midpoint for the quadrature rule
                        amrex::Real midpoint = r[dir] + drHalf;

                        // Rescale the integral and assign it to array of degrees of freedom
                        oneForm(i, j, k, n) = integrate.line(midpoint, drHalf, f);
                    }
#if (AMREX_SPACEDIM < 3)
                    else
                    {
                        oneForm(i, j, k, n) =
                            funcsGpuPtr[n][dir](AMREX_D_DECL(r[xDir], r[yDir], r[zDir]), t);
                    }
#endif
                });
        }
    }

    field.fill_boundary();
}

inline void FDDeRhamComplex::projection(
    amrex::Array<amrex::ParserExecutor<AMREX_SPACEDIM + 1>, 3>& func,
    amrex::Real t,
    DeRhamField<Grid::primal, Space::edge>& field,
    int gaussNodes)
{
    amrex::Vector<amrex::Array<amrex::ParserExecutor<AMREX_SPACEDIM + 1>, 3>> funcs{func};
    this->projection(funcs, t, field, gaussNodes);
}

/**
 * @brief Computes the Restriction operator \f$\tilde{R}_2\f$
 *
 * @details
 * Using the geometric degrees of freedeom:
 * Integrates func at time t over the faces of the dual grid by using a Gauss quadrature rule
 *
 * For dimensions 1 and 2, the 2-form is taken from the "stacked" de Rham complex
 *
 * @param funcs ParserExecutor, functions to be projected
 * @param t Real, time at which func is to be evaluated
 * @param field DeRhamField<dual, Space::face>, 2-form \f$\tilde{u}^2\f$ holding the face
 * integrals
 * @param gaussNodes int, number of Gauss nodes to be used for quadrature
 */
void FDDeRhamComplex::projection (
    const amrex::Vector<amrex::Array<amrex::ParserExecutor<AMREX_SPACEDIM + 1>, 3>>& funcs,
    amrex::Real t,
    DeRhamField<Grid::dual, Space::face>& field,
    int gaussNodes)
{
    BL_PROFILE("Gempic::Forms::FDDeRhamComplex::projection(<dual, face>)");
    auto nComps{static_cast<size_t>(funcs.size())};
    int nCompField{field.m_data[xDir].nComp()};
    std::string msg = "The number of functions (" + std::to_string(nComps) +
                      ") doesn't correspond to the number of DeRhamField components (" +
                      std::to_string(nCompField) + ").";
    if (field.m_data[xDir].nComp() != funcs.size())
    {
        amrex::Assert(msg.c_str(), __FILE__, __LINE__);
    }

    amrex::Gpu::DeviceVector<amrex::Array<amrex::ParserExecutor<AMREX_SPACEDIM + 1>, 3>> funcsGpu{
        nComps};
    amrex::Gpu::copyAsync(amrex::Gpu::hostToDevice, funcs.begin(), funcs.end(), funcsGpu.begin());
    auto* const funcsGpuPtr = funcsGpu.dataPtr();

    GaussQuadrature integrate{gaussNodes};

    const amrex::RealVect dr = m_dr;
    const amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> r0 = m_geom.ProbLoArray();

    std::array<amrex::Real, 3> drHalf{GEMPIC_D_PAD_ONE(dr[xDir] / 2, dr[yDir] / 2, dr[zDir] / 2)};
    for (int i{AMREX_SPACEDIM}; i < 3; i++) drHalf[i] = drHalf[i] / 2.0;

    // x-direction. Plane YZ
    for (amrex::MFIter mfi(field.m_data[xDir], true); mfi.isValid(); ++mfi)
    {
        const amrex::Box& bx = mfi.growntilebox();
        amrex::Array4<amrex::Real> const& twoForm = (field.m_data[xDir])[mfi].array();

        ParallelFor(
            bx, nComps,
            [=] AMREX_GPU_DEVICE(int i, int j, int k, int n)
            {
                // Compute the position of the point i, j, k
                std::array<amrex::Real, 3> r = {GEMPIC_D_PAD(
                    r0[xDir] + drHalf[xDir] + i * dr[xDir], r0[yDir] + drHalf[yDir] + j * dr[yDir],
                    r0[zDir] + drHalf[zDir] + k * dr[zDir])};

                // Midpoint for the quadrature rule. The minus for dual means that it integrates
                // from j-1/2 to j+1/2
                std::array<amrex::Real, 2> midpoint{r[yDir] - drHalf[yDir], r[zDir] - drHalf[zDir]};
                std::array<amrex::Real, 2> drHalfTmp{drHalf[yDir], drHalf[zDir]};

                auto f = [&] (amrex::Real y, amrex::Real z)
                { return funcsGpuPtr[n][xDir](AMREX_D_DECL(r[xDir], y, z), t); };

                twoForm(i, j, k, n) = integrate.surface(midpoint, drHalfTmp, f);
            });
    }

    // y-direction. Plane XZ
    for (amrex::MFIter mfi(field.m_data[yDir], true); mfi.isValid(); ++mfi)
    {
        const amrex::Box& bx = mfi.growntilebox();

        amrex::Array4<amrex::Real> const& twoForm = (field.m_data[yDir])[mfi].array();

        ParallelFor(
            bx, nComps,
            [=] AMREX_GPU_DEVICE(int i, int j, int k, int n)
            {
                // Compute the position of the point i, j, k
                std::array<amrex::Real, 3> r{GEMPIC_D_PAD(r0[xDir] + drHalf[xDir] + i * dr[xDir],
                                                          r0[yDir] + drHalf[yDir] + j * dr[yDir],
                                                          r0[zDir] + drHalf[zDir] + k * dr[zDir])};

                std::array<amrex::Real, 2> midpoint{r[xDir] - drHalf[xDir], r[zDir] - drHalf[zDir]};
                std::array<amrex::Real, 2> drHalfTmp{drHalf[xDir], drHalf[zDir]};

                auto f = [&] (amrex::Real x, amrex::Real z)
                { return funcsGpuPtr[n][yDir](AMREX_D_DECL(x, r[yDir], z), t); };

                twoForm(i, j, k, n) = integrate.surface(midpoint, drHalfTmp, f);
            });
    }

    // z-direction. Plane XY
    for (amrex::MFIter mfi(field.m_data[zDir], true); mfi.isValid(); ++mfi)
    {
        const amrex::Box& bx = mfi.growntilebox();
        amrex::Array4<amrex::Real> const& twoForm = (field.m_data[zDir])[mfi].array();

        ParallelFor(
            bx, nComps,
            [=] AMREX_GPU_DEVICE(int i, int j, int k, int n)
            {
                // Compute the position of the point i, j, k
                std::array<amrex::Real, 3> r{GEMPIC_D_PAD(r0[xDir] + drHalf[xDir] + i * dr[xDir],
                                                          r0[yDir] + drHalf[yDir] + j * dr[yDir],
                                                          r0[zDir] + drHalf[zDir] + k * dr[zDir])};

                // Midpoint for the quadrature rule
                std::array<amrex::Real, 2> midpoint{r[xDir] - drHalf[xDir], r[yDir] - drHalf[yDir]};
                std::array<amrex::Real, 2> drHalfTmp{drHalf[xDir], drHalf[yDir]};

                auto f = [&] (amrex::Real x, amrex::Real y)
                { return funcsGpuPtr[n][zDir](AMREX_D_DECL(x, y, r[zDir]), t); };

                // Rescale the integral and assign it to array of degrees of freedom
                twoForm(i, j, k, n) = integrate.surface(midpoint, drHalfTmp, f);
            });
    }
    field.fill_boundary();
}

inline void FDDeRhamComplex::projection(
    amrex::Array<amrex::ParserExecutor<AMREX_SPACEDIM + 1>, 3>& func,
    amrex::Real t,
    DeRhamField<Grid::dual, Space::face>& field,
    int gaussNodes)
{
    amrex::Vector<amrex::Array<amrex::ParserExecutor<AMREX_SPACEDIM + 1>, 3>> funcs{func};
    this->projection(funcs, t, field, gaussNodes);
}

void FDDeRhamComplex::hodge_scheme_selector (anyFieldConstRef f1, anyFieldRef f2, amrex::Real weight)
{
    BL_PROFILE("Gempic::Forms::FDDeRhamComplex::hodge_scheme_selector()");
    if (m_hodgeScheme == HodgeScheme::FDHodge)
    {
        hodge_degree_selector<HodgeScheme::FDHodge>(f1, f2, weight);
    }
    else if (m_hodgeScheme == HodgeScheme::GDECHodge)
    {
        hodge_degree_selector<HodgeScheme::GDECHodge>(f1, f2, weight);
    }
    else if (m_hodgeScheme == HodgeScheme::GDECLumpHodge)
    {
        hodge_degree_selector<HodgeScheme::GDECLumpHodge>(f1, f2, weight);
    }
}

void FDDeRhamComplex::hodge (DeRhamField<Grid::primal, Space::node>& zeroForm,
                            const DeRhamField<Grid::dual, Space::cell>& threeForm,
                            amrex::Real weight)
{
    BL_PROFILE("Gempic::Forms::FDDeRhamComplex::hodge(primal,node,dual,cell,weight)");
    hodge_scheme_selector(threeForm, zeroForm, weight);
}
void FDDeRhamComplex::hodge (DeRhamField<Grid::dual, Space::node>& zeroForm,
                            const DeRhamField<Grid::primal, Space::cell>& threeForm,
                            amrex::Real weight)
{
    BL_PROFILE("Gempic::Forms::FDDeRhamComplex::hodge(dual,node,primal,cell,weight)");
    hodge_scheme_selector(threeForm, zeroForm, weight);
}
void FDDeRhamComplex::hodge (DeRhamField<Grid::primal, Space::edge>& oneForm,
                            const DeRhamField<Grid::dual, Space::face>& twoForm,
                            amrex::Real weight)
{
    BL_PROFILE("Gempic::Forms::FDDeRhamComplex::hodge(primal,edge,dual,face,weight)");
    hodge_scheme_selector(twoForm, oneForm, weight);
}
void FDDeRhamComplex::hodge (DeRhamField<Grid::dual, Space::edge>& oneForm,
                            const DeRhamField<Grid::primal, Space::face>& twoForm,
                            amrex::Real weight)
{
    BL_PROFILE("Gempic::Forms::FDDeRhamComplex::hodge(dual,edge,primal,face,weight)");
    hodge_scheme_selector(twoForm, oneForm, weight);
}
void FDDeRhamComplex::hodge (DeRhamField<Grid::primal, Space::face>& twoForm,
                            const DeRhamField<Grid::dual, Space::edge>& oneForm,
                            amrex::Real weight)
{
    BL_PROFILE("Gempic::Forms::FDDeRhamComplex::hodge(primal,face,dual,edge,weight)");
    hodge_scheme_selector(oneForm, twoForm, weight);
}
void FDDeRhamComplex::hodge (DeRhamField<Grid::dual, Space::face>& twoForm,
                            const DeRhamField<Grid::primal, Space::edge>& oneForm,
                            amrex::Real weight)
{
    BL_PROFILE("Gempic::Forms::FDDeRhamComplex::hodge(dual,face,primal,edge,weight)");
    hodge_scheme_selector(oneForm, twoForm, weight);
}
void FDDeRhamComplex::hodge (DeRhamField<Grid::primal, Space::cell>& threeForm,
                            const DeRhamField<Grid::dual, Space::node>& zeroForm,
                            amrex::Real weight)
{
    BL_PROFILE("Gempic::Forms::FDDeRhamComplex::hodge(primal,cell,dual,node,weight)");
    hodge_scheme_selector(zeroForm, threeForm, weight);
}

void FDDeRhamComplex::hodge (DeRhamField<Grid::dual, Space::cell>& threeForm,
                            const DeRhamField<Grid::primal, Space::node>& zeroForm,
                            amrex::Real weight)
{
    BL_PROFILE("Gempic::Forms::FDDeRhamComplex::hodge(dual,cell,primal,node,weight)");
    hodge_scheme_selector(zeroForm, threeForm, weight);
}

void FDDeRhamComplex::eval_form (
    const DeRhamField<Grid::primal, Space::cell>& field,
    const amrex::Vector<amrex::GpuArray<amrex::Real, AMREX_SPACEDIM>>& evalShift,
    amrex::MultiFab& pointValues)
{
    this->eval_form_degree_selector(field, evalShift, pointValues);
}

void FDDeRhamComplex::eval_form (
    const DeRhamField<Grid::dual, Space::cell>& field,
    const amrex::Vector<amrex::GpuArray<amrex::Real, AMREX_SPACEDIM>>& evalShift,
    amrex::MultiFab& pointValues)
{
    this->eval_form_degree_selector(field, evalShift, pointValues);
}

void FDDeRhamComplex::eval_form (
    const DeRhamField<Grid::primal, Space::face>& field,
    const amrex::Vector<amrex::GpuArray<amrex::Real, AMREX_SPACEDIM>>& evalShift,
    amrex::MultiFab& pointValues,
    Direction dir)
{
    this->eval_form_direction_selector(field, evalShift, pointValues, dir);
}

void FDDeRhamComplex::eval_form (
    const DeRhamField<Grid::dual, Space::face>& field,
    const amrex::Vector<amrex::GpuArray<amrex::Real, AMREX_SPACEDIM>>& evalShift,
    amrex::MultiFab& pointValues,
    Direction dir)
{
    this->eval_form_direction_selector(field, evalShift, pointValues, dir);
}

void FDDeRhamComplex::eval_form (
    const DeRhamField<Grid::primal, Space::edge>& field,
    const amrex::Vector<amrex::GpuArray<amrex::Real, AMREX_SPACEDIM>>& evalShift,
    amrex::MultiFab& pointValues,
    Direction dir)
{
    this->eval_form_direction_selector(field, evalShift, pointValues, dir);
}

void FDDeRhamComplex::eval_form (
    const DeRhamField<Grid::dual, Space::edge>& field,
    const amrex::Vector<amrex::GpuArray<amrex::Real, AMREX_SPACEDIM>>& evalShift,
    amrex::MultiFab& pointValues,
    Direction dir)
{
    this->eval_form_direction_selector(field, evalShift, pointValues, dir);
}

void FDDeRhamComplex::eval_form (
    const DeRhamField<Grid::primal, Space::node>& field,
    const amrex::Vector<amrex::GpuArray<amrex::Real, AMREX_SPACEDIM>>& evalShift,
    amrex::MultiFab& pointValues)
{
    this->eval_form_degree_selector(field, evalShift, pointValues);
}

void FDDeRhamComplex::eval_form (
    const DeRhamField<Grid::dual, Space::node>& field,
    const amrex::Vector<amrex::GpuArray<amrex::Real, AMREX_SPACEDIM>>& evalShift,
    amrex::MultiFab& pointValues)
{
    this->eval_form_degree_selector(field, evalShift, pointValues);
}

amrex::Real FDDeRhamComplex::scaling_eto_d() { return 1 / (m_sOmega * m_sOmega); }

amrex::Real FDDeRhamComplex::scaling_dto_e() { return m_sOmega * m_sOmega; }

amrex::Real FDDeRhamComplex::scaling_bto_h() { return m_sV * m_sV / (m_sOmega * m_sOmega); }