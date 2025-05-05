#include "GEMPIC_MaxwellInit.H"

AMREX_GPU_HOST_DEVICE inline amrex::GpuArray<amrex::Real, 3> initial_e (
    amrex::GpuArray<amrex::Real, AMREX_SPACEDIM>& r, const amrex::Real time)
{
    static const amrex::Real omega{2 * M_PI};
    static const amrex::Real kx{2 * M_PI};
    return {0.0, 0.0, cos(kx * r[0] - omega * time)};
}

AMREX_GPU_HOST_DEVICE inline amrex::GpuArray<amrex::Real, 3> initial_b (
    amrex::GpuArray<amrex::Real, AMREX_SPACEDIM>& r, const amrex::Real time)
{
    static const amrex::Real kx{2 * M_PI};
    static const amrex::Real omega{2 * M_PI};
    return {0.0, kx / omega * cos(kx * r[0] - omega * time), 0.0};
}

void maxwell_initial_condition (MaxwellDiscretization& disc)
{
    amrex::GpuArray<amrex::Real, 3> ndir;
    amrex::GpuArray<amrex::Real, 3> const dr{disc.m_compDom.cell_size_3darray()};
    amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const domainLength{
        disc.m_compDom.m_geom.ProbSize()};
    for (int idir = 0; idir < 3; idir++)
    {
        for (int jdir = 0; jdir < 3; jdir++)
        {
            ndir[jdir] = 0;
        }
        ndir[idir] = 1;
        // precompute face areas
        amrex::Real const dVol = disc.m_compDom.cell_volume();
        // divide by (normal) edge length to get the area of the normal face
        amrex::Real const dS = dVol / dr[idir];

        // LOOP OVER FACES:
        for (amrex::MFIter mfi(disc.m_myfields->m_e.m_data[idir]); mfi.isValid(); ++mfi)
        {
            const amrex::Box& bx = mfi.growntilebox();
            amrex::Real const edgeLoc{dr[idir]};
            amrex::Real const time{disc.m_time};
            amrex::Array4<amrex::Real> const& edir =
                (disc.m_myfields->m_e.m_data[idir])[mfi].array(); // define pointer to variable

            ParallelFor(bx,
                        [=] AMREX_GPU_DEVICE(int i, int j, int k)
                        {
                            amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> rScaled{
                                AMREX_D_DECL((static_cast<amrex::Real>(i) + 0.5 * ndir[xDir]) *
                                                 dr[xDir] / domainLength[xDir],
                                             (static_cast<amrex::Real>(j) + 0.5 * ndir[yDir]) *
                                                 dr[yDir] / domainLength[yDir],
                                             (static_cast<amrex::Real>(k) + 0.5 * ndir[zDir]) *
                                                 dr[zDir] / domainLength[zDir])};
                            amrex::GpuArray<amrex::Real, 3> eLoc;
                            eLoc = initial_e(rScaled, time);
                            edir(i, j, k) = eLoc[idir] * edgeLoc;
                        });
        }

        // LOOP OVER FACES:
        for (amrex::MFIter mfi(disc.m_myfields->m_b.m_data[idir]); mfi.isValid(); ++mfi)
        {
            const amrex::Box& bx = mfi.growntilebox();
            amrex::Real const time{disc.m_time};
            amrex::Array4<amrex::Real> const& bdir =
                (disc.m_myfields->m_b.m_data[idir])[mfi].array(); // define pointer to variable
            ParallelFor(bx,
                        [=] AMREX_GPU_DEVICE(int i, int j, int k)
                        {
                            amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> rScaled{AMREX_D_DECL(
                                (static_cast<amrex::Real>(i) + 0.5 - 0.5 * ndir[xDir]) * dr[xDir] /
                                    domainLength[xDir],
                                (static_cast<amrex::Real>(j) + 0.5 - 0.5 * ndir[yDir]) * dr[yDir] /
                                    domainLength[yDir],
                                (static_cast<amrex::Real>(k) + 0.5 - 0.5 * ndir[zDir]) * dr[zDir] /
                                    domainLength[zDir])};
                            amrex::GpuArray<amrex::Real, 3> bLoc;
                            bLoc = initial_b(rScaled, time);
                            bdir(i, j, k) = bLoc[idir] * dS;
                        });
        }
    }
}
