.. role:: cpp(code)
   :language: c++

Vlasov-Maxwell
====================
The class ``vlasov_maxwell_simulation`` contains the Vlasov-Maxwell problem which is solved by a field solver.

The method that initializes the Vlasov Maxwell equation is:

:cpp:`initialize_vlasov_maxwell_from_file()`

The following method is used in tests
.. highlight:: c++

::

         initialize_gempic_structures(amrex::GpuArray<int, 2> funcSelectRho,
                                      amrex::GpuArray<int, int(vdim / 2.5) * 2 + 1> funcSelectB)

