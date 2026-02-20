.. _api_Forms:

API: Forms and Operators
========================

In Gempic, the discrete de Rham complexes described in Section :ref:`sec:Maxwell_with_discrete_forms`
are of Finite Difference type, which means that the discrete forms are primarily characterized 
by their degrees of freedom on a (primal or dual) cartesian grid.

.. doxygenclass:: Gempic::DiscreteGrid
   :members:

.. doxygenfunction:: Gempic::fill(DiscreteField&, F const&)

.. doxygenclass:: Gempic::DiscreteField
   :members:

.. doxygenfunction:: Gempic::fill(DiscreteVectorField&, F const&)

.. doxygenclass:: Gempic::DiscreteVectorField
   :members:

.. doxygenclass:: Gempic::DiscreteFieldFunctionParser
.. doxygenclass:: Gempic::DiscreteVectorFieldFunctionParser

.. doxygenfile:: GEMPIC_FDDeRhamComplex.H
.. doxygenfile:: GEMPIC_ExtDerivatives.cpp
   
.. doxygenstruct:: Gempic::Forms::DeRhamField
   :members:

.. doxygenclass:: Gempic::Forms::DeRhamComplex
   :members:

.. doxygenfile:: GEMPIC_BoundaryConditions.H
.. doxygenclass:: Gempic::GaussLegendreQuadrature

.. toctree::
    :maxdepth: 2


