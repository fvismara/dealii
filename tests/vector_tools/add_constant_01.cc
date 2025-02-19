// ---------------------------------------------------------------------
//
// Copyright (C) 2003 - 2023 by the deal.II authors
//
// This file is part of the deal.II library.
//
// The deal.II library is free software; you can use it, redistribute
// it, and/or modify it under the terms of the GNU Lesser General
// Public License as published by the Free Software Foundation; either
// version 2.1 of the License, or (at your option) any later version.
// The full text of the license can be found in the file LICENSE.md at
// the top level directory of deal.II.
//
// ---------------------------------------------------------------------

// Test VectorTools::add_constant()

#include <deal.II/base/function.h>
#include <deal.II/base/quadrature_lib.h>

#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_dgp.h>
#include <deal.II/fe/fe_dgq.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_raviart_thomas.h>
#include <deal.II/fe/fe_simplex_p.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/mapping_fe.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_refinement.h>
#include <deal.II/grid/tria.h>
#include <deal.II/grid/tria_accessor.h>

#include <deal.II/lac/block_vector.h>
#include <deal.II/lac/vector.h>

#include <deal.II/numerics/vector_tools.h>

#include "../tests.h"



// 0 (dim components)
// 1
// 2
// 3
template <int dim>
class Reference : public Function<dim>
{
public:
  Reference()
    : Function<dim>(dim + 3)
  {}

  double
  value(const Point<dim> &p, const unsigned int c) const
  {
    if (c == dim)
      return 1.0;
    if (c == dim + 1)
      return 2.0;
    if (c == dim + 2)
      return 3.0;
    return 0.0;
  }
};



template <int dim, class VectorType>
void
test()
{
  Triangulation<dim> tria;
  GridGenerator::hyper_cube(tria);
  tria.refine_global(2);

  FESystem<dim>   fe(FE_RaviartThomas<dim>(1),
                   1,
                   FE_Q<dim>(2),
                   1,
                   FE_DGQ<dim>(2),
                   1,
                   FE_DGP<dim>(1),
                   1);
  DoFHandler<dim> dofhandler(tria);
  dofhandler.distribute_dofs(fe);

  VectorType vec(dofhandler.n_dofs());
  vec = 0.0;
  for (unsigned int c = dim; c < dim + 3; ++c)
    VectorTools::add_constant(vec, dofhandler, c, 1.0 * (c - dim + 1));

  Vector<double> cellwise_errors(tria.n_active_cells());
  QIterated<dim> quadrature(QTrapezoid<1>(), 5);

  const dealii::Function<dim, double> *w = nullptr;
  VectorTools::integrate_difference(dofhandler,
                                    vec,
                                    Reference<dim>(),
                                    cellwise_errors,
                                    quadrature,
                                    VectorTools::L2_norm);

  const double error = VectorTools::compute_global_error(tria,
                                                         cellwise_errors,
                                                         VectorTools::L2_norm);

  AssertThrow(error < 1e-10, ExcMessage("Error in integrate_difference"));
}

template <int dim, class VectorType>
void
test_simplex()
{
  Triangulation<dim> tria;
  GridGenerator::subdivided_hyper_cube_with_simplices(tria, 4);

  FESystem<dim>   fe(FE_SimplexP<dim>(1),
                   dim,
                   FE_SimplexP<dim>(2),
                   1,
                   FE_SimplexP<dim>(1),
                   1,
                   FE_SimplexDGP<dim>(1),
                   1);
  DoFHandler<dim> dofhandler(tria);
  dofhandler.distribute_dofs(fe);

  VectorType vec(dofhandler.n_dofs());
  vec = 0.0;
  for (unsigned int c = dim; c < dim + 3; ++c)
    VectorTools::add_constant(vec, dofhandler, c, 1.0 * (c - dim + 1));

  Vector<double>     cellwise_errors(tria.n_active_cells());
  QGaussSimplex<dim> quadrature(2);

  const dealii::Function<dim, double> *w = nullptr;
  MappingFE<dim>                       mapping(FE_SimplexP<dim>(1));
  VectorTools::integrate_difference(mapping,
                                    dofhandler,
                                    vec,
                                    Reference<dim>(),
                                    cellwise_errors,
                                    quadrature,
                                    VectorTools::L2_norm);

  const double error = VectorTools::compute_global_error(tria,
                                                         cellwise_errors,
                                                         VectorTools::L2_norm);

  AssertThrow(error < 1e-10, ExcMessage("Error in integrate_difference"));
}


template <int dim>
void
big()
{
  test<dim, Vector<double>>();
  test_simplex<dim, Vector<double>>();
}


int
main(int argc, char **argv)
{
  initlog();

  big<2>();
  big<3>();

  deallog << "OK" << std::endl;
}
