// ---------------------------------------------------------------------
//
// Copyright (C) 2020 - 2023 by the deal.II authors
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



// Test DataOut::write_vtu() for a single simplex subdivided more than once.

#include <deal.II/dofs/dof_handler.h>

#include <deal.II/fe/fe_simplex_p.h>
#include <deal.II/fe/mapping_fe.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/tria.h>

#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/vector_tools.h>

#include "../tests.h"

using namespace dealii;

template <int dim>
class RightHandSideFunction : public Function<dim>
{
public:
  virtual double
  value(const Point<dim> &p, const unsigned int component = 0) const
  {
    return p * p;
  }
};


template <int dim, int spacedim = dim>
void
test()
{
  Triangulation<dim, spacedim> tria;
  GridGenerator::reference_cell(tria, ReferenceCells::get_simplex<dim>());

  // Use a quadratic element; this can correctly represent the
  // quadratic function we are interpolating
  FE_SimplexP<dim> fe(2);
  DoFHandler<dim>  dof_handler(tria);

  dof_handler.distribute_dofs(fe);

  Vector<double> solution(dof_handler.n_dofs());

  MappingFE<dim> mapping(FE_SimplexP<dim>(1));

  VectorTools::interpolate(mapping,
                           dof_handler,
                           RightHandSideFunction<dim>(),
                           solution);

  // Switch off compression to make the resulting file easier to read
  // as a human
  DataOutBase::VtkFlags flags;
  flags.compression_level = DataOutBase::CompressionLevel::plain_text;

  DataOut<dim> data_out;
  data_out.attach_dof_handler(dof_handler);
  data_out.add_data_vector(solution, "solution");
  data_out.build_patches(mapping, 2);

  data_out.set_flags(flags);
  data_out.write_vtu(deallog.get_file_stream());
}

int
main()
{
  initlog();

  test<2>();
  test<3>();
}
