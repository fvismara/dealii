// ---------------------------------------------------------------------
//
// Copyright (C) 2009 - 2023 by the deal.II authors
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



// We take one cell and create the system_matrix and the
// right-hand-side-vector.  But we have a few inhomogeneous
// constraints and want to test if the
// distribute_local_to_global-function supplies the correct
// right-hand-side-vector if the use_inhomogeneities_for_rhs-parameter
// is set to true or false.

#include <deal.II/base/function.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/utilities.h>

#include <deal.II/dofs/dof_accessor.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_q.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_refinement.h>
#include <deal.II/grid/tria.h>
#include <deal.II/grid/tria_accessor.h>
#include <deal.II/grid/tria_iterator.h>

#include <deal.II/lac/affine_constraints.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/vector.h>

#include <deal.II/numerics/error_estimator.h>
#include <deal.II/numerics/matrix_tools.h>
#include <deal.II/numerics/vector_tools.h>

#include <complex>
#include <iostream>

#include "../tests.h"

void
test(bool use_inhomogeneity_for_rhs)
{
  AffineConstraints<double> cm;

  cm.add_line(2);
  cm.set_inhomogeneity(2, 3.0);
  cm.add_line(3);
  cm.set_inhomogeneity(3, 0.0);
  cm.add_line(4);
  cm.add_entry(4, 2, 2.0);

  cm.close();
  cm.print(deallog.get_file_stream());


  DynamicSparsityPattern csp(5, 5);
  for (unsigned int i = 0; i < 5; ++i)
    csp.add(i, i);

  SparsityPattern sp;
  sp.copy_from(csp);
  SparseMatrix<double> mat(sp);
  Vector<double>       rhs(5);

  std::vector<types::global_dof_index> local_dofs;
  for (unsigned int i = 0; i < 5; ++i)
    local_dofs.push_back(i);

  FullMatrix<double> local_mat(5, 5);
  Vector<double>     local_vec(5);
  for (unsigned int i = 0; i < 5; ++i)
    local_mat(i, i) = 2.0;

  local_vec = 0;

  cm.distribute_local_to_global(
    local_mat, local_vec, local_dofs, mat, rhs, use_inhomogeneity_for_rhs);

  mat.print(deallog.get_file_stream());
  rhs.print(deallog.get_file_stream());
}


int
main()
{
  initlog();
  deallog << std::setprecision(2);
  deallog.get_file_stream() << std::setprecision(2);

  // Use the constraints for the right-hand-side
  {
    test(true);
  }

  // Don not use the constraints for the right-hand-side
  {
    test(false);
  }
}
