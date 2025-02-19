// ---------------------------------------------------------------------
//
// Copyright (C) 2003 - 2022 by the deal.II authors
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

// Multigrid for continuous finite elements without MeshWorker. Similar to the
// step-16 test but directly applying the constraints

#include <deal.II/base/function.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/utilities.h>

#include <deal.II/dofs/dof_accessor.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_values.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_refinement.h>
#include <deal.II/grid/manifold_lib.h>
#include <deal.II/grid/tria.h>
#include <deal.II/grid/tria_accessor.h>
#include <deal.II/grid/tria_iterator.h>

#include <deal.II/lac/affine_constraints.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/vector.h>

#include <deal.II/multigrid/mg_coarse.h>
#include <deal.II/multigrid/mg_matrix.h>
#include <deal.II/multigrid/mg_smoother.h>
#include <deal.II/multigrid/mg_tools.h>
#include <deal.II/multigrid/mg_transfer.h>
#include <deal.II/multigrid/multigrid.h>

#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/error_estimator.h>
#include <deal.II/numerics/vector_tools.h>

#include <sstream>

#include "../tests.h"


template <int dim>
class LaplaceProblem
{
public:
  LaplaceProblem(const unsigned int deg);
  void
  run();

private:
  void
  setup_system();
  void
  assemble_system();
  void
  assemble_multigrid();
  void
  solve();
  void
  refine_grid();
  void
  output_results(const unsigned int cycle) const;

  Triangulation<dim> triangulation;
  FE_Q<dim>          fe;
  DoFHandler<dim>    mg_dof_handler;

  SparsityPattern      sparsity_pattern;
  SparseMatrix<double> system_matrix;

  AffineConstraints<double> constraints;

  Vector<double> solution;
  Vector<double> system_rhs;

  const unsigned int degree;

  MGLevelObject<SparsityPattern>      mg_sparsity_patterns;
  MGLevelObject<SparseMatrix<double>> mg_matrices;
  MGLevelObject<SparsityPattern>      mg_interface_sparsity_patterns;
  MGLevelObject<SparseMatrix<double>> mg_interface_matrices;
  MGConstrainedDoFs                   mg_constrained_dofs;
};


template <int dim>
class Coefficient : public Function<dim>
{
public:
  Coefficient()
    : Function<dim>()
  {}

  virtual double
  value(const Point<dim> &p, const unsigned int component = 0) const;

  virtual void
  value_list(const std::vector<Point<dim>> &points,
             std::vector<double>           &values,
             const unsigned int             component = 0) const;
};



template <int dim>
double
Coefficient<dim>::value(const Point<dim> &p, const unsigned int) const
{
  if (p.square() < 0.5 * 0.5)
    return 20;
  else
    return 1;
}



template <int dim>
void
Coefficient<dim>::value_list(const std::vector<Point<dim>> &points,
                             std::vector<double>           &values,
                             const unsigned int             component) const
{
  const unsigned int n_points = points.size();

  Assert(values.size() == n_points,
         ExcDimensionMismatch(values.size(), n_points));

  Assert(component == 0, ExcIndexRange(component, 0, 1));

  for (unsigned int i = 0; i < n_points; ++i)
    values[i] = Coefficient<dim>::value(points[i]);
}


template <int dim>
LaplaceProblem<dim>::LaplaceProblem(const unsigned int degree)
  : triangulation(Triangulation<dim>::limit_level_difference_at_vertices)
  , fe(degree)
  , mg_dof_handler(triangulation)
  , degree(degree)
{}


template <int dim>
void
LaplaceProblem<dim>::setup_system()
{
  mg_dof_handler.distribute_dofs(fe);
  mg_dof_handler.distribute_mg_dofs();
  deallog << "Number of degrees of freedom: " << mg_dof_handler.n_dofs();

  for (unsigned int l = 0; l < triangulation.n_levels(); ++l)
    deallog << "   " << 'L' << l << ": " << mg_dof_handler.n_dofs(l);
  deallog << std::endl;

  sparsity_pattern.reinit(mg_dof_handler.n_dofs(),
                          mg_dof_handler.n_dofs(),
                          mg_dof_handler.max_couplings_between_dofs());
  DoFTools::make_sparsity_pattern(
    static_cast<const DoFHandler<dim> &>(mg_dof_handler), sparsity_pattern);

  solution.reinit(mg_dof_handler.n_dofs());
  system_rhs.reinit(mg_dof_handler.n_dofs());

  constraints.clear();
  DoFTools::make_hanging_node_constraints(mg_dof_handler, constraints);
  std::map<types::boundary_id, const Function<dim> *> dirichlet_boundary;
  Functions::ZeroFunction<dim> homogeneous_dirichlet_bc(1);
  dirichlet_boundary[0] = &homogeneous_dirichlet_bc;
  MappingQ<dim> mapping(1);
  VectorTools::interpolate_boundary_values(mapping,
                                           mg_dof_handler,
                                           dirichlet_boundary,
                                           constraints);
  constraints.close();
  constraints.condense(sparsity_pattern);
  sparsity_pattern.compress();
  system_matrix.reinit(sparsity_pattern);

  mg_constrained_dofs.clear();
  mg_constrained_dofs.initialize(mg_dof_handler);
  mg_constrained_dofs.make_zero_boundary_constraints(mg_dof_handler, {0});
}


template <int dim>
void
LaplaceProblem<dim>::assemble_system()
{
  const QGauss<dim> quadrature_formula(degree + 1);

  FEValues<dim> fe_values(fe,
                          quadrature_formula,
                          update_values | update_gradients |
                            update_quadrature_points | update_JxW_values);

  const unsigned int dofs_per_cell = fe.dofs_per_cell;
  const unsigned int n_q_points    = quadrature_formula.size();

  FullMatrix<double> cell_matrix(dofs_per_cell, dofs_per_cell);
  Vector<double>     cell_rhs(dofs_per_cell);

  std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);

  const Coefficient<dim> coefficient;
  std::vector<double>    coefficient_values(n_q_points);

  typename DoFHandler<dim>::active_cell_iterator cell = mg_dof_handler
                                                          .begin_active(),
                                                 endc = mg_dof_handler.end();
  for (; cell != endc; ++cell)
    {
      cell_matrix = 0;
      cell_rhs    = 0;

      fe_values.reinit(cell);

      coefficient.value_list(fe_values.get_quadrature_points(),
                             coefficient_values);

      for (unsigned int q_point = 0; q_point < n_q_points; ++q_point)
        for (unsigned int i = 0; i < dofs_per_cell; ++i)
          {
            for (unsigned int j = 0; j < dofs_per_cell; ++j)
              cell_matrix(i, j) +=
                (coefficient_values[q_point] *
                 fe_values.shape_grad(i, q_point) *
                 fe_values.shape_grad(j, q_point) * fe_values.JxW(q_point));

            cell_rhs(i) += (fe_values.shape_value(i, q_point) * 1.0 *
                            fe_values.JxW(q_point));
          }

      cell->get_dof_indices(local_dof_indices);
      constraints.distribute_local_to_global(
        cell_matrix, cell_rhs, local_dof_indices, system_matrix, system_rhs);
    }
}


template <int dim>
void
LaplaceProblem<dim>::assemble_multigrid()
{
  const unsigned int n_levels = triangulation.n_levels();

  mg_interface_matrices.resize(0, n_levels - 1);
  mg_interface_matrices.clear_elements();
  mg_matrices.resize(0, n_levels - 1);
  mg_matrices.clear_elements();
  mg_sparsity_patterns.resize(0, n_levels - 1);
  mg_interface_sparsity_patterns.resize(0, n_levels - 1);

  std::vector<AffineConstraints<double>> boundary_constraints(
    triangulation.n_levels());
  AffineConstraints<double> empty_constraints;
  for (unsigned int level = 0; level < n_levels; ++level)
    {
      boundary_constraints[level].add_lines(
        mg_constrained_dofs.get_refinement_edge_indices(level));
      boundary_constraints[level].add_lines(
        mg_constrained_dofs.get_boundary_indices(level));
      boundary_constraints[level].close();

      DynamicSparsityPattern csp;
      csp.reinit(mg_dof_handler.n_dofs(level), mg_dof_handler.n_dofs(level));
      MGTools::make_sparsity_pattern(
        mg_dof_handler, csp, level, boundary_constraints[level], false);
      mg_sparsity_patterns[level].copy_from(csp);

      csp.reinit(mg_dof_handler.n_dofs(level), mg_dof_handler.n_dofs(level));
      MGTools::make_sparsity_pattern(
        mg_dof_handler, csp, level, empty_constraints, true);
      mg_interface_sparsity_patterns[level].copy_from(csp);

      mg_matrices[level].reinit(mg_sparsity_patterns[level]);
      mg_interface_matrices[level].reinit(
        mg_interface_sparsity_patterns[level]);
    }

  QGauss<dim> quadrature_formula(1 + degree);

  FEValues<dim> fe_values(fe,
                          quadrature_formula,
                          update_values | update_gradients |
                            update_quadrature_points | update_JxW_values);

  const unsigned int dofs_per_cell = fe.dofs_per_cell;
  const unsigned int n_q_points    = quadrature_formula.size();

  FullMatrix<double> cell_matrix(dofs_per_cell, dofs_per_cell);

  std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);

  const Coefficient<dim> coefficient;
  std::vector<double>    coefficient_values(n_q_points);

  typename DoFHandler<dim>::cell_iterator cell = mg_dof_handler.begin(),
                                          endc = mg_dof_handler.end();

  for (; cell != endc; ++cell)
    {
      cell_matrix = 0;
      fe_values.reinit(cell);

      coefficient.value_list(fe_values.get_quadrature_points(),
                             coefficient_values);

      for (unsigned int q_point = 0; q_point < n_q_points; ++q_point)
        for (unsigned int i = 0; i < dofs_per_cell; ++i)
          for (unsigned int j = 0; j < dofs_per_cell; ++j)
            cell_matrix(i, j) +=
              (coefficient_values[q_point] * fe_values.shape_grad(i, q_point) *
               fe_values.shape_grad(j, q_point) * fe_values.JxW(q_point));

      cell->get_mg_dof_indices(local_dof_indices);

      boundary_constraints[cell->level()].distribute_local_to_global(
        cell_matrix, local_dof_indices, mg_matrices[cell->level()]);

      // The next step is again slightly more
      // obscure (but explained in the @ref
      // mg_paper): We need the remainder of
      // the operator that we just copied
      // into the <code>mg_matrices</code>
      // object, namely the part on the
      // interface between cells at the
      // current level and cells one level
      // coarser. This matrix exists in two
      // directions: for interior DoFs (index
      // $i$) of the current level to those
      // sitting on the interface (index
      // $j$), and the other way around. Of
      // course, since we have a symmetric
      // operator, one of these matrices is
      // the transpose of the other.
      //
      // The way we assemble these matrices
      // is as follows: since the are formed
      // from parts of the local
      // contributions, we first delete all
      // those parts of the local
      // contributions that we are not
      // interested in, namely all those
      // elements of the local matrix for
      // which not $i$ is an interface DoF
      // and $j$ is not. The result is one of
      // the two matrices that we are
      // interested in, and we then copy it
      // into the
      // <code>mg_interface_matrices</code>
      // object. The
      // <code>boundary_interface_constraints</code>
      // object at the same time makes sure
      // that we delete contributions from
      // all degrees of freedom that are not
      // only on the interface but also on
      // the external boundary of the domain.
      //
      // The last part to remember is how to
      // get the other matrix. Since it is
      // only the transpose, we will later
      // (in the <code>solve()</code>
      // function) be able to just pass the
      // transpose matrix where necessary.
      const unsigned int lvl = cell->level();

      for (unsigned int i = 0; i < dofs_per_cell; ++i)
        for (unsigned int j = 0; j < dofs_per_cell; ++j)
          if (mg_constrained_dofs.at_refinement_edge(lvl,
                                                     local_dof_indices[i]) &&
              !mg_constrained_dofs.at_refinement_edge(lvl,
                                                      local_dof_indices[j]) &&
              ((!mg_constrained_dofs.is_boundary_index(lvl,
                                                       local_dof_indices[i]) &&
                !mg_constrained_dofs.is_boundary_index(
                  lvl,
                  local_dof_indices[j])) // ( !boundary(i) && !boundary(j) )
               ||
               (mg_constrained_dofs.is_boundary_index(lvl,
                                                      local_dof_indices[i]) &&
                local_dof_indices[i] ==
                  local_dof_indices[j]) // ( boundary(i) && boundary(j) &&
                                        // i==j )
               ))
            {
              // do nothing, so add entries to interface matrix
            }
          else
            {
              cell_matrix(i, j) = 0;
              std::cout << i << ' ' << j << "\n";
            }


      empty_constraints.distribute_local_to_global(
        cell_matrix, local_dof_indices, mg_interface_matrices[cell->level()]);
    }
}



// @sect4{LaplaceProblem::solve}

// This is the other function that is
// significantly different in support of the
// multigrid solver (or, in fact, the
// preconditioner for which we use the
// multigrid method).
//
// Let us start out by setting up two of the
// components of multilevel methods: transfer
// operators between levels, and a solver on
// the coarsest level. In finite element
// methods, the transfer operators are
// derived from the finite element function
// spaces involved and can often be computed
// in a generic way independent of the
// problem under consideration. In that case,
// we can use the MGTransferPrebuilt class
// that, given the constraints on the global
// level and an DoFHandler object computes
// the matrices corresponding to these
// transfer operators.
//
// The second part of the following lines
// deals with the coarse grid solver. Since
// our coarse grid is very coarse indeed, we
// decide for a direct solver (a Householder
// decomposition of the coarsest level
// matrix), even if its implementation is not
// particularly sophisticated. If our coarse
// mesh had many more cells than the five we
// have here, something better suited would
// obviously be necessary here.
template <int dim>
void
LaplaceProblem<dim>::solve()
{
  MGTransferPrebuilt<Vector<double>> mg_transfer(mg_constrained_dofs);
  mg_transfer.build(mg_dof_handler);

  FullMatrix<double> coarse_matrix;
  coarse_matrix.copy_from(mg_matrices[0]);
  MGCoarseGridHouseholder<> coarse_grid_solver;
  coarse_grid_solver.initialize(coarse_matrix);

  using Smoother = PreconditionSOR<SparseMatrix<double>>;
  GrowingVectorMemory<> vector_memory;
  MGSmootherRelaxation<SparseMatrix<double>, Smoother, Vector<double>>
    mg_smoother;
  mg_smoother.initialize(mg_matrices);
  mg_smoother.set_steps(2);
  mg_smoother.set_symmetric(true);

  mg::Matrix<> mg_matrix(mg_matrices);
  mg::Matrix<> mg_interface_up(mg_interface_matrices);
  mg::Matrix<> mg_interface_down(mg_interface_matrices);

  Multigrid<Vector<double>> mg(
    mg_matrix, coarse_grid_solver, mg_transfer, mg_smoother, mg_smoother);
  mg.set_edge_matrices(mg_interface_down, mg_interface_up);

  PreconditionMG<dim, Vector<double>, MGTransferPrebuilt<Vector<double>>>
    preconditioner(mg_dof_handler, mg, mg_transfer);

  SolverControl solver_control(1000, 1e-12);
  SolverCG<>    cg(solver_control);

  solution = 0;

  cg.solve(system_matrix, solution, system_rhs, preconditioner);
  constraints.distribute(solution);

  deallog << "   " << solver_control.last_step()
          << " CG iterations needed to obtain convergence." << std::endl;
}



// @sect4{Postprocessing}

// The following two functions postprocess a
// solution once it is computed. In
// particular, the first one refines the mesh
// at the beginning of each cycle while the
// second one outputs results at the end of
// each such cycle. The functions are almost
// unchanged from those in step-6, with the
// exception of two minor differences: The
// KellyErrorEstimator::estimate function
// wants an argument of type DoFHandler, not
// DoFHandler, and so we have to cast from
// derived to base class; and we generate
// output in VTK format, to use the more
// modern visualization programs available
// today compared to those that were
// available when step-6 was written.
template <int dim>
void
LaplaceProblem<dim>::refine_grid()
{
  Vector<float> estimated_error_per_cell(triangulation.n_active_cells());

  KellyErrorEstimator<dim>::estimate(
    mg_dof_handler,
    QGauss<dim - 1>(3),
    std::map<types::boundary_id, const Function<dim> *>(),
    solution,
    estimated_error_per_cell);
  GridRefinement::refine_and_coarsen_fixed_number(triangulation,
                                                  estimated_error_per_cell,
                                                  0.3,
                                                  0.03);
  triangulation.execute_coarsening_and_refinement();
}



template <int dim>
void
LaplaceProblem<dim>::output_results(const unsigned int cycle) const
{
  DataOut<dim> data_out;

  data_out.attach_dof_handler(mg_dof_handler);
  data_out.add_data_vector(solution, "solution");
  data_out.build_patches();

  std::ostringstream filename;
  filename << "solution-" << cycle << ".vtk";

  //  std::ofstream output (filename.str());
  //  data_out.write_vtk (output);
}


// @sect4{LaplaceProblem::run}

// Like several of the functions above, this
// is almost exactly a copy of of the
// corresponding function in step-6. The only
// difference is the call to
// <code>assemble_multigrid</code> that takes
// care of forming the matrices on every
// level that we need in the multigrid
// method.
template <int dim>
void
LaplaceProblem<dim>::run()
{
  for (unsigned int cycle = 0; cycle < 8; ++cycle)
    {
      deallog << "Cycle " << cycle << ':' << std::endl;

      if (cycle == 0)
        {
          GridGenerator::hyper_ball(triangulation);

          static const SphericalManifold<dim> boundary;
          triangulation.set_manifold(0, boundary);

          triangulation.refine_global(1);
        }
      else
        refine_grid();


      deallog << "   Number of active cells:       "
              << triangulation.n_active_cells() << std::endl;

      setup_system();

      deallog << "   Number of degrees of freedom: " << mg_dof_handler.n_dofs()
              << " (by level: ";
      for (unsigned int level = 0; level < triangulation.n_levels(); ++level)
        deallog << mg_dof_handler.n_dofs(level)
                << (level == triangulation.n_levels() - 1 ? ")" : ", ");
      deallog << std::endl;

      assemble_system();
      assemble_multigrid();

      solve();
      output_results(cycle);
    }
}


// @sect3{The main() function}
//
// This is again the same function as
// in step-6:
int
main()
{
  initlog();
  deallog << std::setprecision(4);

  try
    {
      LaplaceProblem<2> laplace_problem(1);
      laplace_problem.run();
    }
  catch (const std::exception &exc)
    {
      std::cerr << std::endl
                << std::endl
                << "----------------------------------------------------"
                << std::endl;
      std::cerr << "Exception on processing: " << std::endl
                << exc.what() << std::endl
                << "Aborting!" << std::endl
                << "----------------------------------------------------"
                << std::endl;

      return 1;
    }
  catch (...)
    {
      std::cerr << std::endl
                << std::endl
                << "----------------------------------------------------"
                << std::endl;
      std::cerr << "Unknown exception!" << std::endl
                << "Aborting!" << std::endl
                << "----------------------------------------------------"
                << std::endl;
      return 1;
    }

  return 0;
}
