// This file solves the Poisson equation -\Delta u = f on the 2d unit square [0,1]x[0,1] using SIPG
// and a matrix free implementation. 

#include <deal.II/base/convergence_table.h>
#include <deal.II/fe/fe_dgq.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/matrix_free/fe_evaluation.h>
#include <deal.II/matrix_free/operators.h>
#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/vector_tools.h>

#include "fe_dgq_aniso.h"
#include "fe_evaluation_aniso.h"

#include <iostream>
#include <fstream>
#include <typeinfo>

using namespace dealii;


const unsigned int dim = 2;
const unsigned int fe_degree_x = 3, fe_degree_z = 2;


/*----------------------------------------------------- Solution -------------------------------------------------------------*/
template <int dim>
class Solution : public Function<dim>
{
public:
  virtual double value(const Point<dim> &p,
                       const unsigned int = 0) const override final
  {
    double val = 1.;
    for (unsigned int d = 0; d < dim; ++d)
      val *= std::cos(numbers::PI * 2.4 * p[d]);
    return val;
  }

  virtual Tensor<1, dim> gradient(const Point<dim> &p,
                                  const unsigned int = 0) const override final
  {
    const double   arg = numbers::PI * 2.4;
    Tensor<1, dim> grad;
    for (unsigned int d = 0; d < dim; ++d)
      {
        grad[d] = 1.;
        for (unsigned int e = 0; e < dim; ++e)
          if (d == e)
            grad[d] *= -arg * std::sin(arg * p[e]);
          else
            grad[d] *= std::cos(arg * p[e]);
      }
    return grad;
  }
};


/*------------------------------------------------------ Right-hand side ---------------------------------------------------*/
template <int dim>
class RightHandSide : public Function<dim>
{
public:
  virtual double value(const Point<dim> &p,
                       const unsigned int = 0) const override final
  {
    const double arg = numbers::PI * 2.4;
    double       val = 1.;
    for (unsigned int d = 0; d < dim; ++d)
      val *= std::cos(arg * p[d]);
    return dim * arg * arg * val;
  }
};



/*---------------------------------------------- LaplaceOperator class -------------------------------------------------------*/
template <int dim, int fe_degree_x, int fe_degree_z, typename number>
class LaplaceOperator : public EnableObserverPointer//MatrixFreeOperators::Base<dim, LinearAlgebra::distributed::Vector<number>>
{
public:
  using value_type = number;

  LaplaceOperator() = default;

  void initialize(std::shared_ptr<const MatrixFree<dim, number>> data);

  void clear();

  void initialize_dof_vector(LinearAlgebra::distributed::Vector<number> &vec) const;

  std::shared_ptr<const MatrixFree<dim, number>> get_matrix_free() const;

  void vmult(LinearAlgebra::distributed::Vector<number>       &dst,
                 const LinearAlgebra::distributed::Vector<number> &src) const;

  number get_penalty_factor() const
  {
    return 1.0 * std::max(fe_degree_x, fe_degree_z) * (std::max(fe_degree_x, fe_degree_z) + 1);
  }

  // epsilon parameter in interior penalized formulation
  const number epsilon_IP = -1; 

private:
  void
  apply_cell(const MatrixFree<dim, number>                    &data,
             LinearAlgebra::distributed::Vector<number>       &dst,
             const LinearAlgebra::distributed::Vector<number> &src,
             const std::pair<unsigned int, unsigned int> &cell_range) const;

  void
  apply_face(const MatrixFree<dim, number>                    &data,
             LinearAlgebra::distributed::Vector<number>       &dst,
             const LinearAlgebra::distributed::Vector<number> &src,
             const std::pair<unsigned int, unsigned int> &face_range) const;

  void
  apply_boundary(const MatrixFree<dim, number>                    &data,
                 LinearAlgebra::distributed::Vector<number>       &dst,
                 const LinearAlgebra::distributed::Vector<number> &src,
                 const std::pair<unsigned int, unsigned int>      &face_range) const;

  std::shared_ptr<const MatrixFree<dim, number>> data;

};

template <int dim, int fe_degree_x, int fe_degree_z, typename number>
void LaplaceOperator<dim, fe_degree_x, fe_degree_z, number>::initialize(
  std::shared_ptr<const MatrixFree<dim, number>> data)
{
  this->data = data;
}

template <int dim, int fe_degree_x, int fe_degree_z, typename number>
void LaplaceOperator<dim, fe_degree_x, fe_degree_z, number>::clear()
{
  data.reset();
}

template <int dim, int fe_degree_x, int fe_degree_z, typename number>
void LaplaceOperator<dim, fe_degree_x, fe_degree_z, number>::initialize_dof_vector(
  LinearAlgebra::distributed::Vector<number> &vec) const
{
  data->initialize_dof_vector(vec);
}

template <int dim, int fe_degree_x, int fe_degree_z, typename number>
std::shared_ptr<const MatrixFree<dim, number>>
LaplaceOperator<dim, fe_degree_x, fe_degree_z, number>::get_matrix_free() const
{
  return data;
}

template <int dim, int fe_degree_x, int fe_degree_z, typename number>
void LaplaceOperator<dim, fe_degree_x, fe_degree_z, number>::vmult(
  LinearAlgebra::distributed::Vector<number>       &dst,
  const LinearAlgebra::distributed::Vector<number> &src) const
{
  data->loop(&LaplaceOperator::apply_cell,
             &LaplaceOperator::apply_face,
             &LaplaceOperator::apply_boundary,
             this,
             dst,
             src,
             /*zero_dst =*/true,
             MatrixFree<dim, number>::DataAccessOnFaces::gradients,
             MatrixFree<dim, number>::DataAccessOnFaces::gradients);
}


template <int dim, int fe_degree_x, int fe_degree_z, typename number>
void LaplaceOperator<dim, fe_degree_x, fe_degree_z, number>::apply_cell(
  const MatrixFree<dim, number>                    &data,
  LinearAlgebra::distributed::Vector<number>       &dst,
  const LinearAlgebra::distributed::Vector<number> &src,
  const std::pair<unsigned int, unsigned int>      &cell_range) const
{
  FEEvaluationAniso<dim, fe_degree_x, fe_degree_x + 1, fe_degree_z, fe_degree_z + 1, 1, number> phi(data);

  for (unsigned int cell = cell_range.first; cell < cell_range.second; ++cell)
    {
      phi.reinit(cell);
      phi.read_dof_values(src);
      phi.evaluate(EvaluationFlags::gradients);
      for (const unsigned int q : phi.quadrature_point_indices())
        phi.submit_gradient(phi.get_gradient(q), q);
      phi.integrate(EvaluationFlags::gradients);
      phi.distribute_local_to_global(dst);
    }
}


template <int dim, int fe_degree_x, int fe_degree_z, typename number>
void LaplaceOperator<dim, fe_degree_x, fe_degree_z, number>::apply_face(
  const MatrixFree<dim, number>                    &data,
  LinearAlgebra::distributed::Vector<number>       &dst,
  const LinearAlgebra::distributed::Vector<number> &src,
  const std::pair<unsigned int, unsigned int>      &face_range) const
{
  FEFaceEvaluationAniso<dim, fe_degree_x, fe_degree_x + 1, fe_degree_z, fe_degree_z + 1, 1, number>
    phi_outer(data, false);
  FEFaceEvaluationAniso<dim, fe_degree_x, fe_degree_x + 1, fe_degree_z, fe_degree_z + 1, 1, number>
    phi_inner(data, true);

  for (unsigned int inner_face = face_range.first; inner_face < face_range.second; ++inner_face) {

    // reinit on face
    phi_inner.reinit(inner_face);
    phi_outer.reinit(inner_face);

    // read dof values
    phi_inner.read_dof_values(src);
    phi_outer.read_dof_values(src);

    // evaluate
    phi_inner.evaluate(EvaluationFlags::values | EvaluationFlags::gradients);
    phi_outer.evaluate(EvaluationFlags::values | EvaluationFlags::gradients);

    // determine the correct number of quadrature points, i.e., whether we are on a vertical or horizontal face
    unsigned int face_q_points = phi_inner.is_vertical ? fe_degree_z + 1 : fe_degree_x + 1;

    const VectorizedArray<number> inverse_length_normal_to_face =
            0.5 * (std::abs((phi_inner.normal_vector(0) *
                             phi_inner.inverse_jacobian(0))[dim - 1]) +
                   std::abs((phi_outer.normal_vector(0) *
                             phi_outer.inverse_jacobian(0))[dim - 1]));
    const VectorizedArray<number> sigma = inverse_length_normal_to_face * get_penalty_factor();

    // submit_values / submit_gradients
    for (unsigned int q = 0; q < face_q_points; q++)
    {
      const VectorizedArray<number> solution_jump = (phi_inner.get_value(q) - phi_outer.get_value(q));
      const VectorizedArray<number> average_normal_derivative = (phi_inner.get_normal_derivative(q) +
                                                                 phi_outer.get_normal_derivative(q)) * number(0.5);
      const VectorizedArray<number> test_by_value = solution_jump * sigma - average_normal_derivative;

      phi_inner.submit_value(test_by_value, q);
      phi_outer.submit_value(-test_by_value, q);

      phi_inner.submit_normal_derivative(epsilon_IP * solution_jump * number(0.5), q);
      phi_outer.submit_normal_derivative(epsilon_IP * solution_jump * number(0.5), q);
    }

    // integrate
    phi_inner.integrate(EvaluationFlags::values | EvaluationFlags::gradients);
    phi_outer.integrate(EvaluationFlags::values | EvaluationFlags::gradients);

    // distribute_local_to_global
    phi_inner.distribute_local_to_global(dst);
    phi_outer.distribute_local_to_global(dst);
  }
}


template <int dim, int fe_degree_x, int fe_degree_z, typename number>
void LaplaceOperator<dim, fe_degree_x, fe_degree_z, number>::apply_boundary(
      const MatrixFree<dim, number>                    &data,
      LinearAlgebra::distributed::Vector<number>       &dst,
      const LinearAlgebra::distributed::Vector<number> &src,
      const std::pair<unsigned int, unsigned int>      &face_range) const
{
  FEFaceEvaluationAniso<dim, fe_degree_x, fe_degree_x + 1, fe_degree_z, fe_degree_z + 1, 1, number>
    phi_inner(data, true);
  for (unsigned int face = face_range.first; face < face_range.second; ++face)
    {
      // reinit on face
    phi_inner.reinit(face);

    // read dof values
    phi_inner.read_dof_values(src);
    phi_inner.evaluate(EvaluationFlags::values | EvaluationFlags::gradients);

    const VectorizedArray<number> inverse_length_normal_to_face = std::abs((
      phi_inner.normal_vector(0) * phi_inner.inverse_jacobian(0))[dim - 1]);
    const VectorizedArray<number> sigma = inverse_length_normal_to_face * get_penalty_factor();

    const bool is_dirichlet = (data.get_boundary_id(face) == 0);

    // get number of quadrature points according to the direction of the face
    unsigned int face_q_points = phi_inner.is_vertical ? fe_degree_z + 1 : fe_degree_x + 1;

    for (unsigned int q = 0; q < face_q_points; q++)
    {
      const VectorizedArray<number> u_inner = phi_inner.get_value(q);
      const VectorizedArray<number> u_outer = is_dirichlet ? -u_inner : u_inner;
      const VectorizedArray<number> normal_derivative_inner = phi_inner.get_normal_derivative(q);
      const VectorizedArray<number> normal_derivative_outer = is_dirichlet ? normal_derivative_inner : -normal_derivative_inner;
      const VectorizedArray<number> solution_jump = (u_inner - u_outer);
      const VectorizedArray<number> average_normal_derivative = (normal_derivative_inner + normal_derivative_outer) * number(0.5);
      const VectorizedArray<number> test_by_value = solution_jump * sigma - average_normal_derivative;
      phi_inner.submit_normal_derivative(epsilon_IP * solution_jump * number(0.5), q);
      phi_inner.submit_value(test_by_value, q);
    }

    // integrate
    phi_inner.integrate(EvaluationFlags::values | EvaluationFlags::gradients);

    // distribute_local_to_global
    phi_inner.distribute_local_to_global(dst);
    }
}


/*---------------------------------------------- LaplaceProblem class -------------------------------------------------------*/
template <int dim>
class LaplaceProblem
{
  public:
    LaplaceProblem();
    void run(const double Lx, const double Lz, const unsigned Nx, const unsigned Nz);
    double get_error() {return L2_error;};

  private:
    void setup_system();
    void assemble_rhs();
    void solve();
    void compute_error();

    Triangulation<dim> triangulation;

    const FE_DGQ_Aniso<dim> fe;
    DoFHandler<dim> dof_handler;

    FE_DGQLegendre<1> fe_x, fe_z;
    QGauss<1> quad_x, quad_z;

    double L2_error = 0.0;

    const MappingQ1<dim> mapping;
    using SystemMatrixType = LaplaceOperator<dim, fe_degree_x, fe_degree_z, double>;
    SystemMatrixType system_matrix;

    LinearAlgebra::distributed::Vector<double> solution;
    LinearAlgebra::distributed::Vector<double> system_rhs;

};


template <int dim>
LaplaceProblem<dim>::LaplaceProblem()
  : fe(fe_degree_x, fe_degree_z,
       AnisotropicPolynomials<dim>({Polynomials::Legendre::generate_complete_basis(fe_degree_x),
                                    Polynomials::Legendre::generate_complete_basis(fe_degree_z)}))
  , dof_handler(triangulation)
  , fe_x(fe_degree_x), fe_z(fe_degree_z)
  , quad_x(fe_degree_x+1), quad_z(fe_degree_z+1)
{}


template <int dim>
void LaplaceProblem<dim>::setup_system()
{
  system_matrix.clear();

  dof_handler.distribute_dofs(fe);

  std::cout << "Number of degrees of freedom: " << dof_handler.n_dofs() << std::endl;

  AffineConstraints<double> dummy;
  dummy.close();

  const std::vector<FE_DGQLegendre<1>> fe_vector{fe_x, fe_z};
  const std::vector<QGauss<1>> q_vector{quad_x, quad_z};

  typename MatrixFree<dim, double>::AdditionalData additional_data;
  additional_data.tasks_parallel_scheme = MatrixFree<dim, double>::AdditionalData::none;
  additional_data.mapping_update_flags = (update_values | update_gradients | update_JxW_values | update_quadrature_points);
  additional_data.mapping_update_flags_inner_faces = (update_values | update_gradients | update_JxW_values | update_normal_vectors);
  additional_data.mapping_update_flags_boundary_faces = (update_values | update_gradients | update_JxW_values | update_normal_vectors | update_quadrature_points);
  const auto system_mf_storage = std::make_shared<MatrixFree<dim, double>>();
  system_mf_storage->reinit_new(mapping, dof_handler, dummy, fe_vector, q_vector, additional_data);

  system_matrix.initialize(system_mf_storage);

  system_matrix.initialize_dof_vector(solution);
  system_matrix.initialize_dof_vector(system_rhs);

}


template <int dim>
void LaplaceProblem<dim>::assemble_rhs()
{
  system_rhs = 0;
  const MatrixFree<dim, double> &data = *system_matrix.get_matrix_free();
  FEEvaluationAniso<dim, fe_degree_x, fe_degree_x + 1, fe_degree_z, fe_degree_z + 1, 1, double> phi(data);
  RightHandSide<dim> rhs_func;
  Solution<dim> exact_solution;

  // volume term
  for (unsigned int cell = 0; cell < data.n_cell_batches(); ++cell)
    {
      phi.reinit(cell);
      for (const unsigned int q : phi.quadrature_point_indices())
        {
          VectorizedArray<double> rhs_val = VectorizedArray<double>();
          // we want to evaluate the rhs at the quadrature points in the physical space
          for (unsigned int v = 0; v < data.n_active_entries_per_cell_batch(cell); ++v)
            {
              typename DoFHandler< dim >::cell_iterator cellit = data.get_cell_iterator(cell, v);
              // when accessing Quadrature objects through the quadrature point index (e.g. Quadrature<2>(quad_x,quad_z).point(q)
              // or Quadrature<2>(quad_x,quad_z).JxW(q)), the x-index is traversed first, then the z-index. In other words, the
              // 2d quadrature is obtained as quad_z \kron quad_x. In our framework, we always consider tensor products of the 
              // form x \kron z, so we need to make sure that we transform the quadrature index q so that point() and JxW()
              // access the correct quantity
              int q_transf = q*(fe_degree_x+1) - int(std::floor(q/(fe_degree_z+1)))*((fe_degree_x+1)*(fe_degree_z+1)-1);
              // get quadrature point in the physical space
              Point<dim> single_point = mapping.transform_unit_to_real_cell(cellit, Quadrature<dim>(quad_x,quad_z).point(q_transf));
              rhs_val[v] = rhs_func.value(single_point);
            }
          phi.submit_value(rhs_val, q);
        }
      phi.integrate(EvaluationFlags::values);
      phi.distribute_local_to_global(system_rhs);
    }

  // contribution of boundary conditions to the rhs vector
  FEFaceEvaluationAniso<dim, fe_degree_x, fe_degree_x + 1, fe_degree_z, fe_degree_z + 1, 1, double> phi_face(data, true);
  for (unsigned int face = data.n_inner_face_batches(); face < data.n_inner_face_batches() + data.n_boundary_face_batches();
           ++face)
  {
    phi_face.reinit(face);

    const VectorizedArray<double> inverse_length_normal_to_face = std::abs(
      (phi_face.normal_vector(0) * phi_face.inverse_jacobian(0))[dim - 1]);
    const VectorizedArray<double> sigma =
      inverse_length_normal_to_face * system_matrix.get_penalty_factor();

    unsigned int face_q_points = phi_face.is_vertical ? fe_degree_z + 1 : fe_degree_x + 1;

    for (unsigned int q = 0; q < face_q_points; q++)
      {
        VectorizedArray<double> test_value = VectorizedArray<double>(),
                                test_normal_derivative = VectorizedArray<double>();
        // we want to evaluate the boundary values at quadrature points in the physical space
        for (unsigned int v = 0; v < data.n_active_entries_per_face_batch(face); ++v)
          {
            Point<dim> single_point;
            std::pair< typename DoFHandler< dim >::cell_iterator, unsigned int > pair_test = data.get_face_iterator(face, v);
            auto faceit = *(pair_test.first->face(pair_test.second));
            // transform the quadrature point from the reference interval [0,1] to the present boundary face
            if (!phi_face.is_vertical)
            {
              // horizontal face
              single_point[0] = quad_x.point(q)[0]*(faceit.vertex(1)[0] - faceit.vertex(0)[0]) + faceit.vertex(0)[0];
              single_point[1] = faceit.vertex(0)[1];
            }
            else
            {
              // vertical face
              single_point[0] = faceit.vertex(0)[0];
              single_point[1] = quad_z.point(q)[0]*(faceit.vertex(1)[1] - faceit.vertex(0)[1]) + faceit.vertex(0)[1];
            }
            // the if-loop above can be made more efficient and/or compact. so far this is specialized to the case 
            // of a cartesian grid. this could also be done via mapping.transform_unit_to_real_cell() as before

            if (data.get_boundary_id(face) == 0)
              // dirichlet
              test_value[v] = 2.0 * exact_solution.value(single_point);
            else
              {
                // neumann
                Tensor<1, dim> normal;
                for (unsigned int d = 0; d < dim; ++d)
                  normal[d] = phi_face.normal_vector(q)[d][v];
                test_normal_derivative[v] = -normal * exact_solution.gradient(single_point);
              }
          }
        phi_face.submit_value(test_value * sigma - test_normal_derivative, q);
        phi_face.submit_normal_derivative(system_matrix.epsilon_IP * 0.5 *test_value, q);
      }
      phi_face.integrate(EvaluationFlags::values | EvaluationFlags::gradients);
      phi_face.distribute_local_to_global(system_rhs);
  }
}


template <int dim>
void LaplaceProblem<dim>::solve()
{
  SolverControl solver_control(10000, 1e-12 * system_rhs.l2_norm());
  SolverCG<LinearAlgebra::distributed::Vector<double>> cg(solver_control);

  cg.solve(system_matrix, solution, system_rhs, PreconditionIdentity());
}


template <int dim>
void LaplaceProblem<dim>::compute_error()
{
  // the code used e.g. in step-59 to compute the L2 error (see commented section below) does not work here. 
  // I suspect this is due to the different indexing of the quadrature points. I'm using this code here which
  // is implemented in a matrix free fashion 
  const MatrixFree<dim, double> &data = *system_matrix.get_matrix_free();
  FEEvaluationAniso<dim, fe_degree_x, fe_degree_x + 1, fe_degree_z, fe_degree_z + 1, 1, double> phi(data);
  Solution<dim> exact_solution;
  for (unsigned int cell = 0; cell < data.n_cell_batches(); ++cell)
    {
      VectorizedArray<double> L2_error_squared_vectorized = VectorizedArray<double>();
      phi.reinit(cell);
      phi.read_dof_values(solution);
      phi.evaluate(EvaluationFlags::values);
      for (const unsigned int q : phi.quadrature_point_indices())
        {
          VectorizedArray<double> exact_solution_at_qp = VectorizedArray<double>();
          // evaluate exact solution at quadrature point in the physical space
          int q_transf = q*(fe_degree_x+1) - int(std::floor(q/(fe_degree_z+1)))*((fe_degree_x+1)*(fe_degree_z+1)-1);
          for (unsigned int v = 0; v < data.n_active_entries_per_cell_batch(cell); ++v)
            {
              typename DoFHandler< dim >::cell_iterator cellit = data.get_cell_iterator(cell, v);
              Point<dim> single_point = mapping.transform_unit_to_real_cell(cellit, Quadrature<dim>(quad_x,quad_z).point(q_transf));
              exact_solution_at_qp[v] = exact_solution.value(single_point);
            }
          phi.submit_value((phi.get_value(q) - exact_solution_at_qp) * (phi.get_value(q) - exact_solution_at_qp), q);
          L2_error_squared_vectorized += phi.get_value(q);
        }
      for (unsigned int v = 0; v < data.n_active_entries_per_cell_batch(cell); ++v)
        L2_error += L2_error_squared_vectorized[v];
    }
  L2_error = std::sqrt(L2_error);

  /*
  Solution<dim> exact_solution;
  Quadrature<dim> quadrature(quad_x, quad_z);
  Vector<double> difference_per_cell(triangulation.n_active_cells());
  VectorTools::integrate_difference(dof_handler, solution, exact_solution, difference_per_cell, 
                                    quadrature, VectorTools::L2_norm);
  L2_error = VectorTools::compute_global_error(triangulation, difference_per_cell, VectorTools::L2_norm);
  */
}


template<int dim>
void LaplaceProblem<dim>::run(const double Lx, const double Lz, const unsigned Nx, const unsigned Nz)
{
  const Point<dim> & p1 = {0, 0};
  const Point<dim> & p2 = {Lx, Lz};
  std::vector<unsigned int> repetitions{Nx, Nz};
  GridGenerator::subdivided_hyper_rectangle(triangulation, repetitions, p1, p2);

  std::cout << "  Number of active cells:       " << triangulation.n_active_cells() << std::endl;

  setup_system();
  assemble_rhs();
  solve();
  compute_error();
}


/*------------------------------------------------- convergence tests -------------------------------------------------------*/
void convergence_test_xz()
{
  const double Lx = 1, Lz = 1;  
  ConvergenceTable convergence_table; 

  for (unsigned int cycle = 1; cycle < 15; cycle++) 
  {
    LaplaceProblem<dim> lp;
    unsigned int Nz = 2*cycle, Nx = 2*cycle;
    lp.run(Lx, Lz, Nx, Nz);

    convergence_table.add_value("Nx", Nx);
    convergence_table.add_value("Nz", Nz);
    convergence_table.add_value("ncells", Nx*Nz);
    convergence_table.add_value("L2err", lp.get_error());
  }

  convergence_table.set_precision("L2err", 3);
  convergence_table.set_scientific("L2err", true);
  convergence_table.evaluate_convergence_rates("L2err", "ncells", ConvergenceTable::reduction_rate_log2);
  convergence_table.write_text(std::cout);
}


/*--------------------------------------------------------- main ------------------------------------------------------------*/
int main()
{
  //const double Lx = 1, Lz = 1;
  //const unsigned int Nx = 8, Nz = 8;

  //LaplaceProblem<dim> lp1;
  //lp1.run(Lx, Lz, Nx, Nz);
  //std::cout << "L2 error: " << lp1.get_error() << std::endl;

  convergence_test_xz();
  
  return 0;
}
