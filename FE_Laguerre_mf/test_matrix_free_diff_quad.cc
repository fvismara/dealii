// This file solves the Poisson equation -\Delta u = f on the 2d unit square [0,1]x[0,1] using DG
// and a matrix free implementation. Still doesn't work because of problems with assemble_rhs. see compiler errors

#include <deal.II/grid/grid_generator.h>
#include <deal.II/matrix_free/fe_evaluation.h>
#include <deal.II/matrix_free/operators.h>
#include <deal.II/fe/fe_dgq.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/numerics/data_out.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/numerics/vector_tools.h>

#include "fe_dgq_aniso.h"
#include "fe_evaluation_aniso.h"

#include <iostream>
#include <fstream>
#include <typeinfo>

using namespace dealii;


const unsigned int dim = 2;
const unsigned int fe_degree_x = 1, fe_degree_z = 2;


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

  //virtual void compute_diagonal() override;

  void vmult(LinearAlgebra::distributed::Vector<number>       &dst,
                 const LinearAlgebra::distributed::Vector<number> &src) const;

  number get_penalty_factor() const
  {
    return 1.0 * std::max(fe_degree_x, fe_degree_z) * (std::max(fe_degree_x, fe_degree_z) + 1);
  }

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

//template <int dim, int fe_degree_x, int fe_degree_z, typename number>
//void
//LaplaceOperator<dim, fe_degree_x, fe_degree_z, number>::compute_diagonal()
//{
//}


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
    unsigned int face_q_points = phi_inner.normal_vector(0)[0][0] == 0. ? fe_degree_x + 1 : fe_degree_z + 1;

    const VectorizedArray<number> inverse_length_normal_to_face =
            0.5 * (std::abs((phi_inner.normal_vector(0) *
                             phi_inner.inverse_jacobian(0))[dim - 1]) +
                   std::abs((phi_outer.normal_vector(0) *
                             phi_outer.inverse_jacobian(0))[dim - 1]));
    const VectorizedArray<number> sigma = inverse_length_normal_to_face *
                                  get_penalty_factor();

    // submit_values / submit_gradients
    for (unsigned int q = 0; q < face_q_points; q++)
    {
      const VectorizedArray<number> solution_jump = (phi_inner.get_value(q) - phi_outer.get_value(q));
      const VectorizedArray<number> average_normal_derivative = (phi_inner.get_normal_derivative(q) +
                                                                 phi_outer.get_normal_derivative(q)) * number(0.5);
      const VectorizedArray<number> test_by_value = solution_jump * sigma - average_normal_derivative;

      phi_inner.submit_value(test_by_value, q);
      phi_outer.submit_value(-test_by_value, q);

      phi_inner.submit_normal_derivative(-solution_jump * number(0.5), q);
      phi_outer.submit_normal_derivative(-solution_jump * number(0.5), q);
    }

    // integrate
    phi_inner.integrate(EvaluationFlags::gradients);
    phi_outer.integrate(EvaluationFlags::gradients);

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

    const VectorizedArray<number> inverse_length_normal_to_face = std::abs((
      phi_inner.normal_vector(0) * phi_inner.inverse_jacobian(0))[dim - 1]);
    const VectorizedArray<number> sigma =
      inverse_length_normal_to_face * get_penalty_factor();

    const bool is_dirichlet = (data.get_boundary_id(face) == 0);

    unsigned int face_q_points = phi_inner.normal_vector(0)[0][0] == 0. ? fe_degree_x + 1 : fe_degree_z + 1;
    for (unsigned int q = 0; q < face_q_points; q++)
    {
      const VectorizedArray<number> u_inner = phi_inner.get_value(q);
      const VectorizedArray<number> u_outer = is_dirichlet ? -u_inner : u_inner;
      const VectorizedArray<number> normal_derivative_inner = phi_inner.get_normal_derivative(q);
      const VectorizedArray<number> normal_derivative_outer = is_dirichlet ? normal_derivative_inner : -normal_derivative_inner;
      const VectorizedArray<number> solution_jump = (u_inner - u_outer);
      const VectorizedArray<number> average_normal_derivative = (normal_derivative_inner + normal_derivative_outer) * number(0.5);
      const VectorizedArray<number> test_by_value = solution_jump * sigma - average_normal_derivative;
      phi_inner.submit_normal_derivative(-solution_jump * number(0.5), q);
      phi_inner.submit_value(test_by_value, q);
    }

    // integrate
    phi_inner.integrate(EvaluationFlags::gradients);

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

  private:
    void setup_system();
    void assemble_rhs();
    void inner_faces_loop();
    void solve();
    void output_results(const unsigned int cycle) const;

    Triangulation<dim> triangulation;

    const FE_DGQ_Aniso<dim> fe;
    DoFHandler<dim> dof_handler;

    const MappingQ1<dim> mapping;

    //AffineConstraints<double> constraints;
    using SystemMatrixType = LaplaceOperator<dim, fe_degree_x, fe_degree_z, double>;
    SystemMatrixType system_matrix;

    LinearAlgebra::distributed::Vector<double> solution;
    LinearAlgebra::distributed::Vector<double> system_rhs;

    double             setup_time;
    //ConditionalOStream pcout;
    //ConditionalOStream time_details;
};


template <int dim>
LaplaceProblem<dim>::LaplaceProblem()
  : fe(fe_degree_x, fe_degree_z,
       AnisotropicPolynomials<dim>({Polynomials::Legendre::generate_complete_basis(fe_degree_x),
                                    Polynomials::Legendre::generate_complete_basis(fe_degree_z)}))
  , dof_handler(triangulation)
  , setup_time(0.)
{}


template <int dim>
void LaplaceProblem<dim>::setup_system()
{
  system_matrix.clear();

  dof_handler.distribute_dofs(fe);

  std::cout << "Number of degrees of freedom: " << dof_handler.n_dofs() << std::endl;

  AffineConstraints<double> dummy;
  dummy.close();

  FE_DGQLegendre<1> fe_x(fe_degree_x), fe_z(fe_degree_z);
  QGauss<1> quad_x(fe_degree_x + 1), quad_z(fe_degree_z + 1);
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

  // arbitrary non-trivial "right hand side": we set it to [0,1,...,Ndofs-1]^T
  //for (unsigned int ii = 0; ii < std::size(system_rhs); ii++)
    //system_rhs[ii] = ii;
}


template <int dim>
void LaplaceProblem<dim>::assemble_rhs()
{
  system_rhs = 0;
  const MatrixFree<dim, double> &data = *system_matrix.get_matrix_free();
  FEEvaluationAniso<dim, fe_degree_x, fe_degree_x + 1, fe_degree_z, fe_degree_z + 1, 1, double> phi(data);
  RightHandSide<dim> rhs_func;
  Solution<dim> exact_solution;
  for (unsigned int cell = 0; cell < data.n_cell_batches(); ++cell)
    {
      phi.reinit(cell);
      std::cout << phi.quadrature_point_indices().size() << std::endl;
      for (const unsigned int q : phi.quadrature_point_indices())
        {
          VectorizedArray<double> rhs_val = VectorizedArray<double>();
          //std::cout << phi.n_quadrature_points << std::endl;
          Point<dim, VectorizedArray<double>> point_batch = phi.quadrature_point(q);
          for (unsigned int v = 0; v < VectorizedArray<double>::size(); ++v)
            {
              Point<dim> single_point;
              for (unsigned int d = 0; d < dim; ++d)
                single_point[d] = point_batch[d][v];
              rhs_val[v] = rhs_func.value(single_point);
            }
          //phi.submit_value(make_vectorized_array<double>(1.0), q);
          phi.submit_value(rhs_val, q);
        }
      phi.integrate(EvaluationFlags::values);
      phi.distribute_local_to_global(system_rhs);
    }
  //system_rhs.compress(VectorOperation::add);

  //constraints.set_zero(system_rhs);

  //for (unsigned int i = 0; i < system_rhs.size(); i++)
    //std::cout << system_rhs[i] << std::endl;

  //std::cout << std::endl;


  FEFaceEvaluationAniso<dim, fe_degree_x, fe_degree_x + 1, fe_degree_z, fe_degree_z + 1, 1, double> phi_face(data, true);
  for (unsigned int face = data.n_inner_face_batches(); face < data.n_inner_face_batches() + data.n_boundary_face_batches();
           ++face)
  {
    phi_face.reinit(face);

    const VectorizedArray<double> inverse_length_normal_to_face = std::abs(
      (phi_face.normal_vector(0) * phi_face.inverse_jacobian(0))[dim - 1]);
    const VectorizedArray<double> sigma =
      inverse_length_normal_to_face * system_matrix.get_penalty_factor();

    unsigned int face_q_points = phi_face.normal_vector(0)[0][0] == 0. ? fe_degree_x + 1 : fe_degree_z + 1;

    for (unsigned int q = 0; q < face_q_points; q++)
      {
        VectorizedArray<double> test_value = VectorizedArray<double>(),
                                test_normal_derivative = VectorizedArray<double>();
        Point<dim, VectorizedArray<double>> point_batch = phi_face.quadrature_point(q);

        for (unsigned int v = 0; v < VectorizedArray<double>::size(); ++v)
          {
            Point<dim> single_point;
            for (unsigned int d = 0; d < dim; ++d)
              single_point[d] = point_batch[d][v];
            if (data.get_boundary_id(face) == 0)
              test_value[v] = 2.0 * exact_solution.value(single_point);
            else
              {
                Tensor<1, dim> normal;
                for (unsigned int d = 0; d < dim; ++d)
                  normal[d] = phi_face.normal_vector(q)[d][v];
                test_normal_derivative[v] = -normal * exact_solution.gradient(single_point);
              }
          }
        phi_face.submit_value(test_value * sigma - test_normal_derivative, q);
        phi_face.submit_normal_derivative(-0.5 * test_value, q);
      }
      phi_face.integrate(EvaluationFlags::values | EvaluationFlags::gradients);
      phi_face.distribute_local_to_global(system_rhs);
  }
}

//GO: How this function differs from the apply_face of the LaplaceOperator?
template <int dim>
void LaplaceProblem<dim>::inner_faces_loop()
{
  FEFaceEvaluationAniso<dim, fe_degree_x, fe_degree_x + 1, fe_degree_z, fe_degree_z + 1, 1, double>
    phi_outer(*system_matrix.get_matrix_free(), false);
  FEFaceEvaluationAniso<dim, fe_degree_x, fe_degree_x + 1, fe_degree_z, fe_degree_z + 1, 1, double>
    phi_inner(*system_matrix.get_matrix_free(), true);

  for (unsigned int inner_face = 0; inner_face < system_matrix.get_matrix_free()->n_inner_face_batches(); ++inner_face) {

    // reinit on face
    phi_inner.reinit(inner_face);
    phi_outer.reinit(inner_face);
    //std::cout << phi_inner.normal_vector(0)[0] << std::endl << std::endl;

    // read dof values
    phi_inner.read_dof_values(system_rhs);
    phi_outer.read_dof_values(system_rhs);
    //for (unsigned int i = 0; i < (fe_degree_x + 1)*(fe_degree_z + 1); i++)
      //std::cout << phi_inner.begin_dof_values()[i] << "            " << phi_outer.begin_dof_values()[i] << std::endl;

    // evaluate
    phi_inner.evaluate(EvaluationFlags::values | EvaluationFlags::gradients);
    phi_outer.evaluate(EvaluationFlags::values | EvaluationFlags::gradients);
    unsigned int face_q_points = phi_inner.normal_vector(0)[0][0] == 0. ? fe_degree_x + 1 : fe_degree_z + 1;
    //for (unsigned int i = 0; i < 2*face_q_points; i++)
      //std::cout << phi_inner.begin_gradients()[i] << "               " << phi_outer.begin_gradients()[i] << std::endl;
    //std::cout << std::endl << std::endl;
    //for (unsigned int i = 0; i < face_q_points; i++)
      //std::cout << phi_inner.get_gradient(i)[0] << "               " << phi_outer.get_gradient(i)[0] << std::endl;

    const VectorizedArray<double> inverse_length_normal_to_face =
            0.5 * (std::abs((phi_inner.normal_vector(0) *
                             phi_inner.inverse_jacobian(0))[dim - 1]) +
                   std::abs((phi_outer.normal_vector(0) *
                             phi_outer.inverse_jacobian(0))[dim - 1]));
    const VectorizedArray<double> sigma = inverse_length_normal_to_face *
                                  double(std::max(fe_degree_x, fe_degree_z) * (std::max(fe_degree_x, fe_degree_z) + 1));
                                  std::cout << sigma << std::endl << std::endl;

    // submit_values / submit_gradients
    for (unsigned int q = 0; q < face_q_points; q++)
    {
      const VectorizedArray<double> solution_jump = (phi_inner.get_value(q) - phi_outer.get_value(q));
      //std::cout << solution_jump << std::endl;
      const VectorizedArray<double> average_normal_derivative = (phi_inner.get_normal_derivative(q) +
                                                                 phi_outer.get_normal_derivative(q)) * double(0.5);
      const VectorizedArray<double> test_by_value = solution_jump * sigma - average_normal_derivative;

      //for (unsigned int i = 0; i < face_q_points; i++)
        //std::cout << phi_inner.get_normal_derivative(i) << "               " << phi_outer.get_normal_derivative(i) << std::endl;
      //std::cout << average_normal_derivative << std::endl;
      //phi_inner.submit_value(solution_jump, q);
      //phi_outer.submit_value(-solution_jump, q);

      phi_inner.submit_value(test_by_value, q);
      phi_outer.submit_value(-test_by_value, q);

      phi_inner.submit_normal_derivative(-solution_jump * double(0.5), q);
      phi_outer.submit_normal_derivative(-solution_jump * double(0.5), q);
    }
    //std::cout << std::endl;

    //for (unsigned int q = 0; q < 2*face_q_points; q++)
      //std::cout << phi_inner.begin_gradients()[q] << "          " << phi_outer.begin_gradients()[q] << std::endl;
    //std::cout << std::endl;

    // integrate
    phi_inner.integrate(EvaluationFlags::gradients);
    phi_outer.integrate(EvaluationFlags::gradients);
    //for (unsigned int q = 0; q < (fe_degree_x + 1)*(fe_degree_z + 1); q++)
      //std::cout << phi_inner.begin_dof_values()[q] << "          " << phi_outer.begin_dof_values()[q] << std::endl;
    //std::cout << std::endl;

    // distribute_local_to_global
    phi_inner.distribute_local_to_global(solution);
    phi_outer.distribute_local_to_global(solution);


    ///////// jump term
    /*
    // reinit on face
    phi_inner.reinit(inner_face);
    phi_outer.reinit(inner_face);
    std::cout << phi_inner.normal_vector(0)[0] << std::endl << std::endl;

    // read dof values
    phi_inner.read_dof_values(system_rhs);
    phi_outer.read_dof_values(system_rhs);

    for (unsigned int i = 0; i < (fe_degree_x + 1)*(fe_degree_z + 1); i++)
      std::cout << phi_inner.begin_dof_values()[i] << "            " << phi_outer.begin_dof_values()[i] << std::endl;

    // evaluate
    phi_inner.evaluate(EvaluationFlags::values);
    phi_outer.evaluate(EvaluationFlags::values);

    unsigned int face_q_points = phi_inner.normal_vector(0)[0][0] == 0. ? fe_degree_x + 1 : fe_degree_z + 1;

    // submit_values
    for (unsigned int q = 0; q < face_q_points; q++)
    {
      const VectorizedArray<double> solution_jump = (phi_inner.get_value(q) - phi_outer.get_value(q));

      phi_inner.submit_value(solution_jump, q);
      phi_outer.submit_value(-solution_jump, q);
    }

    // integrate
    phi_inner.integrate(EvaluationFlags::values);
    phi_outer.integrate(EvaluationFlags::values);

    // distribute_local_to_global
    phi_inner.distribute_local_to_global(solution);
    phi_outer.distribute_local_to_global(solution);

    for (unsigned int i = 0; i < solution.size(); i++)
      std::cout << solution[i] << std::endl;
    */
  }

    for (unsigned int i = 0; i < solution.size(); i++)
      std::cout << solution[i] << std::endl;
}



template <int dim>
void LaplaceProblem<dim>::solve()
{
  SolverControl solver_control(100, 1e-12 * system_rhs.l2_norm());
  SolverCG<LinearAlgebra::distributed::Vector<double>> cg(solver_control);

  //constraints.set_zero(solution);
  cg.solve(system_matrix, solution, system_rhs, PreconditionIdentity());

  //constraints.distribute(solution);
}


template <int dim>
void LaplaceProblem<dim>::output_results(const unsigned int cycle) const
{
  DataOut<dim> data_out;

  solution.update_ghost_values();
  data_out.attach_dof_handler(dof_handler);
  data_out.add_data_vector(solution, "solution");
  data_out.build_patches(mapping);

  DataOutBase::VtkFlags flags;
  flags.compression_level = DataOutBase::CompressionLevel::best_speed;
  data_out.set_flags(flags);
  data_out.write_vtu_with_pvtu_record(
    "./", "solution", cycle, MPI_COMM_WORLD, 3);
}


template<int dim>
void LaplaceProblem<dim>::run(const double Lx, const double Lz, const unsigned Nx, const unsigned Nz)
{
  //std::vector<std::vector<double>> step_sizes;

  const Point<dim> & p1 = {0, 0};
  const Point<dim> & p2 = {Lx, Lz};
  std::vector<unsigned int> repetitions{Nx, Nz};
  GridGenerator::subdivided_hyper_rectangle(triangulation, repetitions, p1, p2);

  std::cout << "  Number of active cells:       " << triangulation.n_active_cells() << std::endl;

  setup_system();
  // arbitrary non-trivial "right hand side": we set it to [0,1,...,Ndofs-1]^T
  //for (unsigned int ii = 0; ii < std::size(system_rhs); ii++)
    //system_rhs[ii] = ii;
  assemble_rhs();
  //inner_faces_loop();

  //system_matrix.vmult(solution, system_rhs);
  //for (unsigned int i = 0; i < solution.size(); i++)
    //std::cout << solution[i] << std::endl;
  //solve();
  //output_results(cycle);
}

/*--------------------------------------------------------- main ------------------------------------------------------------*/
int main()
{
  const double Lx = 1, Lz = 1;
  const unsigned int Nx = 2, Nz = 2;


  LaplaceProblem<dim> lp;
  lp.run(Lx, Lz, Nx, Nz);



  return 0;
}
