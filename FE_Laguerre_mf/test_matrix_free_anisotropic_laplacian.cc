// This file implements and tests the action of the Laplacian matrix on a generic vector in a matrix-free setting
// For the moment we don't care about faces so this code is taken from step 37 which uses continuous finite elements

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


/*---------------------------------------------- LaplaceOperator class -------------------------------------------------------*/
template <int dim, int fe_degree_x, int fe_degree_z, typename number>
class LaplaceOperator : public MatrixFreeOperators::Base<dim, LinearAlgebra::distributed::Vector<number>>
{
public:
  using value_type = number;

  LaplaceOperator() = default;

  void initialize(std::shared_ptr<const MatrixFree<dim, number>> data);

  void clear() override;

  void initialize_dof_vector(LinearAlgebra::distributed::Vector<number> &vec) const;

  std::shared_ptr<const MatrixFree<dim, number>> get_matrix_free() const;

  virtual void compute_diagonal() override;

private:
  virtual void 
  apply_add(
    LinearAlgebra::distributed::Vector<number>       &dst,
    const LinearAlgebra::distributed::Vector<number> &src) const override;

  void
  local_apply(const MatrixFree<dim, number>                    &data,
              LinearAlgebra::distributed::Vector<number>       &dst,
              const LinearAlgebra::distributed::Vector<number> &src,
              const std::pair<unsigned int, unsigned int> &cell_range) const;

};

template <int dim, int fe_degree_x, int fe_degree_z, typename number>
void LaplaceOperator<dim, fe_degree_x, fe_degree_z, number>::initialize(
  std::shared_ptr<const MatrixFree<dim, number>> data)
{
  MatrixFreeOperators::Base<dim, LinearAlgebra::distributed::Vector<number>>::initialize(
    data);
}

template <int dim, int fe_degree_x, int fe_degree_z, typename number>
void LaplaceOperator<dim, fe_degree_x, fe_degree_z, number>::clear()
{
  MatrixFreeOperators::Base<dim, LinearAlgebra::distributed::Vector<number>>::clear();
}

template <int dim, int fe_degree_x, int fe_degree_z, typename number>
void LaplaceOperator<dim, fe_degree_x, fe_degree_z, number>::initialize_dof_vector(
  LinearAlgebra::distributed::Vector<number> &vec) const
{
  MatrixFreeOperators::Base<dim, LinearAlgebra::distributed::Vector<number>>::initialize_dof_vector(vec);
}

template <int dim, int fe_degree_x, int fe_degree_z, typename number>
std::shared_ptr<const MatrixFree<dim, number>>
LaplaceOperator<dim, fe_degree_x, fe_degree_z, number>::get_matrix_free() const
{
  return MatrixFreeOperators::Base<dim, LinearAlgebra::distributed::Vector<number>>::data;
}


template <int dim, int fe_degree_x, int fe_degree_z, typename number>
void
LaplaceOperator<dim, fe_degree_x, fe_degree_z, number>::compute_diagonal()
{
  
}

template <int dim, int fe_degree_x, int fe_degree_z, typename number>
void LaplaceOperator<dim, fe_degree_x, fe_degree_z, number>::local_apply(
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
void LaplaceOperator<dim, fe_degree_x, fe_degree_z, number>::apply_add(
  LinearAlgebra::distributed::Vector<number>       &dst,
  const LinearAlgebra::distributed::Vector<number> &src) const
{
  this->data->cell_loop(&LaplaceOperator::local_apply, this, dst, src);
}


/*--------------------------------------------------------- main ------------------------------------------------------------*/
int main()
{
  const double Lx = 1, Lz = 1;
  const unsigned int Nx = 2, Nz = 2;

  
  // generate grid
  Triangulation<dim> triangulation;

  const Point<dim> & p1 = {0, 0};
  const Point<dim> & p2 = {Lx, Lz};
  std::vector<unsigned int> repetitions{Nx, Nz};
  GridGenerator::subdivided_hyper_rectangle(triangulation, repetitions, p1, p2);
 
  std::cout << "  Number of active cells:       " << triangulation.n_active_cells() << std::endl;

  // setup system
  using SystemMatrixType = LaplaceOperator<dim, fe_degree_x, fe_degree_z, double>;
  SystemMatrixType system_matrix;

  DoFHandler<dim>          dof_handler(triangulation);
  const MappingQ1<dim>     mapping;

  const FE_DGQ_Aniso<dim> fe(fe_degree_x, fe_degree_z, 
                            AnisotropicPolynomials<dim>({Polynomials::Legendre::generate_complete_basis(fe_degree_x), 
                            Polynomials::Legendre::generate_complete_basis(fe_degree_z)}));

  LinearAlgebra::distributed::Vector<double> solution;
  LinearAlgebra::distributed::Vector<double> system_rhs;

  system_matrix.clear();

  dof_handler.distribute_dofs(fe);

  std::cout << "Number of degrees of freedom: " << dof_handler.n_dofs() << std::endl;

  AffineConstraints<double> dummy;
  dummy.close();

  FE_DGQ<1> fe_x(fe_degree_x), fe_z(fe_degree_z);
  QGauss<1> quad_x(fe_degree_x + 1), quad_z(fe_degree_z + 1);
  const std::vector<FE_DGQ<1>> fe_vector{fe_x, fe_z};
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
  for (unsigned int ii = 0; ii < std::size(system_rhs); ii++)
    system_rhs[ii] = ii;

  // construct FEEvaluationAniso object. 
  FEEvaluationAniso<dim, fe_degree_x, fe_degree_x + 1, fe_degree_z, fe_degree_z + 1, 1, double> phi(*system_mf_storage);
  
  
  // local contribution of \int \grad\phi\grad\phi. check that the output is correct. also checked for \int \phi\phi.
  // checked for FE_DGQ and FE_DGLegendre
  phi.reinit(1);
  phi.read_dof_values(system_rhs);

  phi.evaluate(EvaluationFlags::gradients);

  for (const unsigned int q : phi.quadrature_point_indices())
    phi.submit_gradient(phi.get_gradient(q), q);

  phi.integrate(EvaluationFlags::gradients);

  phi.distribute_local_to_global(solution);

  for (unsigned int i = 0; i < solution.size(); i++)
    std::cout << solution[i] << std::endl;
  

  // the functions used above are expanded here for reference (commented)
  /*
  // evaluate
  using Number = VectorizedArray<double>;
  using Number2 = typename FEEvaluationData<dim, Number, false>::shape_info_number_type;

  std::array<const internal::MatrixFreeFunctions::UnivariateShapeData<Number2> *, dim> univariate_shape_data;

  const auto &shape_data = phi.get_shape_info().data;
 
  for (unsigned int i = 0; i < dim; ++i) {
    univariate_shape_data[i] = &shape_data[i];
    //for (unsigned int j = 0; j < univariate_shape_data[i]->shape_gradients.size(); j++)
      //std::cout << univariate_shape_data[i]->shape_gradients[j] << std::endl;
    //std::cout << std::endl;
  }

  const Number *values_dofs = phi.begin_dof_values();
  Number *grads_quad        = phi.begin_gradients();
  Number *temp1             = phi.get_scratch_data().begin();

  static const internal::EvaluatorVariant variant = internal::EvaluatorVariant::evaluate_general;
  apply<variant, dim, fe_degree_x + 1, fe_degree_z + 1, fe_degree_z + 1, 1, false, false, Number, Number2>
        (univariate_shape_data[1]->shape_values.begin(), values_dofs, temp1);
  apply<variant, dim, 1, fe_degree_x + 1, fe_degree_x + 1, fe_degree_z + 1, false, false, Number, Number2>
        (univariate_shape_data[0]->shape_gradients.begin(), temp1, grads_quad);
        
  grads_quad += (fe_degree_x + 1)*(fe_degree_z + 1);

  apply<variant, dim, fe_degree_x + 1, fe_degree_z + 1, fe_degree_z + 1, 1, false, false, Number, Number2>
        (univariate_shape_data[1]->shape_gradients.begin(), values_dofs, temp1);
  apply<variant, dim, 1, fe_degree_x + 1, fe_degree_x + 1, fe_degree_z + 1, false, false, Number, Number2>
        (univariate_shape_data[0]->shape_values.begin(), temp1, grads_quad);
  

  // get_gradient
  grads_quad = phi.begin_gradients();
  for (unsigned int q : phi.quadrature_point_indices())
    for (unsigned int d = 0; d < dim; d++) {
      VectorizedArray<double> temp = 0;
      for (unsigned int dp = 0; dp < dim; dp++)
        temp += phi.inverse_jacobian(q)[d][dp] * grads_quad[dp*phi.quadrature_point_indices().size() + q];
      temp1[q*dim + d] = temp;
    }

  
  // submit_gradient
  for (unsigned int q : phi.quadrature_point_indices())
    for (unsigned int d = 0; d < dim; d++)
      temp1[q*dim + d] *= phi.JxW(q);
 
  for (unsigned int q : phi.quadrature_point_indices())
    for (unsigned int d = 0; d < dim; d++) {
      VectorizedArray<double> temp = 0;
      for (unsigned int dp = 0; dp < dim; dp++)
        temp += phi.inverse_jacobian(q)[dp][d] * temp1[q*dim + dp];
      grads_quad[d*phi.quadrature_point_indices().size() + q] = temp;
    }

  
  // integrate
  VectorizedArray<double> *grad_out = phi.begin_dof_values();
  apply<variant, dim, fe_degree_x + 1, fe_degree_z + 1, fe_degree_z + 1, 1, true, false, Number, Number2>
        (univariate_shape_data[1]->shape_values.begin(), grads_quad, temp1);
  apply<variant, dim, 1, fe_degree_x + 1, fe_degree_x + 1, fe_degree_z + 1, true, false, Number, Number2>
        (univariate_shape_data[0]->shape_gradients.begin(), temp1, grad_out);

  grads_quad += (fe_degree_x + 1)*(fe_degree_z + 1);

  apply<variant, dim, fe_degree_x + 1, fe_degree_z + 1, fe_degree_z + 1, 1, true, false, Number, Number2>
        (univariate_shape_data[1]->shape_gradients.begin(), grads_quad, temp1);
  apply<variant, dim, 1, fe_degree_x + 1, fe_degree_x + 1, fe_degree_z + 1, true, true, Number, Number2>
        (univariate_shape_data[0]->shape_values.begin(), temp1, grad_out);

  //for (unsigned int i = 0; i < (fe_degree_x + 1)*(fe_degree_z + 1); i++)
    //std::cout << grad_out[i] << std::endl;
  //std::cout << std::endl;


  // distribute_local_to_global
  phi.distribute_local_to_global(solution);


  for (unsigned int i = 0; i < solution.size(); i++)
    std::cout << solution[i] << std::endl;
*/
  


  return 0;
}