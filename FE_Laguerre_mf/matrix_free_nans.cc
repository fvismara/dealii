#include <deal.II/grid/grid_generator.h>
#include <deal.II/matrix_free/fe_evaluation.h>

#include "fe_dgq_aniso.h"

#include <iostream>
#include <fstream>
#include <typeinfo>

using namespace dealii;

/*---------------------------------------------- LaplaceOperator class -------------------------------------------------------*/
template <int dim, int fe_degree_x, int fe_degree_z, typename number>
class LaplaceOperator : public EnableObserverPointer
{
public:
  using value_type = number;

  LaplaceOperator() = default;

  void initialize(std::shared_ptr<const MatrixFree<dim, number>> data);

  void clear();

  void initialize_dof_vector(LinearAlgebra::distributed::Vector<number> &vec) const;

  std::shared_ptr<const MatrixFree<dim, number>> get_matrix_free() const;

private:
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


/*--------------------------------------------------------- main ----------------------------------------------------------*/
int main()
{
  const unsigned int dim = 2;
  const double Lx = 1, Lz = 1;
  const unsigned int Nx = 2, Nz = 2;
  const unsigned int fe_degree_x = 1, fe_degree_z = 2;

  const Point<dim> & p1 = {0, 0};
  const Point<dim> & p2 = {Lx, Lz};

  // generate grid
  std::vector<std::vector<double>> step_sizes;
  Triangulation<dim> triangulation;

  std::vector<unsigned int> repetitions{Nx, Nz};
  GridGenerator::subdivided_hyper_rectangle(triangulation, repetitions, p1, p2);
 
  std::cout << "  Number of active cells:       " << triangulation.n_active_cells() << std::endl;

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

  typename MatrixFree<dim, double>::AdditionalData additional_data;
  additional_data.tasks_parallel_scheme = MatrixFree<dim, double>::AdditionalData::none;
  additional_data.mapping_update_flags = (update_values | update_gradients | update_JxW_values | update_quadrature_points);
  additional_data.mapping_update_flags_inner_faces = (update_values | update_gradients | update_JxW_values | update_normal_vectors);
  additional_data.mapping_update_flags_boundary_faces = (update_values | update_gradients | update_JxW_values | update_normal_vectors | update_quadrature_points);
  const auto system_mf_storage = std::make_shared<MatrixFree<dim, double>>();
  system_mf_storage->reinit(mapping, dof_handler, dummy, QGauss<1>(fe.degree + 1), additional_data);
  system_matrix.initialize(system_mf_storage);

  system_matrix.initialize_dof_vector(solution);
  system_matrix.initialize_dof_vector(system_rhs);

  // arbitrary non-trivial "right hand side": we set it to [0,1,...,Ndofs-1]^T
  for (unsigned int ii = 0; ii < std::size(system_rhs); ii++)
    system_rhs[ii] = ii;

  // construct FEEvaluation object. since we need to specify a polynomial degree, we choose the largest of the two
  const unsigned int max_degree = std::max(fe_degree_x, fe_degree_z);
  FEEvaluation<dim, max_degree, max_degree + 1, 1, double> phi(*system_mf_storage);

  // we want to investigate the behaviour of FEEvaluation::evaluate. we first call phi.reinit(0) (reinitialize
  // FEEvaluation object on the first cell on the grid) and then call read_dof_evaluate followed by evaluate.
  // this is equivalent to gather_evaluate
  phi.reinit(0);
  phi.read_dof_values(system_rhs);

  
  /*------------------- everything below this line is the expansion of the call to evaluate ------------------------------*/
  const unsigned int n_q_points_1d = max_degree + 1;

  // variant to evaluate tensor product
  static const internal::EvaluatorVariant variant = 
               internal::EvaluatorSelector<internal::MatrixFreeFunctions::ElementType::tensor_general, 
               (max_degree + n_q_points_1d > 4)>::variant;
  using Number = VectorizedArray<double>;
  using Number2 = typename FEEvaluationData<dim, Number, false>::shape_info_number_type;
 
  // note: Eval is initialized with the maximum degree
  using Eval = internal::EvaluatorTensorProduct<variant, dim, max_degree + 1, n_q_points_1d, Number, Number2>;
  std::array<const internal::MatrixFreeFunctions::UnivariateShapeData<Number2> *, 3> univariate_shape_data;
 
  const auto &shape_data = phi.get_shape_info().data;
  univariate_shape_data.fill(&shape_data.front());
 
  if (shape_data.size() == dim)
    for (unsigned int i = 1; i < dim; ++i)
      univariate_shape_data[i] = &shape_data[i];

  const Number *values_dofs = phi.begin_dof_values();
  Number *values_quad       = phi.begin_values();
  Number *temp1             = phi.get_scratch_data().begin();

  // we want to compute values
  constexpr internal::EvaluatorQuantity value_type = internal::EvaluatorQuantity::value;
 
  // these two lines compute (B^x_{cell}\kron B^y_{cell})*u_cell, where u_cell is in values_dofs and (B_{cell})_{i,j} = \Phi_j(x_i),
  // where \Phi_j is the jth basis function in 1d and x_i is the ith quadrature point in 1d. this is done in two steps
  // (multiply by the two matrices individually). the current implementation of Eval::apply relies on two assumptions:
  // 1) n_rows == n_columns for both matrices
  // 2) n_rows_1 == n_rows_2 and n_columns_1 == n_columns_2
  // however, the product (A\kron B)*x is defined for A and B of arbitrary shape, as long as 
  // n_columns_1*n_columns_2 == std::size(x)
  // the two assumptions above generate issues when polynomial degrees are different. since we initialized Eval with
  // the largest degree, every operation in 'apply' assumes that std::size(values_dofs) = std::size(temp1) = std::size(
  // values_quad) = max_degree + 1, and std::size(univariate_shape_data[i]->shape_values.begin()) == (max_degree+1)^2.
  // when one of the two directions involve a polynomial degree < max_degree we access meaningless values hence the nans

  Eval::apply<0, true, false, false, value_type, 1>(univariate_shape_data[0]->shape_values.begin(), values_dofs, temp1);
  Eval::apply<1, true, false, false, value_type, 1>(univariate_shape_data[1]->shape_values.begin(), temp1, values_quad);
  
  // print the elements of (B^x_{cell}\kron B^y_{cell})*u_cell. in general we expect this object to have
  // n_quadrature_points_x*n_quadrature_points_z = (fe_degree_x+1)*(fe_degree_z+1). if fe_degree_x == fe_degree_z 
  // there's no problem. otherwise we have nans because of the issues mentioned above. also notice that we always
  // get vectors of size (max_degree+1)^2 as expected (but not as desired)
  for (const unsigned int q : phi.quadrature_point_indices()) {
    std::cout << phi.get_value(q) << std::endl;
  }

  return 0;
}