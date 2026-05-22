// we use this file to test the reimplementation of mapping_info.initialize in FEEvaluationAniso.
// we are only interested in calling setup_system() and check that all quantities (i.e., quadrature points,
// jacobians ...) are correctly initialized in the case of different quadrature rules in the 2 direction

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

/*
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
             true,
             MatrixFree<dim, number>::DataAccessOnFaces::gradients,
             MatrixFree<dim, number>::DataAccessOnFaces::gradients);
}*/



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

  /*
  system_rhs = 0;
  const MatrixFree<dim, double> &data = *system_matrix.get_matrix_free();
  FEFaceEvaluationAniso<dim, fe_degree_x, fe_degree_x + 1, fe_degree_z, fe_degree_z + 1, 1, double> phi_face(data, true);
  for (unsigned int face = data.n_inner_face_batches(); face < data.n_inner_face_batches() + data.n_boundary_face_batches(); ++face)
  {
    std::cout << "new face" << std::endl;
    phi_face.reinit(face);
  
    const VectorizedArray<double> inverse_length_normal_to_face = std::abs(
      (phi_face.normal_vector(0) * phi_face.inverse_jacobian(0))[dim - 1]);
    const VectorizedArray<double> sigma =
      inverse_length_normal_to_face * system_matrix.get_penalty_factor();

    unsigned int face_q_points = phi_face.normal_vector(0)[0][0] == 0. ? fe_degree_x + 1 : fe_degree_z + 1;
    std::cout << "number of qpoints: " << face_q_points << std::endl;
  
    for (unsigned int q = 0; q < face_q_points; q++)
      {
        Point<dim, VectorizedArray<double>> point_batch = phi_face.quadrature_point(q);
        std::cout << point_batch << std::endl;
      }
    }
  */
}

/*--------------------------------------------------------- main ------------------------------------------------------------*/
int main()
{
  const double Lx = 1, Lz = 1;
  const unsigned int Nx = 8, Nz = 3;


  LaplaceProblem<dim> lp;
  lp.run(Lx, Lz, Nx, Nz);
  


  return 0;
}