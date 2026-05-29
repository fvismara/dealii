/* ------------------------------------------------------------------------
 *
 * SPDX-License-Identifier: LGPL-2.1-or-later
 * Copyright (C) 1999 - 2026 by the deal.II authors
 *
 * This file is part of the deal.II library.
 *
 * Part of the source code is dual licensed under Apache-2.0 WITH
 * LLVM-exception OR LGPL-2.1-or-later. Detailed license information
 * governing the source code and code contributions can be found in
 * LICENSE.md and CONTRIBUTING.md at the top level directory of deal.II.
 *
 * ------------------------------------------------------------------------
 *
 * Author: Federico Vismara, 2026
 */
#ifndef fe_evaluation_aniso_var_h
#define fe_evaluation_aniso_var_h

//GO: Are all these header files necessary?
#include <deal.II/base/config.h>

#include <deal.II/base/array_view.h>
#include <deal.II/base/exceptions.h>
#include <deal.II/base/observer_pointer.h>
#include <deal.II/base/symmetric_tensor.h>
#include <deal.II/base/template_constraints.h>
#include <deal.II/base/vectorization.h>

#include <deal.II/lac/vector_operation.h>

#include <deal.II/matrix_free/evaluation_flags.h>
#include <deal.II/matrix_free/evaluation_kernels.h>
#include <deal.II/matrix_free/evaluation_kernels_face.h>
#include <deal.II/matrix_free/evaluation_template_factory.h>
#include <deal.II/matrix_free/fe_evaluation_data.h>
#include <deal.II/matrix_free/fe_evaluation.h>
#include <deal.II/matrix_free/hanging_nodes_internal.h>
#include <deal.II/matrix_free/mapping_data_on_the_fly.h>
#include <deal.II/matrix_free/matrix_free.h>
#include <deal.II/matrix_free/shape_info.h>
#include <deal.II/matrix_free/tensor_product_kernels.h>
#include <deal.II/matrix_free/type_traits.h>
#include <deal.II/matrix_free/vector_access_internal.h>

#include <type_traits>


DEAL_II_NAMESPACE_OPEN

/**
 * Implements the product (I \kron B \kron I)*v, where the two identity matrices have dimension p \times p and r \times r, respectively,
 * and B has dimension n_rows x n_columns. shape_data contains the elements of B in row-major format.
 * It is not necessary for p, n_rows, n_columns, and r to be equal.
 * If transpose==true, computes (I \kron B^T \kron I)*v
 */
// GO: Is there a reason to try to mark this routine as inline?
template <internal::EvaluatorVariant variant,
          int dim, int p, int n_rows, int n_columns, int r, bool transpose, bool add,
          typename Number, typename Number2>
inline void apply(const Number2 *DEAL_II_RESTRICT shape_data, const Number *in, Number *out)
{
  for (int i2 = 0; i2 < p; ++i2)
    {
      for (int i1 = 0; i1 < r; ++i1) {
        internal::apply_matrix_vector_product<variant, internal::EvaluatorQuantity::value,
                                              n_rows, n_columns, r, r, transpose, add>(shape_data, in, out);

        in += 1;
        out += 1;
      }
      in += (n_columns-1)*r;
      out += (n_rows-1)*r;
    }
}

/**
 * The class that provides all functions necessary to evaluate functions at quadrature points and cell integrations
 * optimized for matrix-free operators with (possibly) different polynomial degrees along each coordinate direction.
 */
//GO: n_components_ = 1 as for the FEFaceEvaluationAniso
template <int dim, int fe_degree_x, int n_q_points_1d_x, int fe_degree_z, int n_q_points_1d_z,
          int n_components_, typename Number, typename VectorizedArrayType = VectorizedArray<Number>>
class FEEvaluationAniso : public FEEvaluationBase<dim, n_components_, Number, false, VectorizedArrayType>
{
  static_assert(
    std::is_same_v<Number, typename VectorizedArrayType::value_type>,
    "Type of Number and of VectorizedArrayType do not match.");

public:
  using BaseClass = FEEvaluationBase<dim, n_components_, Number, false, VectorizedArrayType>;
  using number_type = Number; //GO: What is its role?
  using value_type = typename BaseClass::value_type;
  using gradient_type = Tensor<1, dim, VectorizedArrayType>; //GO: Why not BaseClass::gradient_type here?

  static constexpr unsigned int dimension = dim;
  static constexpr unsigned int n_components = n_components_;
  static constexpr unsigned int n_lanes = VectorizedArrayType::size();
  static constexpr unsigned int static_n_q_points = n_q_points_1d_x * n_q_points_1d_z;
  static constexpr unsigned int static_dofs_per_component = (fe_degree_x + 1) * (fe_degree_z + 1);
  static constexpr unsigned int tensor_dofs_per_cell = static_dofs_per_component * n_components;
  static constexpr unsigned int static_dofs_per_cell = static_dofs_per_component * n_components;

  /**
   * Class constructor.
   * @param matrix_free Matrix-free structure
   * @param dof_no If multiple DoFHandler objects exist,
                   this parameter selects to which DoFHandler
                   the given evaluator should be attached to.
   * @param quad_no If multiple Quadrature objects exist,
                    this parameter selects to which Quadrature
                    the given evaluator should be attached to.
   * @param first_selected_component If multiple components are involved (FESystem),
                                     this parameter selects to which component
                                     the given evaluator should be attached to.
   * @param active_fe_index FE index in case of hp::FECollections
   * @param active_quad_index Quadrature index in case of hp::Collection
   */
  FEEvaluationAniso(
    const MatrixFree<dim, Number, VectorizedArrayType> &matrix_free,
    const unsigned int                                  dof_no  = 0,
    const unsigned int                                  quad_no = 0,
    const unsigned int first_selected_component                 = 0,
    const unsigned int active_fe_index   = numbers::invalid_unsigned_int,
    const unsigned int active_quad_index = numbers::invalid_unsigned_int);

  //GO: Maybe we can mark these routines as overload to specifiy that this is a re-implementation of the base class?
  /**
   * Initialize the operation pointer to the current cell batch index.
   * @param cell_batch_index Index of the current cell batch
   */
  void reinit(const unsigned int cell_batch_index);

  /**
   * Evaluate the function values, the gradients, and the Hessians of the polynomial interpolation
   * from the DoF values in the input vector to the quadrature points on the unit cell
   * @param evaluation_flag Flags to indicate what has to be evaluated
   */
  void evaluate(const EvaluationFlags::EvaluationFlags evaluation_flag);

  /**
   * Return the value of a finite element function interpolated to the quadrature point with index q_point
   * @param q_point Index of the quadrature point
   */
  value_type get_value(const unsigned int q_point) const;

  /**
   * Return the gradient of a finite element function interpolated to the quadrature point with index q_point
   * @param q_point Index of the quadrature point
   */
  gradient_type get_gradient(const unsigned int q_point) const;

  /**
   * Write a contribution that gets multiplied by the value of the test function
   * to the field containing the values at quadrature points with index q_point
   * @param val_in Contribution to be tested againts test function
   * @param q_point Index of the qaudrature point
   */
  void submit_value(const value_type val_in, const unsigned int q_point);

  /**
   * Write a contribution that gets multiplied by the gradient of the test function
   * to the field containing the gradients at quadrature points with index q_point
   * @param grad_in Contribution to be tested againts gradient of test function
   * @param q_point Index of the qaudrature point
   */
  void submit_gradient(const gradient_type grad_in, const unsigned int q_point);

  /**
   * This function takes the values and/or gradients that are stored on quadrature points,
   * tests them by all the basis functions/gradients on the cell and performs the cell integration
   * @param integration_flag Control of which of the submitted contributions are used during the integration
   */
  void integrate(const EvaluationFlags::EvaluationFlags integration_flag);

  const unsigned int dofs_per_component; /*!< Number of degrees of freedom per component */
  const unsigned int dofs_per_cell;      /*!< Number of degrees of freedom per cell */
  const unsigned int n_q_points;         /*!< Total number of quadrature points */

private:
  void check_template_arguments(const unsigned int fe_no,
                                const unsigned int first_selected_component);
};


// Class constructor
//
template <int dim, int fe_degree_x, int n_q_points_1d_x, int fe_degree_z, int n_q_points_1d_z,
          int n_components_, typename Number, typename VectorizedArrayType>
inline FEEvaluationAniso<dim, fe_degree_x, n_q_points_1d_x, fe_degree_z, n_q_points_1d_z,
                    n_components_, Number, VectorizedArrayType>::
  FEEvaluationAniso(const MatrixFree<dim, Number, VectorizedArrayType> &matrix_free,
                    const unsigned int                                  fe_no,
                    const unsigned int                                  quad_no,
                    const unsigned int first_selected_component,
                    const unsigned int active_fe_index,
                    const unsigned int active_quad_index)
  : BaseClass(matrix_free,
              fe_no,
              first_selected_component,
              quad_no,
              fe_degree_x*fe_degree_z,
              n_q_points_1d_x*n_q_points_1d_z,
              true /*note: this is not a face*/,
              active_fe_index,
              active_quad_index,
              numbers::invalid_unsigned_int /*face_type*/)
  , dofs_per_component((fe_degree_x + 1)*(fe_degree_z + 1))
  , dofs_per_cell((fe_degree_x + 1)*(fe_degree_z + 1) * n_components_)
  , n_q_points(n_q_points_1d_x*n_q_points_1d_z)
  //GO: Why not using the static definitions in the constructor?
{}

// Same as FEEvaluation::reinit
//
//GO: I fail to see what is the difference with the base class and how the anisotropic structure impacts.
//    Does it depend on the use of [0] to initialize with a fixed qaudrature, to be overwritten afterwards?
template <int dim, int fe_degree_x, int n_q_points_1d_x, int fe_degree_z, int n_q_points_1d_z,
          int n_components_, typename Number, typename VectorizedArrayType>
inline void
FEEvaluationAniso<dim, fe_degree_x, n_q_points_1d_x, fe_degree_z, n_q_points_1d_z, n_components_,
                  Number, VectorizedArrayType>::reinit(const unsigned int cell_index)
{
  Assert(this->matrix_free != nullptr,
         ExcMessage("FEEvaluation was initialized without a matrix-free object."
                    " Integer indexing is not possible."));

  Assert(this->dof_info != nullptr, ExcNotInitialized());
  Assert(this->mapping_data != nullptr, ExcNotInitialized());
  this->cell = cell_index;
  this->cell_type = this->matrix_free->get_mapping_info().get_cell_type(cell_index);

  const unsigned int offsets = this->mapping_data->data_index_offsets[cell_index];
  this->jacobian = &this->mapping_data->jacobians[0][offsets];
  this->J_value  = &this->mapping_data->JxW_values[offsets];
  if (!this->mapping_data->jacobian_gradients[0].empty())
    {
      this->jacobian_gradients = this->mapping_data->jacobian_gradients[0].data() + offsets;
      this->jacobian_gradients_non_inverse = this->mapping_data->jacobian_gradients_non_inverse[0].data() + offsets;
    }

  if (this->matrix_free->n_active_entries_per_cell_batch(this->cell) == n_lanes)
    {
      DEAL_II_OPENMP_SIMD_PRAGMA
      for (unsigned int i = 0; i < n_lanes; ++i)
        this->cell_ids[i] = cell_index * n_lanes + i;
    }
  else
    {
      unsigned int i = 0;
      for (; i < this->matrix_free->n_active_entries_per_cell_batch(this->cell); ++i)
        this->cell_ids[i] = cell_index * n_lanes + i;
      for (; i < n_lanes; ++i)
        this->cell_ids[i] = numbers::invalid_unsigned_int;
    }

  //if (this->mapping_data->quadrature_points.empty() == false)
    //this->quadrature_points =
      //&this->mapping_data->quadrature_points[this->mapping_data->quadrature_point_offsets[this->cell]];
  if (this->mapping_data->quadrature_points.empty() == false)
    this->quadrature_points =
      &this->mapping_data->quadrature_points[0];
  //for (unsigned int i = 0; i < this->mapping_data->descriptor[0].n_q_points; i++)
    //this->quadrature_points = &(this->mapping_data->descriptor[0].quadrature.get_points());
    //for (unsigned int i = 0; i < this->n_quadrature_points; i++)
      //std::cout << this-> quadrature_points[i] << "      ";
  //std::cout << std::endl;
  //std::cout << "         " << this->n_quadrature_points << "         " << this->mapping_data->quadrature_point_offsets[this->cell] << std::endl;

  if constexpr (running_in_debug_mode())
    {
      this->is_reinitialized           = true;
      this->dof_values_initialized     = false;
      this->values_quad_initialized    = false;
      this->gradients_quad_initialized = false;
      this->hessians_quad_initialized  = false;
    }
}

// Same as FEEvaluation::evaluate
//
template <int dim, int fe_degree_x, int n_q_points_1d_x, int fe_degree_z, int n_q_points_1d_z,
          int n_components_, typename Number, typename VectorizedArrayType>
inline void
FEEvaluationAniso<dim, fe_degree_x, n_q_points_1d_x, fe_degree_z, n_q_points_1d_z, n_components_,
                  Number, VectorizedArrayType>::evaluate(const EvaluationFlags::EvaluationFlags evaluation_flag)
{
  Assert(evaluation_flag == EvaluationFlags::gradients || evaluation_flag == EvaluationFlags::values,
         ExcMessage("Not yet implemented"));

  using Number2 = typename FEEvaluationData<dim, VectorizedArrayType, false>::shape_info_number_type;

  std::array<const internal::MatrixFreeFunctions::UnivariateShapeData<Number2> *, dim> univariate_shape_data;

  const auto &shape_data = this->get_shape_info().data;

  for (unsigned int i = 0; i < dim; ++i) {
    univariate_shape_data[i] = &shape_data[i];
    //for (unsigned int j = 0; j < univariate_shape_data[i]->shape_values.size(); j++)
      //std::cout << univariate_shape_data[i]->shape_gradients[j] << std::endl;
    //std::cout << std::endl;
  }

  VectorizedArrayType *temp1             = this->get_scratch_data().begin();

  if (evaluation_flag & EvaluationFlags::values)
  {
    const VectorizedArrayType *values_dofs = this->begin_dof_values();
    VectorizedArrayType *vals_quad         = this->begin_values();

    // Here is where the polynomial anisotropy comes into the game
    static const internal::EvaluatorVariant variant = internal::EvaluatorVariant::evaluate_general;
    apply<variant, dim, fe_degree_x + 1, n_q_points_1d_z, fe_degree_z + 1, 1, false, false, VectorizedArrayType, Number2>
          (univariate_shape_data[1]->shape_values.begin(), values_dofs, temp1);
    apply<variant, dim, 1, n_q_points_1d_x, fe_degree_x + 1, n_q_points_1d_z, false, false, VectorizedArrayType, Number2>
          (univariate_shape_data[0]->shape_values.begin(), temp1, vals_quad);
  }

  if (evaluation_flag & EvaluationFlags::gradients)
  {
    const VectorizedArrayType *values_dofs = this->begin_dof_values();
    VectorizedArrayType *grads_quad        = this->begin_gradients();

    // Here is where the polynomial anisotropy comes into the game
    static const internal::EvaluatorVariant variant = internal::EvaluatorVariant::evaluate_general;
    apply<variant, dim, fe_degree_x + 1, n_q_points_1d_z, fe_degree_z + 1, 1, false, false, VectorizedArrayType, Number2>
          (univariate_shape_data[1]->shape_values.begin(), values_dofs, temp1);
    apply<variant, dim, 1, n_q_points_1d_x, fe_degree_x + 1, n_q_points_1d_z, false, false, VectorizedArrayType, Number2>
          (univariate_shape_data[0]->shape_gradients.begin(), temp1, grads_quad);

    grads_quad += n_q_points_1d_x * n_q_points_1d_z; //GO: Why not using static_dofs_per_component?

    apply<variant, dim, fe_degree_x + 1, n_q_points_1d_z, fe_degree_z + 1, 1, false, false, VectorizedArrayType, Number2>
          (univariate_shape_data[1]->shape_gradients.begin(), values_dofs, temp1);
    apply<variant, dim, 1, n_q_points_1d_x, fe_degree_x + 1, n_q_points_1d_z, false, false, VectorizedArrayType, Number2>
          (univariate_shape_data[0]->shape_values.begin(), temp1, grads_quad);
  }
}

//GO: Is this needed?
template <int dim, int fe_degree_x, int n_q_points_1d_x, int fe_degree_z, int n_q_points_1d_z,
          int n_components_, typename Number, typename VectorizedArrayType>
inline typename FEEvaluationAniso<dim, fe_degree_x, n_q_points_1d_x, fe_degree_z, n_q_points_1d_z,
                    n_components_, Number, VectorizedArrayType>::value_type
FEEvaluationAniso<dim, fe_degree_x, n_q_points_1d_x, fe_degree_z, n_q_points_1d_z, n_components_,
                  Number, VectorizedArrayType>::get_value(const unsigned int q_point) const
{
  return this->values_quad[q_point];
}

//GO: Is this needed?
template <int dim, int fe_degree_x, int n_q_points_1d_x, int fe_degree_z, int n_q_points_1d_z,
          int n_components_, typename Number, typename VectorizedArrayType>
inline Tensor<1, dim, VectorizedArrayType>
FEEvaluationAniso<dim, fe_degree_x, n_q_points_1d_x, fe_degree_z, n_q_points_1d_z, n_components_,
                  Number, VectorizedArrayType>::get_gradient(const unsigned int q_point) const
{
  Tensor<1, dim, VectorizedArrayType> grad_out;

  const VectorizedArrayType *grad_in = this->begin_gradients();
  for (unsigned int d = 0; d < dim; d++) {
    VectorizedArrayType temp = 0;
    for (unsigned int dp = 0; dp < dim; dp++)
      temp += this->inverse_jacobian(q_point)[d][dp] * grad_in[dp*this->quadrature_point_indices().size() + q_point];
    grad_out[d] = temp;
  }

  return grad_out;
}

//GO: Is this needed?
template <int dim, int fe_degree_x, int n_q_points_1d_x, int fe_degree_z, int n_q_points_1d_z,
          int n_components_, typename Number, typename VectorizedArrayType>
inline void
FEEvaluationAniso<dim, fe_degree_x, n_q_points_1d_x, fe_degree_z, n_q_points_1d_z, n_components_,
                  Number, VectorizedArrayType>::submit_value(const value_type val_in, const unsigned int q_point)
{
  this->begin_values()[q_point] = this->JxW(q_point)*val_in;
}

//GO: Is this needed?
template <int dim, int fe_degree_x, int n_q_points_1d_x, int fe_degree_z, int n_q_points_1d_z,
          int n_components_, typename Number, typename VectorizedArrayType>
inline void
FEEvaluationAniso<dim, fe_degree_x, n_q_points_1d_x, fe_degree_z, n_q_points_1d_z, n_components_,
                  Number, VectorizedArrayType>::submit_gradient(const gradient_type grad_in, const unsigned int q_point)
{
  for (unsigned int d = 0; d < dim; d++)
  {
    VectorizedArrayType temp = 0;
    for (unsigned int dp = 0; dp < dim; dp++)
      temp += this->inverse_jacobian(q_point)[dp][d] * this->JxW(q_point)*grad_in[dp];
    this->begin_gradients()[d*this->quadrature_point_indices().size() + q_point] = temp;
  }
}

// Same as FEEvaluation::integrate
//
template <int dim, int fe_degree_x, int n_q_points_1d_x, int fe_degree_z, int n_q_points_1d_z,
          int n_components_, typename Number, typename VectorizedArrayType>
inline void
FEEvaluationAniso<dim, fe_degree_x, n_q_points_1d_x, fe_degree_z, n_q_points_1d_z, n_components_,
                  Number, VectorizedArrayType>::integrate(const EvaluationFlags::EvaluationFlags evaluation_flag)
{
  Assert(evaluation_flag == EvaluationFlags::gradients | evaluation_flag == EvaluationFlags::values,
         ExcMessage("Not yet implemented"));

  using Number2 = typename FEEvaluationData<dim, VectorizedArrayType, false>::shape_info_number_type;

  std::array<const internal::MatrixFreeFunctions::UnivariateShapeData<Number2> *, dim> univariate_shape_data;

  const auto &shape_data = this->get_shape_info().data;

  for (unsigned int i = 0; i < dim; ++i)
    univariate_shape_data[i] = &shape_data[i];

  VectorizedArrayType *temp1             = this->get_scratch_data().begin();

  if (evaluation_flag & EvaluationFlags::values)
  {
    VectorizedArrayType *val_out          = this->begin_dof_values();
    VectorizedArrayType *vals_quad        = this->begin_values();

    // Here is where the polynomial anisotropy comes into the game
    static const internal::EvaluatorVariant variant = internal::EvaluatorVariant::evaluate_general;
    apply<variant, dim, fe_degree_x + 1, fe_degree_z + 1, fe_degree_z + 1, 1, true, false, VectorizedArrayType, Number2>
          (univariate_shape_data[1]->shape_values.begin(), vals_quad, temp1);
    apply<variant, dim, 1, fe_degree_x + 1, fe_degree_x + 1, fe_degree_z + 1, true, false, VectorizedArrayType, Number2>
          (univariate_shape_data[0]->shape_values.begin(), temp1, val_out);
  }

  if (evaluation_flag & EvaluationFlags::gradients)
  {
    VectorizedArrayType *grad_out          = this->begin_dof_values();
    VectorizedArrayType *grads_quad        = this->begin_gradients();

    // Here is where the polynomial anisotropy comes into the game
    static const internal::EvaluatorVariant variant = internal::EvaluatorVariant::evaluate_general;
    apply<variant, dim, fe_degree_x + 1, fe_degree_z + 1, n_q_points_1d_z, 1, true, false, VectorizedArrayType, Number2>
          (univariate_shape_data[1]->shape_values.begin(), grads_quad, temp1);
    apply<variant, dim, 1, fe_degree_x + 1, n_q_points_1d_x, fe_degree_z + 1, true, false, VectorizedArrayType, Number2>
          (univariate_shape_data[0]->shape_gradients.begin(), temp1, grad_out);

    grads_quad += (fe_degree_x + 1)*(fe_degree_z + 1); //GO: Why not using static_dofs_per_component?

    apply<variant, dim, fe_degree_x + 1, fe_degree_z + 1, n_q_points_1d_z, 1, true, false, VectorizedArrayType, Number2>
          (univariate_shape_data[1]->shape_gradients.begin(), grads_quad, temp1);
    apply<variant, dim, 1, fe_degree_x + 1, n_q_points_1d_x, fe_degree_z + 1, true, true, VectorizedArrayType, Number2>
          (univariate_shape_data[0]->shape_values.begin(), temp1, grad_out);
  }
}



/**
 * The class that provides all functions necessary to evaluate functions at quadrature points and face integrations
 * optimized for matrix-free operators with (possibly) different polynomial degrees along each coordinate direction.
 */
//GO: I would consider another header file for the FaceEvaluation
template <int dim, int fe_degree_x, int n_q_points_1d_x, int fe_degree_z, int n_q_points_1d_z,
          int n_components_ = 1, typename Number = double, typename VectorizedArrayType = VectorizedArray<Number>>
class FEFaceEvaluationAniso : public FEEvaluationBase<dim, n_components_, Number, true, VectorizedArrayType>
{
  static_assert(
    std::is_same_v<Number, typename VectorizedArrayType::value_type>,
    "Type of Number and of VectorizedArrayType do not match.");

  public:
    using BaseClass = FEEvaluationBase<dim, n_components_, Number, true, VectorizedArrayType>;
    using number_type = Number; //GO: What is its role?
    using value_type = typename BaseClass::value_type;
    using gradient_type = typename BaseClass::gradient_type;
    static constexpr unsigned int dimension = dim;
    static constexpr unsigned int n_components = n_components_;
    static constexpr unsigned int n_lanes = VectorizedArrayType::size();
    /*static constexpr unsigned int static_n_q_points =
      Utilities::pow(n_q_points_1d, dim - 1);
    static constexpr unsigned int static_n_q_points_cell =
      Utilities::pow(n_q_points_1d, dim);
    static constexpr unsigned int static_dofs_per_component =
      Utilities::pow(fe_degree + 1, dim);
    static constexpr unsigned int tensor_dofs_per_cell =
      static_dofs_per_component * n_components;
    static constexpr unsigned int static_dofs_per_cell =
      static_dofs_per_component * n_components;*/
    FEFaceEvaluationAniso(
      const MatrixFree<dim, Number, VectorizedArrayType> &matrix_free,
      const bool                                          is_interior_face = true,
      const unsigned int                                  dof_no           = 0,
      const unsigned int                                  quad_no          = 0,
      const unsigned int first_selected_component                          = 0,
      const unsigned int active_fe_index   = numbers::invalid_unsigned_int,
      const unsigned int active_quad_index = numbers::invalid_unsigned_int,
      const unsigned int face_type         = numbers::invalid_unsigned_int);
    //FEFaceEvaluation(
    //  const MatrixFree<dim, Number, VectorizedArrayType> &matrix_free,
    //  const std::pair<unsigned int, unsigned int>        &range,
    //  const bool                                          is_interior_face = true,
    //  const unsigned int                                  dof_no           = 0,
    //  const unsigned int                                  quad_no          = 0,
    //  const unsigned int first_selected_component                          = 0);
    void
    reinit(const unsigned int face_batch_number);
    //void
    //reinit(const unsigned int cell_batch_number, const unsigned int face_number);
    //static bool
    //fast_evaluation_supported(const unsigned int given_degree,
    //                          const unsigned int given_n_q_points_1d);
    void
    evaluate(const EvaluationFlags::EvaluationFlags evaluation_flag);
    value_type
    get_value(const unsigned int q_point) const;
    gradient_type
    get_gradient(const unsigned int q_point) const;
    value_type
    get_normal_derivative(const unsigned int q_point) const;
    void
    integrate(const EvaluationFlags::EvaluationFlags integration_flag);
    void
    submit_value(const value_type val_in, const unsigned int q_point);
    void
    submit_normal_derivative(const value_type grad_in, const unsigned int q_point);
    std_cxx20::ranges::iota_view<unsigned int, unsigned int>
    dof_indices() const;
    bool
    at_boundary() const;
    types::boundary_id
    boundary_id() const;
    unsigned int
    get_dofs_per_component_projected_to_face();
    unsigned int
    get_dofs_projected_to_face();
    const unsigned int dofs_per_component;
    const unsigned int dofs_per_cell;
    const unsigned int n_q_points;
    unsigned int n_quadrature_points;
    //std::vector<Point<dim-1>> & quadrature_points = nullptr;
};


template <int dim, int fe_degree_x, int n_q_points_1d_x, int fe_degree_z, int n_q_points_1d_z,
          int n_components_, typename Number, typename VectorizedArrayType>
inline FEFaceEvaluationAniso<dim, fe_degree_x, n_q_points_1d_x, fe_degree_z, n_q_points_1d_z, n_components_,
                        Number, VectorizedArrayType>::
  FEFaceEvaluationAniso(
    const MatrixFree<dim, Number, VectorizedArrayType> &matrix_free,
    const bool                                          is_interior_face,
    const unsigned int                                  dof_no,
    const unsigned int                                  quad_no,
    const unsigned int first_selected_component,
    const unsigned int active_fe_index,
    const unsigned int active_quad_index,
    const unsigned int face_type)
  : BaseClass(matrix_free,
              dof_no,
              first_selected_component,
              quad_no,
              fe_degree_x*fe_degree_z,
              n_q_points_1d_x*n_q_points_1d_z,
              is_interior_face,
              active_fe_index,
              active_quad_index,
              face_type)
  , dofs_per_component((fe_degree_x + 1)*(fe_degree_z + 1))
  , dofs_per_cell((fe_degree_x + 1)*(fe_degree_z + 1) * n_components_)
  , n_q_points(n_q_points_1d_x*n_q_points_1d_z)
{}


// same as FEFaceEvaluation::reinit
template <int dim, int fe_degree_x, int n_q_points_1d_x, int fe_degree_z, int n_q_points_1d_z,
          int n_components_, typename Number, typename VectorizedArrayType>
inline void FEFaceEvaluationAniso<dim, fe_degree_x, n_q_points_1d_x, fe_degree_z, n_q_points_1d_z, n_components_,
                        Number, VectorizedArrayType>::reinit(const unsigned int face_index)
{
  Assert(this->mapped_geometry == nullptr,
         ExcMessage("FEEvaluation was initialized without a matrix-free object."
                    " Integer indexing is not possible"));
  if (this->mapped_geometry != nullptr)
    return;

  this->cell = face_index;
  this->dof_access_index =
    this->is_interior_face() ?
      internal::MatrixFreeFunctions::DoFInfo::dof_access_face_interior :
      internal::MatrixFreeFunctions::DoFInfo::dof_access_face_exterior;
  Assert(this->mapping_data != nullptr, ExcNotInitialized());

  if (face_index >=
        this->matrix_free->get_task_info().face_partition_data.back() &&
      face_index <
        this->matrix_free->get_task_info().boundary_partition_data.back())
    Assert(this->is_interior_face(),
           ExcMessage(
             "Boundary faces do not have a neighbor. When looping over "
             "boundary faces use FEFaceEvaluation with the parameter "
             "is_interior_face set to true. "));

  this->reinit_face(this->matrix_free->get_face_info(face_index));

  unsigned int i = 0;
  for (; i < this->matrix_free->n_active_entries_per_face_batch(this->cell); ++i)
    this->face_ids[i] = face_index * n_lanes + i;
  for (; i < n_lanes; ++i)
    this->face_ids[i] = numbers::invalid_unsigned_int;

  this->cell_type = this->matrix_free->get_mapping_info().face_type[face_index];
  const unsigned int offsets = this->mapping_data->data_index_offsets[face_index];
  this->J_value        = &this->mapping_data->JxW_values[offsets];
  this->normal_vectors = &this->mapping_data->normal_vectors[offsets];
  this->jacobian = &this->mapping_data->jacobians[!this->is_interior_face()][offsets];
  this->normal_x_jacobian =
    &this->mapping_data->normals_times_jacobians[!this->is_interior_face()][offsets];
  this->jacobian_gradients =
    this->mapping_data->jacobian_gradients[!this->is_interior_face()].data() + offsets;
  this->jacobian_gradients_non_inverse =
    this->mapping_data->jacobian_gradients_non_inverse[!this->is_interior_face()].data() + offsets;


  if (this->mapping_data->quadrature_point_offsets.empty() == false)
    {
      AssertIndexRange(this->cell,
                       this->mapping_data->quadrature_point_offsets.size());
      this->quadrature_points =
        this->mapping_data->quadrature_points.data() +
        this->mapping_data->quadrature_point_offsets[this->cell];
    }


  /*
  if (this->normal_vector(0)[0][0] == 0.)
  {
    this->n_quadrature_points = this->mapping_data->descriptor[0].n_q_points;
    //std::cout << "testtest " << this->n_quadrature_points << std::endl;
    //this->quadrature_points = &(this->mapping_data->descriptor[0].quadrature.get_points());
  }
  else
  {
    this->n_quadrature_points = this->mapping_data->descriptor[1].n_q_points;
    //std::cout << "testtest " << this->n_quadrature_points << std::endl;
    //this->quadrature_points = &(this->mapping_data->descriptor[1].quadrature.get_points());
  }*/

  if constexpr (running_in_debug_mode())
    {
      this->is_reinitialized           = true;
      this->dof_values_initialized     = false;
      this->values_quad_initialized    = false;
      this->gradients_quad_initialized = false;
      this->hessians_quad_initialized  = false;
    }
}

/*
template <int dim, int fe_degree_x, int n_q_points_1d_x, int fe_degree_z, int n_q_points_1d_z,
          int n_components_, typename Number, typename VectorizedArrayType>
inline Point<dim-1, Number> FEFaceEvaluationAniso<dim, fe_degree_x, n_q_points_1d_x, fe_degree_z, n_q_points_1d_z, n_components_,
                        Number, VectorizedArrayType>::quadrature_point(const unsigned int q) const
{
  AssertIndexRange(q, this->n_quadrature_points);
  Assert(this->quadrature_points != nullptr,
         internal::ExcMatrixFreeAccessToUninitializedMappingField(
           "update_quadrature_points"));

  // Cartesian/affine mesh: only first vertex of cell is stored, we must
  // compute it through the Jacobian (which is stored in non-inverted and
  // non-transposed form as index '1' in the jacobian field)
  if (is_face == false &&
      this->cell_type <= internal::MatrixFreeFunctions::affine)
    {
      Assert(this->jacobian != nullptr, ExcNotInitialized());
      Point<dim, Number> point = this->quadrature_points[0];

      const Tensor<2, dim, Number> &jac = this->jacobian[1];
      if (this->cell_type == internal::MatrixFreeFunctions::cartesian)
        for (unsigned int d = 0; d < dim; ++d)
          point[d] += jac[d][d] * static_cast<typename Number::value_type>(
                                    this->descriptor->quadrature.point(q)[d]);
      else
        for (unsigned int d = 0; d < dim; ++d)
          for (unsigned int e = 0; e < dim; ++e)
            point[d] += jac[d][e] * static_cast<typename Number::value_type>(
                                      this->descriptor->quadrature.point(q)[e]);
      return point;
    }
  else
    return this->quadrature_points[q];
}
*/


template <int dim, int fe_degree_x, int n_q_points_1d_x, int fe_degree_z, int n_q_points_1d_z,
          int n_components_, typename Number, typename VectorizedArrayType>
inline void
FEFaceEvaluationAniso<dim, fe_degree_x, n_q_points_1d_x, fe_degree_z, n_q_points_1d_z, n_components_,
                  Number, VectorizedArrayType>::evaluate(const EvaluationFlags::EvaluationFlags evaluation_flag)
{
  Assert(evaluation_flag | EvaluationFlags::gradients || evaluation_flag == EvaluationFlags::values,
         ExcMessage("Not yet implemented"));

  using Number2 = typename FEEvaluationData<dim, VectorizedArrayType, false>::shape_info_number_type;

  std::array<const internal::MatrixFreeFunctions::UnivariateShapeData<Number2> *, dim> univariate_shape_data;

  const auto &shape_data = this->get_shape_info().data;

  for (unsigned int i = 0; i < dim; ++i) {
    univariate_shape_data[i] = &shape_data[i];
  }

  VectorizedArrayType *temp1             = this->get_scratch_data().begin();

  if (evaluation_flag & EvaluationFlags::values)
  {
    const VectorizedArrayType *values_dofs = this->begin_dof_values();
    VectorizedArrayType *vals_quad         = this->begin_values();

    static const internal::EvaluatorVariant variant = internal::EvaluatorVariant::evaluate_general;
    //GO: Is this normal_vector the physical one or the reference one? Moreover, == 0 or < very small number?
    if (this -> normal_vector(0)[0][0] == 0.)
    {
      // horizontal face
      apply<variant, dim, fe_degree_x + 1, 1, fe_degree_z + 1, 1, false, false, VectorizedArrayType, Number2>
            (univariate_shape_data[1]->shape_data_on_face[this->is_interior_face() ? 1 : 0].begin(), values_dofs, temp1);
      apply<variant, dim, 1, n_q_points_1d_x, fe_degree_x + 1, 1, false, false, VectorizedArrayType, Number2>
            (univariate_shape_data[0]->shape_values.begin(), temp1, vals_quad);
    }
    else
    {
      // vertical face
      apply<variant, dim, fe_degree_x + 1, n_q_points_1d_z, fe_degree_z + 1, 1, false, false, VectorizedArrayType, Number2>
            (univariate_shape_data[1]->shape_values.begin(), values_dofs, temp1);
      apply<variant, dim, 1, 1, fe_degree_x + 1, n_q_points_1d_z, false, false, VectorizedArrayType, Number2>
            (univariate_shape_data[0]->shape_data_on_face[this->is_interior_face() ? 1 : 0].begin(), temp1, vals_quad);
    }
  }

  if (evaluation_flag & EvaluationFlags::gradients)
  {
    const VectorizedArrayType *values_dofs = this->begin_dof_values();
    VectorizedArrayType *grads_quad        = this->begin_gradients();

    static const internal::EvaluatorVariant variant = internal::EvaluatorVariant::evaluate_general;
    if (this -> normal_vector(0)[0][0] == 0.)
    {
      // horizontal face
      apply<variant, dim, fe_degree_x + 1, 1, fe_degree_z + 1, 1, false, false, VectorizedArrayType, Number2>
            (univariate_shape_data[1]->shape_data_on_face[this->is_interior_face() ? 1 : 0].begin(), values_dofs, temp1);
      apply<variant, dim, 1, n_q_points_1d_x, fe_degree_x + 1, 1, false, false, VectorizedArrayType, Number2>
            (univariate_shape_data[0]->shape_gradients.begin(), temp1, grads_quad);

      grads_quad += n_q_points_1d_x;

      apply<variant, dim, fe_degree_x + 1, 1, fe_degree_z + 1, 1, false, false, VectorizedArrayType, Number2>
            (univariate_shape_data[1]->shape_data_on_face[this->is_interior_face() ? 1 : 0].begin() + fe_degree_z + 1, values_dofs, temp1);
      apply<variant, dim, 1, n_q_points_1d_x, fe_degree_x + 1, 1, false, false, VectorizedArrayType, Number2>
            (univariate_shape_data[0]->shape_values.begin(), temp1, grads_quad);
    }
    else
    {
      // vertical face
      apply<variant, dim, fe_degree_x + 1, n_q_points_1d_z, fe_degree_z + 1, 1, false, false, VectorizedArrayType, Number2>
            (univariate_shape_data[1]->shape_values.begin(), values_dofs, temp1);
      apply<variant, dim, 1, 1, fe_degree_x + 1, n_q_points_1d_z, false, false, VectorizedArrayType, Number2>
            (univariate_shape_data[0]->shape_data_on_face[this->is_interior_face() ? 1 : 0].begin() + fe_degree_x + 1, temp1, grads_quad);

      grads_quad += n_q_points_1d_z;

      apply<variant, dim, fe_degree_x + 1, n_q_points_1d_z, fe_degree_z + 1, 1, false, false, VectorizedArrayType, Number2>
            (univariate_shape_data[1]->shape_gradients.begin(), values_dofs, temp1);
      apply<variant, dim, 1, 1, fe_degree_x + 1, n_q_points_1d_z, false, false, VectorizedArrayType, Number2>
            (univariate_shape_data[0]->shape_data_on_face[this->is_interior_face() ? 1 : 0].begin(), temp1, grads_quad);
    }

    // now this->begin_gradients() contains the gradients*dofs evaluated at the quadrature points, where the quadrature
    // point is the index running faster and the dimension is the index running slower. thus we have grad[0][0], grad[0][1],
    // ... grad[0][Q-1], grad[1][0], ..., grad[1][Q-1], ..., grad[d-1][0], ..., grad[d-1][Q-1]
    // where grad[i][j] is the i-th component of the gradient evaluated at the j-th quadrature point.
  }

}


template <int dim, int fe_degree_x, int n_q_points_1d_x, int fe_degree_z, int n_q_points_1d_z,
          int n_components_, typename Number, typename VectorizedArrayType>
inline typename FEFaceEvaluationAniso<dim, fe_degree_x, n_q_points_1d_x, fe_degree_z, n_q_points_1d_z,
                    n_components_, Number, VectorizedArrayType>::value_type
FEFaceEvaluationAniso<dim, fe_degree_x, n_q_points_1d_x, fe_degree_z, n_q_points_1d_z, n_components_,
                  Number, VectorizedArrayType>::get_value(const unsigned int q_point) const
{
  return this->values_quad[q_point];
}


template <int dim, int fe_degree_x, int n_q_points_1d_x, int fe_degree_z, int n_q_points_1d_z,
          int n_components_, typename Number, typename VectorizedArrayType>
inline typename FEFaceEvaluationAniso<dim, fe_degree_x, n_q_points_1d_x, fe_degree_z, n_q_points_1d_z,
                    n_components_, Number, VectorizedArrayType>::gradient_type
FEFaceEvaluationAniso<dim, fe_degree_x, n_q_points_1d_x, fe_degree_z, n_q_points_1d_z, n_components_,
                  Number, VectorizedArrayType>::get_gradient(const unsigned int q_point) const
{
  gradient_type grad_out;

  const VectorizedArrayType *grad_in = this->begin_gradients();
  const unsigned int n_q_face = this->normal_vector(0)[0][0] == 0. ? n_q_points_1d_x : n_q_points_1d_z;
  // specialized to the case dim=2. takes into account the fact that derivatives are reordered. see also 4255 fe_evaluation.h

  if (this->normal_vector(0)[0][0] == 0.)
  {
    // horizontal face. derivatives are in the right order. recall that inverse_jacobian is the inverse transposed
    grad_out[0] = this->inverse_jacobian(0)[0][0] * grad_in[q_point] +
                  this->inverse_jacobian(0)[1][0] * grad_in[n_q_face + q_point];
    grad_out[1] = this->inverse_jacobian(0)[0][1] * grad_in[q_point] +
                  this->inverse_jacobian(0)[1][1] * grad_in[n_q_face + q_point];
  }
  else
  {
    // vertical face. derivatives are swapped
    grad_out[0] = this->inverse_jacobian(0)[0][1] * grad_in[q_point] +
                  this->inverse_jacobian(0)[1][1] * grad_in[n_q_face + q_point];
    grad_out[1] = this->inverse_jacobian(0)[0][0] * grad_in[q_point] +
                  this->inverse_jacobian(0)[1][0] * grad_in[n_q_face + q_point];
  }
  /*
  for (unsigned int d = 0; d < dim; d++) {
    VectorizedArrayType temp = 0;
    for (unsigned int dp = 0; dp < dim; dp++) {
      std::cout << this->inverse_jacobian(0)[d][dp] << "      ";
    }
    std::cout << std::endl;
  }
  */

  return grad_out;
}


template <int dim, int fe_degree_x, int n_q_points_1d_x, int fe_degree_z, int n_q_points_1d_z,
          int n_components_, typename Number, typename VectorizedArrayType>
inline typename FEFaceEvaluationAniso<dim, fe_degree_x, n_q_points_1d_x, fe_degree_z, n_q_points_1d_z,
                    n_components_, Number, VectorizedArrayType>::value_type
FEFaceEvaluationAniso<dim, fe_degree_x, n_q_points_1d_x, fe_degree_z, n_q_points_1d_z, n_components_,
                  Number, VectorizedArrayType>::get_normal_derivative(const unsigned int q_point) const
{
  return this->get_gradient(q_point) * this->normal_vector(0);
}


template <int dim, int fe_degree_x, int n_q_points_1d_x, int fe_degree_z, int n_q_points_1d_z,
          int n_components_, typename Number, typename VectorizedArrayType>
inline void
FEFaceEvaluationAniso<dim, fe_degree_x, n_q_points_1d_x, fe_degree_z, n_q_points_1d_z, n_components_,
                  Number, VectorizedArrayType>::submit_value(const value_type val_in, const unsigned int q_point)
{
  // TODO: use values_quad instead of begin_values
  this->begin_values()[q_point] = this->J_value[0] *
                                  this->mapping_data->descriptor[this->normal_vector(0)[0][0] == 0. ? 0 : 1].quadrature_weights[q_point] *
                                  val_in;
}


template <int dim, int fe_degree_x, int n_q_points_1d_x, int fe_degree_z, int n_q_points_1d_z,
          int n_components_, typename Number, typename VectorizedArrayType>
inline void
FEFaceEvaluationAniso<dim, fe_degree_x, n_q_points_1d_x, fe_degree_z, n_q_points_1d_z, n_components_,
                  Number, VectorizedArrayType>::submit_normal_derivative(const value_type grad_in, const unsigned int q_point)
{
  //VectorizedArrayType *gradients = this -> gradients_quad + q_point*dim;
  //this->begin_gradients()[q_point] = this->J_value[0] *
    //                              this->mapping_data->descriptor[this->normal_vector(0)[0][0] == 0. ? 0 : 1].quadrature_weights[q_point] *
      //                            val_in;

  gradient_type inverse_jacobian_times_n;
    if (this->normal_vector(0)[0][0] == 0.)
    {
      // horizontal face. derivatives are in the right order. recall that inverse_jacobian is the inverse transposed
      inverse_jacobian_times_n[0] = this->inverse_jacobian(0)[0][0] * this->normal_vector(0)[0] +
                    this->inverse_jacobian(0)[1][0] * this->normal_vector(0)[1];
      inverse_jacobian_times_n[1] = this->inverse_jacobian(0)[0][1] * this->normal_vector(0)[0] +
                    this->inverse_jacobian(0)[1][1] * this->normal_vector(0)[1];
    }
    else
    {
      // vertical face. derivatives are swapped
      inverse_jacobian_times_n[0] = this->inverse_jacobian(0)[0][1] * this->normal_vector(0)[0] +
                    this->inverse_jacobian(0)[1][1] * this->normal_vector(0)[1];
      inverse_jacobian_times_n[1] = this->inverse_jacobian(0)[0][0] * this->normal_vector(0)[0] +
                    this->inverse_jacobian(0)[1][0] * this->normal_vector(0)[1];
    }

  unsigned int face_q_points = this -> normal_vector(0)[0][0] == 0. ? n_q_points_1d_x : n_q_points_1d_z;
  unsigned int face_direction = this -> normal_vector(0)[0][0] == 0. ? 0 : 1;
  for (unsigned int d = 0; d < dim; d++)
  {
    this->begin_gradients()[d*face_q_points + q_point] = inverse_jacobian_times_n[d] *
                            this->J_value[0] *
                            this->mapping_data->descriptor[face_direction].quadrature_weights[q_point] *
                            grad_in;
  }
}


template <int dim, int fe_degree_x, int n_q_points_1d_x, int fe_degree_z, int n_q_points_1d_z,
          int n_components_, typename Number, typename VectorizedArrayType>
inline void
FEFaceEvaluationAniso<dim, fe_degree_x, n_q_points_1d_x, fe_degree_z, n_q_points_1d_z, n_components_,
                  Number, VectorizedArrayType>::integrate(const EvaluationFlags::EvaluationFlags evaluation_flag)
{
  //Assert(evaluation_flag | EvaluationFlags::gradients | evaluation_flag == EvaluationFlags::values,
    //     ExcMessage("Not yet implemented"));

  using Number2 = typename FEEvaluationData<dim, VectorizedArrayType, false>::shape_info_number_type;

  std::array<const internal::MatrixFreeFunctions::UnivariateShapeData<Number2> *, dim> univariate_shape_data;

  const auto &shape_data = this->get_shape_info().data;

  for (unsigned int i = 0; i < dim; ++i)
    univariate_shape_data[i] = &shape_data[i];

  VectorizedArrayType *temp1             = this->get_scratch_data().begin();


  if (evaluation_flag & EvaluationFlags::values)
  {
    VectorizedArrayType *val_out          = this->begin_dof_values();
    VectorizedArrayType *vals_quad        = this->begin_values();

    static const internal::EvaluatorVariant variant = internal::EvaluatorVariant::evaluate_general;

    if (this -> normal_vector(0)[0][0] == 0.)
    {
      // horizontal face
      apply<variant, dim, n_q_points_1d_x, fe_degree_z + 1, 1, 1, false, false, VectorizedArrayType, Number2>
            (univariate_shape_data[1]->shape_data_on_face[this->is_interior_face() ? 1 : 0].begin(), vals_quad, temp1);
      apply<variant, dim, 1, fe_degree_x + 1, n_q_points_1d_x, fe_degree_z + 1, true, false, VectorizedArrayType, Number2>
            (univariate_shape_data[0]->shape_values.begin(), temp1, val_out);
    }
    else
    {
      // vertical face
      apply<variant, dim, 1, fe_degree_z + 1, n_q_points_1d_z, 1, true, false, VectorizedArrayType, Number2>
            (univariate_shape_data[1]->shape_values.begin(), vals_quad, temp1);
      apply<variant, dim, 1, fe_degree_x + 1, 1, fe_degree_z + 1, false, false, VectorizedArrayType, Number2>
            (univariate_shape_data[0]->shape_data_on_face[this->is_interior_face() ? 1 : 0].begin(), temp1, val_out);
    }
  }


  if (evaluation_flag & EvaluationFlags::gradients)
  {
    VectorizedArrayType *val_out           = this->begin_dof_values();
    VectorizedArrayType *grads_quad        = this->begin_gradients();

    static const internal::EvaluatorVariant variant = internal::EvaluatorVariant::evaluate_general;
    if (this -> normal_vector(0)[0][0] == 0.)
    {
      // horizontal face
      apply<variant, dim, n_q_points_1d_x, fe_degree_z + 1, 1, 1, false, false, VectorizedArrayType, Number2>
            (univariate_shape_data[1]->shape_data_on_face[this->is_interior_face() ? 1 : 0].begin(), grads_quad, temp1);
      apply<variant, dim, 1, fe_degree_x + 1, n_q_points_1d_x, fe_degree_z + 1, true, false, VectorizedArrayType, Number2>
            (univariate_shape_data[0]->shape_gradients.begin(), temp1, val_out);

      grads_quad += fe_degree_x + 1;

      apply<variant, dim, n_q_points_1d_x, fe_degree_z + 1, 1, 1, false, false, VectorizedArrayType, Number2>
            (univariate_shape_data[1]->shape_data_on_face[this->is_interior_face() ? 1 : 0].begin() + fe_degree_z + 1, grads_quad, temp1);
      apply<variant, dim, 1, fe_degree_x + 1, n_q_points_1d_x, fe_degree_z + 1, true, true, VectorizedArrayType, Number2>
            (univariate_shape_data[0]->shape_values.begin(), temp1, val_out);
    }
    else
    {
      // vertical face
      apply<variant, dim, 1, fe_degree_z + 1, n_q_points_1d_z, 1, true, false, VectorizedArrayType, Number2>
            (univariate_shape_data[1]->shape_values.begin(), grads_quad, temp1);
      apply<variant, dim, 1, fe_degree_x + 1, 1, fe_degree_z + 1, false, false, VectorizedArrayType, Number2>
            (univariate_shape_data[0]->shape_data_on_face[this->is_interior_face() ? 1 : 0].begin() + fe_degree_x + 1, temp1, val_out);

      grads_quad += fe_degree_z + 1;

      apply<variant, dim, 1, fe_degree_z + 1, n_q_points_1d_z, 1, true, false, VectorizedArrayType, Number2>
            (univariate_shape_data[1]->shape_gradients.begin(), grads_quad, temp1);
      apply<variant, dim, 1, fe_degree_x + 1, 1, fe_degree_z + 1, false, true, VectorizedArrayType, Number2>
            (univariate_shape_data[0]->shape_data_on_face[this->is_interior_face() ? 1 : 0].begin(), temp1, val_out);
    }
  }
}




DEAL_II_NAMESPACE_CLOSE

#endif
