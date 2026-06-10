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

};


// Class constructor
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

  if (this->mapping_data->quadrature_points.empty() == false)
    this->quadrature_points =
      &this->mapping_data->quadrature_points[0];

  if constexpr (running_in_debug_mode())
    {
      this->is_reinitialized           = true;
      this->dof_values_initialized     = false;
      this->values_quad_initialized    = false;
      this->gradients_quad_initialized = false;
      this->hessians_quad_initialized  = false;
    }
}


template <int dim, int fe_degree_x, int n_q_points_1d_x, int fe_degree_z, int n_q_points_1d_z,
          int n_components_, typename Number, typename VectorizedArrayType>
inline void
FEEvaluationAniso<dim, fe_degree_x, n_q_points_1d_x, fe_degree_z, n_q_points_1d_z, n_components_,
                  Number, VectorizedArrayType>::evaluate(const EvaluationFlags::EvaluationFlags evaluation_flag)
{
  using Number2 = typename FEEvaluationData<dim, VectorizedArrayType, false>::shape_info_number_type;

  std::array<const internal::MatrixFreeFunctions::UnivariateShapeData<Number2> *, dim> univariate_shape_data;

  const auto &shape_data = this->get_shape_info().data;

  for (unsigned int i = 0; i < dim; ++i)
    univariate_shape_data[i] = &shape_data[i];

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

    grads_quad += dofs_per_component;

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
      temp += this->inverse_jacobian(q_point)[d][dp] * grad_in[dp*n_q_points + q_point];
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
  int q_transf = q_point*(fe_degree_x+1) - int(std::floor(q_point/(fe_degree_z+1)))*((fe_degree_x+1)*(fe_degree_z+1)-1);
  this->begin_values()[q_point] = this->JxW(q_transf)*val_in;
}

//GO: Is this needed?
template <int dim, int fe_degree_x, int n_q_points_1d_x, int fe_degree_z, int n_q_points_1d_z,
          int n_components_, typename Number, typename VectorizedArrayType>
inline void
FEEvaluationAniso<dim, fe_degree_x, n_q_points_1d_x, fe_degree_z, n_q_points_1d_z, n_components_,
                  Number, VectorizedArrayType>::submit_gradient(const gradient_type grad_in, const unsigned int q_point)
{
  int q_transf = q_point*(fe_degree_x+1) - int(std::floor(q_point/(fe_degree_z+1)))*((fe_degree_x+1)*(fe_degree_z+1)-1);
  for (unsigned int d = 0; d < dim; d++)
  {
    VectorizedArrayType temp = 0;
    for (unsigned int dp = 0; dp < dim; dp++) 
      temp += this->inverse_jacobian(q_point)[dp][d] * this->JxW(q_transf)*grad_in[dp];
    
    this->begin_gradients()[d*n_q_points + q_point] = temp;
  }
}


template <int dim, int fe_degree_x, int n_q_points_1d_x, int fe_degree_z, int n_q_points_1d_z,
          int n_components_, typename Number, typename VectorizedArrayType>
inline void
FEEvaluationAniso<dim, fe_degree_x, n_q_points_1d_x, fe_degree_z, n_q_points_1d_z, n_components_,
                  Number, VectorizedArrayType>::integrate(const EvaluationFlags::EvaluationFlags evaluation_flag)
{
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

    grads_quad += dofs_per_component;

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
    
    FEFaceEvaluationAniso(
      const MatrixFree<dim, Number, VectorizedArrayType> &matrix_free,
      const bool                                          is_interior_face = true,
      const unsigned int                                  dof_no           = 0,
      const unsigned int                                  quad_no          = 0,
      const unsigned int first_selected_component                          = 0,
      const unsigned int active_fe_index   = numbers::invalid_unsigned_int,
      const unsigned int active_quad_index = numbers::invalid_unsigned_int,
      const unsigned int face_type         = numbers::invalid_unsigned_int);
      
    void
    reinit(const unsigned int face_batch_number);
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
    unsigned int
    left_or_right_endpoint();

    const unsigned int dofs_per_component;
    const unsigned int dofs_per_cell;
    const unsigned int n_q_points;
    unsigned int n_quadrature_points;

    // if is_vertical == true, the current face (in the physical space) is the image of a vertical face of the
    // reference square [0,1]x[0,1]. this is used to select the correct quadrature rule to use
    bool is_vertical;
    // if is_right_or_top == true, the current face (in the physical space) is the image of the 'right' face (if is_vertical)
    // or of the 'top' face (if !is_vertical) of the reference square [0,1]x[0,1]. this is used to choose whether to
    // evaluate the 1d basis function on the left endpoint 0 or on the right endpoint 1 of the unit interval [0,1]
    // when integrating along faces. this choice is made in left_or_right_endpoint()
    bool is_right_or_top;
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

  // in order to determine whether the current face is the image of a vertical or a horizontal face, we need to
  // know the normal vector on the reference cell. in the general case of a noncartesian grid, this is done by
  // multiplying the normal vector on the physical cell (given by this->normal_vector()) by the jacobian transposed.
  // the result is of the form (c,0)^T or (0,c)^T for some c. in practice, we do not check if the 'zero' component of
  // the result is exactly zero, but rather we check if its absolute value is smaller than 1e-10. then, the sign of the
  // other component will determine the value of is_right_or_top. in the case of a cartesian grid, the normal vector on 
  // the reference cell is a multiple of the normal vector on the physical cell (the jacobian is diagonal and constant)
  // which leads to the simplification that we use here. in general, we need to replace this->normal_vector(0) by the
  // correct quantity obtained after multplication by J^T. notice that it is enough to do this for one of the quadrature
  // points on the face (say the first one, indexed by zero)
  if (abs(this->normal_vector(0)[0][0]) < 1e-10) 
  {
    is_vertical = false;
    is_right_or_top = this->normal_vector(0)[1][0] > 0;
  }
  else 
  {
    is_vertical = true;
    is_right_or_top = this->normal_vector(0)[0][0] > 0;
  }

  if constexpr (running_in_debug_mode())
    {
      this->is_reinitialized           = true;
      this->dof_values_initialized     = false;
      this->values_quad_initialized    = false;
      this->gradients_quad_initialized = false;
      this->hessians_quad_initialized  = false;
    }
}


template <int dim, int fe_degree_x, int n_q_points_1d_x, int fe_degree_z, int n_q_points_1d_z,
          int n_components_, typename Number, typename VectorizedArrayType>
unsigned int
FEFaceEvaluationAniso<dim, fe_degree_x, n_q_points_1d_x, fe_degree_z, n_q_points_1d_z, n_components_,
                  Number, VectorizedArrayType>::left_or_right_endpoint()
{
  // returns 0 if the current physical face is associated with the left endpoint of the reference interval [0,1]
  // returns 1 otherwise
  if (this->cell < this->matrix_free->n_inner_face_batches() && this->is_interior_face())
    return is_right_or_top ? 1 : 0;

  if (this->cell < this->matrix_free->n_inner_face_batches() && !this->is_interior_face())
    return is_right_or_top ? 0 : 1;

  return is_right_or_top ? 1 : 0;

}


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

  for (unsigned int i = 0; i < dim; ++i)
    univariate_shape_data[i] = &shape_data[i];

  VectorizedArrayType *temp1             = this->get_scratch_data().begin();

  if (evaluation_flag & EvaluationFlags::values)
  {
    const VectorizedArrayType *values_dofs = this->begin_dof_values();
    VectorizedArrayType *vals_quad         = this->begin_values();

    static const internal::EvaluatorVariant variant = internal::EvaluatorVariant::evaluate_general;
    
    if (!is_vertical)
    {
      // horizontal face
      unsigned int n_endpoint = left_or_right_endpoint();
      apply<variant, dim, fe_degree_x + 1, 1, fe_degree_z + 1, 1, false, false, VectorizedArrayType, Number2>
            (univariate_shape_data[1]->shape_data_on_face[n_endpoint].begin(), values_dofs, temp1);
      apply<variant, dim, 1, n_q_points_1d_x, fe_degree_x + 1, 1, false, false, VectorizedArrayType, Number2>
            (univariate_shape_data[0]->shape_values.begin(), temp1, vals_quad);
    }
    else
    {
      // vertical face
      unsigned int n_endpoint = left_or_right_endpoint();
      apply<variant, dim, fe_degree_x + 1, n_q_points_1d_z, fe_degree_z + 1, 1, false, false, VectorizedArrayType, Number2>
            (univariate_shape_data[1]->shape_values.begin(), values_dofs, temp1);
      apply<variant, dim, 1, 1, fe_degree_x + 1, n_q_points_1d_z, false, false, VectorizedArrayType, Number2>
            (univariate_shape_data[0]->shape_data_on_face[n_endpoint].begin(), temp1, vals_quad);
    }
  }

  if (evaluation_flag & EvaluationFlags::gradients)
  {
    const VectorizedArrayType *values_dofs = this->begin_dof_values();
    VectorizedArrayType *grads_quad        = this->begin_gradients();

    static const internal::EvaluatorVariant variant = internal::EvaluatorVariant::evaluate_general;
    if (!is_vertical)
    {
      // horizontal face
      unsigned int n_endpoint = left_or_right_endpoint();
      apply<variant, dim, fe_degree_x + 1, 1, fe_degree_z + 1, 1, false, false, VectorizedArrayType, Number2>
            (univariate_shape_data[1]->shape_data_on_face[n_endpoint].begin(), values_dofs, temp1);
      apply<variant, dim, 1, n_q_points_1d_x, fe_degree_x + 1, 1, false, false, VectorizedArrayType, Number2>
            (univariate_shape_data[0]->shape_gradients.begin(), temp1, grads_quad);

      grads_quad += n_q_points_1d_x;

      apply<variant, dim, fe_degree_x + 1, 1, fe_degree_z + 1, 1, false, false, VectorizedArrayType, Number2>
            (univariate_shape_data[1]->shape_data_on_face[n_endpoint].begin() + fe_degree_z + 1, values_dofs, temp1);
      apply<variant, dim, 1, n_q_points_1d_x, fe_degree_x + 1, 1, false, false, VectorizedArrayType, Number2>
            (univariate_shape_data[0]->shape_values.begin(), temp1, grads_quad);
    }
    else
    {
      // vertical face
      unsigned int n_endpoint = left_or_right_endpoint();
      apply<variant, dim, fe_degree_x + 1, n_q_points_1d_z, fe_degree_z + 1, 1, false, false, VectorizedArrayType, Number2>
            (univariate_shape_data[1]->shape_values.begin(), values_dofs, temp1);
      apply<variant, dim, 1, 1, fe_degree_x + 1, n_q_points_1d_z, false, false, VectorizedArrayType, Number2>
            (univariate_shape_data[0]->shape_data_on_face[n_endpoint].begin() + fe_degree_x + 1, temp1, grads_quad);

      grads_quad += n_q_points_1d_z;

      apply<variant, dim, fe_degree_x + 1, n_q_points_1d_z, fe_degree_z + 1, 1, false, false, VectorizedArrayType, Number2>
            (univariate_shape_data[1]->shape_gradients.begin(), values_dofs, temp1);
      apply<variant, dim, 1, 1, fe_degree_x + 1, n_q_points_1d_z, false, false, VectorizedArrayType, Number2>
            (univariate_shape_data[0]->shape_data_on_face[n_endpoint].begin(), temp1, grads_quad);
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
  return this->begin_values()[q_point];
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
  const unsigned int n_q_face = is_vertical ? n_q_points_1d_z : n_q_points_1d_x;
  // specialized to the case dim=2. takes into account the fact that derivatives are reordered. see also 4255 fe_evaluation.h

  if (!is_vertical)
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
  this->begin_values()[q_point] = this->J_value[0] *
                                  this->mapping_data->descriptor[is_vertical ? 1 : 0].quadrature_weights[q_point] *
                                  val_in;
}


template <int dim, int fe_degree_x, int n_q_points_1d_x, int fe_degree_z, int n_q_points_1d_z,
          int n_components_, typename Number, typename VectorizedArrayType>
inline void
FEFaceEvaluationAniso<dim, fe_degree_x, n_q_points_1d_x, fe_degree_z, n_q_points_1d_z, n_components_,
                  Number, VectorizedArrayType>::submit_normal_derivative(const value_type grad_in, const unsigned int q_point)
{
  gradient_type inverse_jacobian_times_n;
    if (!is_vertical)
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

  unsigned int face_q_points = is_vertical ? n_q_points_1d_z : n_q_points_1d_x;
  for (unsigned int d = 0; d < dim; d++)
  {
    this->begin_gradients()[d*face_q_points + q_point] = inverse_jacobian_times_n[d] *
                            this->J_value[0] *
                            this->mapping_data->descriptor[is_vertical ? 1 : 0].quadrature_weights[q_point] *
                            grad_in;
  }
}


template <int dim, int fe_degree_x, int n_q_points_1d_x, int fe_degree_z, int n_q_points_1d_z,
          int n_components_, typename Number, typename VectorizedArrayType>
inline void
FEFaceEvaluationAniso<dim, fe_degree_x, n_q_points_1d_x, fe_degree_z, n_q_points_1d_z, n_components_,
                  Number, VectorizedArrayType>::integrate(const EvaluationFlags::EvaluationFlags evaluation_flag)
{
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

    if (!is_vertical)
    {
      // horizontal face
      // determine whether we need the values of the basis functions at the right-endpoint or at the left-endpoint
      unsigned int n_endpoint = left_or_right_endpoint();
      apply<variant, dim, n_q_points_1d_x, fe_degree_z + 1, 1, 1, false, false, VectorizedArrayType, Number2>
            (univariate_shape_data[1]->shape_data_on_face[n_endpoint].begin(), vals_quad, temp1);
      apply<variant, dim, 1, fe_degree_x + 1, n_q_points_1d_x, fe_degree_z + 1, true, false, VectorizedArrayType, Number2>
            (univariate_shape_data[0]->shape_values.begin(), temp1, val_out);
    }
    else
    {
      // vertical face
      // determine whether we need the values of the basis functions at the right-endpoint or at the left-endpoint
      unsigned int n_endpoint = left_or_right_endpoint();
      apply<variant, dim, 1, fe_degree_z + 1, n_q_points_1d_z, 1, true, false, VectorizedArrayType, Number2>
            (univariate_shape_data[1]->shape_values.begin(), vals_quad, temp1);
      apply<variant, dim, 1, fe_degree_x + 1, 1, fe_degree_z + 1, false, false, VectorizedArrayType, Number2>
            (univariate_shape_data[0]->shape_data_on_face[n_endpoint].begin(), temp1, val_out);
    }
  }


  if (evaluation_flag & EvaluationFlags::gradients)
  {
    VectorizedArrayType *val_out           = this->begin_dof_values();
    VectorizedArrayType *grads_quad        = this->begin_gradients();

    static const internal::EvaluatorVariant variant = internal::EvaluatorVariant::evaluate_general;
    if (!is_vertical)
    {
      // horizontal face
      unsigned int n_endpoint = left_or_right_endpoint();
      apply<variant, dim, n_q_points_1d_x, fe_degree_z + 1, 1, 1, false, false, VectorizedArrayType, Number2>
            (univariate_shape_data[1]->shape_data_on_face[n_endpoint].begin(), grads_quad, temp1);
      apply<variant, dim, 1, fe_degree_x + 1, n_q_points_1d_x, fe_degree_z + 1, true, true, VectorizedArrayType, Number2>
            (univariate_shape_data[0]->shape_gradients.begin(), temp1, val_out);

      grads_quad += fe_degree_x + 1;

      apply<variant, dim, n_q_points_1d_x, fe_degree_z + 1, 1, 1, false, false, VectorizedArrayType, Number2>
            (univariate_shape_data[1]->shape_data_on_face[n_endpoint].begin() + fe_degree_z + 1, grads_quad, temp1);
      apply<variant, dim, 1, fe_degree_x + 1, n_q_points_1d_x, fe_degree_z + 1, true, true, VectorizedArrayType, Number2>
            (univariate_shape_data[0]->shape_values.begin(), temp1, val_out);
    }
    else
    {
      // vertical face
      unsigned int n_endpoint = left_or_right_endpoint();
      apply<variant, dim, 1, fe_degree_z + 1, n_q_points_1d_z, 1, true, false, VectorizedArrayType, Number2>
            (univariate_shape_data[1]->shape_values.begin(), grads_quad, temp1);
      apply<variant, dim, 1, fe_degree_x + 1, 1, fe_degree_z + 1, false, true, VectorizedArrayType, Number2>
            (univariate_shape_data[0]->shape_data_on_face[n_endpoint].begin() + fe_degree_x + 1, temp1, val_out);

      grads_quad += fe_degree_z + 1;

      apply<variant, dim, 1, fe_degree_z + 1, n_q_points_1d_z, 1, true, false, VectorizedArrayType, Number2>
            (univariate_shape_data[1]->shape_gradients.begin(), grads_quad, temp1);
      apply<variant, dim, 1, fe_degree_x + 1, 1, fe_degree_z + 1, false, true, VectorizedArrayType, Number2>
            (univariate_shape_data[0]->shape_data_on_face[n_endpoint].begin(), temp1, val_out);
    }
  }
}

DEAL_II_NAMESPACE_CLOSE

#endif
