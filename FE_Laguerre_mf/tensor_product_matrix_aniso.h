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
#ifndef dealii_tensor_product_matrix_aniso_h
#define dealii_tensor_product_matrix_aniso_h

#include <deal.II/base/config.h>

#include <deal.II/base/array_view.h>
#include <deal.II/base/floating_point_comparator.h>
#include <deal.II/base/mutex.h>
#include <deal.II/base/vectorization.h>

#include <deal.II/lac/lapack_full_matrix.h>

#include <deal.II/matrix_free/tensor_product_kernels.h>

#include <bitset>

DEAL_II_NAMESPACE_OPEN

// Forward declarations
#ifndef DOXYGEN
template <typename>
class Vector;
template <typename>
class FullMatrix;
#endif

//GO: I have not commented this class, because I did not get some routines, I would like to discuss first.

template <int dim, typename Number, int n_rows_1d_x = -1, int n_rows_1d_z = -1>
class TensorProductMatrixSymmetricSumAniso
{
public:
  //using value_type = Number;

  //static constexpr int n_rows_1d_static = n_rows_1d;

  TensorProductMatrixSymmetricSumAniso() = default;

  template <typename T>
  TensorProductMatrixSymmetricSumAniso(const T &mass_matrix,
                                       const T &derivative_matrix);

  template <typename T>
  void
  reinit(const T &mass_matrix, const T &derivative_matrix);

  unsigned int
  m() const;

  unsigned int
  n() const;

  //void
  //vmult(const ArrayView<Number> &dst, const ArrayView<const Number> &src) const;

  //void
  //vmult(const ArrayView<Number>       &dst,
  //      const ArrayView<const Number> &src,
  //      AlignedVector<Number>         &tmp) const;

  void
  apply_inverse(const ArrayView<Number>       &dst,
                const ArrayView<const Number> &src) const;

  //std::size_t
  //memory_consumption() const;

protected:
  std::array<Table<2, Number>, dim> mass_matrix;

  std::array<Table<2, Number>, dim> derivative_matrix;

  std::array<AlignedVector<Number>, dim> eigenvalues;

  std::array<Table<2, Number>, dim> eigenvectors;

private:
  mutable AlignedVector<Number> tmp_array;

  mutable Threads::Mutex mutex;
};


namespace internal
{
  namespace TensorProductMatrixSymmetricSumAniso
  {

    template <typename Number, int n_rows_1d_x, int n_rows_1d_z>
    void
    kron_prod_times_vec(const Number *A, const Number *B, const Number *in, Number *out, bool transposed)
    {
      // Computes (A\kron B)*in, where A is a matrix of size n_rows_1d_x and B is a matrix of size n_rows_1d_z
      // using the identity (A\kron B)*vec(V) = vec(BVA^T).
      // If transposed == true, computes (A^T\kron B^T)*in
      constexpr int m = n_rows_1d_x;
      constexpr int n = n_rows_1d_z;

      // If aliasing, make a full copy of input
      std::vector<Number> input_copy;
      if (in == out)
      {
        input_copy.assign(in, in + m * n);
        in = input_copy.data();
      }

      // Zero output
      for (int i = 0; i < m * n; ++i)
        out[i] = Number(0);

      std::vector<Number> temp(m);

      for (int j = 0; j < n; ++j)
      {
        for (int k = 0; k < n; ++k)
        {
          Number a_jk = !transposed ? A[j*n + k] : A[k*n + j];

          if (a_jk == Number(0))
            continue;

          const Number* x_col = &in[k * m];

          // temp = B * x_col
          for (int i = 0; i < m; ++i)
          {
            temp[i] = Number(0);
            for (int r = 0; r < m; ++r)
            {
              Number B_ir = !transposed ? B[i*m + r] : B[r*m + i];
              temp[i] += B_ir * x_col[r];
            }
          }

          // accumulate
          Number* y_col = &out[j * m];
          for (int i = 0; i < m; ++i)
            y_col[i] += a_jk * temp[i];
        }
      }
    }

    /*
    template <typename Number>
    void
    spectral_assembly(const Number      *mass_matrix,
                      const Number      *derivative_matrix,
                      const unsigned int n_rows,
                      const unsigned int n_cols,
                      Number            *eigenvalues,
                      Number            *eigenvectors)
    {
      Assert(n_rows == n_cols, ExcNotImplemented());

      std::vector<bool> constrained_dofs(n_rows, false);

      for (unsigned int i = 0; i < n_rows; ++i)
        {
          if (mass_matrix[i + i * n_rows] == 0.0)
            {
              Assert(derivative_matrix[i + i * n_rows] == 0.0, ExcInternalError());

              for (unsigned int j = 0; j < n_rows; ++j)
                {
                  Assert(derivative_matrix[i + j * n_rows] == 0, ExcInternalError());
                  Assert(derivative_matrix[j + i * n_rows] == 0, ExcInternalError());
                }

              constrained_dofs[i] = true;
            }
        }

      const auto transpose_fill_nm = [&constrained_dofs](Number            *out,
                                                         const Number      *in,
                                                         const unsigned int n,
                                                         const unsigned int m) {
        for (unsigned int mm = 0, c = 0; mm < m; ++mm)
          for (unsigned int nn = 0; nn < n; ++nn, ++c)
            out[mm + nn * m] = (mm == nn && constrained_dofs[mm]) ? Number(1.0) : in[c];
      };

      std::vector<dealii::Vector<Number>> eigenvecs(n_rows);
      LAPACKFullMatrix<Number>      mass_copy(n_rows, n_cols);
      LAPACKFullMatrix<Number>      deriv_copy(n_rows, n_cols);

      transpose_fill_nm(&(mass_copy(0, 0)), mass_matrix, n_rows, n_cols);
      transpose_fill_nm(&(deriv_copy(0, 0)), derivative_matrix, n_rows, n_cols);

      deriv_copy.compute_generalized_eigenvalues_symmetric(mass_copy, eigenvecs);
      AssertDimension(eigenvecs.size(), n_rows);
      for (unsigned int i = 0, c = 0; i < n_rows; ++i)
        for (unsigned int j = 0; j < n_cols; ++j, ++c)
          if (constrained_dofs[i] == false)
            eigenvectors[c] = eigenvecs[j][i];

      for (unsigned int i = 0; i < n_rows; ++i, ++eigenvalues)
        *eigenvalues = deriv_copy.eigenvalue(i).real();
    }*/

    /*
    template <std::size_t dim, typename Number>
    inline void
    setup(const std::array<Table<2, Number>, dim> &mass_matrix,
          const std::array<Table<2, Number>, dim> &derivative_matrix,
          std::array<Table<2, Number>, dim>       &eigenvectors,
          std::array<AlignedVector<Number>, dim>  &eigenvalues)
    {
      //const unsigned int n_rows_1d = mass_matrix[0].n_cols();

      for (unsigned int dir = 0; dir < dim; ++dir)
        {
          //AssertDimension(n_rows_1d, mass_matrix[dir].n_cols());
          AssertDimension(mass_matrix[dir].n_rows(), mass_matrix[dir].n_cols());
          AssertDimension(mass_matrix[dir].n_rows(), derivative_matrix[dir].n_rows());
          AssertDimension(mass_matrix[dir].n_rows(), derivative_matrix[dir].n_cols());

          eigenvectors[dir].reinit(mass_matrix[dir].n_cols(), mass_matrix[dir].n_rows());
          eigenvalues[dir].resize(mass_matrix[dir].n_cols());
          internal::TensorProductMatrixSymmetricSumAniso::spectral_assembly<Number>(
            &(mass_matrix[dir](0, 0)),
            &(derivative_matrix[dir](0, 0)),
            mass_matrix[dir].n_rows(),
            mass_matrix[dir].n_cols(),
            eigenvalues[dir].begin(),
            &(eigenvectors[dir](0, 0)));
        }
    }*/

    template <std::size_t dim, typename Number, std::size_t n_lanes>
    inline void
    setup(
      const std::array<Table<2, VectorizedArray<Number, n_lanes>>, dim> &mass_matrix,
      const std::array<Table<2, VectorizedArray<Number, n_lanes>>, dim> &derivative_matrix,
      std::array<Table<2, VectorizedArray<Number, n_lanes>>, dim> &eigenvectors,
      std::array<AlignedVector<VectorizedArray<Number, n_lanes>>, dim> &eigenvalues)
    {
      //const unsigned int     n_rows_1d = mass_matrix[0].n_cols();
      constexpr unsigned int macro_size = VectorizedArray<Number, n_lanes>::size();

      for (unsigned int dir = 0; dir < dim; ++dir)
        {
          const unsigned int n_rows = mass_matrix[dir].n_rows();
          const unsigned int n_cols = mass_matrix[dir].n_cols();
          const unsigned int nm     = n_rows * n_cols;

          const std::size_t nm_flat_size_max = n_rows * n_rows * macro_size;
          const std::size_t n_flat_size_max  = n_rows * macro_size;

          std::vector<Number> mass_matrix_flat;
          std::vector<Number> deriv_matrix_flat;
          std::vector<Number> eigenvalues_flat;
          std::vector<Number> eigenvectors_flat;
          mass_matrix_flat.resize(nm_flat_size_max);
          deriv_matrix_flat.resize(nm_flat_size_max);
          eigenvalues_flat.resize(n_flat_size_max);
          eigenvectors_flat.resize(nm_flat_size_max);
          std::array<unsigned int, macro_size> offsets_nm;
          std::array<unsigned int, macro_size> offsets_n;
          for (unsigned int vv = 0; vv < macro_size; ++vv)
            offsets_nm[vv] = nm * vv;

          vectorized_transpose_and_store<Number, n_lanes>(false, nm, &(mass_matrix[dir](0, 0)), offsets_nm.data(), mass_matrix_flat.data());
          vectorized_transpose_and_store<Number, n_lanes>(false, nm, &(derivative_matrix[dir](0, 0)), offsets_nm.data(), deriv_matrix_flat.data());

          const Number *mass_cbegin    = mass_matrix_flat.data();
          const Number *deriv_cbegin   = deriv_matrix_flat.data();
          Number       *eigenvec_begin = eigenvectors_flat.data();
          Number       *eigenval_begin = eigenvalues_flat.data();
          for (unsigned int lane = 0; lane < macro_size; ++lane)
            internal::TensorProductMatrixSymmetricSum::spectral_assembly<
              Number>(mass_cbegin + nm * lane, deriv_cbegin + nm * lane, n_rows, n_cols, eigenval_begin + n_rows * lane,
                eigenvec_begin + nm * lane);

          eigenvalues[dir].resize(n_rows);
          eigenvectors[dir].reinit(n_rows, n_cols);
          for (unsigned int vv = 0; vv < macro_size; ++vv)
            offsets_n[vv] = n_rows * vv;
          vectorized_load_and_transpose<Number, n_lanes>(n_rows, eigenvalues_flat.data(), offsets_n.data(), eigenvalues[dir].begin());
          vectorized_load_and_transpose<Number, n_lanes>(nm, eigenvectors_flat.data(), offsets_nm.data(), &(eigenvectors[dir](0, 0)));
        }
    }

    template <std::size_t dim, typename Number>
    inline std::array<Table<2, Number>, dim>
    convert(const std::array<Table<2, Number>, dim> &mass_matrix)
    {
      return mass_matrix;
    }

    template <std::size_t dim, typename Number>
    inline std::array<Table<2, Number>, dim>
    convert(const std::array<FullMatrix<Number>, dim> &mass_matrix)
    {
      std::array<Table<2, Number>, dim> mass_copy;

      std::transform(mass_matrix.cbegin(),
                     mass_matrix.cend(),
                     mass_copy.begin(),
                     [](const FullMatrix<Number> &m) -> Table<2, Number> {
                       return m;
                     });

      return mass_copy;
    }

    template <std::size_t dim, typename Number>
    inline std::array<Table<2, Number>, dim>
    convert(const Table<2, Number> &matrix)
    {
      std::array<Table<2, Number>, dim> matrices;

      std::fill(matrices.begin(), matrices.end(), matrix);

      return matrices;
    }

    /*
    template <int n_rows_1d_templated, std::size_t dim, typename Number>
    void
    vmult(Number                                *dst,
          const Number                          *src,
          AlignedVector<Number>                 &tmp,
          const unsigned int                     n_rows_1d_non_templated,
          const std::array<const Number *, dim> &mass_matrix,
          const std::array<const Number *, dim> &derivative_matrix)
    {
      const unsigned int n_rows_1d = n_rows_1d_templated == 0 ? n_rows_1d_non_templated : n_rows_1d_templated;
      const unsigned int n         = Utilities::fixed_power<dim>(n_rows_1d);

      tmp.resize_fast(n * 2);
      Number *t = tmp.begin();

      internal::EvaluatorTensorProduct<internal::evaluate_general,
                                       dim,
                                       n_rows_1d_templated,
                                       n_rows_1d_templated,
                                       Number>
        eval({}, {}, {}, n_rows_1d, n_rows_1d);

        const Number *A0 = derivative_matrix[0];
        const Number *M0 = mass_matrix[0];
        const Number *A1 = derivative_matrix[1];
        const Number *M1 = mass_matrix[1];
        eval.template apply<0, false, false>(M0, src, t);
        eval.template apply<1, false, false>(A1, t, dst);
        eval.template apply<0, false, false>(A0, src, t);
        eval.template apply<1, false, true>(M1, t, dst);

    }*/

    /*
    template <int n_rows_1d_templated, std::size_t dim, typename Number>
    void
    apply_inverse(Number            *dst,
                  const Number      *src,
                  const unsigned int n_rows_1d_non_templated,
                  const std::array<const Number *, dim> &eigenvectors,
                  const std::array<const Number *, dim> &eigenvalues,
                  const Number *inverted_eigenvalues = nullptr)
    {
      const unsigned int n_rows_1d = n_rows_1d_templated == 0 ? n_rows_1d_non_templated : n_rows_1d_templated;

      internal::EvaluatorTensorProduct<internal::evaluate_general,
                                       dim,
                                       n_rows_1d_templated,
                                       n_rows_1d_templated,
                                       Number>
        eval({}, {}, {}, n_rows_1d, n_rows_1d);

      // NOTE: dof_to_quad has to be interpreted as 'dof to eigenvalue index'
      //       --> apply<.,true,.> (S,src,dst) calculates dst = S^T * src,
      //       --> apply<.,false,.> (S,src,dst) calculates dst = S * src,
      //       while the eigenvectors are stored column-wise in S, i.e.
      //       rows correspond to dofs whereas columns to eigenvalue indices!
          const Number *S0 = eigenvectors[0];
          const Number *S1 = eigenvectors[1];
          eval.template apply<0, true, false>(S0, src, dst);
          eval.template apply<1, true, false>(S1, dst, dst);

          for (unsigned int i1 = 0, c = 0; i1 < eigenvalues[1].size(); ++i1)
            for (unsigned int i0 = 0; i0 < eigenvalues[0].size(); ++i0, ++c)
              dst[c] /= (eigenvalues[1][i1] + eigenvalues[0][i0]);

          eval.template apply<1, false, false>(S1, dst, dst);
          eval.template apply<0, false, false>(S0, dst, dst);
    }
          */
  } // namespace TensorProductMatrixSymmetricSumAniso
} // namespace internal

template <int dim, typename Number, int n_rows_1d_x, int n_rows_1d_z>
template <typename T>
inline TensorProductMatrixSymmetricSumAniso<dim, Number, n_rows_1d_x, n_rows_1d_z>::
  TensorProductMatrixSymmetricSumAniso(const T &mass_matrix, const T &derivative_matrix)
{
  reinit(mass_matrix, derivative_matrix);
}


template <int dim, typename Number, int n_rows_1d_x, int n_rows_1d_z>
template <typename T>
inline void
TensorProductMatrixSymmetricSumAniso<dim, Number, n_rows_1d_x, n_rows_1d_z>::reinit(const T &mass_matrix, const T &derivative_matrix)
{
  this->mass_matrix = internal::TensorProductMatrixSymmetricSumAniso::convert<dim>(mass_matrix);
  this->derivative_matrix = internal::TensorProductMatrixSymmetricSumAniso::convert<dim>(derivative_matrix);

  internal::TensorProductMatrixSymmetricSumAniso::setup(this->mass_matrix,
                                                        this->derivative_matrix,
                                                        this->eigenvectors,
                                                        this->eigenvalues);
}






////////////////////////////

template <int dim, typename Number, int n_rows_1d_x, int n_rows_1d_z>
inline void
TensorProductMatrixSymmetricSumAniso<dim, Number, n_rows_1d_x, n_rows_1d_z>::apply_inverse(
  const ArrayView<Number>       &dst_view,
  const ArrayView<const Number> &src_view) const
{
  AssertDimension(dst_view.size(), this->n());
  AssertDimension(src_view.size(), this->m());

  Number       *dst = dst_view.begin();
  const Number *src = src_view.begin();

  std::array<const Number *, dim> eigenvectors, eigenvalues;

  for (unsigned int d = 0; d < dim; ++d)
    {
      eigenvectors[d] = &this->eigenvectors[d](0, 0);
      eigenvalues[d]  = this->eigenvalues[d].data();
    }

  //const unsigned int n_rows_1d_x_non_templated = this->mass_matrix[0].n_rows();
  //const unsigned int n_rows_1d_z_non_templated = this->mass_matrix[1].n_rows();

  //const unsigned int n_rows_1d = n_rows_1d_templated == 0 ? n_rows_1d_non_templated : n_rows_1d_templated;

      internal::EvaluatorTensorProduct<internal::evaluate_general, dim, n_rows_1d_x, n_rows_1d_x, Number>
        eval_x({}, {}, {}, n_rows_1d_x, n_rows_1d_z);
      internal::EvaluatorTensorProduct<internal::evaluate_general, dim, n_rows_1d_z, n_rows_1d_z, Number>
        eval_z({}, {}, {}, n_rows_1d_z, n_rows_1d_z);

          const Number *S0 = eigenvectors[0];
          const Number *S1 = eigenvectors[1];

          internal::TensorProductMatrixSymmetricSumAniso::kron_prod_times_vec<Number, n_rows_1d_x, n_rows_1d_z> (S1, S0, src, dst, true);
          //eval_x.template apply<0, true, false>(S0, src, dst);
          //eval_z.template apply<1, true, false>(S1, dst, dst);

          for (unsigned int i1 = 0, c = 0; i1 < n_rows_1d_z; ++i1)
            for (unsigned int i0 = 0; i0 < n_rows_1d_x; ++i0, ++c) {
              dst[c] /= (eigenvalues[1][i1] + eigenvalues[0][i0]);
            }

          internal::TensorProductMatrixSymmetricSumAniso::kron_prod_times_vec<Number, n_rows_1d_x, n_rows_1d_z> (S1, S0, dst, dst, false);
          //eval_z.template apply<1, false, false>(S1, dst, dst);
          //eval_x.template apply<0, false, false>(S0, dst, dst);
}

/*
namespace internal
{

  namespace TensorProductMatrixSymmetricSumAniso
  {
    template <typename Number>

    struct MatrixPairComparator
    {
      using VectorizedArrayTrait =
        ::internal::VectorizedArrayTrait<Number>;
      using ScalarNumber = typename VectorizedArrayTrait::value_type;
      static constexpr std::size_t width = VectorizedArrayTrait::width();

      using MatrixPairType =
        std::pair<std::bitset<width>,
                  std::pair<Table<2, Number>, Table<2, Number>>>;

      MatrixPairComparator()
        : eps(std::sqrt(std::numeric_limits<ScalarNumber>::epsilon()))
      {}

      bool

      operator()(const MatrixPairType &left, const MatrixPairType &right) const
      {
        const auto &M_0 = left.second.first;
        const auto &K_0 = left.second.second;
        const auto &M_1 = right.second.first;
        const auto &K_1 = right.second.second;

        std::bitset<width> mask;

        for (unsigned int v = 0; v < width; ++v)
          mask[v] = left.first[v] && right.first[v];

        const FloatingPointComparator<Number> comparator(
          eps, false, mask);

        if (comparator(M_0, M_1))
          return true;
        else if (comparator(M_1, M_0))
          return false;
        else if (comparator(K_0, K_1))
          return true;
        else
          return false;
      }

    private:
      const ScalarNumber eps;
    };

  } // namespace TensorProductMatrixSymmetricSumAniso

} // namespace internal
*/

#ifndef DOXYGEN

template <int dim, typename Number, int n_rows_1d_x, int n_rows_1d_z>
inline unsigned int
TensorProductMatrixSymmetricSumAniso<dim, Number, n_rows_1d_x, n_rows_1d_z>::m() const
{
  unsigned int m = mass_matrix[0].n_rows();
  for (unsigned int d = 1; d < dim; ++d)
    m *= mass_matrix[d].n_rows();
  return m;
}

template <int dim, typename Number, int n_rows_1d_x, int n_rows_1d_z>
inline unsigned int
TensorProductMatrixSymmetricSumAniso<dim, Number, n_rows_1d_x, n_rows_1d_z>::n() const
{
  unsigned int n = mass_matrix[0].n_cols();
  for (unsigned int d = 1; d < dim; ++d)
    n *= mass_matrix[d].n_cols();
  return n;
}

/*
template <int dim, typename Number, int n_rows_1d>
inline void
TensorProductMatrixSymmetricSumAniso<dim, Number, n_rows_1d>::vmult(
  const ArrayView<Number>       &dst_view,
  const ArrayView<const Number> &src_view) const
{
  std::lock_guard<std::mutex> lock(this->mutex);
  this->vmult(dst_view, src_view, this->tmp_array);
}

template <int dim, typename Number, int n_rows_1d>
inline void
TensorProductMatrixSymmetricSumAniso<dim, Number, n_rows_1d>::vmult(
  const ArrayView<Number>       &dst_view,
  const ArrayView<const Number> &src_view,
  AlignedVector<Number>         &tmp_array) const
{
  AssertDimension(dst_view.size(), this->m());
  AssertDimension(src_view.size(), this->n());

  Number       *dst = dst_view.begin();
  const Number *src = src_view.begin();

  std::array<const Number *, dim> mass_matrix, derivative_matrix;

  for (unsigned int d = 0; d < dim; ++d)
    {
      mass_matrix[d]       = &this->mass_matrix[d](0, 0);
      derivative_matrix[d] = &this->derivative_matrix[d](0, 0);
    }

  const unsigned int n_rows_1d_non_templated = this->mass_matrix[0].n_rows();

  if (n_rows_1d != -1)
    internal::TensorProductMatrixSymmetricSumAniso::vmult<
      n_rows_1d == -1 ? 0 : n_rows_1d>(dst,
                                       src,
                                       tmp_array,
                                       n_rows_1d_non_templated,
                                       mass_matrix,
                                       derivative_matrix);
  else
    internal::TensorProductMatrixSymmetricSumAniso::select_vmult<1>(
      dst,
      src,
      tmp_array,
      n_rows_1d_non_templated,
      mass_matrix,
      derivative_matrix);
}
      */

/*
template <int dim, typename Number, int n_rows_1d>
std::size_t
TensorProductMatrixSymmetricSumAniso<dim, Number, n_rows_1d>::memory_consumption()
  const
{
  return MemoryConsumption::memory_consumption(mass_matrix) +
         MemoryConsumption::memory_consumption(derivative_matrix) +
         MemoryConsumption::memory_consumption(eigenvalues) +
         MemoryConsumption::memory_consumption(eigenvectors) +
         MemoryConsumption::memory_consumption(tmp_array);
}
         */


#endif

DEAL_II_NAMESPACE_CLOSE

#endif
