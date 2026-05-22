#include <deal.II/matrix_free/fe_evaluation.h>

#include <iostream>
#include <fstream>
#include <typeinfo>
#include <cstdlib>

using namespace dealii;

template <internal::EvaluatorVariant variant, int dim, int p, int n_rows, int n_columns, int r, typename Number, typename Number2>
inline void apply(const Number2 *DEAL_II_RESTRICT shape_data, const Number *in, Number *out)
{
  // implements the product (I \kron B \kron I)*v, where the two identity matrices have dimension pxp and rxr, respectively,
  // and B has dimension n_rows x n_columns. shape_data contains the elements of B in row-major format. it is not necessary for p,
  // n_rows, n_columns and r to be equal
  for (int i2 = 0; i2 < p; ++i2)
    {
      for (int i1 = 0; i1 < r; ++i1) {
        internal::apply_matrix_vector_product<variant, internal::EvaluatorQuantity::value, 
                                              n_rows, n_columns, r, r, false, false>(shape_data, in, out);
 
        in += 1;
        out += 1;
      }
      in += (n_columns-1)*r;
      out += (n_rows-1)*r;
    }
}

FullMatrix<double> kron(FullMatrix<double> A, FullMatrix<double> B)
{
  // computes full Kronecker product for debugging
  FullMatrix<double> C(A.m()*B.m(), A.n()*B.n());

  for (unsigned int i = 0; i < A.m(); i++)
    for (unsigned int k = 0; k < B.m(); k++)
      for (unsigned int j = 0; j < A.n(); j++)
        for (unsigned int l = 0; l < B.n(); l++)
          C[B.m()*i+k][B.n()*j+l] = A[i][j] * B[k][l];

  return C;
}


int main() {

  // we want to compute the product (A1 \kron A2 \kron ... \kron Ad)*v, where Ai has size mi x ni for i=1,...,d and v has
  // compatible size. we use the property (A1 \kron Im2 \kron ... \kron Imd)*(In1 \kron A2 \kron ... \kron Imd)*
  // (In1 \kron In2 \kron ... \kron Ad)*v, where Ij is the j-identity matrix, for a generic integer j. thanks to this, 
  // it is sufficient to compute products of the form (I \kron B \kron I)*v, for some B and v of suitable size, which is 
  // done by 'apply' above. this is a test on random matrices A1,A2,A3 and random vector v in the case d=3. in principle the 
  // code can be applied to any dimension.

  constexpr unsigned int n1 = 5, n2 = 3, n3 = 7;
  constexpr unsigned int m1 = 9, m2 = 4, m3 = 2;

  Vector<double> A1(n1*m1), A2(n2*m2), A3(n3*m3);
  Vector<double> v(n1*n2*n3), res(m1*m2*m3);
  Vector<double> temp1(n1*n2*m3);
  Vector<double> temp2(n1*m2*m3);

  // fill A1, A2, A3 with random numbers
  for (unsigned int ii = 0; ii < n1*m1; ii++)
    A1[ii] = ((double) rand() / RAND_MAX);

  for (unsigned int ii = 0; ii < n2*m2; ii++)
    A2[ii] = ((double) rand() / RAND_MAX);

  for (unsigned int ii = 0; ii < n3*m3; ii++)
    A3[ii] = ((double) rand() / RAND_MAX);

  for (unsigned int ii = 0; ii < n1*n2*n3; ii++)
    v[ii] = ((double) rand() / RAND_MAX);

  double *in = v.begin(), *out = res.begin(); 
  double *temp1_p = temp1.begin(), *temp2_p = temp2.begin();
  double *shape_data_1 = A1.begin(), *shape_data_2 = A2.begin(), *shape_data_3 = A3.begin();

  static const internal::EvaluatorVariant variant = internal::EvaluatorVariant::evaluate_general;
  
  // in 3d we have (A1 \kron A2 \kron A3)*v = (A1 \kron I(m2*m3))*(In1 \kron A2 \kron Im3)*(I(n1*n2) \kron A3)*v
  // we compute the three matrix-vector products from right to left
  apply<variant, 3, n1*n2, m3, n3, 1, double, double>(shape_data_3, in, temp1_p);
  apply<variant, 3, n1, m2, n2, m3, double, double>(shape_data_2, temp1_p, temp2_p);
  apply<variant, 3, 1, m1, n1, m2*m3, double, double>(shape_data_1, temp2_p, out);

  // we know compute the full Kronecker product to use as a reference. first construct the full matrices A1,A2,A3
  FullMatrix<double> A1_mat(m1,n1,shape_data_1), A2_mat(m2,n2,shape_data_2), A3_mat(m3,n3,shape_data_3);

  // then compute the reference by first computing the Kronecker product and then multiplying by v
  Vector<double> ex_prod(m1*m2*m3);
  kron(A1_mat, kron(A2_mat, A3_mat)).vmult(ex_prod, v);

  // compute norm of the difference
  double l2_norm = 0;
  for (unsigned int i = 0; i < m1*m2*m3; i++)
    l2_norm += std::pow(ex_prod[i] - out[i], 2);

  l2_norm = std::sqrt(l2_norm);

  std::cout << l2_norm << std::endl;

  return 0;
}