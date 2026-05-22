#ifndef dealii_Laguerre_h
#define dealii_Laguerre_h

#include <deal.II/base/polynomial.h>
#include <deal.II/lac/tridiagonal_matrix.h>

using namespace dealii;

class Laguerre : public Polynomials::Polynomial<double>
  {
  public:
    Laguerre(const unsigned int M, const double mu);
    static std::vector<Polynomial<double>> generate_complete_basis(const unsigned int degree, const double mu);
    static std::vector<Polynomial<double>> generate_complete_basis_scaled(const unsigned int degree, const double mu);
    std::vector<double> get_zeros();
    std::vector<double> get_zeros_of_derivative();
  private:
    unsigned int M;
    double mu;
  };

Laguerre::Laguerre(const unsigned int M, const double mu)
    : Polynomial<double>(0)
  {
    this->coefficients.clear();
    this->in_lagrange_product_form = true;
    this->lagrange_support_points.resize(M);

    this -> M = M;
    this -> mu = mu;
 
    // the roots of a scaled Laguerre polynomial are the eigenvalues of a symmetric tridiagonal matrix
    TridiagonalMatrix<double> T(M, true);
    for (unsigned int i = 0; i < M; i++) 
    {
      T(i, i) = (2*i+1)/mu;
      if (i > 0)
        T(i, i-1) = i/mu;
      if (i < M-1)
        T(i, i+1) = (i+1)/mu;
    }
    T.compute_eigenvalues();

    for (unsigned int i = 0; i < M; i++)
      this -> lagrange_support_points[i] = T.eigenvalue(i);
 
    // compute the coefficient c such that L(x)=c*(x-x1)*...*(x-xM). Since L(0)=1, we have c = (-1)^M/(x1*...*xM)
    double prod = 1.;
    for (unsigned int i = 0; i < M; ++i)
      prod *= this->lagrange_support_points[i];
    
    this->lagrange_weight = std::pow(-1, M) / prod;

  }
 
  std::vector<Polynomials::Polynomial<double>>
  Laguerre::generate_complete_basis(const unsigned int degree, const double mu)
  {
    std::vector<Polynomials::Polynomial<double>> v;
    v.reserve(degree + 1);
    for (unsigned int i = 0; i <= degree; ++i)
      v.push_back(Laguerre(i, mu));

    return v;
  }

  std::vector<Polynomials::Polynomial<double>>
  Laguerre::generate_complete_basis_scaled(const unsigned int degree, const double mu)
  {
    std::vector<Polynomials::Polynomial<double>> v;
    double last_quad_point = Laguerre(degree+1,mu).get_zeros_of_derivative()[degree-1];
    v.reserve(degree + 1);
    for (unsigned int i = 0; i <= degree; ++i)
      v.push_back(Laguerre(i, mu*last_quad_point));

    return v;
  }

  std::vector<double> 
  Laguerre::get_zeros()
  {
    std::vector<double> pts(M);
    for (unsigned int i = 0; i < M; ++i)
      pts[i] = this -> lagrange_support_points[i];
    
    return pts;
  }

  std::vector<double> 
  Laguerre::get_zeros_of_derivative()
  {
    std::vector<double> pts(M-1);

    TridiagonalMatrix<double> T(M-1, true);
    for (unsigned int i = 0; i < M-1; i++) 
    {
      T(i, i) = (2*(i+1))/mu;
      if (i > 0)
        T(i, i-1) = sqrt((i+1)*i)/mu;
      if (i < M-2)
        T(i, i+1) = sqrt((i+1)*(i+2))/mu;
    }
    T.compute_eigenvalues();

    for (unsigned int i = 0; i < M-1; ++i)
      pts[i] = T.eigenvalue(i);
    
    return pts;
  }


#endif