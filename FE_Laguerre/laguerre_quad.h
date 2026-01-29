#ifndef dealii_laguerre_quad_h
#define dealii_laguerre_quad_h

#include <deal.II/base/config.h>
#include <deal.II/base/quadrature.h>
#include "Laguerre.h"

template <int dim>
class QGaussLaguerre : public Quadrature<dim>
{
public:
  QGaussLaguerre(const unsigned int M, const double mu);
};

template <int dim>
class QGaussLaguerreRadau : public Quadrature<dim>
{
public:
  QGaussLaguerreRadau(const unsigned int M, const double mu);
};

template <int dim>
QGaussLaguerre<dim>::QGaussLaguerre(const unsigned int M, const double mu) : Quadrature<dim>(M+1) {

    if (M == 0)
      return;
    
    Laguerre lag_pol_M_plus_1(M+1, mu);
    Laguerre lag_pol_M(M, mu);

    std::vector<double> points = lag_pol_M_plus_1.get_zeros();

    for (unsigned int i = 0; i < M+1; i++) 
    {
      double xi = points[i];
      this -> quadrature_points[i] = Point<dim>(xi);
      this -> weights[i] = xi/(pow((M+1)*lag_pol_M.value(xi),2));
    }

}


template <int dim>
QGaussLaguerreRadau<dim>::QGaussLaguerreRadau(const unsigned int M, const double mu) : Quadrature<dim>(M+1) {

    if (M == 0)
      return;
    
    Laguerre lag_pol_M_plus_1(M+1, mu);
    Laguerre lag_pol_M(M, mu);

    std::vector<double> points = lag_pol_M_plus_1.get_zeros_of_derivative();

    this -> quadrature_points[0] = Point<dim>(0);
    this -> weights[0] = 1/(mu*(M+1));
    for (unsigned int i = 1; i < M+1; i++) 
    {
      double xi = points[i-1];
      this -> quadrature_points[i] = Point<dim>(xi);
      this -> weights[i] = std::exp(mu*xi)/(mu*(M+1)*pow(lag_pol_M.value(xi),2));
    }

}

#endif

