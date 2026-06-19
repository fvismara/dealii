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
#ifndef dealii_laguerre_quad_h
#define dealii_laguerre_quad_h

#include <deal.II/base/config.h>
#include <deal.II/base/quadrature.h>
#include "Laguerre.h"

/**
 * Derived class from Quadrature for quadrature rule based on Laguerre nodes
 * @tparam dim Number of dimensions
 */
template <int dim>
class QGaussLaguerre : public Quadrature<dim>
{
public:
  /**
   * Class constructor.
   * @param M
   * @param mu
   */
  QGaussLaguerre(const unsigned int M, const double mu);
};

/**
 * Derived class from Quadrature for quadrature rule based on Laguerre-Radau nodes
 * @tparam dim Number of dimensions
 */
template <int dim>
class QGaussLaguerreRadau : public Quadrature<dim>
{
public:
  /**
   * Class constructor.
   * @param M
   * @param mu
   */
  QGaussLaguerreRadau(const unsigned int M, const double mu);
  QGaussLaguerreRadau(const unsigned int M, const double mu, const double Lzmax);
};

// Class constructor
//
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

// Class constructor
//
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


template <int dim>
QGaussLaguerreRadau<dim>::QGaussLaguerreRadau(const unsigned int M, const double mu, const double Lzmax) : Quadrature<dim>(M+1) {

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
      this -> quadrature_points[i] = Point<dim>(xi)/Lzmax;
      this -> weights[i] = std::exp(mu*xi)/(mu*(M+1)*pow(lag_pol_M.value(xi),2)*Lzmax);
    }

}

#endif
