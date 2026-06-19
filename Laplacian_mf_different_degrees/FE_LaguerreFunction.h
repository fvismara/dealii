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

#ifndef dealii_fe_LaguerreFunction_h
#define dealii_fe_LaguerreFunction_h

#include <deal.II/base/config.h>

#include <deal.II/base/mutex.h>
#include <deal.II/base/tensor_product_polynomials.h>
#include <deal.II/fe/fe_enriched.h>

#include <deal.II/fe/fe_poly.h>
#include <deal.II/fe/fe_dgq.h>
#include "Laguerre.h"
#include "laguerre_quad.h"

DEAL_II_NAMESPACE_OPEN

namespace internal {
// Enrichment function, exp(-mu*z/2)
template <int dim>
class ScaledExponential: public Function<dim> {
public:
  ScaledExponential(const double mm, const double _Lz) {this -> mu = mm; this -> Lz = _Lz;};
  double value(const Point<dim>& p, const unsigned) const override {
    return std::exp(-mu*(p[dim-1]-Lz)/2)*(p[dim-1] >= Lz);
  }
  Tensor<1,dim,double> gradient(const Point<dim>& p, const unsigned) const override {
    Tensor<1,dim,double> grad;
    //grad[0] = 0;
    grad[dim-1] = -(mu/2)*std::exp(-mu*(p[dim-1]-Lz)/2)*(p[dim-1] >= Lz);
    return grad;
  }
private:
  double mu, Lz;
};
}

template <int dim, int spacedim = dim>
class FE_LaguerreFunction : public FE_Enriched<dim, spacedim>
{
public:
  /**
   * Class constructor.
   * @param M
   * @param mu
   */
  FE_LaguerreFunction(const unsigned int M, const double mu, const double Lz);

  double shape_value(const unsigned int i, const Point<dim> &p) const override;
  Tensor<1,dim> shape_grad(const unsigned int i, const Point<dim> &p) const override;
  Tensor<2,dim> shape_grad_grad(const unsigned int i, const Point<dim> &p) const override;

  unsigned int M;
  double mu, Lz;

private:
  internal::ScaledExponential<dim> scaling_function;
};

// Class constructor
//
template <int dim, int spacedim>
FE_LaguerreFunction<dim, spacedim>::FE_LaguerreFunction(const unsigned int _M, const double _mu, const double _Lz)
  : FE_Enriched<dim, spacedim>(FE_Nothing<dim>(), FE_Laguerre<dim>(_M, _mu), &scaling_function)
  , M(_M), mu(_mu), Lz(_Lz)
  , scaling_function(_mu, 0)
{}

template <int dim, int spacedim>
double FE_LaguerreFunction<dim, spacedim>::shape_value(const unsigned int i, const Point<dim> &p) const 
{
  return std::exp(-mu*p[dim-1]/2) * FE_Laguerre<1>(M,mu).shape_value(i,p);
}

template <int dim, int spacedim>
Tensor<1,dim> FE_LaguerreFunction<dim, spacedim>::shape_grad(const unsigned int i, const Point<dim> &p) const 
{
  Tensor<1,dim> grad;
  grad[dim-1] =  std::exp(-mu*p[dim-1]/2) * (FE_Laguerre<1>(M,mu).shape_grad(i,p)[dim-1] - mu*FE_Laguerre<1>(M,mu).shape_value(i,p)/2);
  return grad;
}

template <int dim, int spacedim>
Tensor<2,dim> FE_LaguerreFunction<dim, spacedim>::shape_grad_grad(const unsigned int i, const Point<dim> &p) const 
{
  Tensor<2,dim> grad_grad;
  grad_grad[dim-1][dim-1] =  std::exp(-mu*p[dim-1]/2) * ((mu*mu/4)*FE_Laguerre<1>(M,mu).shape_value(i,p)
                                                  -mu*FE_Laguerre<1>(M,mu).shape_grad(i,p)[dim-1]
                                                  +FE_Laguerre<1>(M,mu).shape_grad_grad(i,p)[dim-1][dim-1]);
  return grad_grad;
}


DEAL_II_NAMESPACE_CLOSE

#endif
