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
#ifndef dealii_fe_Laguerre_h
#define dealii_fe_Laguerre_h

#include <deal.II/base/config.h>

#include <deal.II/base/mutex.h>
#include <deal.II/base/tensor_product_polynomials.h>

#include <deal.II/fe/fe_poly.h>
#include <deal.II/fe/fe_dgq.h>
#include "Laguerre.h"
#include "laguerre_quad.h"

DEAL_II_NAMESPACE_OPEN

/**
 * Derived class from FE_DGQ for discontinuous finite elements with Laguerre polynomials
 * @tparam dim Number of dimensions
 * @tparam spacedim Number of spatial dimensions
 */
//GO: Derive from FiniteElement? Since FE_DGQ uses Lagrange polynomials...
template <int dim, int spacedim = dim>
class FE_Laguerre : public FE_DGQ<dim, spacedim>
{
public:
  /**
   * Class constructor.
   * @param M
   * @param mu
   */
  FE_Laguerre(const unsigned int M, const double mu);
};

// Class constructor
//
template <int dim, int spacedim>
FE_Laguerre<dim, spacedim>::FE_Laguerre(const unsigned int M, const double mu)
  : FE_DGQ<dim, spacedim>(Laguerre::generate_complete_basis_scaled(M, mu))
{}


DEAL_II_NAMESPACE_CLOSE

#endif
