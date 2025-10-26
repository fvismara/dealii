/* ------------------------------------------------------------------------
 *
 * SPDX-License-Identifier: LGPL-2.1-or-later
 * Copyright (C) 1999 - 2024 by the deal.II authors
 *
 * This file is part of the deal.II library.
 *
 * Part of the source code is dual licensed under Apache-2.0 WITH
 * LLVM-exception OR LGPL-2.1-or-later. Detailed license information
 * governing the source code and code contributions can be found in
 * LICENSE.md and CONTRIBUTING.md at the top level directory of deal.II.
 *
 * ------------------------------------------------------------------------
 */
/*--- Author: Giuseppe Orlando, 2025 ---*/

#pragma once

#include <deal.II/fe/fe_poly.h>
#include <deal.II/fe/fe_data.h>

using namespace dealii;

// Derived class from FE_Poly to represent anisotropic DG finite elements
//
template<int dim>
class FE_DGQ_Aniso: public FE_Poly<dim> {
public:
  static_assert(dim <= 3, "FE_DGQ_Aniso is available only for dim = 1, 2 or 3");

  FE_DGQ_Aniso(const unsigned degree_x,
               const unsigned degree_z,
               const AnisotropicPolynomials<dim>& polynomials); /*--- Class constructor ---*/

  std::string get_name() const override; /*--- Get name of this finite element space ---*/

  std::unique_ptr<FiniteElement<dim>> clone() const override; /*--- Clone function ---*/

private:
  std::vector<unsigned> build_dpo(const unsigned degree_x,
                                  const unsigned degree_z); /*--- Build dof per object ---*/
};

// Class constructor
//
template<int dim>
FE_DGQ_Aniso<dim>::FE_DGQ_Aniso(const unsigned degree_x,
                                const unsigned degree_z,
                                const AnisotropicPolynomials<dim>& polynomials) :
  FE_Poly<dim>(polynomials,
               FiniteElementData<dim>(this->build_dpo(degree_x, degree_z),
                                      ReferenceCells::Quadrilateral,
                                      1,
                                      std::max(degree_x, degree_z),
                                      FiniteElementData<dim>::L2),
                std::vector<bool>(this->build_dpo(degree_x, degree_z).back(), true),
                std::vector<ComponentMask>(this->build_dpo(degree_x, degree_z).back(),
                                           ComponentMask(std::vector<bool>(1, true))))
  {
    /*--- Numeber of nodes along each coordinate direction ---*/
    const unsigned nx = degree_x + 1;
    const unsigned nz = degree_z + 1;

    /*--- Each shape function will have a corresponding support point ---*/
    this->unit_support_points.clear();
    this->unit_support_points.reserve(nx*nz);

    /*--- Cartesian product [0,1]^2 coherent with LagrangeEquidistant ---*/
    for(unsigned j = 0; j < nz; ++j) {
      const double z = (nz == 1 ? 0.5 : static_cast<double>(j)/degree_z);
      for(unsigned i = 0; i < nx; ++i) {
        const double x = (nx == 1 ? 0.5 : static_cast<double>(i)/degree_x);
        this->unit_support_points.emplace_back(x, z);
      }
    }
  }

// Auxiliary function necessary to initalize the dofs per object
//
template<int dim>
std::vector<unsigned>
FE_DGQ_Aniso<dim>::build_dpo(const unsigned degree_x,
                             const unsigned degree_z) {
  std::vector<unsigned> dpo(dim + 1, 0);

  if(dim == 1) {
    dpo[dim] = degree_x + 1;
  }
  else if(dim == 2) {
    dpo[dim] = (degree_x + 1)*(degree_z + 1);
  }
  else if(dim == 3) {
    dpo[dim] = (degree_x + 1)*(degree_x + 1)*(degree_z + 1);
  }
  
  return dpo;
}

// Set name of the function (pure virtual in FiniteElement)
//
template<int dim>
std::string FE_DGQ_Aniso<dim>::get_name() const {
  return "FE_DGQ_Aniso";
}

// Clone function (pure virtual in FiniteElement)
//
template<int dim>
std::unique_ptr<FiniteElement<dim>>
FE_DGQ_Aniso<dim>::clone() const {
  return std::make_unique<FE_DGQ_Aniso<dim>>(*this);
}
