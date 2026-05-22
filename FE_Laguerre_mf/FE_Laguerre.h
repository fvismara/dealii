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

template <int dim, int spacedim = dim>
class FE_Laguerre : public FE_DGQ<dim, spacedim>
{
public:
  FE_Laguerre(const unsigned int M, const double mu);
};

template <int dim, int spacedim>
FE_Laguerre<dim, spacedim>::FE_Laguerre(const unsigned int M, const double mu)
  : FE_DGQ<dim, spacedim>(Laguerre::generate_complete_basis_scaled(M, mu))
{}


DEAL_II_NAMESPACE_CLOSE

#endif