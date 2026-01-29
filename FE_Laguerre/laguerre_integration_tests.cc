#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/function.h>
#include <deal.II/lac/vector.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/grid/tria.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_out.h>
#include <deal.II/grid/grid_refinement.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/fe/mapping_q1.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/numerics/vector_tools.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/numerics/data_out.h>
#include <deal.II/fe/mapping_q1.h>
#include <deal.II/fe/fe_dgq.h>
#include <deal.II/fe/fe_enriched.h>
#include <deal.II/fe/fe_nothing.h>
#include <deal.II/fe/fe_interface_values.h>
#include <deal.II/lac/solver_gmres.h>
#include <deal.II/lac/precondition_block.h>
#include <deal.II/numerics/derivative_approximation.h>
 
#include <deal.II/meshworker/mesh_loop.h>

#include "fe_dgq_aniso.h"
#include "Laguerre.h"
#include "laguerre_quad.h"
 
#include <iostream>
#include <fstream>

double f(const Point<1> z) {
    return std::pow(z[0],3);
}

template <int dim>
class ScaledExponential: public Function<dim> {
public:
  ScaledExponential(const double mm) {this -> mu = mm;};
  double value(const Point<dim>& p, const unsigned) const override {
    return std::exp(-mu*p[1]/2);
  }
  Tensor<1,2,double> gradient(const Point<dim>& p, const unsigned) const override {
    Tensor<1,2,double> grad;
    grad[0] = 0;
    grad[1] = -(mu/2)*std::exp(-mu*p[1]/2);
    return grad;
  }
private:
  double mu;
};

int main() {

    int M = 5, px = 1;
    double mu = 1.357;
    Tensor<1,2> beta({3,1});

    double Lmax = Laguerre(M+1,mu).get_zeros_of_derivative()[M-1];

    const Point<2> & p1 = {0, 0};
    const Point<2> & p2 = {1, Lmax};
    Triangulation<2>   triangulation;
    GridGenerator::hyper_rectangle(triangulation, p1, p2);
    DoFHandler<2> dof_handler(triangulation);

    const Quadrature<2> quadrature(QGauss<1>(px+1), 
                                   QGaussLaguerreRadau<1>(M,mu*Lmax));

    /*double res = 0;
    for (unsigned int ii = 0; ii < quadrature.size(); ii++)
        res += f(quadrature.point(ii)) * quadrature.weight(ii);

    std::cout << res << std::endl;*/
    ScaledExponential<2> enrichment_function(mu);
    FE_Enriched<2> fe(FE_Nothing<2>(), FE_DGQ_Aniso<2>(px, M, 
        AnisotropicPolynomials<2>({Polynomials::Legendre::generate_complete_basis(px),
        Laguerre::generate_complete_basis_scaled(M, mu)})), &enrichment_function);

    FEValues<2> fe_values(fe, quadrature, update_values | update_gradients | update_quadrature_points | update_JxW_values);
    
    for (const auto &cell : dof_handler.active_cell_iterators())
        fe_values.reinit(cell);

    FullMatrix<double> phi_i_phi_j((M+1)*(px+1), (M+1)*(px+1));
    FullMatrix<double> beta_grad_phi_i_phi_j((M+1)*(px+1), (M+1)*(px+1));
    FullMatrix<double> grad_phi_i_grad_phi_j((M+1)*(px+1), (M+1)*(px+1));

    for (unsigned int jj = 0; jj < (M+1)*(px+1); jj++) {
        for (unsigned int kk = 0; kk < (M+1)*(px+1); kk++) {
            for (unsigned int ii = 0; ii < quadrature.size(); ii++) {
                phi_i_phi_j(jj,kk) += fe_values.shape_value(jj, ii) 
                    * fe_values.shape_value(kk, ii) 
                    * fe_values.JxW(ii);
                beta_grad_phi_i_phi_j(jj,kk) += beta*fe_values.shape_grad(jj, ii) 
                    * fe_values.shape_value(kk, ii) 
                    * fe_values.JxW(ii);
                grad_phi_i_grad_phi_j(jj,kk) += fe_values.shape_grad(jj, ii) 
                    * fe_values.shape_grad(kk, ii) 
                    * fe_values.JxW(ii);
            }
        }
    }

    for (unsigned int jj = 0; jj < (M+1)*(px+1); jj++) 
        for (unsigned int kk = 0; kk < (M+1)*(px+1); kk++) {
            if (std::abs(phi_i_phi_j[jj][kk]) < 1e-10)
                phi_i_phi_j[jj][kk] = 0;
            if (std::abs(beta_grad_phi_i_phi_j[jj][kk]) < 1e-10)
                beta_grad_phi_i_phi_j[jj][kk] = 0;
            if (std::abs(grad_phi_i_grad_phi_j[jj][kk]) < 1e-10)
                grad_phi_i_grad_phi_j[jj][kk] = 0;
        }
    phi_i_phi_j.print(std::cout,10,4);
    std::cout << std::endl;
    beta_grad_phi_i_phi_j.print(std::cout,10,4);
    std::cout << std::endl;
    grad_phi_i_grad_phi_j.print(std::cout,10,4);

    return 0;
}