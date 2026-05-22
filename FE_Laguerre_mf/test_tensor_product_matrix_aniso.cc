#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/function.h>
#include <deal.II/base/timer.h>

#include <deal.II/lac/affine_constraints.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/la_parallel_vector.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/lac/tensor_product_matrix.h>

#include <deal.II/fe/fe_dgq.h>
#include <deal.II/fe/fe_tools.h>
#include <deal.II/fe/mapping_q1.h>
#include <deal.II/fe/fe_enriched.h>
#include <deal.II/fe/fe_nothing.h>

#include <deal.II/grid/tria.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_tools.h>

#include <deal.II/multigrid/multigrid.h>
#include <deal.II/multigrid/mg_transfer_matrix_free.h>
#include <deal.II/multigrid/mg_tools.h>
#include <deal.II/multigrid/mg_coarse.h>
#include <deal.II/multigrid/mg_smoother.h>
#include <deal.II/multigrid/mg_matrix.h>

#include <deal.II/numerics/vector_tools.h>

#include <deal.II/matrix_free/matrix_free.h>
#include <deal.II/matrix_free/fe_evaluation.h>

#include "tensor_product_matrix_aniso.h"

#include <iostream>
#include <fstream>

using namespace dealii;

int main ()
{
    const unsigned int fe_degree_x = 2, fe_degree_z = 3, dim = 2;

    std::vector<TensorProductMatrixSymmetricSumAniso<dim, double, fe_degree_x + 1, fe_degree_z + 1>> cell_matrices;

    std::unique_ptr<FiniteElement<1>> fe_1d_x = std::make_unique<FE_DGQ<1>>(fe_degree_x);
    std::unique_ptr<FiniteElement<1>> fe_1d_z = std::make_unique<FE_DGQ<1>>(fe_degree_z);

    const unsigned int                                 N_x = fe_degree_x + 1;
    const unsigned int                                 N_z = fe_degree_z + 1;
    //std::array<Table<2, double>, dim> mass_matrices;
    //std::array<Table<2, double>, dim> laplace_matrices;
    std::array<FullMatrix<double>, dim> mass_matrices;
    std::array<FullMatrix<double>, dim> laplace_matrices;

    mass_matrices[0].reinit(N_x, N_x);
    laplace_matrices[0].reinit(N_x, N_x);
    mass_matrices[1].reinit(N_z, N_z);
    laplace_matrices[1].reinit(N_z, N_z);
    
    const QGauss<1> quadrature_x(N_x);
    for (unsigned int i = 0; i < N_x; ++i)
      for (unsigned int j = 0; j < N_x; ++j)
        {
          double sum_mass = 0, sum_laplace = 0;
          for (unsigned int q = 0; q < quadrature_x.size(); ++q)
            {
              sum_mass += (fe_1d_x->shape_value(i, quadrature_x.point(q)) *
                           fe_1d_x->shape_value(j, quadrature_x.point(q))) *
                           quadrature_x.weight(q);
              sum_laplace += (fe_1d_x->shape_grad(i, quadrature_x.point(q))[0] *
                              fe_1d_x->shape_grad(j, quadrature_x.point(q))[0]) *
                              quadrature_x.weight(q);
            }
          mass_matrices[0](i, j) = sum_mass;
            
          sum_laplace +=
            (1. * fe_1d_x->shape_value(i, Point<1>()) *
               fe_1d_x->shape_value(j, Point<1>()) +
             0.5 * fe_1d_x->shape_grad(i, Point<1>())[0] *
               fe_1d_x->shape_value(j, Point<1>()) +
             0.5 * fe_1d_x->shape_grad(j, Point<1>())[0] *
               fe_1d_x->shape_value(i, Point<1>()));

          sum_laplace +=
            (1. * fe_1d_x->shape_value(i, Point<1>(1.0)) *
               fe_1d_x->shape_value(j, Point<1>(1.0)) -
             0.5 * fe_1d_x->shape_grad(i, Point<1>(1.0))[0] *
               fe_1d_x->shape_value(j, Point<1>(1.0)) -
             0.5 * fe_1d_x->shape_grad(j, Point<1>(1.0))[0] *
               fe_1d_x->shape_value(i, Point<1>(1.0)));

          laplace_matrices[0](i, j) = sum_laplace;
        }

    const QGauss<1> quadrature_z(N_z);
    for (unsigned int i = 0; i < N_z; ++i)
      for (unsigned int j = 0; j < N_z; ++j)
        {
          double sum_mass = 0, sum_laplace = 0;
          for (unsigned int q = 0; q < quadrature_z.size(); ++q)
            {
              sum_mass += (fe_1d_z->shape_value(i, quadrature_z.point(q)) *
                           fe_1d_z->shape_value(j, quadrature_z.point(q))) *
                           quadrature_z.weight(q);
              sum_laplace += (fe_1d_z->shape_grad(i, quadrature_z.point(q))[0] *
                              fe_1d_z->shape_grad(j, quadrature_z.point(q))[0]) *
                              quadrature_z.weight(q);
            }
          mass_matrices[1](i, j) = sum_mass;
            
          sum_laplace +=
            (1. * fe_1d_z->shape_value(i, Point<1>()) *
               fe_1d_z->shape_value(j, Point<1>()) +
             0.5 * fe_1d_z->shape_grad(i, Point<1>())[0] *
               fe_1d_z->shape_value(j, Point<1>()) +
             0.5 * fe_1d_z->shape_grad(j, Point<1>())[0] *
               fe_1d_z->shape_value(i, Point<1>()));

          sum_laplace +=
            (1. * fe_1d_z->shape_value(i, Point<1>(1.0)) *
               fe_1d_z->shape_value(j, Point<1>(1.0)) -
             0.5 * fe_1d_z->shape_grad(i, Point<1>(1.0))[0] *
               fe_1d_z->shape_value(j, Point<1>(1.0)) -
             0.5 * fe_1d_z->shape_grad(j, Point<1>(1.0))[0] *
               fe_1d_z->shape_value(i, Point<1>(1.0)));

          laplace_matrices[1](i, j) = sum_laplace;
        }

    cell_matrices.resize(1);
    cell_matrices[0].reinit(mass_matrices, laplace_matrices);

    mass_matrices[0].print(std::cout, 20, 6);
    std::cout << std::endl << std::endl << std::endl;
    mass_matrices[1].print(std::cout, 20, 6);
    std::cout << std::endl << std::endl << std::endl;
    laplace_matrices[0].print(std::cout, 20, 6);
    std::cout << std::endl << std::endl << std::endl;
    laplace_matrices[1].print(std::cout, 20, 6);
    std::cout << std::endl << std::endl << std::endl;

    Vector<double> src((fe_degree_x+1)*(fe_degree_z+1)); 
    for (unsigned int i = 0; i < src.size(); i++)
      src[i] = i;
    
    Vector<double> dst((fe_degree_x+1)*(fe_degree_z+1)); 
    cell_matrices[0].apply_inverse(dst, src);

    for (unsigned int i = 0; i < dst.size(); i++)
      std::cout << src[i] << "          " << dst[i] << std::endl;

    return 0;
}