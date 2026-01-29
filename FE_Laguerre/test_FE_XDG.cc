#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/function.h>
#include <deal.II/lac/vector.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/sparse_matrix.h>
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
#include <deal.II/base/convergence_table.h>

#include <deal.II/hp/fe_collection.h>
#include <deal.II/hp/fe_values.h>
#include <deal.II/hp/refinement.h>
 
#include <deal.II/meshworker/mesh_loop.h>

#include "fe_dgq_aniso.h"
#include "Laguerre.h"
#include "laguerre_quad.h"
 
#include <iostream>
#include <fstream>

// Enrichment function, exp(-mu*z/2)
template <int dim>
class ScaledExponential: public Function<dim> {
public:
  ScaledExponential(const double mm, const double _Lz) {this -> mu = mm; this -> Lz = _Lz;};
  double value(const Point<dim>& p, const unsigned) const override {
    return std::exp(-mu*(p[1]-Lz)/2)*(p[1] >= Lz);
  }
  Tensor<1,2,double> gradient(const Point<dim>& p, const unsigned) const override {
    Tensor<1,2,double> grad;
    grad[0] = 0;
    grad[1] = -(mu/2)*std::exp(-mu*(p[1]-Lz)/2)*(p[1] >= Lz);
    return grad;
  }
private:
  double mu, Lz;
};


int main () 
{
    const unsigned int dim = 2;
    unsigned int px = 1;
    unsigned int pz = 2;//1;
    unsigned int M = 4;//3;
    unsigned int Mquad = M;
    double mu = 1.437;
    double Lx = 1.317;//1;
    double Lz = 0.498;//2;
    unsigned int Nx = 2;
    unsigned int Nz = 2;//1;
    Tensor<1,dim> beta({3,1});

    ScaledExponential<dim> enrichment_function(mu, Lz);

    Triangulation<dim>   triangulation;
    const MappingQ1<dim> mapping;

    // finite element collection
    const FE_Enriched<dim> fe_laguerre(FE_Nothing<dim>(), FE_DGQ_Aniso<dim>(px, M, AnisotropicPolynomials<dim>({Polynomials::Legendre::generate_complete_basis(px),
         Laguerre::generate_complete_basis_scaled(M, mu)})), &enrichment_function);
    //const FE_DGQLegendre<dim> fe_legendre(px);
    const FE_DGQ_Aniso<dim> fe_legendre(px, pz, AnisotropicPolynomials<dim>({Polynomials::Legendre::generate_complete_basis(px),
         Polynomials::Legendre::generate_complete_basis(pz)}));

    hp::FECollection<dim> fe_collection;
    fe_collection.push_back(fe_legendre);
    fe_collection.push_back(fe_laguerre);

    // generate grid
    double Lmax = Laguerre(Mquad+1,mu).get_zeros_of_derivative()[Mquad-1];
    const Point<dim> & p1 = {0, 0};
    const Point<dim> & p2 = {Lx, Lz+Lmax};
    double dx = Lx/Nx;
    double dz = Lz/Nz;

    std::vector<std::vector<double>> step_sizes;

    std::vector<double> step_sizes_x(Nx);
    std::vector<double> step_sizes_z(Nz+1);
    for (unsigned int i = 0; i < Nx; i++)
        step_sizes_x[i] = dx;
    for (unsigned int j = 0; j < Nz; j++)
        step_sizes_z[j] = dz;
    step_sizes_z[Nz] = Lmax;

    step_sizes.push_back(step_sizes_x);
    step_sizes.push_back(step_sizes_z);

    GridGenerator::subdivided_hyper_rectangle(triangulation, step_sizes, p1, p2);

    /*std::ofstream out("grid-1.svg");
    GridOut grid_out;
    grid_out.write_svg(triangulation, out);*/

    // assign different finite elements to different cells
    DoFHandler<dim> dof_handler(triangulation);
    for (auto &cell: dof_handler.active_cell_iterators()) {
        if ((cell -> center())[1] > Lz)
            cell -> set_active_fe_index(1);
        for (auto &face : cell->face_iterators()) {
          if (face -> vertex(0)[1] == face -> vertex(1)[1])
            face -> set_manifold_id(0); // horizontal face
          if (face -> vertex(0)[0] == face -> vertex(1)[0]) {
            if (face -> center()[1] < Lz)
              face -> set_manifold_id(1); // vertical face in the bounded region
            else 
              face -> set_manifold_id(2); // vertical face in the unbounded region
          }
        }
      }
    dof_handler.distribute_dofs(fe_collection);

    // quadrature collection
    hp::QCollection<dim> quadrature_collection;
    hp::QCollection<dim-1> face_quadrature_collection;
    quadrature_collection.push_back(QGauss<dim>(px+1));
    quadrature_collection.push_back(Quadrature<dim>(QGauss<dim-1>(px+1), QGaussLaguerreRadau<1>(Mquad,mu*Lmax)));
    face_quadrature_collection.push_back(QGauss<dim-1>(px+1));
    face_quadrature_collection.push_back(QGauss<dim-1>(pz+1));
    face_quadrature_collection.push_back(QGaussLaguerreRadau<1>(Mquad,mu*Lmax));

    // construct fe space
    hp::FEValues<dim> hp_fe_values(fe_collection, quadrature_collection, 
                                    update_values | update_gradients | update_quadrature_points | update_JxW_values);
    hp::FEFaceValues<dim> hp_fe_interface_values(fe_collection, face_quadrature_collection,
                                    update_values | update_gradients | update_quadrature_points | update_JxW_values | update_normal_vectors);

    // print number of dofs on each cell
    /*for (const auto &cell : dof_handler.active_cell_iterators())
      std::cout << cell->get_fe().n_dofs_per_cell() << std::endl;*/

    // check local matrices (in the laplacian case, i.e. grad\dot grad)
    FullMatrix<double> cell_matrix;
  
    std::vector<types::global_dof_index> local_dof_indices;
  
    for (const auto &cell : dof_handler.active_cell_iterators())
      {
        const unsigned int dofs_per_cell = cell->get_fe().n_dofs_per_cell();
  
        cell_matrix.reinit(dofs_per_cell, dofs_per_cell);
        cell_matrix = 0;
  
        hp_fe_values.reinit(cell);
  
        const FEValues<dim> &fe_values = hp_fe_values.get_present_fe_values();
  
        for (unsigned int q_point = 0; q_point < fe_values.n_quadrature_points; ++q_point) {
          //std::cout << fe_values.quadrature_point(q_point)[0] << "     " << fe_values.quadrature_point(q_point)[1] << std::endl;
          for (unsigned int i = 0; i < dofs_per_cell; ++i) {
            //std::cout << fe_values.shape_grad(i, q_point) << std::endl;
            {
              for (unsigned int j = 0; j < dofs_per_cell; ++j)
                cell_matrix(i, j) +=
                  (fe_values.shape_grad(i, q_point) * // grad phi_i(x_q)
                   fe_values.shape_grad(j, q_point) * // grad phi_j(x_q)
                   fe_values.JxW(q_point));           // dx
            }
          }
          }
  
        for (unsigned int i = 0; i < dofs_per_cell; i++)
          for (unsigned int j = 0; j < dofs_per_cell; j++)
            if (std::abs(cell_matrix(i, j)) < 1e-10)
              cell_matrix(i, j) = 0;
        cell_matrix.print(std::cout,20,4);
        std::cout << std::endl << std::endl;
      }

    // check local matrices in the advection case
    /*FullMatrix<double> cell_matrix;
  
    std::vector<types::global_dof_index> local_dof_indices;
  
    for (const auto &cell : dof_handler.active_cell_iterators())
      {
        const unsigned int dofs_per_cell = cell->get_fe().n_dofs_per_cell();
  
        cell_matrix.reinit(dofs_per_cell, dofs_per_cell);
        cell_matrix = 0;
  
        hp_fe_values.reinit(cell);
  
        const FEValues<dim> &fe_values = hp_fe_values.get_present_fe_values();
  
        for (unsigned int q_point = 0; q_point < fe_values.n_quadrature_points; ++q_point)
          for (unsigned int i = 0; i < dofs_per_cell; ++i)
            {
              for (unsigned int j = 0; j < dofs_per_cell; ++j)
                cell_matrix(i, j) +=
                  (-beta*fe_values.shape_grad(i, q_point) * // -beta*grad phi_i(x_q)
                   fe_values.shape_value(j, q_point) * // phi_j(x_q)
                   fe_values.JxW(q_point));           // dx
            }
  
        for (unsigned int i = 0; i < dofs_per_cell; i++)
          for (unsigned int j = 0; j < dofs_per_cell; j++)
            if (std::abs(cell_matrix(i, j)) < 1e-10)
              cell_matrix(i, j) = 0;
        cell_matrix.print(std::cout,20,4);
        std::cout << std::endl << std::endl;
      }*/

    // print number of quadrature points on each face
    /*for (const auto &cell : dof_handler.active_cell_iterators())
      for (auto &face : cell->face_iterators()) {
        hp_fe_interface_values.reinit(cell, face, face -> manifold_id());
        const FEFaceValues<dim> &fe_face_values = hp_fe_interface_values.get_present_fe_values();
        std::cout << fe_face_values.n_quadrature_points << std::endl;
      }*/




    return 0;
}