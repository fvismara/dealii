#include <deal.II/fe/fe_dgq.h>

#include <deal.II/base/function.h>
#include <deal.II/fe/fe_enriched.h>

#include <deal.II/base/polynomial.h>
#include <deal.II/base/tensor_product_polynomials.h>

#include <deal.II/grid/tria.h>
#include <deal.II/grid/grid_generator.h>

#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/mapping_q1.h>

#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/precondition.h>

#include <deal.II/numerics/data_out.h>

#include "fe_anisotropic.h"
#include "Laplace_SIPG_DG.h"

#include <iostream>
#include <set>
#include <cmath>

using namespace dealii;

// Check if a 2D finite element space has the tensor product structure
//
bool is_tensor_product_fe_2D(const FiniteElement<2>& fe, unsigned& nx, unsigned& nz) {
  const double tol = 1e-12; /*--- Tolerance to verify location of support points ---*/

  /*--- Check if support points are available and save them ---*/
  const auto unit_support = fe.get_unit_support_points();
  if(unit_support.empty()) {
    nx = 0;
    nz = 0;
    std::cerr << "No support point available (not nodal FE?) " << std::endl;
    return false;
  }

  /*--- Extract and save support points coordinates ---*/
  std::set<double> xs,
                   zs;
  for(const auto& p: unit_support) {
    xs.insert(p[0]);
    zs.insert(p[1]);
  }

  /*--- Preliminary check in terms of dimensions ---*/
  nx = xs.size();
  nz = zs.size();

  if(nx*nz != unit_support.size()) {
    return false;
  }

  /*--- Complete check of tensor product structure ---*/
  for(auto x: xs) {
    for(auto z : zs) {
      bool found = false;
      for(const auto &p : unit_support) {
        if(std::abs(p[0] - x) < tol && std::abs(p[1] - z) < tol) {
          found = true;
          break;
        }
      }
      if(!found) {
        return false;
      }
    }
  }

  return true;
}

// Main function
//
int main() {
  constexpr unsigned dim = 2;
  const unsigned p = 1; /*--- Polynomial degree along x direction ---*/
  const unsigned r = 2; /*--- Polynomial degree along z direction ---*/

  /*--- Prelminary check with standard finite element ---*/
  std::cout << "Starting with standard nodal discontinuous finite elements" << std::endl;

  FE_DGQ<dim> fe_dgq(std::max(p,r));

  std::cout << "DoFs per cell: " << fe_dgq.dofs_per_cell << std::endl;

  unsigned nx, nz;
  bool is_tensor_product = is_tensor_product_fe_2D(fe_dgq, nx, nz);
  std::cout << "Tensor product structure? " << (is_tensor_product ? "✅ YES" : "❌ NO") << std::endl;
  std::cout << "Number of support points along each direction, nx = " << nx << ", nz = " << nz << std::endl;
  std::cout << "Support points and shape values:" << std::endl;
  for(const auto& pnt : fe_dgq.get_unit_support_points()) {
    std::cout << pnt << std::endl;
    for(unsigned i = 0; i < fe_dgq.dofs_per_cell; ++i) {
      std::cout << fe_dgq.shape_value(i, pnt) << std::endl;
    }
  }

  /*--- Check enriched finite element ---*/
  std::cout << std::endl;
  std::cout << "Focus now on enriched discontinuous finite elements" << std::endl;

  Functions::ConstantFunction<dim> enrichment_function(2.0);
  FE_Enriched<dim> fe_enriched(FE_DGQ<dim>(p), FE_DGQ<dim>(r), &enrichment_function);

  std::cout << "DoFs per cell: " << fe_enriched.dofs_per_cell << std::endl;

  is_tensor_product = is_tensor_product_fe_2D(fe_enriched, nx, nz);
  std::cout << "Tensor product structure? " << (is_tensor_product ? "✅ YES" : "❌ NO") << std::endl;
  std::cout << "Number of support points along each direction, nx = " << nx << ", nz = " << nz << std::endl;
  std::cout << "Support points:" << std::endl;
  for(const auto& pnt : fe_enriched.get_unit_support_points()) {
    std::cout << pnt << std::endl;
    /*for(unsigned i = 0; i < fe_enriched.dofs_per_cell; ++i) {
      std::cout << fe_enriched.shape_value(i, pnt) << std::endl;
    }*/ /*--- NOTE: Enirched finite elements require evaluation of the enrichment function at the point in real-space, not valid on reference element ---*/
  }

  std::cout << std::endl;
  std::cout << "Test FE_Enriched with FE_Nothing to verify if it is possible to keep tesor product structure" << std::endl;
  FE_Enriched<dim> fe_fake_enriched(FE_Nothing<dim>(ReferenceCells::Quadrilateral), FE_DGQ<dim>(r), &enrichment_function);

  std::cout << "DoFs per cell: " << fe_fake_enriched.dofs_per_cell << std::endl;

  is_tensor_product = is_tensor_product_fe_2D(fe_fake_enriched, nx, nz);
  std::cout << "Tensor product structure? " << (is_tensor_product ? "✅ YES" : "❌ NO") << std::endl;
  std::cout << "Number of support points along each direction, nx = " << nx << ", nz = " << nz << std::endl;
  std::cout << "Support points:" << std::endl;
  for(const auto& pnt : fe_fake_enriched.get_unit_support_points()) {
    std::cout << pnt << std::endl;
  }

  /*--- Try to analyze AnisotropicPolynomials ---*/
  std::cout << std::endl;
  std::cout << "Analyze AnisotropicPolynomials" << std::endl;

  const auto poly_x = Polynomials::LagrangeEquidistant::generate_complete_basis(p);
  const auto poly_z = Polynomials::LagrangeEquidistant::generate_complete_basis(r);

  std::vector<std::vector<Polynomials::Polynomial<double>>> polys(dim);
  polys[0] = std::move(poly_x);
  polys[1] = std::move(poly_z);

  AnisotropicPolynomials<dim> scalar_poly_tensor_product(polys);

  // Let's evalaute all the polynomials in a random point
  const Point<dim> pnt(0.5, 0.25);

  std::cout << "Degree along x = " << p << ", degree along z = " << r << std::endl;
  std::cout << "Maximum degree according to the class: "<< scalar_poly_tensor_product.degree() << std::endl;
  std::cout << "Number of generated polynomials: " << scalar_poly_tensor_product.n() << std::endl;
  std::cout << "Value in point = " << pnt << std::endl;
  for(unsigned i = 0; i < scalar_poly_tensor_product.n(); ++i) {
    std::cout << "phi[" << i << "] = " << scalar_poly_tensor_product.compute_value(i, pnt) << std::endl;
  }

  /*--- Try to analyze dofs per object (dpo) ---*/
  std::cout << std::endl;
  std::cout << "Testing dofs per object" << std::endl;

  internal::GenericDoFsPerObject dpo;
  dpo.dofs_per_object_exclusive.resize(4);
  dpo.dofs_per_object_inclusive.resize(4);
  dpo.object_index.resize(4);
  dpo.first_object_index_on_face.resize(3);

  // Vertices (4 in 2D)
  dpo.dofs_per_object_exclusive[0] = {1, 1, 1, 1};
  dpo.dofs_per_object_inclusive[0] = {1, 1, 1, 1};

  // Edges
  // edge 0 = bottom, edge 2 = top --> horizontal -> degree_x
  // edge 1 = right, edge 3 = left --> vertical -> degree_z
  unsigned e_h = (p >= 2 ? p - 1 : 0);
  unsigned e_v = (r >= 2 ? r - 1 : 0);
  dpo.dofs_per_object_exclusive[1] = {e_h, e_v, e_h, e_v};
  dpo.dofs_per_object_inclusive[1] = {p + 1, r + 1, p + 1, r + 1};

  // Inner faces
  dpo.dofs_per_object_exclusive[2] = {(p >= 2 && r >= 2) ?
                                      (p - 1)*(r - 1) : 0};
  dpo.dofs_per_object_inclusive[2] = {(p + 1)*(r + 1)};

  // Volume (included for coherence)
  dpo.dofs_per_object_exclusive[3] = {0};
  dpo.dofs_per_object_inclusive[3] = {0};

  // Focus now on object_index. This should be the global index of my geoemtric entity,
  // i.e., first vertices, then edgses and so on...
  // However not fully coherent with what I say in fe_data.cc
  dpo.object_index[0] = {0, 1, 2, 3};
  dpo.object_index[1] = {4, 5, 6, 7};
  dpo.object_index[2] = {8};
  dpo.object_index[3] = {0};

  dpo.first_object_index_on_face[0] = {0};
  dpo.first_object_index_on_face[1] = {4};
  dpo.first_object_index_on_face[2] = {8};

  std::cout << "Size vector of vectors dofs_per_object_inclusive imposed = "
            << dpo.dofs_per_object_inclusive.size() << std::endl;
  std::cout << "Size vector of vectors dofs_per_object_inclusive (vertex) imposed = "
            << dpo.dofs_per_object_inclusive[0].size() << std::endl;
  for(const auto elem: dpo.dofs_per_object_inclusive[0]) {
    std::cout << elem << "  ";
  }
  std::cout << "\nSize vector of vectors dofs_per_object_inclusive (edge) imposed = "
            << dpo.dofs_per_object_inclusive[1].size() << std::endl;
  for(const auto elem: dpo.dofs_per_object_inclusive[1]) {
    std::cout << elem << "  ";
  }
  std::cout << "\nSize vector of vectors dofs_per_object_inclusive (inner face) imposed = "
            << dpo.dofs_per_object_inclusive[2].size() << std::endl;
  for(const auto elem: dpo.dofs_per_object_inclusive[2]) {
    std::cout << elem << "  ";
  }
  std::cout << std::endl;

  /*--- Check if I can build a FiniteElementData from that ---*/
  std::cout << std::endl;
  std::cout << "Checking FiniteElementData constructor" << std::endl;

  FiniteElementData<dim> fe_data_test(dpo,
                                      ReferenceCells::Quadrilateral,
                                      1,
                                      std::max(p,r),
                                      FiniteElementData<dim>::L2);
  std::cout << "Dofs per vertex = " << fe_data_test.n_dofs_per_vertex() << std::endl;
  std::cout << "Dofs per line = "   << fe_data_test.n_dofs_per_line()   << std::endl;
  std::cout << "Dofs per cell = "   << fe_data_test.n_dofs_per_cell()   << std::endl;
  dpo = internal::GenericDoFsPerObject::generate(fe_data_test);
  std::cout << "Size vector of vectors dofs_per_object_inclusive from generate "
            << "(starting from the FiniteElementData built using GenericDoFsPerObject) = "
            << dpo.dofs_per_object_inclusive.size()
            << std::endl;
  std::cout << "Size vector of vectors dofs_per_object_inclusive (vertex) from generate "
            << "(starting from the FiniteElementData built using GenericDoFsPerObject) = "
            << dpo.dofs_per_object_inclusive[0].size()
            << std::endl;
  for(const auto elem: dpo.dofs_per_object_inclusive[0]) {
    std::cout << elem << "  ";
  }
  std::cout << "\nSize vector of vectors dofs_per_object_inclusive (edge) from generate "
            << "(starting from the FiniteElementData built using GenericDoFsPerObject) = "
            << dpo.dofs_per_object_inclusive[1].size()
            << std::endl;
  for(const auto elem: dpo.dofs_per_object_inclusive[1]) {
    std::cout << elem << "  ";
  }
  std::cout << "\nSize vector of vectors dofs_per_object_inclusive (inner face) from generate "
            << "(starting from the FiniteElementData built using GenericDoFsPerObject) = "
            << dpo.dofs_per_object_inclusive[2].size()
            << std::endl;
  for(const auto elem: dpo.dofs_per_object_inclusive[2]) {
    std::cout << elem << "  ";
  }
  std::cout << std::endl;

  std::cout << std::endl;
  std::cout << "Building a classic FiniteElementData based on DG" << std::endl;
  const auto dpo_vector = FE_Anisotropic<dim>::build_dpo(std::max(p,r));
  std::cout << "Checking vector of dofs_per_object" << std::endl;
  for(const auto elem: dpo_vector) {
    std::cout << elem << "  ";
  }
  FiniteElementData<dim> fe_data_test_dg(dpo_vector,
                                         ReferenceCells::Quadrilateral,
                                         1,
                                         std::max(p,r),
                                         FiniteElementData<dim>::L2);
  std::cout << "\nDofs per vertex = " << fe_data_test_dg.n_dofs_per_vertex() << std::endl;
  std::cout << "Dofs per line = "   << fe_data_test_dg.n_dofs_per_line()   << std::endl;
  std::cout << "Dofs per cell = "   << fe_data_test_dg.n_dofs_per_cell()   << std::endl;
  dpo = internal::GenericDoFsPerObject::generate(fe_data_test_dg);
  std::cout << "Size vector of vectors dofs_per_object_inclusive from generate "
            << "(starting from the FiniteElementData built as for DG) = "
            << dpo.dofs_per_object_inclusive.size()
            << std::endl;
  std::cout << "Size vector of vectors dofs_per_object_inclusive (vertex) from generate "
            << "(starting from the FiniteElementData built as for DG) = "
            << dpo.dofs_per_object_inclusive[0].size()
            << std::endl;
  for(const auto elem: dpo.dofs_per_object_inclusive[0]) {
    std::cout << elem << "  ";
  }
  std::cout << "\nSize vector of vectors dofs_per_object_inclusive (edge) from generate "
            << "(starting from the FiniteElementData built as for DG) = "
            << dpo.dofs_per_object_inclusive[1].size()
            << std::endl;
  for(const auto elem: dpo.dofs_per_object_inclusive[1]) {
    std::cout << elem << "  ";
  }
  std::cout << "\nSize vector of vectors dofs_per_object_inclusive (inner face) from generate "
            << "(starting from the FiniteElementData built as for DG) = "
            << dpo.dofs_per_object_inclusive[2].size()
            << std::endl;
  for(const auto elem: dpo.dofs_per_object_inclusive[2]) {
    std::cout << elem << "  ";
  }
  std::cout << "\nSize vector of vectors object_index from generate "
            << "(starting from the FiniteElementData built as for DG) = "
            << dpo.object_index.size()
            << std::endl;
  std::cout << "Size vector of vectors object_index (vertex) from generate "
            << "(starting from the FiniteElementData built as for DG) = "
            << dpo.object_index[0].size()
            << std::endl;
  for(const auto elem: dpo.object_index[0]) {
    std::cout << elem << "  ";
  }
  std::cout << "\nSize vector of vectors object_index (edge) from generate "
            << "(starting from the FiniteElementData built as for DG) = "
            << dpo.object_index[1].size()
            << std::endl;
  for(const auto elem: dpo.object_index[1]) {
    std::cout << elem << "  ";
  }
  std::cout << "\nSize vector of vectors object_index (inner face) from generate "
            << "(starting from the FiniteElementData built as for DG) = "
            << dpo.object_index[2].size()
            << std::endl;
  for(const auto elem: dpo.object_index[2]) {
    std::cout << elem << "  ";
  }
  std::cout << std::endl;

  /*--- Try with anisotropic finite element space ---*/
  std::cout << std::endl;
  std::cout << "Focus now on anisotropic discontinuous finite elements" << std::endl;

  FE_Anisotropic<dim> fe_aniso(p, r, scalar_poly_tensor_product);

  std::cout << "DoFs per cell stored: " << fe_aniso.dofs_per_cell << std::endl;

  is_tensor_product = is_tensor_product_fe_2D(fe_aniso, nx, nz);
  std::cout << "Tensor product structure? " << (is_tensor_product ? "✅ YES" : "❌ NO") << std::endl;
  std::cout << "Number of support points along each direction, nx = " << nx << ", nz = " << nz << std::endl;
  std::cout << "Degree of the finite element space = " << fe_aniso.degree << std::endl;
  std::cout << "Support points and shape values:" << std::endl;
  for(const auto& pnt : fe_aniso.get_unit_support_points()) {
    std::cout << pnt << std::endl;
    for(unsigned i = 0; i < fe_aniso.dofs_per_cell; ++i) {
      std::cout << fe_aniso.shape_value(i, pnt) << std::endl;
    }
  }

  /*--- Check the newly created finite element space ---*/
  // Create the mesh and refine as desired
  Triangulation<dim> triangulation;
  GridGenerator::hyper_cube(triangulation);
  triangulation.refine_global(3);

  // Create the associated dof handler
  DoFHandler<dim> dof_handler(triangulation);
  DoFHandler<dim> dof_handler_ref(triangulation);

  dof_handler.distribute_dofs(fe_aniso);

  const auto dof_location_map = DoFTools::map_dofs_to_support_points(MappingQ1<dim>(), dof_handler);
  std::ofstream dof_location_file("dofs_location.gnuplot");
  DoFTools::write_gnuplot_dof_support_point_info(dof_location_file,
                                                 dof_location_map);

  // Associate (fictitious) constraints
  AffineConstraints<double> constraints;
  constraints.close();

  // Sparsity pattern
  const unsigned n_dofs = dof_handler.n_dofs();
  DynamicSparsityPattern dsp(n_dofs);
  DoFTools::make_flux_sparsity_pattern(dof_handler, dsp);
  SparsityPattern sparsity_pattern;
  sparsity_pattern.copy_from(dsp);

  // Assemble matrix and right-hand-side vector
  SparseMatrix<double> sparse_matrix;
  sparse_matrix.reinit(sparsity_pattern);

  Vector<double> rhs_vector(n_dofs);
  Vector<double> solution_vector(n_dofs);

  const unsigned degree = fe_aniso.degree;
  QGauss<dim>     cell_quad(degree + 1);
  QGauss<dim - 1> face_quad(degree + 1);

  assemble_system(dof_handler, sparse_matrix, rhs_vector, cell_quad, face_quad);

  // Solve the linear system
  SolverControl solver_control(1000, 1e-12*rhs_vector.l2_norm());
  SolverCG<Vector<double>> solver(solver_control);
  solver.solve(sparse_matrix, solution_vector, rhs_vector, PreconditionIdentity());

  // Output results
  DataOut<dim> data_out;

  DataOutBase::VtkFlags flags;
  flags.write_higher_order_cells = true;
  data_out.set_flags(flags);

  data_out.add_data_vector(dof_handler, solution_vector, "solution");
  data_out.build_patches(fe_aniso.degree + 1);

  std::string   file_name = "solution.vtu";
  std::ofstream file(file_name);
  data_out.write_vtu(file);

  return 0;
}
