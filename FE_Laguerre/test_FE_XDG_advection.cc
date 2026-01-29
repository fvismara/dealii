#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/function.h>
#include <deal.II/base/convergence_table.h>
#include <deal.II/lac/vector.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/solver_gmres.h>
#include <deal.II/lac/precondition_block.h>
#include <deal.II/grid/tria.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_out.h>
#include <deal.II/grid/grid_refinement.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/fe/mapping_q1.h>
#include <deal.II/fe/mapping_q1.h>
#include <deal.II/fe/fe_dgq.h>
#include <deal.II/fe/fe_enriched.h>
#include <deal.II/fe/fe_nothing.h>
#include <deal.II/fe/fe_interface_values.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/numerics/vector_tools.h>
#include <deal.II/numerics/data_out.h>
#include <deal.II/meshworker/mesh_loop.h>
#include <deal.II/hp/fe_collection.h>
#include <deal.II/hp/fe_values.h>

#include "fe_dgq_aniso.h"
#include "Laguerre.h"
#include "laguerre_quad.h"
 
#include <iostream>
#include <fstream>

/*
Solves the advection equation div(beta . grad(u)) = f with Dirichlet (inflow) boundary conditions. Builds on step 12
*/
 
 
namespace XDGAdvection
{
  using namespace dealii;

  // Enrichment function, exp(-mu*z/2)
  template <int dim>
  class ScaledExponential: public Function<dim> {
  public:
    ScaledExponential(const double _mu, const double _Lz) : mu(_mu), Lz(_Lz) {};
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
    double mu, Lz;
  };

  // exact solution
  template <int dim>
  class ExactSolution: public Function<dim> {
  public:
    ExactSolution(const double _Lx) : Lx(_Lx) {};
    double value(const Point<dim>& p, const unsigned) const override {
      return p[1]*std::exp(-p[1])*std::sin(2*numbers::PI*p[0]/Lx);
    }
    Tensor<1,2,double> gradient(const Point<dim>& p, const unsigned) const override {
      Tensor<1,2,double> grad;
      grad[0] = p[1]*std::exp(-p[1])*2*numbers::PI*std::cos(2*numbers::PI*p[0]/Lx)/Lx;
      grad[1] = (1-p[1])*std::exp(-p[1])*std::sin(2*numbers::PI*p[0]/Lx);
      return grad;
    }
  private:
    double Lx;
  };
 
  // boundary data
  template <int dim>
  class BoundaryValues : public Function<dim>
  {
  public:
    BoundaryValues() = default;
    virtual void value_list(const std::vector<Point<dim>> &points,
                            std::vector<double>           &values,
                            const unsigned int component = 0) const override;
  };
 
  template <int dim>
  void BoundaryValues<dim>::value_list(const std::vector<Point<dim>> &points,
                                       std::vector<double>           &values,
                                       const unsigned int component) const
  {
    (void)component;
    AssertIndexRange(component, 1);
    AssertDimension(values.size(), points.size());
 
    for (unsigned int i = 0; i < values.size(); ++i)
      values[i] = 0;
  }

  // advection field
  template <int dim>
  Tensor<1, dim> beta(const Point<dim> &p)
  {
    Assert(dim >= 2, ExcNotImplemented());
 
    Tensor<1, dim> wind_field;
    wind_field[0] = 3;
    wind_field[1] = 1;
  
    return wind_field;
  }

  // divergence of advection field
  template <int dim>
  double divergence_beta(const Point<dim> &p)
  {
    return 0;
  }

  // right hand side f
  template <int dim>
  class RhsFunction : public Function<dim>
  {
  public:
    RhsFunction() = default;
    RhsFunction(const double _Lx) : Lx(_Lx) {};
    virtual void value_list(const std::vector<Point<dim>> &points,
                            std::vector<double>           &values,
                            const unsigned int component = 0) const override;
  private:
    double Lx;
  };
 
  template <int dim>
  void RhsFunction<dim>::value_list(const std::vector<Point<dim>> &points, std::vector<double> &values, const unsigned int component) const
  {
    (void)component;
    AssertIndexRange(component, 1);
    AssertDimension(values.size(), points.size());
 
    for (unsigned int i = 0; i < values.size(); ++i)
      {
        values[i] = std::exp(-points[i][1])*(beta(points[i])[0]*2*numbers::PI*points[i][1]*std::cos(2*numbers::PI*points[i][0]/Lx)/Lx
          + beta(points[i])[1]*(1-points[i][1])*std::sin(2*numbers::PI*points[i][0]/Lx))
          + divergence_beta(points[i])*points[i][1]*std::exp(-points[i][1])*std::sin(2*numbers::PI*points[i][0]/Lx);
      }
  }
 
 
  template <int dim>
  struct ScratchData
  {
    ScratchData(const hp::FECollection<dim>   &fe_collection,
                const hp::QCollection<dim>    &quadrature_collection,
                const hp::QCollection<dim-1>  &quadrature_face_collection,
                const UpdateFlags          update_flags = update_values | update_gradients | update_quadrature_points | update_JxW_values,
                const UpdateFlags          face_update_flags = update_values | update_gradients | update_quadrature_points |
                                                           update_JxW_values | update_normal_vectors,
                const UpdateFlags          interface_update_flags = update_values | update_gradients | update_quadrature_points |
                                                           update_JxW_values | update_normal_vectors)
      : fe_values(fe_collection, quadrature_collection, update_flags)
      , fe_face_values(fe_collection, quadrature_face_collection, face_update_flags)
      , fe_interface_values(fe_collection, quadrature_face_collection, interface_update_flags)
    {}
 
    ScratchData(const ScratchData<dim> &scratch_data)
      : fe_values(scratch_data.fe_values.get_fe_collection(),
                  scratch_data.fe_values.get_quadrature_collection(),
                  scratch_data.fe_values.get_update_flags())
      , fe_face_values(scratch_data.fe_face_values.get_fe_collection(),
                            scratch_data.fe_face_values.get_quadrature_collection(),
                            scratch_data.fe_face_values.get_update_flags())
      , fe_interface_values(scratch_data.fe_interface_values.get_fe_collection(),
                            scratch_data.fe_interface_values.get_quadrature_collection(),
                            scratch_data.fe_interface_values.get_update_flags())

    {}
 
    hp::FEValues<dim>          fe_values;
    hp::FEFaceValues<dim>      fe_face_values;
    FEInterfaceValues<dim>     fe_interface_values;
  };
 
 
  struct CopyDataFace
  {
    FullMatrix<double>                   cell_matrix;
    std::vector<types::global_dof_index> joint_dof_indices;
  };
 
 
  struct CopyData
  {
    FullMatrix<double>                   cell_matrix;
    Vector<double>                       cell_rhs;
    std::vector<types::global_dof_index> local_dof_indices;
    std::vector<CopyDataFace>            face_data;
 
    template <class Iterator>
    void reinit(const Iterator &cell, unsigned int dofs_per_cell)
    {
      cell_matrix.reinit(dofs_per_cell, dofs_per_cell);
      cell_rhs.reinit(dofs_per_cell);
 
      local_dof_indices.resize(dofs_per_cell);
      cell->get_dof_indices(local_dof_indices);
    }
  };
 
  template <int dim>
  class AdvectionProblem
  {
  public:
    AdvectionProblem(unsigned int _px, unsigned int _pz, unsigned int _M, unsigned int _Nx, unsigned int _Nz,
      double _mu, double _Lx, double _Lz);
    void run();
    double get_L2_error() {return this -> L2_error;};
 
  private:
    void setup_system();
    void assemble_system();
    void solve();
    void output_results() const;
    void compute_error();
 
    Triangulation<dim>   triangulation;
    const MappingQ1<dim> mapping;
    DoFHandler<dim>      dof_handler;

    const unsigned px, pz, M, Nx, Nz;
    const double mu, Lx, Lz;
    const double Lzmax; // last Laguerre quadrature point
    ScaledExponential<dim> enrichment_function;

    RhsFunction<dim>    rhs_function;
    BoundaryValues<dim> boundary_function;
    ExactSolution<dim>  exact_solution;

    hp::FECollection<dim>  fe_collection;
    hp::QCollection<dim>   quadrature_collection;
    hp::QCollection<dim-1> face_quadrature_collection;
 
    SparsityPattern      sparsity_pattern;
    SparseMatrix<double> system_matrix;
 
    Vector<double> solution;
    Vector<double> right_hand_side;

    double L2_error;
  };
 
 
  template <int dim>
  AdvectionProblem<dim>::AdvectionProblem(unsigned int _px, unsigned int _pz, unsigned int _M, unsigned int _Nx, 
                                          unsigned int _Nz, double _mu, double _Lx, double _Lz)
    : mapping()
    , dof_handler(triangulation)
    , px(_px), pz(_pz), M(_M), Nx(_Nx), Nz(_Nz), mu(_mu), Lx(_Lx), Lz(_Lz), Lzmax(Laguerre(_M+1,_mu).get_zeros_of_derivative()[_M-1])
    , enrichment_function(_mu, _Lz), rhs_function(_Lx), exact_solution(_Lx)
  {}
 
 
  template <int dim>
  void AdvectionProblem<dim>::setup_system()
  {
    const FE_DGQ_Aniso<dim> fe_legendre(px, pz, AnisotropicPolynomials<dim>({Polynomials::Legendre::generate_complete_basis(px),
         Polynomials::Legendre::generate_complete_basis(pz)}));
    const FE_Enriched<dim> fe_laguerre(FE_Nothing<dim>(), FE_DGQ_Aniso<dim>(px, M, AnisotropicPolynomials<dim>({Polynomials::Legendre::generate_complete_basis(px),
         Laguerre::generate_complete_basis_scaled(M, mu)})), &enrichment_function);

    fe_collection.push_back(fe_legendre);
    fe_collection.push_back(fe_laguerre);

    quadrature_collection.push_back(Quadrature<dim>(QGauss<dim-1>(px+1), QGauss<dim-1>(pz+1)));
    quadrature_collection.push_back(Quadrature<dim>(QGauss<dim-1>(px+1), QGaussLaguerreRadau<1>(M,mu*Lzmax)));

    face_quadrature_collection.push_back(QGauss<dim-1>(px+1));
    face_quadrature_collection.push_back(QGauss<dim-1>(pz+1));
    face_quadrature_collection.push_back(QGaussLaguerreRadau<1>(M,mu*Lzmax));

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
 
    DynamicSparsityPattern dsp(dof_handler.n_dofs());
    DoFTools::make_flux_sparsity_pattern(dof_handler, dsp);
    sparsity_pattern.copy_from(dsp);
 
    system_matrix.reinit(sparsity_pattern);
    solution.reinit(dof_handler.n_dofs());
    right_hand_side.reinit(dof_handler.n_dofs());
  }

 
  template <int dim>
  void AdvectionProblem<dim>::assemble_system()
  {
    using Iterator = typename DoFHandler<dim>::active_cell_iterator;
 
    // cell worker
    const auto cell_worker = [&](const Iterator &cell, ScratchData<dim> &scratch_data, CopyData &copy_data) {

      scratch_data.fe_values.reinit(cell);

      const FEValues<dim> &fe_cell = scratch_data.fe_values.get_present_fe_values();
      
      const auto &q_points = fe_cell.get_quadrature_points();
      const unsigned int n_dofs = fe_cell.get_fe().n_dofs_per_cell();
      
      copy_data.reinit(cell, n_dofs);

      std::vector<double> f(q_points.size());
      rhs_function.value_list(q_points, f);
      
      for (unsigned int qpoint = 0; qpoint < q_points.size(); ++qpoint)
        {
          auto beta_q = beta(q_points[qpoint]);
          for (unsigned int i = 0; i < n_dofs; ++i)
            for (unsigned int j = 0; j < n_dofs; ++j)
                copy_data.cell_matrix(i, j) +=
                  -beta_q                          // -\beta
                  * fe_cell.shape_grad(i, qpoint)  // \nabla \phi_i
                  * fe_cell.shape_value(j, qpoint) // \phi_j
                  * fe_cell.JxW(qpoint);           // dx

          for (unsigned int i = 0; i < n_dofs; ++i)
            copy_data.cell_rhs(i) += f[qpoint] * fe_cell.shape_value(i, qpoint) * fe_cell.JxW(qpoint);
        }
    };
 
    // boundary worker
    const auto boundary_worker = [&](const Iterator &cell, const unsigned int &face_no, ScratchData<dim> &scratch_data, CopyData &copy_data) {
      
      scratch_data.fe_face_values.reinit(cell, face_no, cell -> face(face_no) -> manifold_id());

      const FEFaceValues<dim> &fe_face = scratch_data.fe_face_values.get_present_fe_values();

      const auto &q_points = fe_face.get_quadrature_points();
      const unsigned int n_dofs = fe_face.get_fe().n_dofs_per_cell();
      const std::vector<Tensor<1, dim>> &normals = fe_face.get_normal_vectors();
 
      std::vector<double> g(q_points.size());
      boundary_function.value_list(q_points, g);

      for (unsigned int qpoint = 0; qpoint < q_points.size(); ++qpoint)
        {
          const double beta_dot_n = beta(q_points[qpoint]) * normals[qpoint];
          
          if (beta_dot_n > 0) // outflow
            {
              for (unsigned int i = 0; i < n_dofs; ++i)
                for (unsigned int j = 0; j < n_dofs; ++j)
                  copy_data.cell_matrix(i, j) +=
                    fe_face.shape_value(i, qpoint)   // \phi_i
                    * fe_face.shape_value(j, qpoint) // \phi_j
                    * beta_dot_n                     // \beta . n
                    * fe_face.JxW(qpoint);           // dx
            }
          else
            for (unsigned int i = 0; i < n_dofs; ++i)
              copy_data.cell_rhs(i) += -fe_face.shape_value(i, qpoint) // \phi_i
                                       * g[qpoint]                     // g
                                       * beta_dot_n                    // \beta . n
                                       * fe_face.JxW(qpoint);          // dx
        }
    };
 
    // face worker
    const auto face_worker = [&](const Iterator     &cell,
                                 const unsigned int &f,
                                 const unsigned int &sf,
                                 const Iterator     &ncell,
                                 const unsigned int &nf,
                                 const unsigned int &nsf,
                                 ScratchData<dim>   &scratch_data,
                                 CopyData           &copy_data) {

      scratch_data.fe_interface_values.reinit(cell, f, sf, ncell, nf, nsf, cell -> face(f) -> manifold_id(),
                                              0, cell -> active_fe_index(), ncell -> active_fe_index());

      FEInterfaceValues<dim> &fe_iv = scratch_data.fe_interface_values;
      fe_iv.reinit(cell, f, sf, ncell, nf, nsf, cell -> face(f) -> manifold_id(),
                    0, cell -> active_fe_index(), ncell -> active_fe_index());

      const auto &q_points = fe_iv.get_quadrature_points();
 
      copy_data.face_data.emplace_back();
      CopyDataFace &copy_data_face = copy_data.face_data.back();
 
      const unsigned int n_dofs        = fe_iv.n_current_interface_dofs();
      copy_data_face.joint_dof_indices = fe_iv.get_interface_dof_indices();
 
      copy_data_face.cell_matrix.reinit(n_dofs, n_dofs);
 
      const std::vector<Tensor<1, dim>> &normals = fe_iv.get_normal_vectors();
 
      for (unsigned int qpoint = 0; qpoint < q_points.size(); ++qpoint)
        {
          const double beta_dot_n = beta(q_points[qpoint]) * normals[qpoint];
          
          for (unsigned int i = 0; i < n_dofs; ++i)
            for (unsigned int j = 0; j < n_dofs; ++j)
              copy_data_face.cell_matrix(i, j) +=
                fe_iv.jump_in_shape_values(i, qpoint)             // [\phi_i]
                * fe_iv.shape_value((beta_dot_n > 0), j, qpoint)  // phi_j^{upwind}
                * beta_dot_n                                      // (\beta . n)
                * fe_iv.JxW(qpoint);                              // dx
        }
    };
 
    // copy local contribution to global matrix and vector
    const AffineConstraints<double> constraints;
    const auto copier = [&](const CopyData &c) {
      constraints.distribute_local_to_global(c.cell_matrix, c.cell_rhs, c.local_dof_indices, system_matrix, right_hand_side);
 
      for (const auto &cdf : c.face_data)
        constraints.distribute_local_to_global(cdf.cell_matrix, cdf.joint_dof_indices, system_matrix);
    };
 
    // mesh loop
    ScratchData<dim> scratch_data(fe_collection, quadrature_collection, face_quadrature_collection);
    CopyData         copy_data;
 
    MeshWorker::mesh_loop(dof_handler.begin_active(),
                          dof_handler.end(),
                          cell_worker,
                          copier,
                          scratch_data,
                          copy_data,
                          MeshWorker::assemble_own_cells |
                          MeshWorker::assemble_boundary_faces |
                          MeshWorker::assemble_own_interior_faces_once,
                          boundary_worker,
                          face_worker);
  }

 
  template <int dim>
  void AdvectionProblem<dim>::solve()
  {
    SolverControl solver_control(1000, 1e-6 * right_hand_side.l2_norm());
 
    SolverGMRES<Vector<double>>::AdditionalData additional_data;
    additional_data.max_basis_size = 100;
    SolverGMRES<Vector<double>> solver(solver_control, additional_data);
 
    PreconditionBlockSSOR<SparseMatrix<double>> preconditioner;
    preconditioner.initialize(system_matrix, 1);
 
    solver.solve(system_matrix, solution, right_hand_side, preconditioner);

    std::cout << "  Solver converged in " << solver_control.last_step() << " iterations." << std::endl;
  }
 

  template <int dim>
  void AdvectionProblem<dim>::output_results() const
  {
    const std::string filename = "solution-" + std::to_string(0) + ".vtk";
    std::cout << "  Writing solution to <" << filename << '>' << std::endl;
    std::ofstream output(filename);
 
    DataOut<dim> data_out;
    data_out.attach_dof_handler(dof_handler);
    data_out.add_data_vector(solution, "u", DataOut<dim>::type_dof_data);
 
    data_out.build_patches(mapping,20);
 
    data_out.write_vtk(output);
  }


  template <int dim>
  void AdvectionProblem<dim>::compute_error()
  {
    Vector<double> difference_per_cell(triangulation.n_active_cells());
    VectorTools::integrate_difference(dof_handler, solution, exact_solution, difference_per_cell, 
                                      quadrature_collection, VectorTools::L2_norm);
    L2_error = VectorTools::compute_global_error(triangulation, difference_per_cell, VectorTools::L2_norm);
  }
 
 
  template <int dim>
  void AdvectionProblem<dim>::run()
  {
    const Point<dim> & p1 = {0, 0};
    const Point<dim> & p2 = {Lx, Lz+Lzmax};
    double dx = Lx/Nx;
    double dz = Lz/Nz;

    // generate grid
    std::vector<std::vector<double>> step_sizes;

    std::vector<double> step_sizes_x(Nx);
    std::vector<double> step_sizes_z(Nz+1);
    for (unsigned int i = 0; i < Nx; i++)
        step_sizes_x[i] = dx;
    for (unsigned int j = 0; j < Nz; j++)
        step_sizes_z[j] = dz;
    step_sizes_z[Nz] = Lzmax;

    step_sizes.push_back(step_sizes_x);
    step_sizes.push_back(step_sizes_z);

    GridGenerator::subdivided_hyper_rectangle(triangulation, step_sizes, p1, p2);
 
    std::cout << "  Number of active cells:       " << triangulation.n_active_cells() << std::endl;
 
    setup_system();
 
    std::cout << "  Number of degrees of freedom: " << dof_handler.n_dofs() << std::endl;
 
    assemble_system();
    solve();
    output_results();
    compute_error();
  }

} // namespace XDGAdvection


void convergence_analysis (unsigned int px, unsigned int pz, unsigned int M, unsigned int Nx, unsigned int Nz,
  double mu, double Lx, double Lz, unsigned int n_refinements, unsigned int variable) {

  ConvergenceTable convergence_table; 

  unsigned int _Nx, _Nz, _M;
  switch (variable) {
    case 1:
      _Nx = 1, _Nz = Nz, _M = M;
      break;
    case 2:
      _Nx = Nx, _Nz = 1, _M = M;
      break;
    case 3:
      _Nx = Nx, _Nz = Nz, _M = 1;
      break;
  }

  for (unsigned int cycle = 0; cycle < n_refinements; cycle++) {
    XDGAdvection::AdvectionProblem<2> XDG_advection(px, pz, _M, _Nx, _Nz, mu, Lx, Lz);
    XDG_advection.run();

    convergence_table.add_value("Nx", _Nx);
    convergence_table.add_value("Nz", _Nz);
    convergence_table.add_value("M", _M);
    convergence_table.add_value("L2err", XDG_advection.get_L2_error());

    switch (variable) {
    case 1:
      _Nx *= 2;
      break;
    case 2:
      _Nz *= 2;
      break;
    case 3:
      _M += 1;
      break;
    }
  }

  convergence_table.set_precision("L2err", 3);
  convergence_table.set_scientific("L2err", true);
  convergence_table.evaluate_convergence_rates("L2err", ConvergenceTable::reduction_rate_log2);
  convergence_table.write_text(std::cout);

}
 
 
int main()
{

  unsigned int px = 1, pz = 3, M = 20, Nx = 17, Nz = 30; // simply set Nz = 0 for standalone DG-Laguerre
  double mu = 1.835, Lx = 1, Lz = 2;

  unsigned int n_refinements = 8;  // number of refinements in the covergence test
  unsigned int variable = 1;       // quantity to refine: Nx -> 1, Nz -> 2, M -> 3 

  bool run_convergence_test = true; // set to true for convergence test, false to solve the problem once

  
  if (run_convergence_test)

    convergence_analysis(px, pz, M, Nx, Nz, mu, Lx, Lz, n_refinements, variable);

  else {

    XDGAdvection::AdvectionProblem<2> XDG_advection(px, pz, M, Nx, Nz, mu, Lx, Lz);
    XDG_advection.run();
    std::cout << XDG_advection.get_L2_error() << std::endl;

  }
 
  return 0;

}