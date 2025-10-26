// Minimal DG SIPG using MeshWorker + LocalIntegrators::Laplace (very similar to step-39)
//
#include <deal.II/meshworker/integration_info.h>
#include <deal.II/meshworker/mesh_loop.h>
#include <deal.II/meshworker/simple.h>

#include <deal.II/integrators/laplace.h>

using namespace dealii;

// Auxiliary class for the rhs (just for the sake of generality)
//
template <int dim>
class RHS_Func: public Function<dim> {
public:
  double value(const Point<dim>& p,
               const unsigned    comp = 0) const override {
    (void)p;
    (void)comp;

    return 1.0;
  }
};

// Auxiliary class for the the boundary condition (just for the sake of generality)
//
template <int dim>
class Boundary_Func: public Function<dim> {
public:
  double value(const Point<dim>& p,
               const unsigned    comp = 0) const override {
    (void)p;
    (void)comp;

    return 0.0;
  }
};

// Assemble_system uses MeshWorker as well as LocalIntegrators::Laplace::cell_matrix and
// LocalIntegrators::Laplace::ip_matrix
//
template<int dim>
void assemble_system(DoFHandler<dim>& dof_handler,
                     SparseMatrix<double>& system_matrix,
                     Vector<double>& system_rhs,
                     const Quadrature<dim>& quad,
                     const Quadrature<dim - 1>& face_quad) {
  /*--- MeshWorker helpers ---*/
  using InfoBox = MeshWorker::IntegrationInfoBox<dim>;
  InfoBox info_box;

  const UpdateFlags cell_update_flags = update_values | update_gradients | update_JxW_values;
  const UpdateFlags face_update_flags = update_values | update_gradients |
                                        update_JxW_values | update_normal_vectors;
  info_box.add_update_flags_cell(cell_update_flags);
  info_box.add_update_flags_face(face_update_flags);
  info_box.add_update_flags_boundary(face_update_flags);

  info_box.cell_quadrature     = quad;
  info_box.face_quadrature     = face_quad;
  info_box.boundary_quadrature = face_quad;

  const FiniteElement<dim>& fe = dof_handler.get_fe();
  MappingQ1<dim> mapping;
  info_box.initialize(fe, mapping);

  MeshWorker::DoFInfo<dim> dof_info(dof_handler);

  /*--- Assembler: simple assembler for SPD system ---*/
  MeshWorker::Assembler::MatrixSimple<SparseMatrix<double>> assembler;
  assembler.initialize(system_matrix);

  /*--- Cell worker: compute local stiffness with LocalIntegrators::Laplace::cell_matrix ---*/
  auto cell_worker = [&](MeshWorker::DoFInfo<dim>& dinfo,
                         MeshWorker::IntegrationInfo<dim>& info)
  {
    FullMatrix<double>& M = dinfo.matrix(0, false).matrix;

    // Extract FEValuesBase from IntegrationInfo
    const FEValuesBase<dim>& fe_v = info.fe_values();

    // LocalIntegrators builds the local stiffness matrix
    LocalIntegrators::Laplace::cell_matrix(M, fe_v);
  };

  /*--- Interior face worker: use ip_matrix to get 4 blocks and assemble ---*/
  auto face_worker = [&](MeshWorker::DoFInfo<dim>& dinfo_minus,
                         MeshWorker::DoFInfo<dim>& dinfo_plus,
                         MeshWorker::IntegrationInfo<dim>& info_minus,
                         MeshWorker::IntegrationInfo<dim>& info_plus)
  {
    FullMatrix<double>& M11 = dinfo_minus.matrix(0, false).matrix;
    FullMatrix<double>& M12 = dinfo_minus.matrix(0, true).matrix;
    FullMatrix<double>& M21 = dinfo_plus.matrix(0, true).matrix;
    FullMatrix<double>& M22 = dinfo_plus.matrix(0, false).matrix;

    // Access FEValuesBase from IntegrationInfo
    const FEValuesBase<dim>& fe_f_minus = info_minus.fe_values(0);
    const FEValuesBase<dim>& fe_f_plus  = info_plus.fe_values(0);

    // Compute interior penalty parameter
    const double ip_penalty = LocalIntegrators::Laplace::compute_penalty<dim>(dinfo_minus,
                                                                              dinfo_plus,
                                                                              fe_f_minus.get_fe().tensor_degree(),
                                                                              fe_f_plus.get_fe().tensor_degree());

    // LocalIntegrators provides ip_matrix overload producing four blocks
    LocalIntegrators::Laplace::ip_matrix(M11, M12, M21, M22,
                                         fe_f_minus, fe_f_plus,
                                         ip_penalty);
  };

  /*--- Boundary face worker: single block (imposes Dirichlet homogeneous via SIPG) ---*/
  auto boundary_worker = [&](MeshWorker::DoFInfo<dim> &dinfo,
                             MeshWorker::IntegrationInfo<dim> &info)
  {
    FullMatrix<double>& M = dinfo.matrix(0, false).matrix;

    // Extract FEValuesBase from IntegrationInfo
    const FEValuesBase<dim>& fe_f = info.fe_values(0);

    // Compute interior penalty parameter
    const unsigned degree   = fe_f.get_fe().tensor_degree();
    const double ip_penalty = LocalIntegrators::Laplace::compute_penalty<dim>(dinfo,
                                                                              dinfo,
                                                                              degree,
                                                                              degree);

    // Assemble boundary contribution (weak imposition through Nitsche)
    LocalIntegrators::Laplace::nitsche_matrix(M,
                                              fe_f,
                                              ip_penalty);
  };

  /*--- Finally, launch the loop which iterates over cells and faces and does assembly ---*/
  MeshWorker::loop<dim, dim>(dof_handler.begin_active(),
                             dof_handler.end(),
                             dof_info,
                             info_box,
                             cell_worker,
                             boundary_worker,
                             face_worker,
                             assembler);

  /*--- Focus now on rhs. Start by declaring and initializing the proper assembler ---*/
  MeshWorker::Assembler::ResidualSimple<Vector<double>> assembler_rhs;
  AnyData                                               data;
  data.add<Vector<double>*>(&system_rhs, "RHS");
  assembler_rhs.initialize(data);

  InfoBox info_box_rhs;

  const UpdateFlags rhs_cell_update_flags = update_values | update_quadrature_points | update_JxW_values;
  const UpdateFlags rhs_boundary_update_flags = update_values | update_gradients |
                                                update_quadrature_points | update_JxW_values | update_normal_vectors;
  info_box_rhs.add_update_flags_cell(rhs_cell_update_flags);
  info_box_rhs.add_update_flags_boundary(rhs_boundary_update_flags);

  info_box_rhs.cell_quadrature     = quad;
  info_box_rhs.boundary_quadrature = face_quad;

  info_box_rhs.initialize(fe, mapping);

  MeshWorker::DoFInfo<dim> dof_info_rhs(dof_handler);

  /*--- Cell worker for the rhs ---*/
  auto rhs_cell_worker = [&](MeshWorker::DoFInfo<dim>& dinfo,
                             MeshWorker::IntegrationInfo<dim>& info)
  {
    Vector<double>& local_vector = dinfo.vector(0).block(0);

    // Extract FEValuesBase from IntegrationInfo
    const FEValuesBase<dim>& fe_v = info.fe_values();

    // Assemble volumetric contribution of rhs
    RHS_Func<dim> rhs_function;
    for(unsigned q = 0; q < fe_v.n_quadrature_points; ++q) {
      const double rhs_val = rhs_function.value(fe_v.quadrature_point(q));
      for(unsigned i = 0; i < fe_v.dofs_per_cell; ++i) {
        local_vector(i) += rhs_val*fe_v.shape_value(i,q)*fe_v.JxW(q);
      }
    }
  };

  /*--- Fictitious face worker for the rhs ---*/
  auto rhs_face_worker = [&](MeshWorker::DoFInfo<dim>& dinfo_minus,
                             MeshWorker::DoFInfo<dim>& dinfo_plus,
                             MeshWorker::IntegrationInfo<dim>& info_minus,
                             MeshWorker::IntegrationInfo<dim>& info_plus) {};

  /*--- Boundary worker for the rhs ---*/
  auto rhs_boundary_worker = [&](MeshWorker::DoFInfo<dim>& dinfo,
                                 MeshWorker::IntegrationInfo<dim>& info)
  {
    Vector<double>& local_vector = dinfo.vector(0).block(0);

    // Extract FEValuesBase from IntegrationInfo
    const FEValuesBase<dim>& fe_f = info.fe_values();

    // Compute interior penalty parameter
    const unsigned degree   = fe_f.get_fe().tensor_degree();
    const double ip_penalty = LocalIntegrators::Laplace::compute_penalty<dim>(dinfo,
                                                                              dinfo,
                                                                              degree,
                                                                              degree);

    // Assemble boundary conitrubtion of rhs
    Boundary_Func<dim> boundary_function;
    for(unsigned q = 0; q < fe_f.n_quadrature_points; ++q) {
      const double boundary_val = boundary_function.value(fe_f.quadrature_point(q));
      for(unsigned int i = 0; i < fe_f.dofs_per_cell; ++i) {
        local_vector(i) += (ip_penalty*fe_f.shape_value(i,q) - // (sigma * v_i(x_q)
                            fe_f.normal_vector(q)*fe_f.shape_grad(i,q))* // - n * grad v_i(x_q))
                            boundary_val*fe_f.JxW(q); // u^D(x_k) * dx
      }
    }
  };

  /*--- Finally, launch the loop which iterates over cells and faces and does assembly ---*/
  MeshWorker::loop<dim, dim>(dof_handler.begin_active(),
                             dof_handler.end(),
                             dof_info_rhs,
                             info_box_rhs,
                             rhs_cell_worker,
                             rhs_boundary_worker,
                             rhs_face_worker,
                             assembler_rhs);
}
