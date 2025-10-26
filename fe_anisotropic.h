#include <deal.II/fe/fe_poly.h>
#include <deal.II/fe/fe_data.h>

using namespace dealii;

// Derived class from FE_Poly to represent anisotropic finite elements
//
template<int dim>
class FE_Anisotropic: public FE_Poly<dim> {
public:
  FE_Anisotropic(const unsigned degree_x,
                 const unsigned degree_z,
                 const AnisotropicPolynomials<dim>& polynomials); /*--- Class constructor ---*/

  std::string get_name() const override; /*--- Get name of this finite element space ---*/

  std::unique_ptr<FiniteElement<dim>> clone() const override; /*--- Clone function ---*/

  static std::vector<unsigned> build_dpo(const unsigned degree); /*--- Build auxiliary vector of dof per object for FiniteElementData ---*/

private:
  internal::GenericDoFsPerObject build_dpo(const unsigned degree_x,
                                           const unsigned degree_z); /*--- Build dof per object for FiniteElementData ---*/

  /*unsigned count_dofs_per_cell_from_degrees(const unsigned degree_x,
                                            const unsigned degree_z);*/ /*--- Get the degrees of freedom per cell from polynomial degrees ---*/
};

// Class constructor
//
template<int dim>
FE_Anisotropic<dim>::FE_Anisotropic(const unsigned degree_x,
                                    const unsigned degree_z,
                                    const AnisotropicPolynomials<dim>& polynomials) :
  FE_Poly<dim>(polynomials,
               FiniteElementData<dim>(this->build_dpo(degree_x, degree_z),
                                      ReferenceCells::Quadrilateral,
                                      1,
                                      std::max(degree_x, degree_z),
                                      FiniteElementData<dim>::L2),
                std::vector<bool>((degree_x + 1)*(degree_z + 1), true),
                std::vector<ComponentMask>((degree_x + 1)*(degree_z + 1),
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
    } /*--- TODO: Cartesian order employed (as in FE_DGQ),
                  not the same of lexographic order employed in dpo!!! ----*/

    /*--- Employ lexographic order. ---*/
    // Start with vertices
    /*if(degree_x == 0 && degree_z == 0) {
      this->unit_support_points.emplace_back(0.5, 0.5);
    }
    else if(degree_x == 0) {
      this->unit_support_points.emplace_back(0.5, 0.0);
      this->unit_support_points.emplace_back(0.5, 1.0);
    }
    else if(degree_z == 0) {
      this->unit_support_points.emplace_back(0.0, 0.5);
      this->unit_support_points.emplace_back(0.0, 0.5);
    }
    else {
      this->unit_support_points.emplace_back(0.0, 0.0);
      this->unit_support_points.emplace_back(1.0, 0.0);
      this->unit_support_points.emplace_back(1.0, 1.0);
      this->unit_support_points.emplace_back(0.0, 1.0);
    }

    // Focus now on edge points (which are not vertices)
    // Bottom edge (z=0)
    for(unsigned i = 1; i < degree_x; ++i) {
      this->unit_support_points.emplace_back(static_cast<double>(i)/degree_x, 0.0);
    }
    // Right edge (x=1)
    for(unsigned j = 1; j < degree_z; ++j) {
      this->unit_support_points.emplace_back(1.0, static_cast<double>(j)/degree_z);
    }
    // Top edge (z=1)
    for(unsigned i = 1; i < degree_x; ++i) {
      this->unit_support_points.emplace_back(static_cast<double>(i)/degree_x, 1.0);
    }
    // Left edge (x=0)
    for(unsigned j = 1; j < degree_z; ++j)
      this->unit_support_points.emplace_back(0.0, static_cast<double>(j)/degree_z);

    // Focus now on internal points
    for(unsigned j = 1; j < degree_z; ++j) {
      for(unsigned i = 1; i < degree_x; ++i) {
        this->unit_support_points.emplace_back(static_cast<double>(i)/degree_x,
                                               static_cast<double>(j)/degree_z);
      }
    }*/
  }

// Auxiliary function necessary to initalize the dofs per object
//
template<int dim>
internal::GenericDoFsPerObject
FE_Anisotropic<dim>::build_dpo(const unsigned degree_x,
                               const unsigned degree_z) {
  internal::GenericDoFsPerObject dpo;
  dpo.dofs_per_object_exclusive.resize(4); /*--- In 2D, 0: vertices, 1: edges, 2: faces ---*/
  dpo.dofs_per_object_inclusive.resize(4);
  dpo.object_index.resize(4);
  dpo.first_object_index_on_face.resize(3);

  /*--- Focus on dofs_per_object_exclusive and dofs_per_object_inclusive ---*/
  // Vertices (4 in 2D)
  /*dpo.dofs_per_object_exclusive[0] = {1, 1, 1, 1};
  dpo.dofs_per_object_inclusive[0] = {1, 1, 1, 1};*/
  dpo.dofs_per_object_exclusive[0] = {0, 0, 0, 0};
  dpo.dofs_per_object_inclusive[0] = {0, 0, 0, 0};

  // Edges
  // edge 0 = bottom, edge 2 = top --> horizontal -> degree_x
  // edge 1 = right, edge 3 = left --> vertical -> degree_z
  /*unsigned e_h = (degree_x >= 2 ? degree_x - 1 : 0);
  unsigned e_v = (degree_z >= 2 ? degree_z - 1 : 0);
  dpo.dofs_per_object_exclusive[1] = {e_h, e_v, e_h, e_v};
  dpo.dofs_per_object_inclusive[1] = {degree_x + 1, degree_z + 1, degree_x + 1, degree_z + 1};*/
  dpo.dofs_per_object_exclusive[1] = {0, 0, 0, 0};
  dpo.dofs_per_object_inclusive[1] = {0, 0, 0, 0};

  // Inner faces
  /*dpo.dofs_per_object_exclusive[2] = {(degree_x >= 2 && degree_z >= 2) ?
                                      (degree_x - 1)*(degree_z - 1) : 0};*/
  dpo.dofs_per_object_exclusive[2] = {(degree_x + 1)*(degree_z + 1)};
  dpo.dofs_per_object_inclusive[2] = {(degree_x + 1)*(degree_z + 1)};

  // Inner volumes (none in 2D)
  dpo.dofs_per_object_exclusive[3] = {0};
  dpo.dofs_per_object_inclusive[3] = {0};

  /*--- Focus on object_index. This should be the global index of my geoemtric entity,
        i.e., first vertices, then edgses and so on...
        However not fully coherent with what I say in fe_data.cc ---*/
  /*dpo.object_index[0] = {0, 1, 2, 3};
  dpo.object_index[1] = {4, 5, 6, 7};
  dpo.object_index[2] = {8};*/
  dpo.object_index[0] = {0};
  dpo.object_index[1] = {0};
  dpo.object_index[2] = {0};
  dpo.object_index[3] = {0};

  dpo.first_object_index_on_face[0] = {0};
  dpo.first_object_index_on_face[1] = {0};
  dpo.first_object_index_on_face[2] = {0};
  /*dpo.first_object_index_on_face[1] = {4};
  dpo.first_object_index_on_face[2] = {8};*/

  return dpo;
}

// Auxiliary function necessary to initalize the dofs per object (in the case of same degree)
//
template<int dim>
std::vector<unsigned>
FE_Anisotropic<dim>::build_dpo(const unsigned degree) {
  std::vector<unsigned> dpo(dim + 1, 0);
  dpo[dim] = degree + 1;
  for(unsigned int i = 1; i < dim; ++i) {
    dpo[dim] *= degree + 1;
  }

  return dpo;
}

// Count the number of dofs per cell
//
/*template<int dim>
unsigned FE_Anisotropic<dim>::count_dofs_per_cell_from_degrees(const unsigned degree_x,
                                                               const unsigned degree_z) {
  const auto dpo = this->build_dpo(degree_x, degree_z);

  unsigned dofs_per_cell = 0;
  for(const auto& vec: dpo.dofs_per_object_exclusive) {
    for(unsigned n : vec) {
      dofs_per_cell += n;
    }
  }

  return dofs_per_cell;
}*/

// Set name of the function (pure virtual in FiniteElement)
//
template<int dim>
std::string FE_Anisotropic<dim>::get_name() const {
  return "FE_Anisotropic";
}

// Clone function (pure virtual in FiniteElement)
//
template<int dim>
std::unique_ptr<FiniteElement<dim>>
FE_Anisotropic<dim>::clone() const {
  return std::make_unique<FE_Anisotropic<dim>>(*this);
}
