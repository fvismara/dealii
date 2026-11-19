// ---------------------------------------------------------------------------
//  Matrix-free DG-Laguerre solver for the Euler equations with reaction,
//  following 'laguerre_systems_2.pdf':
//
//      dU/dt + div F(U) = b(U, x, z, t)     in  Omega = Omega_b x (0, inf)
//      U -> U_inf                           as  z -> inf
//
//  in conservative variables U = (rho, rho u, E), u in R^d, d = dim_b + 1,
//  with the source
//      b = ( -g1 (rho - rho_inf),
//             rho g - g2 .* (rho u - rho_inf u_inf),
//             rho u . g - g3 (E - E_inf) )
//  and reaction coefficients gamma = (g1, g2, g3)(x, z, t).
//
//  Discretization ("column approach"):
//   * Omega_b = (0,1)^dim_b is meshed once; ONE MatrixFree<dim_b> object.
//   * z is discretized by a 1D space shared by all columns:
//       - N_z intervals of [0, L_z], orthonormal Legendre polynomials of
//         degree p_z^b on each (DG in z),
//       - the tail [L_z, inf) with Laguerre functions
//         exp(-mu (z-L_z)/2) L_i(mu (z-L_z)), i = 0..p_z.
//   * The unknown is the perturbation W = V - V_inf of the variables
//       V = U = (rho, rho u, E)   (Variables::conservative, default), or
//       V = w = (rho, u, E)       (Variables::primitive, as in
//                                  laguerre_systems.cc),
//     expanded as
//       W(x,z,t) = sum_j W_j(x,t) phi^z_j(z),  W_j in [DG_Q_{p_b}(Omega_b)]^c.
//     Fluxes and sources are always evaluated on the conservative state,
//     U = U(V_inf + W), pointwise at the quadrature points.
//   * Time derivative. Conservative variables: M dW/dt = R(W) with the
//     diagonal-in-z mass M = M_x (x) M_z. Primitive variables: the equations
//     for rho and E are the same; the momentum equation reads
//       d/dt int (rho u) phi_i dz = R^m_i,  i.e.
//       sum_k [int rho phi_i phi_k] u'_k = R^m_i - int rho' u phi_i,
//     (' = d/dt) with rho' known from the density equation. Every z-mode lives
//     on one segment, so the rho-weighted matrix couples only the modes of one
//     segment of one column. Its rho-weight is evaluated (collocated) at the
//     p+1 Gauss points in x, at which the cell mass matrix is exact; then it is
//     a dense SPD system of size n_modes(segment) per x-point and segment,
//     solved by Cholesky. (laguerre_systems.cc uses Gauss(p+3) in x and a CG
//     method preconditioned by exactly this collocated operator.) In z, the
//     rho_inf part is the exact diagonal mass and the rho - rho_inf part uses
//     the segment's z-quadrature.
//   * Storage: BlockVector with c * n_z blocks; block j*c + m holds
//     component m of z-mode j. FEEvaluation with n_components = c reads one
//     z-mode at a time via read_dof_values(src, j*c).
//   * Cell kernel (on Omega_b cells): volume terms of all z-segments plus
//     the "horizontal" faces z = z_l (incl. the interface at L_z and the
//     bottom z = 0), which are local to each column.
//   * Face kernels (on Omega_b faces): the "vertical" faces, integrated
//     over all of [0, inf) with the z-quadrature.
//   * Local Lax-Friedrichs flux everywhere; explicit low-storage RK (5,4).
//   * Time step: dt = courant / max over (x_q, z_p) of the wave speeds times
//     the spectral norm (largest singular value) of the 1D DG/Laguerre
//     advection operators, recomputed every 10 steps.
//   * Reaction coefficients gamma(x, z, t) = constant + sigmoidal damping
//     layer  dgamma / (1 + exp((alpha - z) / sigma)).
//
//  Tests: free-stream preservation and convergence with a manufactured
//  solution (as before), and an isentropic vortex advected along x (exact
//  solution, convergence) or along +z into the damping layer (long-time,
//  qualitative; with and without damping, resolved and under-resolved in z).
//
//  dim_b = 1 gives d = 2, dim_b = 2 gives d = 3.
// ---------------------------------------------------------------------------

#include <deal.II/base/aligned_vector.h>
#include <deal.II/base/mpi.h>
#include <deal.II/base/point.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/tensor.h>
#include <deal.II/base/time_stepping.h>
#include <deal.II/base/timer.h>
#include <deal.II/base/vectorization.h>

#include <deal.II/dofs/dof_handler.h>

#include <deal.II/fe/fe_dgq.h>
#include <deal.II/fe/mapping_q.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/tria.h>

#include <deal.II/lac/affine_constraints.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/la_parallel_block_vector.h>
#include <deal.II/lac/la_parallel_vector.h>
#include <deal.II/lac/lapack_full_matrix.h>

#include <deal.II/matrix_free/fe_evaluation.h>
#include <deal.II/matrix_free/matrix_free.h>
#include <deal.II/matrix_free/operators.h>

#include <array>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <vector>

using namespace dealii;

constexpr double gamma_gas = 1.4;


// ===========================================================================
//  1D discretization in z: DG-Legendre on [0, L_z] + Laguerre tail
// ===========================================================================

// Gauss-Laguerre rule for the weight exp(-y): nodes y_q and scaled weights
// w_q exp(y_q), so that  int_0^inf g(y) dy ~= sum_q wscaled_q g(y_q).
void compute_gauss_laguerre(const unsigned int   n,
                            std::vector<double> &nodes,
                            std::vector<double> &scaled_weights)
{
  AssertThrow(n >= 1 && n <= 150,
              ExcMessage("Gauss-Laguerre rule implemented for 1 <= n <= 150"));
  nodes.resize(n);
  scaled_weights.resize(n);
  double z = 0.;
  for (unsigned int i = 0; i < n; ++i)
    {
      if (i == 0)
        z = 3.0 / (1.0 + 2.4 * n);
      else if (i == 1)
        z += 15.0 / (1.0 + 2.5 * n);
      else
        {
          const double ai = i - 1;
          z += (1.0 + 2.55 * ai) / (1.9 * ai) * (z - nodes[i - 2]);
        }
      double p1 = 0., p2 = 0., pp = 0.;
      for (unsigned int it = 0; it < 200; ++it)
        {
          p1 = 1.0;
          p2 = 0.0;
          for (unsigned int j = 1; j <= n; ++j)
            {
              const double p3 = p2;
              p2              = p1;
              p1 = ((2.0 * j - 1.0 - z) * p2 - (j - 1.0) * p3) / j;
            }
          pp              = n * (p1 - p2) / z;
          const double z1 = z;
          z               = z1 - p1 / pp;
          if (std::abs(z - z1) <= 1e-15 * std::max(1.0, std::abs(z)))
            break;
        }
      nodes[i]          = z;
      scaled_weights[i] = -std::exp(z) / (pp * n * p2);
    }
}

// Legendre polynomials P_0..P_n and derivatives at t in [-1,1]
void legendre(const double         t,
              const unsigned int   n,
              std::vector<double> &P,
              std::vector<double> &dP)
{
  P.assign(n + 1, 0.);
  dP.assign(n + 1, 0.);
  P[0] = 1.;
  if (n >= 1)
    {
      P[1]  = t;
      dP[1] = 1.;
    }
  for (unsigned int k = 1; k < n; ++k)
    {
      P[k + 1]  = ((2. * k + 1.) * t * P[k] - k * P[k - 1]) / (k + 1.);
      dP[k + 1] = dP[k - 1] + (2. * k + 1.) * P[k];
    }
}

// Laguerre polynomials L_0..L_n and derivatives at t >= 0
void laguerre(const double         t,
              const unsigned int   n,
              std::vector<double> &L,
              std::vector<double> &dL)
{
  L.assign(n + 1, 0.);
  dL.assign(n + 1, 0.);
  L[0] = 1.;
  if (n >= 1)
    {
      L[1]  = 1. - t;
      dL[1] = -1.;
    }
  for (unsigned int k = 1; k < n; ++k)
    {
      L[k + 1]  = ((2. * k + 1. - t) * L[k] - k * L[k - 1]) / (k + 1.);
      dL[k + 1] = ((2. * k + 1. - t) * dL[k] - L[k] - k * dL[k - 1]) / (k + 1.);
    }
}


class ZDiscretization
{
public:
  // A segment is either an interval of [0, L_z] or the tail [L_z, inf).
  // Each z-mode lives on exactly one segment.
  struct Segment
  {
    bool                is_tail = false;
    double              z_lo = 0., z_hi = 0.;
    unsigned int        first_mode = 0, n_modes = 0;
    std::vector<double> points, weights;     // physical quadrature in z
    FullMatrix<double>  values, derivatives; // (local mode, point)
    std::vector<double> trace_lo, trace_hi;  // values at z_lo^+, z_hi^-
    double              mass = 0.;           // diagonal mass entry
    double              rate = 0.;           // 1D upwind operator norm
  };

  ZDiscretization(const double       L_z,
                  const unsigned int N_z,
                  const unsigned int degree_intervals,
                  const unsigned int degree_tail,
                  const double       mu,
                  const unsigned int n_q_intervals,
                  const unsigned int n_q_tail)
    : L_z(L_z)
    , N_z(N_z)
    , degree_intervals(degree_intervals)
    , degree_tail(degree_tail)
    , mu(mu)
  {
    unsigned int        mode = 0;
    std::vector<double> P, dP, L, dL;

    // intervals: phi_i(z) = sqrt(2i+1) P_i(2(z - z_lo)/h - 1),
    // int_I phi_i phi_j dz = h delta_ij
    const double    h = (N_z > 0 ? L_z / N_z : 0.);
    const QGauss<1> gauss(n_q_intervals);
    for (unsigned int l = 0; l < N_z; ++l)
      {
        Segment s;
        s.z_lo       = l * h;
        s.z_hi       = (l + 1) * h;
        s.first_mode = mode;
        s.n_modes    = degree_intervals + 1;
        s.mass       = h;
        s.values.reinit(s.n_modes, gauss.size());
        s.derivatives.reinit(s.n_modes, gauss.size());
        for (unsigned int q = 0; q < gauss.size(); ++q)
          {
            const double xi = gauss.point(q)[0];
            s.points.push_back(s.z_lo + h * xi);
            s.weights.push_back(h * gauss.weight(q));
            legendre(2. * xi - 1., degree_intervals, P, dP);
            for (unsigned int i = 0; i < s.n_modes; ++i)
              {
                const double c       = std::sqrt(2. * i + 1.);
                s.values(i, q)       = c * P[i];
                s.derivatives(i, q)  = c * dP[i] * 2. / h;
              }
          }
        for (unsigned int i = 0; i < s.n_modes; ++i)
          {
            const double c = std::sqrt(2. * i + 1.);
            s.trace_lo.push_back(c * (i % 2 == 0 ? 1. : -1.));
            s.trace_hi.push_back(c);
          }
        mode += s.n_modes;
        segments.push_back(s);
      }

    // tail: phi_i(z) = exp(-y) L_i(2y),  y = mu (z - L_z)/2,
    // int phi_i phi_j dz = delta_ij / mu, and phi_i(L_z) = 1
    {
      Segment s;
      s.is_tail    = true;
      s.z_lo       = L_z;
      s.z_hi       = std::numeric_limits<double>::infinity();
      s.first_mode = mode;
      s.n_modes    = degree_tail + 1;
      s.mass       = 1. / mu;
      std::vector<double> y, w;
      compute_gauss_laguerre(n_q_tail, y, w);
      s.values.reinit(s.n_modes, y.size());
      s.derivatives.reinit(s.n_modes, y.size());
      for (unsigned int q = 0; q < y.size(); ++q)
        {
          s.points.push_back(L_z + 2. * y[q] / mu);
          s.weights.push_back(2. / mu * w[q]);
          laguerre(2. * y[q], degree_tail, L, dL);
          const double e = std::exp(-y[q]);
          for (unsigned int i = 0; i < s.n_modes; ++i)
            {
              s.values(i, q)      = e * L[i];
              s.derivatives(i, q) = 0.5 * mu * e * (2. * dL[i] - L[i]);
            }
        }
      s.trace_lo.assign(s.n_modes, 1.);
      mode += s.n_modes;
      segments.push_back(s);
    }

    n_total_modes = mode;
    mass_of_mode.resize(n_total_modes);
    for (auto &s : segments)
      {
        for (unsigned int i = 0; i < s.n_modes; ++i)
          mass_of_mode[s.first_mode + i] = s.mass;
        s.rate = compute_rate(s);
      }
  }

  unsigned int                n_modes() const { return n_total_modes; }

  // Values of the modes of the segment containing z (local numbering);
  // returns the index of that segment.
  unsigned int evaluate(const double z, std::vector<double> &values) const
  {
    std::vector<double> P, dP;
    const double        h = (N_z > 0 ? L_z / N_z : 0.);
    if (z < L_z && N_z > 0)
      {
        const unsigned int l  = std::min(N_z - 1, static_cast<unsigned int>(z / h));
        const double       xi = (z - l * h) / h;
        legendre(2. * xi - 1., degree_intervals, P, dP);
        values.resize(degree_intervals + 1);
        for (unsigned int i = 0; i <= degree_intervals; ++i)
          values[i] = std::sqrt(2. * i + 1.) * P[i];
        return l;
      }
    const double y = 0.5 * mu * (z - L_z);
    laguerre(2. * y, degree_tail, P, dP);
    values.resize(degree_tail + 1);
    for (unsigned int i = 0; i <= degree_tail; ++i)
      values[i] = std::exp(-y) * P[i];
    return segments.size() - 1;
  }
  double                      mass(const unsigned int j) const { return mass_of_mode[j]; }
  const std::vector<Segment> &get_segments() const { return segments; }

  // Spectral norm (largest singular value) of the 1D upwind DG advection
  // operator (unit speed, both directions) on a segment, divided by the mass,
  // used for the CFL estimate. (The Frobenius norm used before overestimates
  // it by a factor that grows with the number of modes.)
  static double compute_rate(const Segment &s)
  {
    const unsigned int n = s.n_modes;
    FullMatrix<double> Kp(n, n), Km(n, n);
    for (unsigned int i = 0; i < n; ++i)
      for (unsigned int j = 0; j < n; ++j)
        {
          double a = 0;
          for (unsigned int p = 0; p < s.points.size(); ++p)
            a += s.weights[p] * s.derivatives(i, p) * s.values(j, p);
          Kp(i, j) = -a + (s.is_tail ? 0. : s.trace_hi[i] * s.trace_hi[j]);
          Km(i, j) = a + s.trace_lo[i] * s.trace_lo[j];
        }
    const auto spectral_norm = [n](const FullMatrix<double> &K) {
      LAPACKFullMatrix<double> A(n, n);
      A = K;
      A.compute_svd();
      return A.singular_value(0);
    };
    return std::max(spectral_norm(Kp), spectral_norm(Km)) / s.mass;
  }

  // same quantity for a DG interval of length h and given degree
  // (used for the x-direction)
  static double interval_rate(const unsigned int degree, const double h)
  {
    const ZDiscretization tmp(h, 1, degree, 0, 1., degree + 2, 2);
    return tmp.get_segments()[0].rate;
  }

private:
  double               L_z;
  unsigned int         N_z, degree_intervals, degree_tail;
  double               mu;
  std::vector<Segment> segments;
  unsigned int         n_total_modes = 0;
  std::vector<double>  mass_of_mode;
};


// ===========================================================================
//  Euler physics (templated on double or VectorizedArray<double>)
// ===========================================================================

template <int n_components, int dim, typename T>
inline Tensor<1, n_components, T>
contract_normal(const Tensor<1, n_components, Tensor<1, dim, T>> &F,
                const Tensor<1, dim, T>                           &n)
{
  Tensor<1, n_components, T> r;
  for (unsigned int c = 0; c < n_components; ++c)
    {
      r[c] = F[c][0] * n[0];
      for (unsigned int d = 1; d < dim; ++d)
        r[c] += F[c][d] * n[d];
    }
  return r;
}

template <int dim, typename T>
inline T euler_pressure(const Tensor<1, dim + 2, T> &U)
{
  T kinetic = U[1] * U[1];
  for (unsigned int d = 1; d < dim; ++d)
    kinetic += U[1 + d] * U[1 + d];
  return (gamma_gas - 1.) * (U[dim + 1] - 0.5 * kinetic / U[0]);
}

template <int dim, typename T>
inline Tensor<1, dim + 2, Tensor<1, dim, T>>
euler_flux(const Tensor<1, dim + 2, T> &U)
{
  const T           inv_rho = 1. / U[0];
  Tensor<1, dim, T> u;
  for (unsigned int d = 0; d < dim; ++d)
    u[d] = U[1 + d] * inv_rho;
  const T p = euler_pressure<dim>(U);

  Tensor<1, dim + 2, Tensor<1, dim, T>> F;
  for (unsigned int d = 0; d < dim; ++d)
    {
      F[0][d] = U[1 + d];
      for (unsigned int e = 0; e < dim; ++e)
        F[1 + e][d] = U[1 + e] * u[d];
      F[1 + d][d] += p;
      F[dim + 1][d] = u[d] * (U[dim + 1] + p);
    }
  return F;
}

template <int dim, typename T>
inline T max_normal_wave_speed(const Tensor<1, dim + 2, T> &U,
                               const Tensor<1, dim, T>     &n)
{
  T un = U[1] * n[0];
  for (unsigned int d = 1; d < dim; ++d)
    un += U[1 + d] * n[d];
  un        = un / U[0];
  const T p = euler_pressure<dim>(U);
  return std::abs(un) + std::sqrt(gamma_gas * p / U[0]);
}

// local Lax-Friedrichs flux F^(U-, U+) n, with n pointing from - to +
template <int dim, typename T>
inline Tensor<1, dim + 2, T>
lax_friedrichs_flux(const Tensor<1, dim + 2, T> &Um,
                    const Tensor<1, dim + 2, T> &Up,
                    const Tensor<1, dim, T>     &n)
{
  const auto Fm     = euler_flux<dim>(Um);
  const auto Fp     = euler_flux<dim>(Up);
  const T    lambda = std::max(max_normal_wave_speed<dim>(Um, n),
                            max_normal_wave_speed<dim>(Up, n));
  return 0.5 * (contract_normal(Fm, n) + contract_normal(Fp, n)) +
         0.5 * lambda * (Um - Up);
}


// ===========================================================================
//  Problem data: far field, reaction, gravity, boundary conditions and a
//  manufactured solution for verification
// ===========================================================================

enum class Variables
{
  conservative, // (rho, rho u, E)
  primitive     // (rho, u, E), as in laguerre_systems.cc
};

// (rho, u, E) <-> (rho, rho u, E), for double or VectorizedArray
template <int dim, typename T>
inline Tensor<1, dim + 2, T> primitive_to_conservative(const Tensor<1, dim + 2, T> &w)
{
  Tensor<1, dim + 2, T> U = w;
  for (unsigned int d = 0; d < dim; ++d)
    U[1 + d] = w[0] * w[1 + d];
  return U;
}

template <int dim, typename T>
inline Tensor<1, dim + 2, T> conservative_to_primitive(const Tensor<1, dim + 2, T> &U)
{
  Tensor<1, dim + 2, T> w   = U;
  const T               inv = 1. / U[0];
  for (unsigned int d = 0; d < dim; ++d)
    w[1 + d] = U[1 + d] * inv;
  return w;
}

enum class InitialData
{
  far_field,        // W = 0 (free-stream test), Dirichlet data U_inf
  manufactured,     // manufactured solution with forcing
  isentropic_vortex // exact solution without source (if gamma = g = 0)
};

enum class BoundaryType
{
  dirichlet, // exterior state = Dirichlet data (U_inf, or the exact solution)
  wall       // slip wall: reflect the normal momentum
};

template <int dim>
struct ProblemData
{
  static constexpr int dim_b  = dim - 1;
  static constexpr int n_comp = dim + 2;

  Tensor<1, n_comp> U_inf;
  Tensor<1, dim>    gravity;
  Tensor<1, n_comp> reaction; // constant part of (gamma_1, gamma_2, gamma_3)
  InitialData       initial = InitialData::far_field;
  bool              manufactured = false; // = (initial == manufactured)

  // damping layer added to all components:
  //   dgamma / (1 + exp((alpha - z) / sigma))
  double damping_dgamma = 0., damping_alpha = 0., damping_sigma = 1.;

  // isentropic vortex: center (x_c, z_c) at t = 0, advection velocity
  // a = a_x e_0 + a_z e_z (= far-field velocity)
  double vortex_x_c = 0., vortex_z_c = 0., vortex_a_x = 0., vortex_a_z = 0.;
  BoundaryType      lateral_bc   = BoundaryType::dirichlet;
  BoundaryType      bottom_bc    = BoundaryType::dirichlet;

  ProblemData()
  {
    const double   rho = 1., p = 1.;
    Tensor<1, dim> u;
    u[0]     = 0.5;
    U_inf[0] = rho;
    for (unsigned int d = 0; d < dim; ++d)
      U_inf[1 + d] = rho * u[d];
    U_inf[dim + 1] = p / (gamma_gas - 1.) + 0.5 * rho * u.norm_square();
    for (unsigned int c = 0; c < n_comp; ++c)
      reaction[c] = 0.5;
  }

  // Isentropic vortex advected with velocity (a_x, a_z); no reaction, no
  // gravity unless set afterwards.
  void set_isentropic_vortex(const double x_c,
                             const double z_c,
                             const double a_x,
                             const double a_z)
  {
    initial      = InitialData::isentropic_vortex;
    manufactured = false;
    vortex_x_c   = x_c;
    vortex_z_c   = z_c;
    vortex_a_x   = a_x;
    vortex_a_z   = a_z;
    gravity      = Tensor<1, dim>();
    reaction     = Tensor<1, n_comp>();
    U_inf        = Tensor<1, n_comp>();
    U_inf[0]     = 1.;
    U_inf[1]     = a_x;
    U_inf[dim]   = a_z;
    U_inf[dim + 1] = 1. / (gamma_gas - 1.) + 0.5 * (a_x * a_x + a_z * a_z);
  }

  void set_damping_layer(const double dgamma, const double alpha, const double sigma)
  {
    damping_dgamma = dgamma;
    damping_alpha  = alpha;
    damping_sigma  = sigma;
  }

  double damping(const double z) const
  {
    if (damping_dgamma == 0.)
      return 0.;
    // exp overflows to inf far below the layer: damping -> 0, as it should
    return damping_dgamma / (1. + std::exp((damping_alpha - z) / damping_sigma));
  }

  double max_reaction() const
  {
    double m = 0.;
    for (unsigned int c = 0; c < n_comp; ++c)
      m = std::max(m, reaction[c]);
    return m + damping_dgamma;
  }

  // gamma(x, z, t) = constant + damping layer in z
  template <typename T>
  Tensor<1, n_comp, T> reaction_coefficients(const Point<dim_b, T> &,
                                             const double z,
                                             const double) const
  {
    const double         layer = damping(z);
    Tensor<1, n_comp, T> g;
    for (unsigned int c = 0; c < n_comp; ++c)
      g[c] = reaction[c] + layer;
    return g;
  }

  template <typename T>
  Tensor<1, n_comp, T> source(const Tensor<1, n_comp, T> &U,
                              const Tensor<1, n_comp, T> &gam) const
  {
    Tensor<1, n_comp, T> b;
    b[0]       = -gam[0] * (U[0] - U_inf[0]);
    T m_dot_g  = U[1] * gravity[0];
    for (unsigned int d = 0; d < dim; ++d)
      {
        b[1 + d] = U[0] * gravity[d] - gam[1 + d] * (U[1 + d] - U_inf[1 + d]);
        if (d > 0)
          m_dot_g += U[1 + d] * gravity[d];
      }
    b[dim + 1] = m_dot_g - gam[dim + 1] * (U[dim + 1] - U_inf[dim + 1]);
    return b;
  }

  // Smooth perturbation of U_inf decaying like exp(-z) (not exactly
  // representable in the Laguerre tail because of the sin(z) factor).
  Tensor<1, n_comp> exact_solution(const Point<dim_b> &x,
                                   const double        z,
                                   const double        t) const
  {
    if (initial == InitialData::isentropic_vortex)
      return vortex_solution(x, z, t);
    const double two_pi     = 2. * numbers::PI;
    double       transverse = 1.;
    for (unsigned int k = 1; k < dim_b; ++k)
      transverse *= 1. + 0.5 * std::sin(two_pi * x[k]);
    const double D  = std::exp(-z) * (1. + 0.5 * std::sin(z)) * transverse;
    const double D2 = z * std::exp(-z) * transverse;

    Tensor<1, n_comp> U = U_inf;
    U[0] += 0.1 * std::sin(two_pi * x[0] - t) * D;
    for (unsigned int k = 0; k < dim_b; ++k)
      U[1 + k] += 0.1 * std::cos(two_pi * x[0] + t + k) * D;
    U[dim] += 0.1 * std::sin(two_pi * x[0]) * std::cos(t) * D2;
    U[dim + 1] += 0.2 * std::cos(two_pi * x[0] - t) * D;
    return U;
  }

  // Isentropic vortex of step-67 in the (x_0, z) plane (uniform in the
  // other directions); exact solution of the Euler equations without source
  Tensor<1, n_comp> vortex_solution(const Point<dim_b> &x,
                                    const double        z,
                                    const double        t) const
  {
    const double beta = 5.;
    const double dx   = x[0] - vortex_x_c - vortex_a_x * t;
    const double dz   = z - vortex_z_c - vortex_a_z * t;
    const double factor =
      beta / (2. * numbers::PI) * std::exp(1. - dx * dx - dz * dz);
    const double density_log = std::log2(
      std::abs(1. - (gamma_gas - 1.) / gamma_gas * 0.25 * factor * factor));
    const double rho = std::exp2(density_log / (gamma_gas - 1.));
    const double p   = std::exp2(density_log * gamma_gas / (gamma_gas - 1.));
    const double u   = vortex_a_x - factor * dz;
    const double v   = vortex_a_z + factor * dx;
    Tensor<1, n_comp> U;
    U[0]           = rho;
    U[1]           = rho * u;
    U[dim]         = rho * v;
    U[dim + 1]     = p / (gamma_gas - 1.) + 0.5 * rho * (u * u + v * v);
    return U;
  }

  // s = dU/dt + div F(U) - b(U), by 4th-order central differences, so that
  // exact_solution solves dU/dt + div F(U) = b(U) + s
  Tensor<1, n_comp> manufactured_forcing(const Point<dim_b> &x,
                                         const double        z,
                                         const double        t) const
  {
    const double h  = 1e-3;
    auto         d4 = [h](const auto &f) {
      return (-f(2. * h) + 8. * f(h) - 8. * f(-h) + f(-2. * h)) / (12. * h);
    };
    auto flux_column = [](const Tensor<1, n_comp> &U, const unsigned int k) {
      const auto        F = euler_flux<dim>(U);
      Tensor<1, n_comp> f;
      for (unsigned int c = 0; c < n_comp; ++c)
        f[c] = F[c][k];
      return f;
    };

    Tensor<1, n_comp> r =
      d4([&](const double s) { return exact_solution(x, z, t + s); });
    for (unsigned int k = 0; k < dim_b; ++k)
      r += d4([&](const double s) {
        Point<dim_b> xs = x;
        xs[k] += s;
        return flux_column(exact_solution(xs, z, t), k);
      });
    r += d4([&](const double s) {
      return flux_column(exact_solution(x, z + s, t), dim - 1);
    });
    r -= source(exact_solution(x, z, t), reaction_coefficients(x, z, t));
    return r;
  }

  Tensor<1, n_comp> dirichlet_value(const Point<dim_b> &x,
                                    const double        z,
                                    const double        t) const
  {
    return initial == InitialData::far_field ? U_inf : exact_solution(x, z, t);
  }
};


// ===========================================================================
//  The matrix-free column operator
// ===========================================================================

template <int dim_b, int degree>
class EulerColumnOperator
{
public:
  static constexpr int dim    = dim_b + 1;
  static constexpr int n_comp = dim + 2;
  static constexpr int n_q_1d = degree + 2; // over-integration as in step-67
  using Number                = double;
  using VA                    = VectorizedArray<Number>;
  using State                 = Tensor<1, n_comp, VA>;
  using FluxX                 = Tensor<1, n_comp, Tensor<1, dim_b, VA>>;
  using BlockVectorType       = LinearAlgebra::distributed::BlockVector<Number>;
  using Segment               = ZDiscretization::Segment;

  EulerColumnOperator(const MatrixFree<dim_b, Number> &matrix_free,
                      const ZDiscretization           &z,
                      const ProblemData<dim>          &problem,
                      const Variables variables = Variables::conservative)
    : mf(matrix_free)
    , z(z)
    , problem(problem)
    , variables(variables)
  {
    for (unsigned int c = 0; c < n_comp; ++c)
      U_inf[c] = problem.U_inf[c];
    // base state of the expansion: U_inf or w_inf
    const Tensor<1, n_comp> V = primitive() ?
                                  conservative_to_primitive<dim>(problem.U_inf) :
                                  problem.U_inf;
    for (unsigned int c = 0; c < n_comp; ++c)
      V_inf[c] = V[c];
    e_z[dim - 1] = 1.;

    TimeStepping::LowStorageRungeKutta<LinearAlgebra::distributed::Vector<Number>>
      lsrk(TimeStepping::LOW_STORAGE_RK_STAGE5_ORDER4);
    lsrk.get_coefficients(rk_a, rk_b, rk_c);
  }

  void initialize_vector(BlockVectorType &v) const
  {
    v.reinit(n_comp * z.n_modes());
    for (unsigned int b = 0; b < v.n_blocks(); ++b)
      mf.initialize_dof_vector(v.block(b));
    v.collect_sizes();
  }

  // dst = R(src): right-hand side of  M dW/dt = R(W)
  void apply(const double t, const BlockVectorType &src, BlockVectorType &dst) const
  {
    time = t;
    mf.loop(&EulerColumnOperator::local_cell,
            &EulerColumnOperator::local_face,
            &EulerColumnOperator::local_boundary,
            this,
            dst,
            src,
            true,
            MatrixFree<dim_b, Number>::DataAccessOnFaces::values,
            MatrixFree<dim_b, Number>::DataAccessOnFaces::values);
  }

  bool primitive() const { return variables == Variables::primitive; }

  // dst = (M_x (x) M_z)^{-1} src: cell-wise inverse in x, diagonal in z
  void apply_inverse_mass(const BlockVectorType &src, BlockVectorType &dst) const
  {
    mf.cell_loop(&EulerColumnOperator::local_inverse_mass, this, dst, src);
  }

  // dst = dW/dt from the residual rhs = R(state): the inverse mass for
  // conservative variables, the rho-weighted momentum solve for primitive
  // variables (see the header)
  void compute_rates(const BlockVectorType &state,
                     const BlockVectorType &rhs,
                     BlockVectorType       &dst) const
  {
    if (!primitive())
      {
        apply_inverse_mass(rhs, dst);
        return;
      }
    current_state = &state;
    mf.cell_loop(&EulerColumnOperator::local_primitive_rates, this, dst, rhs);
    current_state = nullptr;
  }

  // one step of the low-storage RK scheme (5 stages, order 4), as in step-67
  void time_step(const double     t,
                 const double     dt,
                 BlockVectorType &solution,
                 BlockVectorType &ri,
                 BlockVectorType &ki,
                 BlockVectorType &rhs) const
  {
    ri = solution;
    for (unsigned int s = 0; s < rk_b.size(); ++s)
      {
        apply(t + rk_c[s] * dt, ri, rhs);
        compute_rates(ri, rhs, ki);
        if (s + 1 < rk_b.size())
          {
            ri = solution;
            ri.add(rk_a[s] * dt, ki);
          }
        solution.add(rk_b[s] * dt, ki);
      }
  }

  // L2 projection of exact_solution(t) - V_inf (in the chosen variables),
  // using the (finer) z-rule
  // of z_fine, which must have the same modes as the operator's z.
  void project_exact(const double           t,
                     const ZDiscretization &z_fine,
                     BlockVectorType       &dst) const
  {
    BlockVectorType rhs;
    initialize_vector(rhs);
    FEEvaluation<dim_b, degree, n_q_1d, n_comp, Number> phi(mf);
    const unsigned int nq = phi.n_q_points, NZ = z_fine.n_modes();
    AlignedVector<State> acc(NZ * nq);
    for (unsigned int cell = 0; cell < mf.n_cell_batches(); ++cell)
      {
        phi.reinit(cell);
        for (auto &a : acc)
          a = State();
        for (unsigned int q = 0; q < nq; ++q)
          {
            const Point<dim_b, VA> x = phi.quadrature_point(q);
            for (const Segment &s : z_fine.get_segments())
              for (unsigned int p = 0; p < s.points.size(); ++p)
                {
                  State e;
                  for (unsigned int v = 0; v < VA::size(); ++v)
                    {
                      const auto Ue = problem.exact_solution(lane(x, v), s.points[p], t);
                      for (unsigned int c = 0; c < n_comp; ++c)
                        e[c][v] = (primitive() ? conservative_to_primitive<dim>(Ue)[c] :
                                                 Ue[c]) -
                                  V_inf[c][v];
                    }
                  for (unsigned int m = 0; m < s.n_modes; ++m)
                    acc[(s.first_mode + m) * nq + q] +=
                      (s.weights[p] * s.values(m, p)) * e;
                }
          }
        for (unsigned int j = 0; j < NZ; ++j)
          {
            for (unsigned int q = 0; q < nq; ++q)
              phi.submit_value(acc[j * nq + q], q);
            phi.integrate(EvaluationFlags::values);
            phi.distribute_local_to_global(rhs, j * n_comp);
          }
      }
    rhs.compress(VectorOperation::add);
    apply_inverse_mass(rhs, dst);
  }

  // L2 error of U_inf + W against exact_solution(t) over the whole Omega
  double l2_error(const BlockVectorType &solution,
                  const double           t,
                  const ZDiscretization &z_fine) const
  {
    FEEvaluation<dim_b, degree, n_q_1d, n_comp, Number> phi(mf);
    const unsigned int nq = phi.n_q_points, NZ = z_fine.n_modes();
    AlignedVector<State> Uhat(NZ * nq);
    double               error_sq = 0;
    for (unsigned int cell = 0; cell < mf.n_cell_batches(); ++cell)
      {
        phi.reinit(cell);
        read_all_modes(phi, solution, Uhat);
        for (unsigned int q = 0; q < nq; ++q)
          {
            const Point<dim_b, VA> x   = phi.quadrature_point(q);
            const VA               JxW = phi.JxW(q);
            for (const Segment &s : z_fine.get_segments())
              for (unsigned int p = 0; p < s.points.size(); ++p)
                {
                  const State U = reconstruct(Uhat, nq, q, s, p);
                  for (unsigned int v = 0;
                       v < mf.n_active_entries_per_cell_batch(cell);
                       ++v)
                    {
                      const auto Ue =
                        problem.exact_solution(lane(x, v), s.points[p], t);
                      double e2 = 0;
                      for (unsigned int c = 0; c < n_comp; ++c)
                        e2 += Utilities::fixed_power<2>(Ue[c] - U[c][v]);
                      error_sq += JxW[v] * s.weights[p] * e2;
                    }
                }
          }
      }
    return std::sqrt(error_sq);
  }

  // dt = courant / max over all (x_q, z_p) of
  //      sum_k (|u_k| + a) rate_x + (|u_z| + a) rate_z(segment)
  double compute_time_step(const BlockVectorType &solution,
                           const double           courant,
                           const double           rate_x) const
  {
    FEEvaluation<dim_b, degree, n_q_1d, n_comp, Number> phi(mf);
    const unsigned int   nq = phi.n_q_points;
    AlignedVector<State> Uhat(z.n_modes() * nq);
    double               max_rate = 0;
    for (unsigned int cell = 0; cell < mf.n_cell_batches(); ++cell)
      {
        phi.reinit(cell);
        read_all_modes(phi, solution, Uhat);
        for (unsigned int q = 0; q < nq; ++q)
          for (const Segment &s : z.get_segments())
            for (unsigned int p = 0; p < s.points.size(); ++p)
              {
                const State U = reconstruct(Uhat, nq, q, s, p);
                const VA    a = std::sqrt(gamma_gas * euler_pressure<dim>(U) / U[0]);
                VA          r = (std::abs(U[dim] / U[0]) + a) * s.rate;
                for (unsigned int k = 0; k < dim_b; ++k)
                  r += (std::abs(U[1 + k] / U[0]) + a) * rate_x;
                for (unsigned int v = 0;
                     v < mf.n_active_entries_per_cell_batch(cell);
                     ++v)
                  max_rate = std::max(max_rate, r[v]);
              }
      }
    double dt = courant / max_rate;
    // explicit reaction: dt * max(gamma) <= 1 is well inside the stability
    // region of RK(5,4) on the negative real axis
    if (problem.max_reaction() > 0.)
      dt = std::min(dt, 1. / problem.max_reaction());
    return dt;
  }

  // L2(Omega_b x Z) norms of the density and of the momentum of
  // U_inf + W - U_ref, with a z-rule (z_points, z_weights) on Z. U_ref =
  // exact_solution(t) if subtract_exact, else U_inf.
  std::array<double, 2> l2_norms(const BlockVectorType     &solution,
                                 const double               t,
                                 const std::vector<double> &z_points,
                                 const std::vector<double> &z_weights,
                                 const bool                 subtract_exact) const
  {
    FEEvaluation<dim_b, degree, n_q_1d, n_comp, Number> phi(mf);
    const unsigned int   nq = phi.n_q_points;
    AlignedVector<State> Uhat(z.n_modes() * nq);
    // basis values at the z-points
    const auto              &segs = z.get_segments();
    std::vector<unsigned int> seg_of(z_points.size());
    std::vector<std::vector<double>> vals(z_points.size());
    for (unsigned int p = 0; p < z_points.size(); ++p)
      seg_of[p] = z.evaluate(z_points[p], vals[p]);

    std::array<double, 2> sq = {{0., 0.}};
    for (unsigned int cell = 0; cell < mf.n_cell_batches(); ++cell)
      {
        phi.reinit(cell);
        read_all_modes(phi, solution, Uhat);
        for (unsigned int q = 0; q < nq; ++q)
          {
            const Point<dim_b, VA> x   = phi.quadrature_point(q);
            const VA               JxW = phi.JxW(q);
            for (unsigned int p = 0; p < z_points.size(); ++p)
              {
                const Segment &s = segs[seg_of[p]];
                State          V = V_inf;
                for (unsigned int m = 0; m < s.n_modes; ++m)
                  V += vals[p][m] * Uhat[(s.first_mode + m) * nq + q];
                const State U = to_conservative(V);
                for (unsigned int v = 0;
                     v < mf.n_active_entries_per_cell_batch(cell);
                     ++v)
                  {
                    const auto Ur =
                      subtract_exact ?
                        problem.exact_solution(lane(x, v), z_points[p], t) :
                        problem.U_inf;
                    double e_m = 0;
                    for (unsigned int d = 0; d < dim; ++d)
                      e_m += Utilities::fixed_power<2>(U[1 + d][v] - Ur[1 + d]);
                    const double w = JxW[v] * z_weights[p];
                    sq[0] += w * Utilities::fixed_power<2>(U[0][v] - Ur[0]);
                    sq[1] += w * e_m;
                  }
              }
          }
      }
    return {{std::sqrt(sq[0]), std::sqrt(sq[1])}};
  }

private:
  // ---- helpers ------------------------------------------------------------

  static Point<dim_b> lane(const Point<dim_b, VA> &x, const unsigned int v)
  {
    Point<dim_b> xv;
    for (unsigned int k = 0; k < dim_b; ++k)
      xv[k] = x[k][v];
    return xv;
  }

  template <typename Evaluator>
  void read_all_modes(Evaluator             &phi,
                      const BlockVectorType &src,
                      AlignedVector<State>  &Uhat) const
  {
    const unsigned int nq = phi.n_q_points;
    for (unsigned int j = 0; j < z.n_modes(); ++j)
      {
        phi.read_dof_values(src, j * n_comp);
        phi.evaluate(EvaluationFlags::values);
        for (unsigned int q = 0; q < nq; ++q)
          Uhat[j * nq + q] = phi.get_value(q);
      }
  }

  // conservative state from V = V_inf + W
  State to_conservative(const State &V) const
  {
    return primitive() ? primitive_to_conservative<dim>(V) : V;
  }

  // U(x_q, z_p) from V_inf + sum_m phi_m(z_p) W_m(x_q), over the modes of s
  State reconstruct(const AlignedVector<State> &Uhat,
                    const unsigned int          nq,
                    const unsigned int          q,
                    const Segment              &s,
                    const unsigned int          p) const
  {
    State V = V_inf;
    for (unsigned int m = 0; m < s.n_modes; ++m)
      V += s.values(m, p) * Uhat[(s.first_mode + m) * nq + q];
    return to_conservative(V);
  }

  State reconstruct_trace(const AlignedVector<State> &Uhat,
                          const unsigned int          nq,
                          const unsigned int          q,
                          const Segment              &s,
                          const std::vector<double>  &trace) const
  {
    State V = V_inf;
    for (unsigned int m = 0; m < s.n_modes; ++m)
      V += trace[m] * Uhat[(s.first_mode + m) * nq + q];
    return to_conservative(V);
  }

  State dirichlet(const Point<dim_b, VA> &x, const double zc) const
  {
    State U;
    for (unsigned int v = 0; v < VA::size(); ++v)
      {
        const auto Ue = problem.dirichlet_value(lane(x, v), zc, time);
        for (unsigned int c = 0; c < n_comp; ++c)
          U[c][v] = Ue[c];
      }
    return U;
  }

  static State reflect(const State &U, const Tensor<1, dim, VA> &n)
  {
    VA mn = U[1] * n[0];
    for (unsigned int d = 1; d < dim; ++d)
      mn += U[1 + d] * n[d];
    State R = U;
    for (unsigned int d = 0; d < dim; ++d)
      R[1 + d] -= 2. * mn * n[d];
    return R;
  }

  State source_term(const State &U, const Point<dim_b, VA> &x, const double zc) const
  {
    State b = problem.source(U, problem.reaction_coefficients(x, zc, time));
    if (problem.manufactured)
      for (unsigned int v = 0; v < VA::size(); ++v)
        {
          const auto f = problem.manufactured_forcing(lane(x, v), zc, time);
          for (unsigned int c = 0; c < n_comp; ++c)
            b[c][v] += f[c];
        }
    return b;
  }

  // ---- kernels ------------------------------------------------------------

  // Volume terms of all z-segments of a column, plus the horizontal faces.
  //   R_j += int F_b(U) phi_j . grad_x v + (f(U) phi_j' + b phi_j) v
  //   R   -= F^ n phi at horizontal faces
  void local_cell(const MatrixFree<dim_b, Number>             &data,
                  BlockVectorType                             &dst,
                  const BlockVectorType                       &src,
                  const std::pair<unsigned int, unsigned int> &range) const
  {
    FEEvaluation<dim_b, degree, n_q_1d, n_comp, Number> phi(data);
    const unsigned int   nq = phi.n_q_points, NZ = z.n_modes();
    AlignedVector<State> Uhat(NZ * nq), val(NZ * nq);
    AlignedVector<FluxX> grad(NZ * nq);
    const auto          &segs = z.get_segments();

    for (unsigned int cell = range.first; cell < range.second; ++cell)
      {
        phi.reinit(cell);
        read_all_modes(phi, src, Uhat);
        for (unsigned int i = 0; i < NZ * nq; ++i)
          {
            val[i]  = State();
            grad[i] = FluxX();
          }

        for (unsigned int q = 0; q < nq; ++q)
          {
            const Point<dim_b, VA> x = phi.quadrature_point(q);

            // volume integrals over each segment
            for (const Segment &s : segs)
              for (unsigned int p = 0; p < s.points.size(); ++p)
                {
                  const State U = reconstruct(Uhat, nq, q, s, p);
                  const auto  F = euler_flux<dim>(U);
                  const State b = source_term(U, x, s.points[p]);
                  for (unsigned int m = 0; m < s.n_modes; ++m)
                    {
                      const unsigned int i  = (s.first_mode + m) * nq + q;
                      const double       wv = s.weights[p] * s.values(m, p);
                      const double       wd = s.weights[p] * s.derivatives(m, p);
                      for (unsigned int c = 0; c < n_comp; ++c)
                        {
                          for (unsigned int k = 0; k < dim_b; ++k)
                            grad[i][c][k] += wv * F[c][k];
                          val[i][c] += wd * F[c][dim - 1] + wv * b[c];
                        }
                    }
                }

            // horizontal faces: face l lies below segment l
            for (unsigned int l = 0; l < segs.size(); ++l)
              {
                const Segment &above = segs[l];
                const State    Ua =
                  reconstruct_trace(Uhat, nq, q, above, above.trace_lo);
                State Ub;
                if (l == 0) // bottom boundary z = 0
                  Ub = (problem.bottom_bc == BoundaryType::wall) ?
                         reflect(Ua, e_z) :
                         dirichlet(x, 0.);
                else
                  Ub = reconstruct_trace(Uhat, nq, q, segs[l - 1], segs[l - 1].trace_hi);

                const State Fhat = lax_friedrichs_flux<dim>(Ub, Ua, e_z);
                if (l > 0)
                  {
                    const Segment &below = segs[l - 1];
                    for (unsigned int m = 0; m < below.n_modes; ++m)
                      val[(below.first_mode + m) * nq + q] -= below.trace_hi[m] * Fhat;
                  }
                for (unsigned int m = 0; m < above.n_modes; ++m)
                  val[(above.first_mode + m) * nq + q] += above.trace_lo[m] * Fhat;
              }
          }

        for (unsigned int j = 0; j < NZ; ++j)
          {
            for (unsigned int q = 0; q < nq; ++q)
              {
                phi.submit_value(val[j * nq + q], q);
                phi.submit_gradient(grad[j * nq + q], q);
              }
            phi.integrate(EvaluationFlags::values | EvaluationFlags::gradients);
            phi.distribute_local_to_global(dst, j * n_comp);
          }
      }
  }

  // Vertical faces between two Omega_b cells, integrated over [0, inf)
  void local_face(const MatrixFree<dim_b, Number>             &data,
                  BlockVectorType                             &dst,
                  const BlockVectorType                       &src,
                  const std::pair<unsigned int, unsigned int> &range) const
  {
    FEFaceEvaluation<dim_b, degree, n_q_1d, n_comp, Number> phi_m(data, true);
    FEFaceEvaluation<dim_b, degree, n_q_1d, n_comp, Number> phi_p(data, false);
    const unsigned int   nq = phi_m.n_q_points, NZ = z.n_modes();
    AlignedVector<State> Um(NZ * nq), Up(NZ * nq), flux(NZ * nq);

    for (unsigned int face = range.first; face < range.second; ++face)
      {
        phi_m.reinit(face);
        phi_p.reinit(face);
        read_all_modes(phi_m, src, Um);
        read_all_modes(phi_p, src, Up);
        for (auto &f : flux)
          f = State();

        for (unsigned int q = 0; q < nq; ++q)
          {
            const auto         nb = phi_m.normal_vector(q);
            Tensor<1, dim, VA> n;
            for (unsigned int k = 0; k < dim_b; ++k)
              n[k] = nb[k];
            for (const Segment &s : z.get_segments())
              for (unsigned int p = 0; p < s.points.size(); ++p)
                {
                  const State Fhat = lax_friedrichs_flux<dim>(
                    reconstruct(Um, nq, q, s, p), reconstruct(Up, nq, q, s, p), n);
                  for (unsigned int m = 0; m < s.n_modes; ++m)
                    flux[(s.first_mode + m) * nq + q] +=
                      (s.weights[p] * s.values(m, p)) * Fhat;
                }
          }

        for (unsigned int j = 0; j < NZ; ++j)
          {
            for (unsigned int q = 0; q < nq; ++q)
              {
                phi_m.submit_value(-flux[j * nq + q], q);
                phi_p.submit_value(flux[j * nq + q], q);
              }
            phi_m.integrate(EvaluationFlags::values);
            phi_m.distribute_local_to_global(dst, j * n_comp);
            phi_p.integrate(EvaluationFlags::values);
            phi_p.distribute_local_to_global(dst, j * n_comp);
          }
      }
  }

  // Vertical faces on the boundary of Omega_b, integrated over [0, inf)
  void local_boundary(const MatrixFree<dim_b, Number>             &data,
                      BlockVectorType                             &dst,
                      const BlockVectorType                       &src,
                      const std::pair<unsigned int, unsigned int> &range) const
  {
    FEFaceEvaluation<dim_b, degree, n_q_1d, n_comp, Number> phi(data, true);
    const unsigned int   nq = phi.n_q_points, NZ = z.n_modes();
    AlignedVector<State> Um(NZ * nq), flux(NZ * nq);

    for (unsigned int face = range.first; face < range.second; ++face)
      {
        phi.reinit(face);
        read_all_modes(phi, src, Um);
        for (auto &f : flux)
          f = State();

        for (unsigned int q = 0; q < nq; ++q)
          {
            const Point<dim_b, VA> x  = phi.quadrature_point(q);
            const auto             nb = phi.normal_vector(q);
            Tensor<1, dim, VA>     n;
            for (unsigned int k = 0; k < dim_b; ++k)
              n[k] = nb[k];
            for (const Segment &s : z.get_segments())
              for (unsigned int p = 0; p < s.points.size(); ++p)
                {
                  const State Ui = reconstruct(Um, nq, q, s, p);
                  const State Ue = (problem.lateral_bc == BoundaryType::wall) ?
                                     reflect(Ui, n) :
                                     dirichlet(x, s.points[p]);
                  const State Fhat = lax_friedrichs_flux<dim>(Ui, Ue, n);
                  for (unsigned int m = 0; m < s.n_modes; ++m)
                    flux[(s.first_mode + m) * nq + q] +=
                      (s.weights[p] * s.values(m, p)) * Fhat;
                }
          }

        for (unsigned int j = 0; j < NZ; ++j)
          {
            for (unsigned int q = 0; q < nq; ++q)
              phi.submit_value(-flux[j * nq + q], q);
            phi.integrate(EvaluationFlags::values);
            phi.distribute_local_to_global(dst, j * n_comp);
          }
      }
  }

  void local_inverse_mass(const MatrixFree<dim_b, Number>             &data,
                          BlockVectorType                             &dst,
                          const BlockVectorType                       &src,
                          const std::pair<unsigned int, unsigned int> &range) const
  {
    // quadrature index 1 = Gauss(degree+1): exact cell mass matrix
    FEEvaluation<dim_b, degree, degree + 1, n_comp, Number> phi(data, 0, 1);
    MatrixFreeOperators::CellwiseInverseMassMatrix<dim_b, degree, n_comp, Number>
      inverse(phi);
    for (unsigned int cell = range.first; cell < range.second; ++cell)
      {
        phi.reinit(cell);
        for (unsigned int j = 0; j < z.n_modes(); ++j)
          {
            phi.read_dof_values(src, j * n_comp);
            inverse.apply(phi.begin_dof_values(), phi.begin_dof_values());
            const Number scale = 1. / z.mass(j);
            for (unsigned int i = 0; i < phi.dofs_per_cell; ++i)
              phi.begin_dof_values()[i] *= scale;
            phi.set_dof_values(dst, j * n_comp);
          }
      }
  }

  // Rates for primitive variables (see the header): src = R(W), the
  // residual; *current_state = W; dst = dW/dt. Per column (cell batch):
  //  * g_j = M_x^{-1} R_j at the p+1 Gauss points (collocation, i.e. the
  //    values D^{-1} S^{-T} R_j of the residual density),
  //  * rho and E: rate_j = g_j / m_j,
  //  * velocity: per x-point and segment, solve
  //      sum_k A_ik u'_k = g_i^m - u_inf rho'_i m_s - sum_p w_p rho'(z_p)
  //                        (u - u_inf)(z_p) phi_i(z_p),
  //      A_ik = rho_inf m_s delta_ik
  //             + sum_p w_p (rho - rho_inf)(z_p) phi_i(z_p) phi_k(z_p),
  //    by Cholesky (vectorized over the lanes),
  //  * back to the DG basis with S^{-1}.
  void local_primitive_rates(const MatrixFree<dim_b, Number>             &data,
                             BlockVectorType                             &dst,
                             const BlockVectorType                       &src,
                             const std::pair<unsigned int, unsigned int> &range) const
  {
    FEEvaluation<dim_b, degree, degree + 1, n_comp, Number> phi(data, 0, 1);
    MatrixFreeOperators::CellwiseInverseMassMatrix<dim_b, degree, n_comp, Number>
                       inverse(phi);
    const unsigned int NZ = z.n_modes(), nq = phi.n_q_points;
    const auto        &segs = z.get_segments();
    unsigned int       max_modes = 0, max_points = 0;
    for (const Segment &s : segs)
      {
        max_modes  = std::max(max_modes, s.n_modes);
        max_points = std::max<unsigned int>(max_points, s.points.size());
      }
    AlignedVector<State> g(NZ * nq), Wv(NZ * nq), rate(NZ * nq);
    AlignedVector<VA> A(max_modes * max_modes), dinv(max_modes),
      x(dim * max_modes), rho_p(max_points), rhot_p(max_points),
      u_p(dim * max_points), vals(n_comp * nq);
    const VA rho_inf = V_inf[0];

    for (unsigned int cell = range.first; cell < range.second; ++cell)
      {
        phi.reinit(cell);
        for (unsigned int j = 0; j < NZ; ++j)
          {
            phi.read_dof_values(src, j * n_comp);
            inverse.apply(phi.begin_dof_values(), phi.begin_dof_values());
            phi.evaluate(EvaluationFlags::values);
            for (unsigned int q = 0; q < nq; ++q)
              g[j * nq + q] = phi.get_value(q);
            phi.read_dof_values(*current_state, j * n_comp);
            phi.evaluate(EvaluationFlags::values);
            for (unsigned int q = 0; q < nq; ++q)
              Wv[j * nq + q] = phi.get_value(q);
          }

        for (unsigned int q = 0; q < nq; ++q)
          for (const Segment &s : segs)
            {
              const unsigned int N = s.n_modes, P = s.points.size(),
                                 f = s.first_mode;
              const Number       m_s = s.mass;
              const auto gq = [&](const unsigned int i) -> const State & {
                return g[(f + i) * nq + q];
              };

              // rho - rho_inf, rho' and u - u_inf at the z-points
              for (unsigned int p = 0; p < P; ++p)
                {
                  VA r = Number(0.), rt = Number(0.);
                  for (unsigned int d = 0; d < dim; ++d)
                    u_p[d * P + p] = Number(0.);
                  for (unsigned int m = 0; m < N; ++m)
                    {
                      const Number v  = s.values(m, p);
                      const State &W  = Wv[(f + m) * nq + q];
                      r += v * W[0];
                      rt += v * gq(m)[0];
                      for (unsigned int d = 0; d < dim; ++d)
                        u_p[d * P + p] += v * W[1 + d];
                    }
                  rho_p[p]  = r * Number(s.weights[p]);
                  rhot_p[p] = rt * Number(s.weights[p] / m_s);
                }

              // A (lower triangle) and its Cholesky factorization; the pivot
              // is bounded from below so that non-physical states cannot
              // break the factorization
              for (unsigned int i = 0; i < N; ++i)
                for (unsigned int k = 0; k <= i; ++k)
                  {
                    VA a = Number(0.);
                    for (unsigned int p = 0; p < P; ++p)
                      a += rho_p[p] * (s.values(i, p) * s.values(k, p));
                    A[i * N + k] = a;
                  }
              for (unsigned int i = 0; i < N; ++i)
                A[i * N + i] += rho_inf * m_s;
              for (unsigned int j = 0; j < N; ++j)
                {
                  VA d = A[j * N + j];
                  for (unsigned int k = 0; k < j; ++k)
                    d -= A[j * N + k] * A[j * N + k];
                  d            = std::max(d, Number(1e-12) * rho_inf * m_s);
                  A[j * N + j] = std::sqrt(d);
                  dinv[j]      = Number(1.) / A[j * N + j];
                  for (unsigned int i = j + 1; i < N; ++i)
                    {
                      VA t = A[i * N + j];
                      for (unsigned int k = 0; k < j; ++k)
                        t -= A[i * N + k] * A[j * N + k];
                      A[i * N + j] = t * dinv[j];
                    }
                }

              // right-hand sides and solves, one per velocity component
              for (unsigned int d = 0; d < dim; ++d)
                {
                  VA *b = &x[d * N];
                  for (unsigned int i = 0; i < N; ++i)
                    {
                      VA r = gq(i)[1 + d] - V_inf[1 + d] * gq(i)[0];
                      for (unsigned int p = 0; p < P; ++p)
                        r -= rhot_p[p] * u_p[d * P + p] * Number(s.values(i, p));
                      b[i] = r;
                    }
                  for (unsigned int i = 0; i < N; ++i)
                    {
                      VA t = b[i];
                      for (unsigned int k = 0; k < i; ++k)
                        t -= A[i * N + k] * b[k];
                      b[i] = t * dinv[i];
                    }
                  for (unsigned int i = N; i-- > 0;)
                    {
                      VA t = b[i];
                      for (unsigned int k = i + 1; k < N; ++k)
                        t -= A[k * N + i] * b[k];
                      b[i] = t * dinv[i];
                    }
                }

              for (unsigned int i = 0; i < N; ++i)
                {
                  State &r = rate[(f + i) * nq + q];
                  r[0]       = gq(i)[0] / m_s;
                  r[dim + 1] = gq(i)[dim + 1] / m_s;
                  for (unsigned int d = 0; d < dim; ++d)
                    r[1 + d] = x[d * N + i];
                }
            }

        // S^{-1}: from the values at the Gauss points to the DG basis
        for (unsigned int j = 0; j < NZ; ++j)
          {
            for (unsigned int c = 0; c < n_comp; ++c)
              for (unsigned int q = 0; q < nq; ++q)
                vals[c * nq + q] = rate[j * nq + q][c];
            inverse.transform_from_q_points_to_basis(n_comp,
                                                     vals.data(),
                                                     phi.begin_dof_values());
            phi.set_dof_values(dst, j * n_comp);
          }
      }
  }

  const MatrixFree<dim_b, Number> &mf;
  const ZDiscretization           &z;
  const ProblemData<dim>          &problem;
  const Variables                  variables;
  State                            U_inf, V_inf; // V_inf: base of the expansion
  mutable const BlockVectorType   *current_state = nullptr;
  Tensor<1, dim, VA>               e_z;
  std::vector<double>              rk_a, rk_b, rk_c;
  mutable double                   time = 0.;
};


// ===========================================================================
//  Driver
// ===========================================================================

// variables used by all studies of this run (command-line option)
Variables selected_variables = Variables::conservative;

const char *variables_name(const Variables v)
{
  return v == Variables::primitive ? "primitive (rho, u, E)" :
                                     "conservative (rho, rho u, E)";
}

struct Parameters
{
  double       L_x         = 1.;  // Omega_b = (0, L_x)^dim_b
  unsigned int n_cells_x   = 8;   // subdivisions per direction of Omega_b
  double       L_z         = 2.;  // height of the DG-in-z region
  unsigned int N_z         = 8;   // number of intervals in [0, L_z]
  unsigned int degree_z    = 3;   // polynomial degree on the intervals
  unsigned int degree_tail = 16;  // Laguerre degree p_z (p_z + 1 modes)
  double       mu          = 2.;  // Laguerre scaling
  unsigned int n_q_tail    = 0;   // 0: 2 (p_z + 1) + 10
  double       final_time  = 0.1;
  // (initialized from the command-line option, see main())
  // relative to the spectral-norm bound of compute_time_step(). Measured
  // stability limit for the vortex (mode "cfl"): between 2.5 and 3 with the
  // hybrid basis, above 4 with pure Laguerre; 1.5 leaves a margin.
  double       courant     = 1.5;
  std::vector<double> monitor_times; // times at which monitor() is recorded
  Variables    variables = selected_variables;
};

// ||rho - rho_inf||, ||m - m_inf|| over Omega_b x (0,inf), and deviation
// ||rho - rho_exact||, ||m - m_exact|| over Omega_b x (0,3) from the exact
// (source-free) solution
struct MonitorEntry
{
  double time, pert_rho, pert_m, window_rho, window_m;
};

struct Result
{
  types::global_dof_index   n_dofs;
  unsigned int              n_z_modes, n_steps;
  double                    dt, error, residual_of_far_field, max_perturbation, wall_time;
  bool                      blew_up = false;
  double                    blow_up_time = 0.;
  std::vector<MonitorEntry> monitor;
};

template <int dim_b, int degree>
Result run(const Parameters &prm, const ProblemData<dim_b + 1> &problem)
{
  using Op = EulerColumnOperator<dim_b, degree>;

  Triangulation<dim_b> tria;
  GridGenerator::subdivided_hyper_cube(tria, prm.n_cells_x, 0., prm.L_x);
  const FE_DGQ<dim_b> fe(degree);
  DoFHandler<dim_b>   dof_handler(tria);
  dof_handler.distribute_dofs(fe);
  AffineConstraints<double> constraints;
  constraints.close();
  const MappingQ<dim_b> mapping(1);

  typename MatrixFree<dim_b, double>::AdditionalData data;
  data.tasks_parallel_scheme = MatrixFree<dim_b, double>::AdditionalData::none;
  data.mapping_update_flags =
    update_gradients | update_JxW_values | update_quadrature_points;
  data.mapping_update_flags_inner_faces = update_JxW_values | update_normal_vectors;
  data.mapping_update_flags_boundary_faces =
    update_JxW_values | update_normal_vectors | update_quadrature_points;
  const std::vector<Quadrature<1>> quads = {QGauss<1>(degree + 2),
                                            QGauss<1>(degree + 1)};
  const std::vector<const DoFHandler<dim_b> *>         dofs = {&dof_handler};
  const std::vector<const AffineConstraints<double> *> cons = {&constraints};
  MatrixFree<dim_b, double>                            mf;
  mf.reinit(mapping, dofs, cons, quads, data);

  const unsigned int q_tail =
    prm.n_q_tail > 0 ? prm.n_q_tail : std::min(150u, 2 * (prm.degree_tail + 1) + 10);
  const ZDiscretization z(prm.L_z, prm.N_z, prm.degree_z, prm.degree_tail,
                          prm.mu, prm.degree_z + 2, q_tail);
  const ZDiscretization z_fine(prm.L_z, prm.N_z, prm.degree_z, prm.degree_tail,
                               prm.mu, prm.degree_z + 4, std::min(150u, q_tail + 40));

  const Op op(mf, z, problem, prm.variables);
  typename Op::BlockVectorType solution, ri, ki, rhs;
  op.initialize_vector(solution);
  op.initialize_vector(ri);
  op.initialize_vector(ki);
  op.initialize_vector(rhs);

  Result r;
  r.n_dofs    = solution.size();
  r.n_z_modes = z.n_modes();

  // residual of the far-field state W = 0 (zero for free-stream preservation)
  solution = 0.;
  op.apply(0., solution, rhs);
  r.residual_of_far_field = rhs.l2_norm();

  if (problem.initial != InitialData::far_field)
    op.project_exact(0., z_fine, solution);

  // z-rules for monitoring: the fine rule over (0, inf), Gauss on (0, 3)
  std::vector<double> z_all, w_all, z_win, w_win;
  for (const auto &s : z_fine.get_segments())
    for (unsigned int p = 0; p < s.points.size(); ++p)
      {
        z_all.push_back(s.points[p]);
        w_all.push_back(s.weights[p]);
      }
  const QGauss<1> gauss_window(60);
  for (unsigned int p = 0; p < gauss_window.size(); ++p)
    {
      z_win.push_back(3. * gauss_window.point(p)[0]);
      w_win.push_back(3. * gauss_window.weight(p));
    }
  const auto monitor = [&](const double t) {
    const auto pert = op.l2_norms(solution, t, z_all, w_all, false);
    const auto win  = op.l2_norms(solution, t, z_win, w_win, true);
    r.monitor.push_back({t, pert[0], pert[1], win[0], win[1]});
  };

  // time loop: dt recomputed every 10 steps, landing exactly on the
  // monitor times and on the final time
  const double rate_x =
    ZDiscretization::interval_rate(degree, prm.L_x / prm.n_cells_x);
  double       dt   = op.compute_time_step(solution, prm.courant, rate_x);
  double       t    = 0.;
  unsigned int step = 0, next_monitor = 0;
  const double eps  = 1e-12 * prm.final_time;
  if (!prm.monitor_times.empty() && prm.monitor_times[0] < eps)
    {
      monitor(0.);
      ++next_monitor;
    }

  Timer timer;
  while (t < prm.final_time - eps)
    {
      if (step % 10 == 0 && step > 0)
        dt = op.compute_time_step(solution, prm.courant, rate_x);
      double stop = prm.final_time;
      if (next_monitor < prm.monitor_times.size())
        stop = std::min(stop, prm.monitor_times[next_monitor]);
      const double h = std::min(dt, stop - t);
      op.time_step(t, h, solution, ri, ki, rhs);
      t += h;
      ++step;
      if (!std::isfinite(solution.l2_norm()))
        {
          r.blew_up      = true;
          r.blow_up_time = t;
          break;
        }
      while (next_monitor < prm.monitor_times.size() &&
             std::abs(t - prm.monitor_times[next_monitor]) < eps)
        {
          monitor(t);
          ++next_monitor;
        }
    }
  r.wall_time = timer.wall_time();
  r.n_steps   = step;
  r.dt        = dt;

  r.error = (problem.initial != InitialData::far_field && !r.blew_up) ?
              op.l2_error(solution, prm.final_time, z_fine) :
              0.;
  r.max_perturbation = solution.linfty_norm();
  return r;
}


template <int dim_b, int degree>
void run_studies(const std::string &title, const bool full)
{
  constexpr int dim = dim_b + 1;
  std::cout << "\n=== " << title << " (d = " << dim << ", DG degree " << degree
            << " in x and in z on [0, L_z]) ===\n";

  // ---- free-stream preservation: W = 0 must stay 0 ----------------------
  {
    ProblemData<dim> problem; // gravity = 0, reaction = 0.5, U_inf on boundaries
    Parameters       prm;
    prm.n_cells_x   = 4;
    prm.N_z         = 4;
    prm.degree_tail = 12;
    prm.final_time  = 0.05;
    const Result r  = run<dim_b, degree>(prm, problem);
    std::cout << "Free stream: |R(0)| = " << std::scientific << std::setprecision(2)
              << r.residual_of_far_field << ",  max|W| after " << r.n_steps
              << " steps = " << r.max_perturbation << "\n";
  }

  ProblemData<dim> problem;
  problem.manufactured        = true;
  problem.initial             = InitialData::manufactured;
  problem.gravity[dim - 1]    = -0.5;

  // ---- refinement of the bounded part (x and z intervals together) -----
  std::cout << "\nRefinement in x and in [0, L_z] (h_x = h_z, L_z = 2, p_z = 20, T = 0.1)\n"
            << "   cells_x  N_z    n_dofs  steps        dt      L2 error   rate   time[s]\n";
  double prev = -1;
  for (const unsigned int n : (full ? std::vector<unsigned int>{2, 4, 8, 16} :
                                      std::vector<unsigned int>{2, 4}))
    {
      Parameters prm;
      prm.n_cells_x   = n;
      prm.N_z         = 2 * n;
      prm.degree_tail = 20;
      const Result r  = run<dim_b, degree>(prm, problem);
      std::cout << std::setw(10) << n << std::setw(5) << prm.N_z << std::setw(10)
                << r.n_dofs << std::setw(7) << r.n_steps << std::scientific
                << std::setprecision(2) << std::setw(10) << r.dt << std::setw(14)
                << r.error << std::fixed << std::setprecision(2) << std::setw(7)
                << (prev > 0 ? std::log2(prev / r.error) : 0.) << std::setw(10)
                << r.wall_time << "\n";
      prev = r.error;
    }

  // ---- Laguerre degree in the tail ---------------------------------------
  if (full)
    {
      std::cout << "\nLaguerre degree p_z in the tail (cells_x = 8, N_z = 16, mu = 2, T = 0.1)\n"
                << "  p_z  z-modes      L2 error\n";
      for (const unsigned int pz : {0u, 2u, 4u, 6u, 8u, 12u, 16u, 24u})
        {
          Parameters prm;
          prm.n_cells_x   = 8;
          prm.N_z         = 16;
          prm.degree_tail = pz;
          const Result r  = run<dim_b, degree>(prm, problem);
          std::cout << std::setw(5) << pz << std::setw(9) << r.n_z_modes
                    << std::scientific << std::setprecision(2) << std::setw(14)
                    << r.error << "\n";
        }
    }
}


// ---- isentropic vortex tests (2D, Omega_b = (0, 10)) ----------------------

struct ZSetup
{
  const char  *name;
  double       L_z;
  unsigned int N_z, p_z;
  double       mu;
};

Parameters vortex_parameters(const ZSetup &zs, const double T)
{
  Parameters prm;
  prm.L_x         = 10.;
  prm.n_cells_x   = 80;
  prm.L_z         = zs.L_z;
  prm.N_z         = zs.N_z;
  prm.degree_tail = zs.p_z;
  prm.mu          = zs.mu;
  prm.final_time  = T;
  return prm;
}

void print_vortex_row(const ZSetup     &zs,
                      const Parameters &prm,
                      const Result     &r,
                      const char       *err_label = "err")
{
  std::cout << std::defaultfloat << std::setprecision(6) << "  " << std::left
            << std::setw(14) << zs.name << std::right
            << " L_z=" << zs.L_z << " N_z=" << std::setw(2) << zs.N_z
            << " p_z=" << std::setw(2) << zs.p_z << " mu=" << std::setw(2) << zs.mu
            << " courant=" << std::fixed << std::setprecision(1) << prm.courant
            << std::setw(9) << r.n_dofs << " dofs" << std::setw(6) << r.n_steps
            << " steps  ";
  if (r.blew_up)
    std::cout << "BLEW UP at t = " << std::setprecision(3) << r.blow_up_time;
  else
    std::cout << err_label << " = " << std::scientific << std::setprecision(2) << r.error;
  std::cout << std::fixed << std::setprecision(2) << "  time " << r.wall_time
            << " s" << std::endl;
}

void print_monitor(const char *title, const Result &a, const Result &b)
{
  std::cout << "\n  " << title
            << "\n      t |  ||rho-rho_inf|| (0,inf)  |  ||m-m_inf|| (0,inf)    |"
               " ||rho-rho_ex|| (0,3)\n"
               "        |  no damping     damping  |  no damping     damping  |"
               "  no damping     damping\n";
  const double nan = std::numeric_limits<double>::quiet_NaN();
  for (unsigned int i = 0; i < std::max(a.monitor.size(), b.monitor.size()); ++i)
    {
      const MonitorEntry m{nan, nan, nan, nan, nan};
      const auto &x = i < a.monitor.size() ? a.monitor[i] : m;
      const auto &y = i < b.monitor.size() ? b.monitor[i] : m;
      std::cout << std::fixed << std::setprecision(2) << std::setw(7)
                << (std::isfinite(x.time) ? x.time : y.time) << " |"
                << std::scientific << std::setprecision(3) << std::setw(12)
                << x.pert_rho << std::setw(12) << y.pert_rho << "  |"
                << std::setw(12) << x.pert_m << std::setw(12) << y.pert_m
                << "  |" << std::setw(12) << x.window_rho << std::setw(12)
                << y.window_rho << std::endl;
    }
}

void run_vortex_studies(const std::string &mode)
{
  const bool all = mode == "all";
  const ZSetup laguerre48{"pure Laguerre", 0., 0, 48, 10.};

  // (1) stability limit of the new time-step estimate
  if (all || mode == "cfl")
    {
      std::cout << "\n=== stability limit of the time step: vortex along x, T = 1 "
                   "(spectral-norm estimate, recomputed every 10 steps) ===\n";
      for (const ZSetup zs : {laguerre48, ZSetup{"hybrid", 6., 32, 16, 3.}})
        for (const double courant : {1., 1.5, 2., 2.5, 3., 4.})
          {
            ProblemData<2> problem;
            problem.set_isentropic_vortex(3., 2., 1., 0.);
            Parameters prm = vortex_parameters(zs, 1.);
            prm.courant    = courant;
            print_vortex_row(zs, prm, run<1, 3>(prm, problem));
          }
    }

  // (2) accuracy: vortex along x, exact solution
  if (all || mode == "vortex")
    {
      std::cout << "\n=== isentropic vortex along x (exact solution), T = 1, 80 "
                   "cells Q3; err = L2 error of all components ===\n";
      for (const ZSetup zs : {ZSetup{"pure Laguerre", 0., 0, 24, 10.},
                              ZSetup{"pure Laguerre", 0., 0, 32, 10.},
                              laguerre48,
                              ZSetup{"hybrid", 6., 8, 16, 3.},
                              ZSetup{"hybrid", 6., 16, 16, 3.},
                              ZSetup{"hybrid", 6., 32, 16, 3.}})
        {
          ProblemData<2> problem;
          problem.set_isentropic_vortex(3., 2., 1., 0.);
          const Parameters prm = vortex_parameters(zs, 1.);
          print_vortex_row(zs, prm, run<1, 3>(prm, problem));
        }
    }

  // (3) long-time runs: vortex along +z, with and without the damping layer
  // dgamma = 3, alpha = 4.5, sigma = 0.4 (as in laguerre_systems.cc).
  // Without damping the vortex rises into the region that the z-basis
  // cannot resolve: a stability test for the aliasing of the pointwise
  // z-quadrature. "under-resolved" setups are deliberately too coarse.
  if (all || mode == "damping")
    {
      const double T = 6.;
      std::cout << "\n=== vortex along +z, T = " << T
                << ", damping layer dgamma = 3, alpha = 4.5, sigma = 0.4 ===\n";
      for (const ZSetup zs : {laguerre48,
                              ZSetup{"hybrid", 6., 24, 16, 3.},
                              ZSetup{"under-resolved", 0., 0, 16, 10.},
                              ZSetup{"under-resolved", 3., 6, 8, 3.}})
        {
          Result results[2];
          for (const bool damped : {false, true})
            {
              ProblemData<2> problem;
              problem.set_isentropic_vortex(5., 2., 0., 1.);
              if (damped)
                problem.set_damping_layer(3., 4.5, 0.4);
              Parameters prm = vortex_parameters(zs, T);
              for (unsigned int i = 0; i <= 12; ++i)
                prm.monitor_times.push_back(i * T / 12);
              std::cout << (damped ? "  damped:  " : "  undamped:");
              results[damped] = run<1, 3>(prm, problem);
              print_vortex_row(zs, prm, results[damped], "err vs undamped exact (t=6)");
            }
          const std::string title = std::string(zs.name) + " (L_z = " +
                                    std::to_string(zs.L_z) + ", N_z = " +
                                    std::to_string(zs.N_z) + ", p_z = " +
                                    std::to_string(zs.p_z) + ")";
          print_monitor(title.c_str(), results[0], results[1]);
          std::cout << std::endl;
        }
    }
}


int main(int argc, char **argv)
{
  try
    {
      Utilities::MPI::MPI_InitFinalize mpi(argc, argv, 1);
      // usage: laguerre_systems_2 [manufactured|cfl|vortex|damping|all]
      //                           [conservative|primitive]
      const std::string mode = argc > 1 ? argv[1] : "all";
      // optional second argument: conservative (default) | primitive
      if (argc > 2)
        {
          const std::string v = argv[2];
          AssertThrow(v == "conservative" || v == "primitive",
                      ExcMessage("variables must be conservative or primitive"));
          selected_variables =
            v == "primitive" ? Variables::primitive : Variables::conservative;
        }
      std::cout << "SIMD width: " << VectorizedArray<double>::size() << " doubles\n"
                << "variables: " << variables_name(selected_variables) << "\n";
      if (mode == "all" || mode == "manufactured")
        {
          run_studies<1, 3>("2D reaction-Euler on [0,1] x [0,inf)", true);
          run_studies<2, 2>("3D reaction-Euler on [0,1]^2 x [0,inf)", false);
        }
      run_vortex_studies(mode);
    }
  catch (std::exception &exc)
    {
      std::cerr << "Exception: " << exc.what() << std::endl;
      return 1;
    }
  return 0;
}
