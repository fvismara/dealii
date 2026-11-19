// ---------------------------------------------------------------------------
//  Matrix-free DG-Laguerre solver for systems of balance laws on a half-space,
//  following 'laguerre_systems.pdf':
//
//      d_t (M(w) w) + div F(w) = b(w, x, z, t),   (x, z) in Omega_b x [0, inf)
//
//  specialized to the Euler equations with reaction (damping) and gravity,
//
//      b(w) = (0, rho g, rho u.g) - Gamma(z) (M(w) w - M(w_inf) w_inf),
//      Gamma_ii(z) = dgamma_i / (1 + exp((alpha_i - z) / sigma_i)),
//
//  with w = (rho, u, E) and M(w) w = (rho, rho u, E). dim (= d) is the full
//  dimension; the DG discretization lives on the (dim-1)-dimensional mesh of
//  Omega_b. This file extends laguerre_euler.cc (same discretization of the
//  flux terms) by the source term b and by the projections of the notes.
//
//  * Expansion: w^i(x,z,t) ~ w^i_inf + sum_{j=0}^{pz} w^i_j(x,t) L_j(z),
//    L_j(z) = exp(-mu z/2) L_j(mu z), one scaling mu for all components.
//    Each w_j is an FE_DGQ function with dim+2 components on Omega_b; the
//    modes are the blocks of a BlockVector.
//
//  * Projected system: d_t a~ + div_x B~ = h~ with
//        a_j = int M(w) w L_j,  B_j = int F_b(w) L_j,
//        h_j = f^_0 + int f(w) L_j' + int b(w) L_j.
//
//  * Exact z-integrals. Every entry of M(w) w, F(w) and of
//    the gravity part of b is a polynomial of degree N <= 4 in the entries
//    of w, so
//        int (prod_k w~^k) L_i dz = sum_sigma (prod_{l not in sigma} w_inf^l)
//                     sum_q prod_{l in sigma} (sum_j w^l_j P^(s)_jq)
//                           P^(s)_iq omega^(s)_q
//    with Gauss-Laguerre rules of Q_s = ceil((1 + (s+1) pz)/2) points. The
//    sum over subsets is generated automatically by evaluating the
//    polynomial on "graded" states w_inf + t w'(z_q) (truncated polynomials
//    in t); the coefficient of t^s is the sum over |sigma| = s. In the code
//    the exponentials are absorbed into the Laguerre *functions*
//    (P^(s)_jq omega^(s)_q = L_j(z_q) * exp-scaled weight), which is
//    algebraically identical but avoids overflow/underflow. The integrals
//    against L_j' follow from (8): L_i' = -mu/2 L_i - mu sum_{l<i} L_l.
//
//  * Reaction term. With
//    w = w_inf + w', M(w)w - M(w_inf)w_inf = (rho', rho_inf u' + u_inf rho'
//    + rho' u', E'). The terms linear in w' need
//        G_i(j,k) = int Gamma_ii L_j L_k dz,
//    which do not depend on time and are precomputed once (Gauss-Laguerre in
//    y = mu z, more points than the minimum ceil((3pz+1)/2) of the notes
//    since this is done only once). The quadratic momentum term rho' u'
//    gives int Gamma_ii L_j L_k L_l dz, whose (pz+1)^3 storage we avoid: it
//    is evaluated at every call with the s = 2 rule (scaling 3 mu/2) of
//    Q = 2 pz + 1 points, weights premultiplied by Gamma_ii(z_q). Note that
//    with the expansion around w_inf the integrals int Gamma_ii L_j dz of
//    the notes are not needed: there is no constant term in
//    M(w)w - M(w_inf)w_inf. The reaction integrals are the only ones that
//    are not exact (Gamma is not a polynomial); their quadrature error is
//    reported by LaguerreDGEulerOperator::verify_source_integrals().
//
//  * Initial condition and lateral Dirichlet data:
//        w^i_k(x) = 2 sum_q w^i(x, 2 z_q/mu) L_k(2 z_q) omega_q
//                   - 2 (-1)^k w^i_inf,
//    Gauss-Laguerre in y = mu z / 2. The notes choose Q = pz + 1 points;
//    this limits the accuracy of the convergence test (error floor ~2e-3
//    for the vortex), so by default 40 more points are used (command-line
//    option, see main()).
//
//  * DG in Omega_b: Rusanov flux of the projected system on the faces of
//    Omega_b (all pieces are exact z-integrals); at z = 0 the boundary flux
//    f^_0 is the local Lax-Friedrichs flux of step-67 with Dirichlet data or
//    a slip wall. The time derivative of a~ is inverted as in
//    laguerre_euler.cc: (1/mu) M_x for rho and E, and a cell-local,
//    preconditioned CG for the rho-weighted mass matrix of the momentum.
//    Time stepping: low-storage RK(5,4) of step-67; with damping additionally
//    dt <= 1 / max dgamma.
//
//  Test cases:
//    isentropic_vortex : vortex of step-67 in the (x_0, z) plane advected
//                        along x_0, no source term, exact Dirichlet data.
//                        Convergence study (spectral in pz, h in x).
//    damped_vortex     : vortex advected along +z (u_inf = e_z) into a
//                        sigmoidal damping layer. Qualitative test: compared
//                        to the same run without damping.
// ---------------------------------------------------------------------------

#include <deal.II/base/aligned_vector.h>
#include <deal.II/base/data_out_base.h>
#include <deal.II/base/mpi.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/time_stepping.h>
#include <deal.II/base/timer.h>
#include <deal.II/base/vectorization.h>

#include <deal.II/dofs/dof_handler.h>

#include <deal.II/fe/fe_dgq.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/fe/mapping_q.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/tria.h>

#include <deal.II/lac/affine_constraints.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/la_parallel_block_vector.h>
#include <deal.II/lac/la_parallel_vector.h>
#include <deal.II/lac/lapack_full_matrix.h>
#include <deal.II/lac/vector.h>

#include <deal.II/matrix_free/fe_evaluation.h>
#include <deal.II/matrix_free/matrix_free.h>
#include <deal.II/matrix_free/operators.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

using namespace dealii;

constexpr double gamma_gas = 1.4;


// ===========================================================================
//  Euler physics
// ===========================================================================

// Pointwise functions in the conserved variables U = (rho, rho u, E), used
// for the numerical flux at z = 0 (taken over from step-67).
template <int dim, typename Number>
inline DEAL_II_ALWAYS_INLINE Tensor<1, dim, Number>
euler_velocity(const Tensor<1, dim + 2, Number> &U)
{
  const Number           inverse_density = Number(1.) / U[0];
  Tensor<1, dim, Number> velocity;
  for (unsigned int d = 0; d < dim; ++d)
    velocity[d] = U[1 + d] * inverse_density;
  return velocity;
}

template <int dim, typename Number>
inline DEAL_II_ALWAYS_INLINE Number
euler_pressure(const Tensor<1, dim + 2, Number> &U)
{
  const Tensor<1, dim, Number> velocity    = euler_velocity<dim>(U);
  Number                       rho_u_dot_u = U[1] * velocity[0];
  for (unsigned int d = 1; d < dim; ++d)
    rho_u_dot_u += U[1 + d] * velocity[d];
  return (gamma_gas - 1.) * (U[dim + 1] - 0.5 * rho_u_dot_u);
}

template <int dim, typename Number>
inline DEAL_II_ALWAYS_INLINE Tensor<1, dim + 2, Tensor<1, dim, Number>>
euler_flux(const Tensor<1, dim + 2, Number> &U)
{
  const Tensor<1, dim, Number> velocity = euler_velocity<dim>(U);
  const Number                 pressure = euler_pressure<dim>(U);

  Tensor<1, dim + 2, Tensor<1, dim, Number>> flux;
  for (unsigned int d = 0; d < dim; ++d)
    {
      flux[0][d] = U[1 + d];
      for (unsigned int e = 0; e < dim; ++e)
        flux[e + 1][d] = U[e + 1] * velocity[d];
      flux[d + 1][d] += pressure;
      flux[dim + 1][d] = velocity[d] * (U[dim + 1] + pressure);
    }
  return flux;
}

template <int n_components, int dim, typename Number>
inline DEAL_II_ALWAYS_INLINE Tensor<1, n_components, Number>
operator*(const Tensor<1, n_components, Tensor<1, dim, Number>> &matrix,
          const Tensor<1, dim, Number>                          &vector)
{
  Tensor<1, n_components, Number> result;
  for (unsigned int c = 0; c < n_components; ++c)
    result[c] = matrix[c] * vector;
  return result;
}

// Modified local Lax-Friedrichs flux of step-67; normal points from U_m to U_p
template <int dim, typename Number>
inline DEAL_II_ALWAYS_INLINE Tensor<1, dim + 2, Number>
euler_numerical_flux(const Tensor<1, dim + 2, Number> &U_m,
                     const Tensor<1, dim + 2, Number> &U_p,
                     const Tensor<1, dim, Number>     &normal)
{
  const auto velocity_m = euler_velocity<dim>(U_m);
  const auto velocity_p = euler_velocity<dim>(U_p);
  const auto pressure_m = euler_pressure<dim>(U_m);
  const auto pressure_p = euler_pressure<dim>(U_p);
  const auto flux_m     = euler_flux<dim>(U_m);
  const auto flux_p     = euler_flux<dim>(U_p);

  const auto lambda =
    0.5 * std::sqrt(std::max(velocity_p.norm_square() +
                               gamma_gas * pressure_p * (1. / U_p[0]),
                             velocity_m.norm_square() +
                               gamma_gas * pressure_m * (1. / U_m[0])));

  return 0.5 * (flux_m * normal + flux_p * normal) +
         0.5 * lambda * (U_m - U_p);
}


// Conversion between w = (rho, u, E) and U = (rho, rho u, E)
template <int dim, typename Number>
inline DEAL_II_ALWAYS_INLINE Tensor<1, dim + 2, Number>
to_conserved(const Tensor<1, dim + 2, Number> &w)
{
  Tensor<1, dim + 2, Number> U = w;
  for (unsigned int d = 0; d < dim; ++d)
    U[1 + d] = w[0] * w[1 + d];
  return U;
}

template <int dim, typename Number>
inline DEAL_II_ALWAYS_INLINE Tensor<1, dim + 2, Number>
to_rho_u_E(const Tensor<1, dim + 2, Number> &U)
{
  Tensor<1, dim + 2, Number> w               = U;
  const Number               inverse_density = Number(1.) / U[0];
  for (unsigned int d = 0; d < dim; ++d)
    w[1 + d] = U[1 + d] * inverse_density;
  return w;
}

// Pressure p = (gamma - 1) (E - rho |u|^2 / 2) from w = (rho, u, E)
template <int dim, typename Number>
inline DEAL_II_ALWAYS_INLINE Number
pressure_from_rho_u_E(const Tensor<1, dim + 2, Number> &w)
{
  Number u_dot_u = w[1] * w[1];
  for (unsigned int d = 1; d < dim; ++d)
    u_dot_u += w[1 + d] * w[1 + d];
  return (gamma_gas - 1.) * (w[dim + 1] - 0.5 * w[0] * u_dot_u);
}


// ---------------------------------------------------------------------------
//  Polynomial form of the Euler fluxes in w = (rho, u, E):
//    F_rho     = rho u
//    F_rho u   = rho u u^T + p I,   p = (gamma - 1)(E - rho |u|^2 / 2)
//    F_E       = (E + p) u = gamma E u - (gamma - 1)/2 rho |u|^2 u
//  Written only with +, -, * so that it can be evaluated on the graded
//  numbers below (degree <= 4 in the entries of w).
// ---------------------------------------------------------------------------
template <int dim, typename T>
inline DEAL_II_ALWAYS_INLINE void
euler_flux_rho_u_E(const std::array<T, dim + 2>            &w,
                   std::array<std::array<T, dim>, dim + 2> &F)
{
  const T &rho = w[0];
  const T &E   = w[dim + 1];

  std::array<T, dim> rho_u;
  for (unsigned int d = 0; d < dim; ++d)
    rho_u[d] = rho * w[1 + d];
  T u_sqr = w[1] * w[1];
  for (unsigned int d = 1; d < dim; ++d)
    u_sqr = u_sqr + w[1 + d] * w[1 + d];
  const T p   = (E - (rho * u_sqr) * 0.5) * (gamma_gas - 1.);
  const T E_p = E + p;

  for (unsigned int d = 0; d < dim; ++d)
    {
      F[0][d] = rho_u[d];
      for (unsigned int c = 0; c < dim; ++c)
        F[1 + c][d] = rho_u[c] * w[1 + d];
      F[1 + d][d]     = F[1 + d][d] + p;
      F[dim + 1][d]   = E_p * w[1 + d];
    }
}

// M(w) w = (rho, rho u, E), polynomial of degree 2
template <int dim, typename T>
inline DEAL_II_ALWAYS_INLINE void
conserved_rho_u_E(const std::array<T, dim + 2> &w, std::array<T, dim + 2> &U)
{
  U[0] = w[0];
  for (unsigned int d = 0; d < dim; ++d)
    U[1 + d] = w[0] * w[1 + d];
  U[dim + 1] = w[dim + 1];
}

// Gravity part of b: (0, rho g, rho u.g), polynomial of degree 2
template <int dim, typename T>
inline DEAL_II_ALWAYS_INLINE void
gravity_source_rho_u_E(const std::array<T, dim + 2> &w,
                       const Tensor<1, dim>         &g,
                       std::array<T, dim + 2>       &S)
{
  S[0]       = T();
  S[dim + 1] = T();
  for (unsigned int d = 0; d < dim; ++d)
    {
      S[1 + d]   = w[0] * g[d];
      S[dim + 1] = S[dim + 1] + (w[0] * w[1 + d]) * g[d];
    }
}


// ---------------------------------------------------------------------------
//  Graded numbers: truncated polynomials c_0 + c_1 t + ... + c_4 t^4 in a
//  bookkeeping variable t. Evaluating a polynomial f on w_inf + t w' gives,
//  as coefficient of t^m, the sum over all subsets sigma with |sigma| = m of
//  prod_{l not in sigma} w^l_inf prod_{l in sigma} w'^l, i.e. exactly the
//  terms grouped together in Proposition 1. The maximum degree 4 is the
//  maximum number of factors in the Euler fluxes.
// ---------------------------------------------------------------------------
constexpr unsigned int max_factors = 4;

template <typename Number>
struct Graded
{
  static constexpr unsigned int size = max_factors + 1;
  Number                        c[size];

  Graded()
  {
    for (unsigned int i = 0; i < size; ++i)
      c[i] = Number(0.);
  }
};

template <typename Number>
inline DEAL_II_ALWAYS_INLINE Graded<Number>
operator+(const Graded<Number> &a, const Graded<Number> &b)
{
  Graded<Number> r;
  for (unsigned int i = 0; i < Graded<Number>::size; ++i)
    r.c[i] = a.c[i] + b.c[i];
  return r;
}

template <typename Number>
inline DEAL_II_ALWAYS_INLINE Graded<Number>
operator-(const Graded<Number> &a, const Graded<Number> &b)
{
  Graded<Number> r;
  for (unsigned int i = 0; i < Graded<Number>::size; ++i)
    r.c[i] = a.c[i] - b.c[i];
  return r;
}

template <typename Number>
inline DEAL_II_ALWAYS_INLINE Graded<Number>
operator*(const Graded<Number> &a, const Graded<Number> &b)
{
  Graded<Number> r;
  for (unsigned int k = 0; k < Graded<Number>::size; ++k)
    {
      r.c[k] = a.c[0] * b.c[k];
      for (unsigned int i = 1; i <= k; ++i)
        r.c[k] += a.c[i] * b.c[k - i];
    }
  return r;
}

template <typename Number>
inline DEAL_II_ALWAYS_INLINE Graded<Number>
operator*(const Graded<Number> &a, const double s)
{
  Graded<Number> r;
  for (unsigned int i = 0; i < Graded<Number>::size; ++i)
    r.c[i] = a.c[i] * s;
  return r;
}

template <typename Number>
inline DEAL_II_ALWAYS_INLINE Graded<VectorizedArray<Number>>
operator*(const Graded<VectorizedArray<Number>> &a,
          const VectorizedArray<Number>         &s)
{
  Graded<VectorizedArray<Number>> r;
  for (unsigned int i = 0; i < Graded<VectorizedArray<Number>>::size; ++i)
    r.c[i] = a.c[i] * s;
  return r;
}


// ===========================================================================
//  1D Laguerre machinery on (0, inf)
// ===========================================================================

// Laguerre functions exp(-t/2) L_k(t), k = 0..n, by the (stable) three-term
// recurrence applied directly to the functions
void laguerre_functions(const unsigned int   n,
                        const double         t,
                        std::vector<double> &psi)
{
  psi.resize(n + 1);
  psi[0] = std::exp(-0.5 * t);
  if (n >= 1)
    psi[1] = (1. - t) * psi[0];
  for (unsigned int k = 1; k < n; ++k)
    psi[k + 1] = ((2. * k + 1. - t) * psi[k] - k * psi[k - 1]) / (k + 1.);
}

// Gauss-Laguerre rule for the weight exp(-t); returns the nodes and the
// *scaled* weights omega_p exp(t_p), i.e. int_0^inf g dt ~= sum_p w_p g(t_p).
// The scaled weights are computed as t / ((n+1)^2 psi_{n+1}(t)^2) with the
// Laguerre function psi_{n+1} = exp(-t/2) L_{n+1}(t), which avoids the
// overflow of exp(t) for large rules.
void compute_gauss_laguerre(const unsigned int   n,
                            std::vector<double> &nodes,
                            std::vector<double> &scaled_weights)
{
  AssertThrow(n >= 1 && n <= 300,
              ExcMessage("Gauss-Laguerre rule implemented for 1 <= n <= 300"));
  nodes.resize(n);
  scaled_weights.resize(n);

  double              z = 0.;
  std::vector<double> psi;
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

      for (unsigned int it = 0; it < 200; ++it)
        {
          double p1 = 1.0, p2 = 0.0;
          for (unsigned int j = 1; j <= n; ++j)
            {
              const double p3 = p2;
              p2              = p1;
              p1 = ((2.0 * j - 1.0 - z) * p2 - (j - 1.0) * p3) / j;
            }
          const double pp = n * (p1 - p2) / z; // L_n'(z)
          const double z1 = z;
          z               = z1 - p1 / pp;
          if (std::abs(z - z1) <= 1e-15 * std::max(1.0, std::abs(z)))
            break;
        }
      nodes[i] = z;
      laguerre_functions(n + 1, z, psi);
      scaled_weights[i] = z / ((n + 1.) * (n + 1.) * psi[n + 1] * psi[n + 1]);
    }
}


// Laguerre functions L_k(z) = exp(-mu z/2) L_k(mu z), k = 0..pz, with a
// Gauss-Laguerre rule in physical z (weight exp(-mu z)). This rule is used
// only where the integrand is not a polynomial in the expansion
// coefficients: projection of initial/boundary data, time step estimate,
// error computation and output. Closed-form 1D matrices:
//   M_z(j,k) = int L_j L_k  dz = delta_jk / mu
//   C_z(k,j) = int L_j L_k' dz = -1 (j<k), -1/2 (j=k), 0 (j>k)
//   L_k(0) = 1,  int_0^inf L_k dz = 2 (-1)^k / mu.
class ClassicalLaguerreBasis
{
public:
  ClassicalLaguerreBasis(const unsigned int pz,
                         const double       mu,
                         const unsigned int n_quadrature_points,
                         const double       drop_tolerance = 1e-15)
    : pz(pz)
    , mu(mu)
  {
    compute_gauss_laguerre(n_quadrature_points, t_all, w_all);
    for (unsigned int p = 0; p < t_all.size(); ++p)
      {
        t_all[p] /= mu; // physical z
        w_all[p] /= mu; // dz = dt / mu
      }

    // Drop the nodes at which all projection weights w_p L_k(z_p) are
    // negligible (far tail of the rule, where L_k underflows).
    std::vector<double> v, max_weight(t_all.size(), 0.);
    double              global_max = 0;
    for (unsigned int p = 0; p < t_all.size(); ++p)
      {
        values(t_all[p], v);
        for (unsigned int k = 0; k <= pz; ++k)
          max_weight[p] = std::max(max_weight[p], std::abs(w_all[p] * v[k]));
        global_max = std::max(global_max, max_weight[p]);
      }
    for (unsigned int p = 0; p < t_all.size(); ++p)
      if (max_weight[p] > drop_tolerance * global_max)
        {
          z_q.push_back(t_all[p]);
          w_q.push_back(w_all[p]);
        }

    phi.reinit(pz + 1, z_q.size());
    for (unsigned int p = 0; p < z_q.size(); ++p)
      {
        values(z_q[p], v);
        for (unsigned int k = 0; k <= pz; ++k)
          phi(k, p) = v[k];
      }

    // mu ||C_z||_2 bounds the field of values of the 1D z-advection operator
    // M_z^{-1} C_z per unit wave speed; it plays the role of (p+1)^2/h in
    // the CFL condition.
    LAPACKFullMatrix<double> C(pz + 1, pz + 1);
    for (unsigned int k = 0; k <= pz; ++k)
      for (unsigned int j = 0; j <= pz; ++j)
        C(k, j) = j < k ? -1. : (j == k ? -0.5 : 0.);
    C.compute_svd();
    advection_bound = mu * C.singular_value(0);
  }

  unsigned int get_pz() const { return pz; }
  unsigned int n_modes() const { return pz + 1; }
  double       get_mu() const { return mu; }
  unsigned int n_quadrature_points() const { return z_q.size(); }
  double quadrature_point(const unsigned int p) const { return z_q[p]; }
  double quadrature_weight(const unsigned int p) const { return w_q[p]; }
  double value(const unsigned int k, const unsigned int p) const
  {
    return phi(k, p);
  }
  double advection_operator_bound() const { return advection_bound; }
  double integral(const unsigned int k) const
  {
    return 2. / mu * (k % 2 == 0 ? 1. : -1.);
  }

  // L_k(z), k = 0..pz
  void values(const double z, std::vector<double> &out) const
  {
    laguerre_functions(pz, mu * z, out);
  }

private:
  unsigned int        pz;
  double              mu;
  std::vector<double> t_all, w_all; // full rule (physical z)
  std::vector<double> z_q, w_q;     // truncated rule
  FullMatrix<double>  phi;          // L_k(z_p)
  double              advection_bound = 0;
};


// Quadrature rules of Proposition 2. For s = |sigma| = 0..max_factors the
// integrand of the sigma-term is
//     exp(-(s+1) mu z / 2) * (polynomial of degree s pz + i in mu z),
// which Gauss-Laguerre in y = (s+1) mu z / 2 with Q_s = ceil((1 + (s+1) pz)/2)
// points integrates exactly. We store
//     z_q      = 2 y_q / ((s+1) mu)                (physical nodes)
//     weight_q = omega_q exp(y_q) 2 / ((s+1) mu)   (= omega^(s)_q exp(y_q))
//     phi(j,q) = L_j(z_q) = exp(-mu z_q / 2) L_j(mu z_q),
// so that, since exp(-y_q) = prod over the s+1 factors exp(-mu z_q/2),
//     sum_q omega^(s)_q (prod_l sum_j w^l_j P^(s)_jq) P^(s)_iq
//   = sum_q weight_q (prod_l w'^l(z_q)) L_i(z_q),
// which is formula (7) without the overflow-prone factors.
// Besides the rules s = 1..max_factors (the s = 0 term is f(w_inf) int L_i),
// the class provides the projection rule of the notes (s = 0, Q = pz + 1).
class LaguerreProductRules
{
public:
  struct Rule
  {
    unsigned int        m = 0, Q = 0;
    std::vector<double> z, weight;
    std::vector<double> phi; // phi[j * Q + q] = L_j(z_q)
  };

  // n_projection_extra: points of the projection rule beyond the pz + 1 of
  // the notes
  LaguerreProductRules(const unsigned int pz,
                       const double       mu,
                       const unsigned int n_projection_extra = 0)
    : pz(pz)
    , mu(mu)
  {
    for (unsigned int m = 1; m <= max_factors; ++m)
      rules.push_back(make_rule(m, n_points(m)));
    projection = make_rule(0, pz + 1 + n_projection_extra);
  }

  // Rule for the projection of initial and boundary data (pz + 1 points
  // unless n_projection_extra > 0),
  //   w_k = mu int (w - w_inf) L_k dz
  //       = mu sum_q weight_q w(z_q) L_k(z_q) - 2 (-1)^k w_inf,
  // i.e. the Gauss-Laguerre rule with Q = pz + 1 points in y = mu z / 2 of
  // "Computation of the initial condition".
  const Rule &projection_rule() const { return projection; }

  // Number of points for the precomputed reaction matrices
  // int Gamma L_j L_k dz (rule s = 1). The notes suggest at least
  // ceil((3 pz + 1)/2); since the matrices are computed only once we use
  // more points to resolve the sigmoid.
  unsigned int n_points_reaction_matrix() const { return 3 * pz + 40; }

  // Number of points for int Gamma L_j L_k L_i dz (rule s = 2), evaluated
  // at every call: the exact count for polynomials plus pz/2 points for
  // Gamma, in analogy to the choice of the notes for the matrices.
  unsigned int n_points_reaction_triple() const { return 2 * pz + 1; }

  const Rule &rule(const unsigned int m) const
  {
    AssertIndexRange(m - 1, rules.size());
    return rules[m - 1];
  }

  unsigned int n_points(const unsigned int m) const
  {
    return (1 + (m + 1) * pz + 1) / 2; // ceil((1 + (m+1) pz) / 2)
  }

  // Check exactness: (i) rule 1 reproduces int L_i L_j = delta_ij / mu,
  // (ii) for every m the highest-degree product int (L_pz)^m L_pz dz (and a
  // mixed one) does not change when the rule is enlarged by 8 points.
  void verify(std::ostream &out) const
  {
    double err_mass = 0;
    {
      const Rule &r = rule(1);
      for (unsigned int i = 0; i <= pz; ++i)
        for (unsigned int j = 0; j <= pz; ++j)
          {
            double sum = 0;
            for (unsigned int q = 0; q < r.Q; ++q)
              sum += r.weight[q] * r.phi[i * r.Q + q] * r.phi[j * r.Q + q];
            err_mass =
              std::max(err_mass, std::abs(mu * sum - (i == j ? 1. : 0.)));
          }
    }
    out << std::scientific << std::setprecision(2)
        << "  Proposition-2 rules (pz=" << pz << ", mu=" << mu
        << "): |mu M_rule1 - I| = " << err_mass << "; Q_m =";
    for (const Rule &r : rules)
      out << " " << r.Q;
    out << "; |rule_m - rule_(m,Q+8)| =";
    for (unsigned int m = 1; m <= max_factors; ++m)
      {
        const Rule a = rule(m), b = make_rule(m, n_points(m) + 8);
        double     err = 0;
        for (const unsigned int shift : {0u, 1u})
          {
            auto integrate = [&](const Rule &r) {
              double sum = 0;
              for (unsigned int q = 0; q < r.Q; ++q)
                {
                  double prod = r.phi[pz * r.Q + q];
                  for (unsigned int l = 0; l < m; ++l)
                    prod *= r.phi[(pz - (l % 2) * shift) * r.Q + q];
                  sum += r.weight[q] * prod;
                }
              return sum;
            };
            err = std::max(err, mu * std::abs(integrate(a) - integrate(b)));
          }
        out << " " << err;
      }
    out << std::endl;
  }

  // Gauss-Laguerre rule with Q points in y = (m+1) mu z / 2
  Rule make_rule(const unsigned int m, const unsigned int Q) const
  {
    Rule r;
    r.m = m;
    r.Q = Q;
    std::vector<double> y, w, psi;
    compute_gauss_laguerre(Q, y, w);
    const double s_m = 0.5 * (m + 1) * mu;
    r.z.resize(Q);
    r.weight.resize(Q);
    r.phi.resize((pz + 1) * Q);
    for (unsigned int q = 0; q < Q; ++q)
      {
        r.z[q]      = y[q] / s_m;
        r.weight[q] = w[q] / s_m;
        laguerre_functions(pz, mu * r.z[q], psi);
        for (unsigned int j = 0; j <= pz; ++j)
          r.phi[j * Q + q] = psi[j];
      }
    return r;
  }

private:
  unsigned int      pz;
  double            mu;
  std::vector<Rule> rules;
  Rule              projection;
};


// ===========================================================================
//  Test cases. Points are full-dimensional, z = p[dim-1] is the unbounded
//  direction.
// ===========================================================================

enum class TestCase
{
  isentropic_vortex,
  damped_vortex
};

enum class BottomBoundary
{
  dirichlet, // all components of w prescribed at z = 0
  slip_wall  // v = 0 at z = 0 (mirror state)
};

// Parameters of the source term
//   b(w) = (0, rho g, rho u.g) - Gamma(z) (M(w) w - M(w_inf) w_inf),
//   Gamma_ii(z) = dgamma_i / (1 + exp((alpha_i - z) / sigma_i)),
// per component i of w = (rho, u, E). dgamma_i = 0: component not damped.
template <int dim>
struct SourceParameters
{
  Tensor<1, dim>              gravity;
  std::array<double, dim + 2> delta_gamma, alpha, sigma;

  SourceParameters()
  {
    delta_gamma.fill(0.);
    alpha.fill(0.);
    sigma.fill(1.);
  }

  // same damping layer for all components
  static SourceParameters uniform_damping(const double dgamma,
                                          const double alpha_,
                                          const double sigma_)
  {
    SourceParameters p;
    p.delta_gamma.fill(dgamma);
    p.alpha.fill(alpha_);
    p.sigma.fill(sigma_);
    return p;
  }

  bool has_gravity() const { return gravity.norm() > 0.; }
  bool has_damping() const { return max_damping() > 0.; }
  double max_damping() const
  {
    return *std::max_element(delta_gamma.begin(), delta_gamma.end());
  }

  double gamma(const unsigned int i, const double z) const
  {
    if (delta_gamma[i] == 0.)
      return 0.;
    // exp overflows to inf far below the layer: gamma -> 0, as it should
    return delta_gamma[i] / (1. + std::exp((alpha[i] - z) / sigma[i]));
  }
};

template <int dim>
class EulerTestCase
{
public:
  EulerTestCase(const TestCase                test,
                const SourceParameters<dim> &source = SourceParameters<dim>())
    : source(source)
    , test(test)
  {
    lower          = Point<dim - 1>();
    repetitions    = std::vector<unsigned int>(dim - 1, 1);
    upper[0]       = 10.;
    repetitions[0] = 10;
    for (unsigned int d = 1; d < dim - 1; ++d)
      upper[d] = 1.;

    if (test == TestCase::isentropic_vortex)
      {
        // vortex of step-67, center (3, 2), advected along x_0
        x_c = 3.;
        z_c = 2.;
        a_x = 1.;
        a_z = 0.;
      }
    else
      {
        // vortex centered at (5, 2), advected along +z into the damping
        // layer
        x_c = 5.;
        z_c = 2.;
        a_x = 0.;
        a_z = 1.;
      }
  }

  std::string name() const
  {
    return test == TestCase::isentropic_vortex ? "isentropic_vortex" :
                                                 "damped_vortex";
  }

  // The damped vortex has no exact solution; state() is then the undamped
  // (pure advection) solution, used as initial condition, as Dirichlet data
  // and as reference below the damping layer.
  bool has_exact_solution() const
  {
    return test == TestCase::isentropic_vortex && !source.has_damping() &&
           !source.has_gravity();
  }

  BottomBoundary bottom_boundary() const { return BottomBoundary::dirichlet; }

  // Far-field state in conserved variables: rho = 1, p = 1, u = a
  Tensor<1, dim + 2> far_field_state() const
  {
    Tensor<1, dim + 2> U;
    U[0]       = 1.;
    U[1]       = a_x;
    U[dim]     = a_z;
    U[dim + 1] = 1. / (gamma_gas - 1.) + 0.5 * (a_x * a_x + a_z * a_z);
    return U;
  }

  // Isentropic vortex of step-67 in the (x_0, z) plane, center (x_c, z_c)
  // at t = 0, advected with velocity a = a_x e_0 + a_z e_z, in conserved
  // variables. Exact solution of the Euler equations without source term.
  Tensor<1, dim + 2> state(const Point<dim> &p, const double t) const
  {
    Tensor<1, dim + 2> U;
    const double       z    = p[dim - 1];
    const double       beta = 5.;
    const double       dx   = p[0] - x_c - a_x * t;
    const double       dz   = z - z_c - a_z * t;
    const double       factor =
      beta / (2. * numbers::PI) * std::exp(1. - dx * dx - dz * dz);
    const double density_log = std::log2(
      std::abs(1. - (gamma_gas - 1.) / gamma_gas * 0.25 * factor * factor));
    const double density = std::exp2(density_log / (gamma_gas - 1.));
    const double pressure =
      std::exp2(density_log * gamma_gas / (gamma_gas - 1.));
    const double u = a_x - factor * dz;
    const double v = a_z + factor * dx;
    U[0]           = density;
    U[1]           = density * u;
    U[dim]         = density * v;
    U[dim + 1] = pressure / (gamma_gas - 1.) + 0.5 * density * (u * u + v * v);
    return U;
  }

  // Height of the region written for visualization
  double output_height() const
  {
    return test == TestCase::isentropic_vortex ? 6. : 10.;
  }

  Point<dim - 1>            lower, upper;
  std::vector<unsigned int> repetitions;
  SourceParameters<dim>     source;

private:
  TestCase test;
  double   x_c, z_c, a_x, a_z;
};

// Vectorized evaluation of EulerTestCase::state at (x, z), x in Omega_b.
// Returns the conserved variables U = (rho, rho u, E).
template <int dim, typename Number>
Tensor<1, dim + 2, VectorizedArray<Number>>
evaluate_state(const EulerTestCase<dim>                      &test_case,
               const Point<dim - 1, VectorizedArray<Number>> &x,
               const double                                   z,
               const double                                   t)
{
  Tensor<1, dim + 2, VectorizedArray<Number>> result;
  for (unsigned int v = 0; v < VectorizedArray<Number>::size(); ++v)
    {
      Point<dim> p;
      for (unsigned int d = 0; d < dim - 1; ++d)
        p[d] = x[d][v];
      p[dim - 1]                     = z;
      const Tensor<1, dim + 2> state = test_case.state(p, t);
      for (unsigned int c = 0; c < dim + 2; ++c)
        result[c][v] = state[c];
    }
  return result;
}


// ===========================================================================
//  Matrix-free DG-Laguerre Euler operator:  dst = d_t w = D(src)^{-1} s(src)
//  for the unknowns w = (rho, u, E). States named "w" are in these
//  variables, states named "U" are conserved variables (rho, rho u, E).
// ===========================================================================

template <int dim, int fe_degree, int n_q_1d, typename Number = double>
class LaguerreDGEulerOperator
{
public:
  static_assert(dim >= 2, "dim is the full dimension: Omega_b x (0,inf)");
  static constexpr int          dim_x  = dim - 1;
  static constexpr unsigned int n_comp = dim + 2;

  using BlockVectorType = LinearAlgebra::distributed::BlockVector<Number>;
  using VA              = VectorizedArray<Number>;
  using G               = Graded<VA>;
  using State           = Tensor<1, n_comp, VA>;
  using FluxX           = Tensor<1, n_comp, Tensor<1, dim_x, VA>>;

  void initialize(std::shared_ptr<const MatrixFree<dim_x, Number>> mf,
                  const ClassicalLaguerreBasis                    &basis,
                  const LaguerreProductRules                      &rules,
                  const EulerTestCase<dim>                        &test_case)
  {
    matrix_free     = mf;
    this->rules     = &rules;
    this->test_case = &test_case;
    n_modes         = basis.n_modes();
    mu              = basis.get_mu();
    lambda_z        = basis.advection_operator_bound();

    // fine z-nodes, used for the time step estimate: L_j(z_p)
    n_fine = basis.n_quadrature_points();
    fine_phi.resize(n_modes * n_fine);
    for (unsigned int p = 0; p < n_fine; ++p)
      for (unsigned int k = 0; k < n_modes; ++k)
        fine_phi[k * n_fine + p] = basis.value(k, p);

    // projection rule of the notes (Q = pz + 1): weight_q L_k(z_q)
    const auto &proj = rules.projection_rule();
    n_proj           = proj.Q;
    proj_z           = proj.z;
    proj_w_phi.resize(n_modes * n_proj);
    for (unsigned int k = 0; k < n_modes; ++k)
      for (unsigned int q = 0; q < n_proj; ++q)
        proj_w_phi[k * n_proj + q] = proj.weight[q] * proj.phi[k * n_proj + q];

    integral_phi.resize(n_modes);
    for (unsigned int k = 0; k < n_modes; ++k)
      integral_phi[k] = basis.integral(k);

    const Tensor<1, n_comp> w = to_rho_u_E<dim>(test_case.far_field_state());
    for (unsigned int c = 0; c < n_comp; ++c)
      w_inf[c] = w[c];

    // source term
    source      = &test_case.source;
    has_gravity = source->has_gravity();
    has_damping = source->has_damping();
    buffer_source.resize(n_modes);
    buffer_linear.resize(n_modes);
    if (has_damping)
      {
        // G(i,k)_c = int Gamma_cc L_i L_k dz, precomputed once (rule s = 1)
        const auto rule_G =
          rules.make_rule(1, rules.n_points_reaction_matrix());
        reaction_matrix = compute_reaction_matrices(rule_G);

        // int Gamma_cc rho' u'_d L_i dz (rule s = 2): weights with Gamma
        const auto rule_C =
          rules.make_rule(2, rules.n_points_reaction_triple());
        Q_cubic = rule_C.Q;
        cubic_phi.assign(rule_C.phi.begin(), rule_C.phi.end());
        cubic_weight.resize(dim * Q_cubic);
        for (unsigned int d = 0; d < dim; ++d)
          for (unsigned int q = 0; q < Q_cubic; ++q)
            cubic_weight[d * Q_cubic + q] =
              rule_C.weight[q] * source->gamma(1 + d, rule_C.z[q]);
      }

    inverse_h.resize(matrix_free->n_cell_batches());
    for (unsigned int cell = 0; cell < matrix_free->n_cell_batches(); ++cell)
      {
        inverse_h[cell] = Number(1.);
        for (unsigned int v = 0;
             v < matrix_free->n_active_entries_per_cell_batch(cell);
             ++v)
          inverse_h[cell][v] =
            1. / matrix_free->get_cell_iterator(cell, v)
                   ->minimum_vertex_distance();
      }
  }

  void initialize_vector(BlockVectorType &v) const
  {
    v.reinit(n_modes);
    for (unsigned int k = 0; k < n_modes; ++k)
      matrix_free->initialize_dof_vector(v.block(k));
    v.collect_sizes();
  }

  void apply(const double           current_time,
             const BlockVectorType &src,
             BlockVectorType       &dst) const
  {
    time = current_time;
    // s(y): DG discretization in x of the pz+1 systems (4)
    matrix_free->loop(&LaguerreDGEulerOperator::local_apply_cell,
                      &LaguerreDGEulerOperator::local_apply_face,
                      &LaguerreDGEulerOperator::local_apply_boundary_face,
                      this,
                      dst,
                      src,
                      /*zero_dst =*/true,
                      MatrixFree<dim_x, Number>::DataAccessOnFaces::values,
                      MatrixFree<dim_x, Number>::DataAccessOnFaces::values);
    // y' = D(y)^{-1} s(y), cell by cell
    matrix_free->cell_loop(&LaguerreDGEulerOperator::local_compute_rates,
                           this,
                           dst,
                           src);
  }

  // w_i(x, 0) = mu int (w(x,z,0) - w_inf) L_i(z) dz, then L2 projection in x
  void project_initial_condition(BlockVectorType &solution) const
  {
    BlockVectorType rhs;
    initialize_vector(rhs);
    FEEvaluation<dim_x, fe_degree, n_q_1d, n_comp, Number> phi(*matrix_free);
    AlignedVector<State> f(n_proj), modes(n_modes * phi.n_q_points);
    for (unsigned int cell = 0; cell < matrix_free->n_cell_batches(); ++cell)
      {
        phi.reinit(cell);
        for (unsigned int q = 0; q < phi.n_q_points; ++q)
          {
            for (unsigned int p = 0; p < n_proj; ++p)
              f[p] = to_rho_u_E<dim>(evaluate_state<dim, Number>(
                *test_case, phi.quadrature_point(q), proj_z[p], 0.));
            // (1/mu) w_k, the factor mu is applied with the mass matrix
            project_in_z(f.data(), &modes[q], phi.n_q_points, Number(1.));
          }
        for (unsigned int k = 0; k < n_modes; ++k)
          {
            for (unsigned int q = 0; q < phi.n_q_points; ++q)
              phi.submit_value(modes[k * phi.n_q_points + q], q);
            phi.integrate(EvaluationFlags::values);
            phi.distribute_local_to_global(rhs.block(k));
          }
      }
    // (1/mu M_x)^{-1}
    matrix_free->cell_loop(
      &LaguerreDGEulerOperator::local_apply_inverse_mass_matrix,
      this,
      solution,
      rhs);
  }

  // Explicit time step  dt = cfl / max( (p+1)^2 (|u_x| + c) / h
  //                                     + mu ||C_z|| (|v| + c) )
  // with the max over all cells, x-quadrature points and z-nodes (incl. 0).
  double compute_time_step(const BlockVectorType &solution,
                           const double           cfl) const
  {
    FEEvaluation<dim_x, fe_degree, fe_degree + 1, n_comp, Number> phi(
      *matrix_free, 0, 1);
    AlignedVector<State> W(n_modes * phi.n_q_points);
    Number               max_rate   = 0;
    const Number         deg_factor = (fe_degree + 1) * (fe_degree + 1);

    for (unsigned int cell = 0; cell < matrix_free->n_cell_batches(); ++cell)
      {
        phi.reinit(cell);
        evaluate_modes(phi, solution, W);
        const VA inv_h = inverse_h[cell];
        VA       rate  = Number(0.);
        for (unsigned int q = 0; q < phi.n_q_points; ++q)
          for (unsigned int p = 0; p <= n_fine; ++p) // p == n_fine: z = 0
            {
              State w = w_inf;
              for (unsigned int j = 0; j < n_modes; ++j)
                w += (p < n_fine ? fine_phi[j * n_fine + p] : Number(1.)) *
                     W[j * phi.n_q_points + q];
              const VA c = std::sqrt(
                std::abs(gamma_gas * pressure_from_rho_u_E<dim>(w) / w[0]));
              VA u_x_sqr = Number(0.);
              for (unsigned int d = 0; d < dim_x; ++d)
                u_x_sqr += w[1 + d] * w[1 + d];
              rate = std::max(rate,
                              deg_factor * inv_h * (std::sqrt(u_x_sqr) + c) +
                                Number(lambda_z) * (std::abs(w[dim]) + c));
            }
        for (unsigned int v = 0;
             v < matrix_free->n_active_entries_per_cell_batch(cell);
             ++v)
          max_rate = std::max(max_rate, rate[v]);
      }
    double dt = cfl / max_rate;
    // explicit treatment of the reaction term: dt * max(dgamma) <= 1 lies
    // well inside the stability region of RK(5,4) on the negative real axis
    if (has_damping)
      dt = std::min(dt, 1. / source->max_damping());
    return dt;
  }

  // Errors ||rho - rho_ex||, ||u - u_ex||, ||E - E_ex|| (the unknowns) in
  // L2(Omega_b x (0,inf)), with Gauss(p+3) in x and a finer z rule
  std::array<double, 3> compute_errors(const BlockVectorType        &solution,
                                       const ClassicalLaguerreBasis &fine,
                                       const double                  t) const
  {
    AssertThrow(fine.n_modes() == n_modes, ExcInternalError());
    std::vector<double> z(fine.n_quadrature_points()),
      wz(fine.n_quadrature_points());
    for (unsigned int p = 0; p < z.size(); ++p)
      {
        z[p]  = fine.quadrature_point(p);
        wz[p] = fine.quadrature_weight(p);
      }
    return compute_l2_norms(solution, z, wz, fine, t, true);
  }

  // L2(Omega_b x Z) norms of rho, u, E of w_h - w_ref, with a z-rule
  // (z_points, z_weights) on Z. w_ref = test_case.state(t) (the exact
  // solution without source) if subtract_state, otherwise w_inf.
  std::array<double, 3>
  compute_l2_norms(const BlockVectorType        &solution,
                   const std::vector<double>    &z_points,
                   const std::vector<double>    &z_weights,
                   const ClassicalLaguerreBasis &basis,
                   const double                  t,
                   const bool                    subtract_state) const
  {
    FEEvaluation<dim_x, fe_degree, fe_degree + 3, n_comp, Number> phi(
      *matrix_free, 0, 2);
    AlignedVector<State>  W(n_modes * phi.n_q_points);
    std::array<double, 3> err_sq = {{0., 0., 0.}};
    const unsigned int    n_z    = z_points.size();
    std::vector<double>   phi_z(n_modes * n_z), v;
    for (unsigned int p = 0; p < n_z; ++p)
      {
        basis.values(z_points[p], v);
        for (unsigned int j = 0; j < n_modes; ++j)
          phi_z[j * n_z + p] = v[j];
      }

    for (unsigned int cell = 0; cell < matrix_free->n_cell_batches(); ++cell)
      {
        phi.reinit(cell);
        evaluate_modes(phi, solution, W);
        std::array<VA, 3> local;
        local.fill(Number(0.));
        for (unsigned int q = 0; q < phi.n_q_points; ++q)
          for (unsigned int p = 0; p < n_z; ++p)
            {
              State e;
              if (subtract_state)
                e = to_rho_u_E<dim>(evaluate_state<dim, Number>(
                      *test_case, phi.quadrature_point(q), z_points[p], t)) -
                    w_inf;
              for (unsigned int j = 0; j < n_modes; ++j)
                e -= Number(phi_z[j * n_z + p]) * W[j * phi.n_q_points + q];
              const VA weight = phi.JxW(q) * Number(z_weights[p]);
              local[0] += e[0] * e[0] * weight;
              for (unsigned int d = 0; d < dim; ++d)
                local[1] += e[1 + d] * e[1 + d] * weight;
              local[2] += e[dim + 1] * e[dim + 1] * weight;
            }
        for (unsigned int v = 0;
             v < matrix_free->n_active_entries_per_cell_batch(cell);
             ++v)
          for (unsigned int i = 0; i < 3; ++i)
            err_sq[i] += local[i][v];
      }
    return {
      {std::sqrt(err_sq[0]), std::sqrt(err_sq[1]), std::sqrt(err_sq[2])}};
  }

  // Mass of the perturbation, int int (rho - rho_inf) dx dz
  //   = sum_k (int L_k dz) (int rho_k dx)
  double compute_mass_perturbation(const BlockVectorType &solution) const
  {
    FEEvaluation<dim_x, fe_degree, fe_degree + 1, n_comp, Number> phi(
      *matrix_free, 0, 1);
    AlignedVector<State> W(n_modes * phi.n_q_points);
    double               mass = 0;
    for (unsigned int cell = 0; cell < matrix_free->n_cell_batches(); ++cell)
      {
        phi.reinit(cell);
        evaluate_modes(phi, solution, W);
        VA local = Number(0.);
        for (unsigned int k = 0; k < n_modes; ++k)
          for (unsigned int q = 0; q < phi.n_q_points; ++q)
            local += Number(integral_phi[k]) *
                     W[k * phi.n_q_points + q][0] * phi.JxW(q);
        for (unsigned int v = 0;
             v < matrix_free->n_active_entries_per_cell_batch(cell);
             ++v)
          mass += local[v];
      }
    return mass;
  }

  // Largest number of CG iterations of the momentum mass solve since the
  // last call (statistics)
  unsigned int get_and_reset_max_cg_iterations() const
  {
    const unsigned int n = max_cg_iterations;
    max_cg_iterations    = 0;
    return n;
  }

  // Check of the z-integrals of the source term: int b(w~) L_i dz, i =
  // 0..pz, computed as in the scheme (exact formulas of Proposition 2 for
  // gravity; precomputed matrices and the s = 2 rule for the reaction) is
  // compared with a brute-force composite Gauss-Legendre quadrature of the
  // pointwise source b(w~(z), z) L_i(z), for a fixed set of modes w_j.
  void verify_source_integrals(std::ostream &out) const
  {
    AlignedVector<State> W(n_modes);
    for (unsigned int j = 0; j < n_modes; ++j)
      for (unsigned int c = 0; c < n_comp; ++c)
        W[j][c] = Number(0.3 * std::sin(1.3 * j + 0.7 * c + 0.4) /
                         (1. + 0.5 * j));

    // brute force: the slowest decay is exp(-mu z / 2) times a polynomial
    // of degree <= 3 pz in mu z
    const unsigned int  pz = n_modes - 1;
    const double        z_max = (12. * pz + 200.) / mu;
    const unsigned int  n_intervals = 4000;
    const QGauss<1>     gauss(8);
    std::vector<double> phi_z;
    Tensor<1, n_comp>   w_inf_s;
    for (unsigned int c = 0; c < n_comp; ++c)
      w_inf_s[c] = w_inf[c][0];
    const Tensor<1, n_comp> U_inf = to_conserved<dim>(w_inf_s);

    std::vector<Tensor<1, n_comp>> brute_g(n_modes), brute_r(n_modes);
    for (unsigned int iv = 0; iv < n_intervals; ++iv)
      for (unsigned int qp = 0; qp < gauss.size(); ++qp)
        {
          const double h  = z_max / n_intervals;
          const double z  = (iv + gauss.point(qp)[0]) * h;
          const double wq = gauss.weight(qp) * h;
          laguerre_functions(pz, mu * z, phi_z);
          Tensor<1, n_comp> w = w_inf_s;
          for (unsigned int j = 0; j < n_modes; ++j)
            for (unsigned int c = 0; c < n_comp; ++c)
              w[c] += phi_z[j] * W[j][c][0];
          const Tensor<1, n_comp> U = to_conserved<dim>(w);
          Tensor<1, n_comp>       S_g, S_r;
          for (unsigned int d = 0; d < dim; ++d)
            {
              S_g[1 + d] = U[0] * source->gravity[d];
              S_g[dim + 1] += U[1 + d] * source->gravity[d];
            }
          for (unsigned int c = 0; c < n_comp; ++c)
            S_r[c] = -source->gamma(c, z) * (U[c] - U_inf[c]);
          for (unsigned int i = 0; i < n_modes; ++i)
            {
              brute_g[i] += (wq * phi_z[i]) * S_g;
              brute_r[i] += (wq * phi_z[i]) * S_r;
            }
        }

    const auto compare = [&](const bool                            gravity,
                             const std::vector<Tensor<1, n_comp>> &brute) {
      AlignedVector<State> result(n_modes);
      add_source_terms(W.data(), 1, result.data(), gravity, !gravity);
      double err = 0, scale = 0;
      for (unsigned int i = 0; i < n_modes; ++i)
        for (unsigned int c = 0; c < n_comp; ++c)
          {
            err   = std::max(err, std::abs(result[i][c][0] - brute[i][c]));
            scale = std::max(scale, std::abs(brute[i][c]));
          }
      return scale > 0 ? err / scale : err;
    };
    out << std::scientific << std::setprecision(2)
        << "  source integrals vs. brute-force quadrature (pz = " << pz
        << "): rel. error gravity (exact) = "
        << (has_gravity ? compare(true, brute_g) : 0.)
        << ", reaction (quadrature) = "
        << (has_damping ? compare(false, brute_r) : 0.) << std::endl;
  }

private:
  // out[k * stride] = factor * (w_k / mu), w_k = mu int (w - w_inf) L_k dz,
  // from the values f[p] = w(z_p) at the nodes of the projection rule:
  //   w_k / mu = sum_p weight_p L_k(z_p) w(z_p) - w_inf int L_k dz,
  // with int L_k dz = 2 (-1)^k / mu.
  void project_in_z(const State       *f,
                    State             *out,
                    const unsigned int stride,
                    const Number       factor) const
  {
    for (unsigned int k = 0; k < n_modes; ++k)
      {
        State s = Number(-integral_phi[k]) * w_inf;
        for (unsigned int p = 0; p < n_proj; ++p)
          s += proj_w_phi[k * n_proj + p] * f[p];
        out[k * stride] = factor * s;
      }
  }

  // G_c(i,k) = int Gamma_cc L_i L_k dz, stored at [(i * N + k) * n_comp + c]
  std::vector<Number>
  compute_reaction_matrices(const LaguerreProductRules::Rule &rule) const
  {
    const unsigned int  N = n_modes, Q = rule.Q;
    std::vector<double> Gm(N * N * n_comp, 0.);
    for (unsigned int c = 0; c < n_comp; ++c)
      if (source->delta_gamma[c] > 0.)
        for (unsigned int q = 0; q < Q; ++q)
          {
            const double gq = rule.weight[q] * source->gamma(c, rule.z[q]);
            for (unsigned int i = 0; i < N; ++i)
              for (unsigned int k = 0; k < N; ++k)
                Gm[(i * N + k) * n_comp + c] +=
                  gq * rule.phi[i * Q + q] * rule.phi[k * Q + q];
          }
    return std::vector<Number>(Gm.begin(), Gm.end());
  }

  // out[i] += int b(w~) L_i dz, i = 0..pz, for the modes W[j * stride]
  //  * gravity (0, rho g, rho u.g): exact, Proposition 2 (N <= 2)
  //  * reaction -Gamma (M(w)w - M(w_inf)w_inf), with w = w_inf + w':
  //      linear part   -G_c (rho', rho_inf u' + u_inf rho', E')_c
  //      quadratic     -int Gamma_cc rho' u'_d L_i dz   (momentum only)
  void add_source_terms(const State       *W,
                        const unsigned int stride,
                        State             *out,
                        const bool         with_gravity = true,
                        const bool         with_damping = true) const
  {
    if (with_gravity && has_gravity)
      {
        const Tensor<1, dim> g = source->gravity;
        const auto           gravity_function =
          [&g](const std::array<G, n_comp> &w, std::array<G, n_comp> &S) {
            gravity_source_rho_u_E<dim>(w, g, S);
          };
        integrate_in_z<n_comp>(W,
                               stride,
                               gravity_function,
                               buffer_source.data(),
                               [](const State &) {});
        for (unsigned int i = 0; i < n_modes; ++i)
          for (unsigned int c = 0; c < n_comp; ++c)
            out[i][c] += buffer_source[i][c];
      }

    if (with_damping && has_damping)
      {
        // linear part
        for (unsigned int k = 0; k < n_modes; ++k)
          {
            const State &Wk = W[k * stride];
            State        X  = Wk;
            for (unsigned int d = 0; d < dim; ++d)
              X[1 + d] = w_inf[0] * Wk[1 + d] + w_inf[1 + d] * Wk[0];
            buffer_linear[k] = X;
          }
        for (unsigned int i = 0; i < n_modes; ++i)
          {
            State s;
            for (unsigned int k = 0; k < n_modes; ++k)
              {
                const Number *Gik =
                  &reaction_matrix[(i * n_modes + k) * n_comp];
                for (unsigned int c = 0; c < n_comp; ++c)
                  s[c] += Gik[c] * buffer_linear[k][c];
              }
            out[i] -= s;
          }

        // quadratic momentum part rho' u'
        for (unsigned int n = 0; n < Q_cubic; ++n)
          {
            VA                 rho_n = Number(0.);
            Tensor<1, dim, VA> u_n;
            for (unsigned int j = 0; j < n_modes; ++j)
              {
                const Number phi_j = cubic_phi[j * Q_cubic + n];
                rho_n += phi_j * W[j * stride][0];
                for (unsigned int d = 0; d < dim; ++d)
                  u_n[d] += phi_j * W[j * stride][1 + d];
              }
            for (unsigned int d = 0; d < dim; ++d)
              u_n[d] *= cubic_weight[d * Q_cubic + n] * rho_n;
            for (unsigned int i = 0; i < n_modes; ++i)
              {
                const Number phi_i = cubic_phi[i * Q_cubic + n];
                for (unsigned int d = 0; d < dim; ++d)
                  out[i][1 + d] -= phi_i * u_n[d];
              }
          }
      }
  }

  // W[j * n_q + q] = w_j(x_q) for all modes j
  template <typename Evaluator>
  void evaluate_modes(Evaluator             &phi,
                      const BlockVectorType &src,
                      AlignedVector<State>  &W) const
  {
    for (unsigned int j = 0; j < n_modes; ++j)
      {
        // Pass the block: with an FESystem and a BlockVector, FEEvaluation
        // would otherwise read n_comp consecutive blocks.
        phi.read_dof_values(src.block(j));
        phi.evaluate(EvaluationFlags::values);
        for (unsigned int q = 0; q < phi.n_q_points; ++q)
          W[j * phi.n_q_points + q] = phi.get_value(q);
      }
  }

  // Exact z-integrals (Proposition 1)
  //     out[i][o] = int_0^inf f_o(w~(z)) L_i(z) dz,   i = 0..pz,
  // of a polynomial map f of degree <= max_factors in w = (rho, u, E), where
  // w~ = w_inf + sum_j W_j L_j and W_j = W[j * stride]. f is called on
  // graded states w_inf + t w'(z_q); the coefficient of t^m is integrated
  // with the rule for |sigma| = m. node_callback(w) is called with the
  // (plain) state at every node of the rules.
  template <unsigned int n_out, typename Func, typename NodeCallback>
  void integrate_in_z(const State                   *W,
                      const unsigned int             stride,
                      const Func                    &f,
                      std::array<VA, n_out>         *out,
                      const NodeCallback            &node_callback) const
  {
    std::array<G, n_comp> g;
    std::array<G, n_out>  fo;
    for (unsigned int c = 0; c < n_comp; ++c)
      g[c].c[0] = w_inf[c];

    // sigma = {}: f(w_inf) int L_i dz
    f(g, fo);
    for (unsigned int i = 0; i < n_modes; ++i)
      for (unsigned int o = 0; o < n_out; ++o)
        out[i][o] = fo[o].c[0] * Number(integral_phi[i]);

    // |sigma| = m >= 1
    for (unsigned int m = 1; m <= max_factors; ++m)
      {
        const auto        &rule = rules->rule(m);
        const unsigned int Q    = rule.Q;
        for (unsigned int q = 0; q < Q; ++q)
          {
            State w_pert;
            for (unsigned int j = 0; j < n_modes; ++j)
              w_pert += Number(rule.phi[j * Q + q]) * W[j * stride];
            for (unsigned int c = 0; c < n_comp; ++c)
              g[c].c[1] = w_pert[c];
            node_callback(w_inf + w_pert);

            f(g, fo);
            std::array<VA, n_out> val;
            for (unsigned int o = 0; o < n_out; ++o)
              val[o] = fo[o].c[m] * Number(rule.weight[q]);
            for (unsigned int i = 0; i < n_modes; ++i)
              {
                const Number phi_i = rule.phi[i * Q + q];
                for (unsigned int o = 0; o < n_out; ++o)
                  out[i][o] += phi_i * val[o];
              }
          }
      }
  }

  // Exterior state at the boundary z = 0 (outward normal -e_z), in
  // conserved variables
  State bottom_exterior_state(const State &U, const Point<dim_x, VA> &x) const
  {
    if (test_case->bottom_boundary() == BottomBoundary::dirichlet)
      return evaluate_state<dim, Number>(*test_case, x, 0., time);
    State U_ext = U;
    U_ext[dim]  = -U[dim]; // mirror the normal momentum
    return U_ext;
  }

  // Cell terms of (4):  (B_i, grad phi) + (h_i, phi), with
  //   B_i = int F_b L_i,  h_i = int f L_i' + f^_0 + int b L_i
  void local_apply_cell(const MatrixFree<dim_x, Number>             &data,
                        BlockVectorType                             &dst,
                        const BlockVectorType                       &src,
                        const std::pair<unsigned int, unsigned int> &range) const
  {
    FEEvaluation<dim_x, fe_degree, n_q_1d, n_comp, Number> phi(data);
    const unsigned int   n_q = phi.n_q_points;
    AlignedVector<State> W(n_modes * n_q), value(n_modes * n_q),
      source_q(n_modes);
    AlignedVector<FluxX> gradient(n_modes * n_q);
    AlignedVector<std::array<VA, n_comp * dim>> A(n_modes);

    Tensor<1, dim, VA> normal_bottom;
    normal_bottom[dim - 1] = Number(-1.);

    const auto flux_function = [](const std::array<G, n_comp>   &w,
                                  std::array<G, n_comp * dim> &out) {
      std::array<std::array<G, dim>, n_comp> F;
      euler_flux_rho_u_E<dim>(w, F);
      for (unsigned int c = 0; c < n_comp; ++c)
        for (unsigned int d = 0; d < dim; ++d)
          out[c * dim + d] = F[c][d];
    };
    const auto no_callback = [](const State &) {};

    for (unsigned int cell = range.first; cell < range.second; ++cell)
      {
        phi.reinit(cell); // once per cell, shared by all modes
        evaluate_modes(phi, src, W);

        for (unsigned int q = 0; q < n_q; ++q)
          {
            // numerical flux at z = 0, F^_z(0) L_i(0) with L_i(0) = 1
            State w0 = w_inf;
            for (unsigned int j = 0; j < n_modes; ++j)
              w0 += W[j * n_q + q];
            const State U0 = to_conserved<dim>(w0);
            const State U0_ext =
              bottom_exterior_state(U0, phi.quadrature_point(q));
            const State bottom_flux =
              -euler_numerical_flux<dim>(U0, U0_ext, normal_bottom);

            integrate_in_z<n_comp * dim>(
              &W[q], n_q, flux_function, A.data(), no_callback);

            // B_i = -mu/2 A_i - mu sum_{l<i} A_l for the z-flux
            State prefix;
            for (unsigned int i = 0; i < n_modes; ++i)
              {
                State Az;
                for (unsigned int c = 0; c < n_comp; ++c)
                  Az[c] = A[i][c * dim + dim - 1];
                value[i * n_q + q] =
                  Number(-0.5 * mu) * Az - Number(mu) * prefix + bottom_flux;
                prefix += Az;
                for (unsigned int c = 0; c < n_comp; ++c)
                  for (unsigned int d = 0; d < dim_x; ++d)
                    gradient[i * n_q + q][c][d] = A[i][c * dim + d];
              }

            // + int b(w~) L_i dz
            if (has_gravity || has_damping)
              {
                for (unsigned int i = 0; i < n_modes; ++i)
                  source_q[i] = State();
                add_source_terms(&W[q], n_q, source_q.data());
                for (unsigned int i = 0; i < n_modes; ++i)
                  value[i * n_q + q] += source_q[i];
              }
          }

        for (unsigned int k = 0; k < n_modes; ++k)
          {
            for (unsigned int q = 0; q < n_q; ++q)
              {
                phi.submit_value(value[k * n_q + q], q);
                phi.submit_gradient(gradient[k * n_q + q], q);
              }
            phi.integrate(EvaluationFlags::values | EvaluationFlags::gradients);
            phi.distribute_local_to_global(dst.block(k));
          }
      }
  }

  // Rusanov flux of the projected system (4) for every test mode i:
  //   F^_i = 1/2 (A_i[F.n](w-) + A_i[F.n](w+)) + lambda/2 (a_i(w-) - a_i(w+))
  // where a_i = A_i[M(w) w] and lambda = max(|u.n| + c) over both sides and
  // all z-nodes. Computes flux[i] for the trace modes W_m, W_p.
  void face_flux(const State              *W_m,
                 const unsigned int        stride_m,
                 const State              *W_p,
                 const unsigned int        stride_p,
                 const Tensor<1, dim, VA> &normal,
                 State                    *flux) const
  {
    const auto face_function = [&normal](const std::array<G, n_comp> &w,
                                         std::array<G, 2 * n_comp>   &out) {
      std::array<std::array<G, dim>, n_comp> F;
      euler_flux_rho_u_E<dim>(w, F);
      for (unsigned int c = 0; c < n_comp; ++c)
        {
          G Fn = F[c][0] * normal[0];
          for (unsigned int d = 1; d < dim - 1; ++d) // normal[dim-1] = 0
            Fn = Fn + F[c][d] * normal[d];
          out[c] = Fn;
        }
      std::array<G, n_comp> U;
      conserved_rho_u_E<dim>(w, U);
      for (unsigned int c = 0; c < n_comp; ++c)
        out[n_comp + c] = U[c];
    };
    VA         lambda = Number(0.);
    const auto max_speed = [&](const State &w) {
      const VA c =
        std::sqrt(std::abs(gamma_gas * pressure_from_rho_u_E<dim>(w) / w[0]));
      VA u_n = Number(0.);
      for (unsigned int d = 0; d < dim_x; ++d)
        u_n += w[1 + d] * normal[d];
      lambda = std::max(lambda, std::abs(u_n) + c);
    };

    integrate_in_z<2 * n_comp>(
      W_m, stride_m, face_function, buffer_m.data(), max_speed);
    integrate_in_z<2 * n_comp>(
      W_p, stride_p, face_function, buffer_p.data(), max_speed);

    for (unsigned int i = 0; i < n_modes; ++i)
      for (unsigned int c = 0; c < n_comp; ++c)
        flux[i][c] =
          Number(0.5) * (buffer_m[i][c] + buffer_p[i][c]) +
          Number(0.5) * lambda *
            (buffer_m[i][n_comp + c] - buffer_p[i][n_comp + c]);
  }

  // Interior faces of Omega_b
  void local_apply_face(const MatrixFree<dim_x, Number>             &data,
                        BlockVectorType                             &dst,
                        const BlockVectorType                       &src,
                        const std::pair<unsigned int, unsigned int> &range) const
  {
    FEFaceEvaluation<dim_x, fe_degree, n_q_1d, n_comp, Number> phi_m(data,
                                                                    true);
    FEFaceEvaluation<dim_x, fe_degree, n_q_1d, n_comp, Number> phi_p(data,
                                                                    false);
    const unsigned int   n_q = phi_m.n_q_points;
    AlignedVector<State> W_m(n_modes * n_q), W_p(n_modes * n_q),
      flux(n_modes * n_q), flux_q(n_modes);
    buffer_m.resize(n_modes);
    buffer_p.resize(n_modes);

    for (unsigned int face = range.first; face < range.second; ++face)
      {
        phi_m.reinit(face);
        phi_p.reinit(face);
        evaluate_modes(phi_m, src, W_m);
        evaluate_modes(phi_p, src, W_p);

        for (unsigned int q = 0; q < n_q; ++q)
          {
            face_flux(&W_m[q],
                      n_q,
                      &W_p[q],
                      n_q,
                      embed_normal(phi_m.normal_vector(q)),
                      flux_q.data());
            for (unsigned int i = 0; i < n_modes; ++i)
              flux[i * n_q + q] = flux_q[i];
          }

        for (unsigned int k = 0; k < n_modes; ++k)
          {
            for (unsigned int q = 0; q < n_q; ++q)
              {
                phi_m.submit_value(-flux[k * n_q + q], q);
                phi_p.submit_value(flux[k * n_q + q], q);
              }
            phi_m.integrate(EvaluationFlags::values);
            phi_m.distribute_local_to_global(dst.block(k));
            phi_p.integrate(EvaluationFlags::values);
            phi_p.distribute_local_to_global(dst.block(k));
          }
      }
  }

  // Lateral boundary dOmega_b x (0,inf): the Dirichlet data are projected
  // onto the Laguerre modes, w+_j = mu int (w_bc - w_inf) L_j dz (projection
  // rule of the notes, Q = pz + 1), and used as
  // exterior state of the Rusanov flux
  void local_apply_boundary_face(
    const MatrixFree<dim_x, Number>             &data,
    BlockVectorType                             &dst,
    const BlockVectorType                       &src,
    const std::pair<unsigned int, unsigned int> &range) const
  {
    FEFaceEvaluation<dim_x, fe_degree, n_q_1d, n_comp, Number> phi(data, true);
    const unsigned int   n_q = phi.n_q_points;
    AlignedVector<State> W(n_modes * n_q), W_bc(n_modes), data_z(n_proj),
      flux(n_modes * n_q), flux_q(n_modes);
    buffer_m.resize(n_modes);
    buffer_p.resize(n_modes);

    for (unsigned int face = range.first; face < range.second; ++face)
      {
        phi.reinit(face);
        evaluate_modes(phi, src, W);

        for (unsigned int q = 0; q < n_q; ++q)
          {
            const Point<dim_x, VA> x = phi.quadrature_point(q);
            for (unsigned int p = 0; p < n_proj; ++p)
              data_z[p] = to_rho_u_E<dim>(
                evaluate_state<dim, Number>(*test_case, x, proj_z[p], time));
            project_in_z(data_z.data(), W_bc.data(), 1, Number(mu));

            face_flux(&W[q],
                      n_q,
                      W_bc.data(),
                      1,
                      embed_normal(phi.normal_vector(q)),
                      flux_q.data());
            for (unsigned int i = 0; i < n_modes; ++i)
              flux[i * n_q + q] = flux_q[i];
          }

        for (unsigned int k = 0; k < n_modes; ++k)
          {
            for (unsigned int q = 0; q < n_q; ++q)
              phi.submit_value(-flux[k * n_q + q], q);
            phi.integrate(EvaluationFlags::values);
            phi.distribute_local_to_global(dst.block(k));
          }
      }
  }

  // dst_k = ((1/mu) M_x)^{-1} src_k for every mode k
  void local_apply_inverse_mass_matrix(
    const MatrixFree<dim_x, Number>             &data,
    BlockVectorType                             &dst,
    const BlockVectorType                       &src,
    const std::pair<unsigned int, unsigned int> &range) const
  {
    FEEvaluation<dim_x, fe_degree, fe_degree + 1, n_comp, Number> phi(data,
                                                                      0,
                                                                      1);
    MatrixFreeOperators::CellwiseInverseMassMatrix<dim_x,
                                                   fe_degree,
                                                   n_comp,
                                                   Number>
      inverse(phi);

    for (unsigned int cell = range.first; cell < range.second; ++cell)
      {
        phi.reinit(cell);
        for (unsigned int k = 0; k < n_modes; ++k)
          {
            phi.read_dof_values(src.block(k));
            inverse.apply(phi.begin_dof_values(), phi.begin_dof_values());
            for (unsigned int i = 0; i < phi.dofs_per_cell; ++i)
              phi.begin_dof_values()[i] *= Number(mu);
            phi.set_dof_values(dst.block(k));
          }
      }
  }

  // Solve D(y) y' = s(y) cell by cell. On input dst holds s(y) for all
  // components; on output the rates (rho', u', E') of the unknowns.
  //  * rho', E':  ((1/mu) M_x)^{-1} s
  //  * u':        sum_k (A_i^{rho, u'_k L_k}, phi) = s_2,i - (A_i^{rho', u}, phi)
  //    with A_i^{rho, u'} = rho_inf / mu u'_i
  //                         + sum_q weight_q rho'(z_q) u''(z_q) L_i(z_q)
  //    (rule m = 2 of Proposition 1, exact), solved by preconditioned CG
  //    with all lanes of the cell batch at once.
  //
  //  Collocated preconditioner. Per velocity component the operator is
  //      A = sum_q JxW_q phi(x_q) phi(x_q)^T (x) M_rho(x_q),
  //      M_rho(x)_{ik} = rho_inf/mu delta_ik
  //                      + sum_n weight_n rho'(x, z_n) L_i(z_n) L_k(z_n),
  //  with q running over the Gauss(p+3) points. Using instead the p+1 Gauss
  //  points (collocation with the dofs) gives P = S^T D S, where S evaluates
  //  the DG basis at these points (square, invertible) and D_q = JxW_q
  //  M_rho(x_q). With M_x = S^T diag(JxW) S,
  //      P^{-1} = S^{-1} diag_q(M_rho(x_q)^{-1}) S M_x^{-1},
  //  i.e. inverse mass matrix, evaluation in the p+1 Gauss points, one
  //  Cholesky solve of size pz+1 per point, and the change of basis
  //  CellwiseInverseMassMatrix::transform_from_q_points_to_basis(). P only
  //  differs from A in the x-quadrature of the density weight; for
  //  rho' = 0 it reduces to (mu / rho_inf) M_x^{-1} = A^{-1}.
  void local_compute_rates(const MatrixFree<dim_x, Number>             &data,
                           BlockVectorType                             &dst,
                           const BlockVectorType                       &src,
                           const std::pair<unsigned int, unsigned int> &range) const
  {
    using Velocity = Tensor<1, dim, VA>;
    FEEvaluation<dim_x, fe_degree, fe_degree + 1, n_comp, Number> phi_mass(
      data, 0, 1);
    MatrixFreeOperators::CellwiseInverseMassMatrix<dim_x,
                                                   fe_degree,
                                                   n_comp,
                                                   Number>
      inverse(phi_mass);
    // Gauss(p+3) integrates rho_j u'_k phi exactly up to p = 5
    FEEvaluation<dim_x, fe_degree, fe_degree + 3, n_comp, Number> phi(data,
                                                                      0,
                                                                      2);

    const unsigned int n_dofs = phi.dofs_per_cell;
    const unsigned int dpc    = n_dofs / n_comp;
    const unsigned int n_q    = phi.n_q_points;
    const unsigned int n_vel  = n_modes * dim * dpc; // size of the system
    const auto        &rule   = rules->rule(2);
    const unsigned int Q      = rule.Q;
    const VA           rho_inf = w_inf[0];

    AlignedVector<VA>       rates(n_modes * n_dofs), b(n_vel), x(n_vel),
      r(n_vel), z(n_vel), p(n_vel), Ap(n_vel), rho_z(n_q * Q);
    AlignedVector<State>    W(n_modes * n_q);
    AlignedVector<VA>       rho_t(n_modes * n_q);
    AlignedVector<Velocity> vel(n_modes * n_q), acc(n_modes * n_q);

    // integrate acc[i * n_q + q] against the velocity test functions and
    // return the velocity dofs in out (layout [(i * dim + d) * dpc + a])
    const auto integrate_velocity = [&](VA *out) {
      for (unsigned int i = 0; i < n_modes; ++i)
        {
          for (unsigned int q = 0; q < n_q; ++q)
            {
              State s;
              for (unsigned int d = 0; d < dim; ++d)
                s[1 + d] = acc[i * n_q + q][d];
              phi.submit_value(s, q);
            }
          phi.integrate(EvaluationFlags::values);
          for (unsigned int d = 0; d < dim; ++d)
            for (unsigned int a = 0; a < dpc; ++a)
              out[(i * dim + d) * dpc + a] =
                phi.begin_dof_values()[(1 + d) * dpc + a];
        }
    };

    // Y = rho-weighted mass matrix applied to the velocity dofs X
    const auto apply_mass = [&](const VA *X, VA *Y) {
      for (unsigned int k = 0; k < n_modes; ++k)
        {
          for (unsigned int i = 0; i < n_dofs; ++i)
            phi.begin_dof_values()[i] = Number(0.);
          for (unsigned int d = 0; d < dim; ++d)
            for (unsigned int a = 0; a < dpc; ++a)
              phi.begin_dof_values()[(1 + d) * dpc + a] =
                X[(k * dim + d) * dpc + a];
          phi.evaluate(EvaluationFlags::values);
          for (unsigned int q = 0; q < n_q; ++q)
            for (unsigned int d = 0; d < dim; ++d)
              vel[k * n_q + q][d] = phi.get_value(q)[1 + d];
        }
      for (unsigned int q = 0; q < n_q; ++q)
        {
          for (unsigned int i = 0; i < n_modes; ++i)
            acc[i * n_q + q] = (rho_inf / Number(mu)) * vel[i * n_q + q];
          for (unsigned int n = 0; n < Q; ++n)
            {
              Velocity u_n;
              for (unsigned int k = 0; k < n_modes; ++k)
                u_n += Number(rule.phi[k * Q + n]) * vel[k * n_q + q];
              u_n *= Number(rule.weight[n]) * rho_z[q * Q + n];
              for (unsigned int i = 0; i < n_modes; ++i)
                acc[i * n_q + q] += Number(rule.phi[i * Q + n]) * u_n;
            }
        }
      integrate_velocity(Y);
    };

    // Collocated preconditioner (see above): Cholesky factors of
    // M_rho(x_q) at the p+1 Gauss points (quadrature 1 = points of phi_mass)
    const unsigned int N    = n_modes;
    const unsigned int n_q1 = phi_mass.n_q_points; // = dpc (collocation)
    AlignedVector<VA>  chol(n_q1 * N * N), diag_inv(n_q1 * N),
      rho1(N * n_q1), vq(n_q1 * dim * N), vals(n_comp * n_q1), c_z(Q);
    std::vector<Number> phi_t(Q * N); // phi_t[n * N + j] = L_j(z_n)
    for (unsigned int n = 0; n < Q; ++n)
      for (unsigned int j = 0; j < N; ++j)
        phi_t[n * N + j] = rule.phi[j * Q + n];
    // L_j(z_n) row-wise, padded with zero rows to a multiple of 4 modes
    const unsigned int  N_pad = (N + 3) / 4 * 4;
    std::vector<Number> phi_pad(N_pad * Q, Number(0.));
    for (unsigned int j = 0; j < N; ++j)
      for (unsigned int n = 0; n < Q; ++n)
        phi_pad[j * Q + n] = rule.phi[j * Q + n];
    AlignedVector<VA> a_iz(N_pad * Q);
    for (unsigned int i = 0; i < N_pad * Q; ++i)
      a_iz[i] = Number(0.); // rows >= N stay zero

    // v <- M_rho(x_q)^{-1} v
    const auto cholesky_solve = [&](const unsigned int q, VA *v) {
      const VA *L    = &chol[q * N * N];
      const VA *dinv = &diag_inv[q * N];
      for (unsigned int i = 0; i < N; ++i)
        {
          VA t = v[i];
          for (unsigned int k = 0; k < i; ++k)
            t -= L[i * N + k] * v[k];
          v[i] = t * dinv[i];
        }
      for (unsigned int i = N; i-- > 0;)
        {
          VA t = v[i];
          for (unsigned int k = i + 1; k < N; ++k)
            t -= L[k * N + i] * v[k];
          v[i] = t * dinv[i];
        }
    };

    // Z = P^{-1} R = S^{-1} diag(M_rho^{-1}) S M_x^{-1} R on the velocity dofs
    const auto precondition = [&](const VA *R, VA *Z) {
      VA *d_ptr = phi_mass.begin_dof_values();
      // S M_x^{-1} R: values at the p+1 Gauss points
      for (unsigned int k = 0; k < N; ++k)
        {
          for (unsigned int i = 0; i < n_dofs; ++i)
            d_ptr[i] = Number(0.);
          for (unsigned int d = 0; d < dim; ++d)
            for (unsigned int a = 0; a < dpc; ++a)
              d_ptr[(1 + d) * dpc + a] = R[(k * dim + d) * dpc + a];
          inverse.apply(d_ptr, d_ptr);
          phi_mass.evaluate(EvaluationFlags::values);
          for (unsigned int q = 0; q < n_q1; ++q)
            for (unsigned int d = 0; d < dim; ++d)
              vq[(q * dim + d) * N + k] = phi_mass.get_value(q)[1 + d];
        }
      // pointwise solves with M_rho(x_q), coupling the modes
      for (unsigned int q = 0; q < n_q1; ++q)
        for (unsigned int d = 0; d < dim; ++d)
          cholesky_solve(q, &vq[(q * dim + d) * N]);
      // S^{-1}: back from point values to the DG basis
      for (unsigned int k = 0; k < N; ++k)
        {
          for (unsigned int i = 0; i < n_comp * n_q1; ++i)
            vals[i] = Number(0.);
          for (unsigned int d = 0; d < dim; ++d)
            for (unsigned int q = 0; q < n_q1; ++q)
              vals[(1 + d) * n_q1 + q] = vq[(q * dim + d) * N + k];
          inverse.transform_from_q_points_to_basis(n_comp,
                                                   vals.data(),
                                                   d_ptr);
          for (unsigned int d = 0; d < dim; ++d)
            for (unsigned int a = 0; a < dpc; ++a)
              Z[(k * dim + d) * dpc + a] = d_ptr[(1 + d) * dpc + a];
        }
    };

    const auto dot = [&](const AlignedVector<VA> &u,
                         const AlignedVector<VA> &v) {
      VA s = Number(0.);
      for (unsigned int i = 0; i < n_vel; ++i)
        s += u[i] * v[i];
      return s;
    };
    const auto safe_divide = [](const VA &num, const VA &den) {
      VA r;
      for (unsigned int v = 0; v < VA::size(); ++v)
        r[v] = den[v] > 0. ? num[v] / den[v] : 0.;
      return r;
    };

    for (unsigned int cell = range.first; cell < range.second; ++cell)
      {
        phi_mass.reinit(cell);
        phi.reinit(cell);
        const unsigned int n_lanes =
          data.n_active_entries_per_cell_batch(cell);

        // residual s(y): momentum part -> b; rho', E' by the inverse mass
        for (unsigned int k = 0; k < n_modes; ++k)
          {
            phi_mass.read_dof_values(dst.block(k));
            VA *d_ptr = phi_mass.begin_dof_values();
            for (unsigned int d = 0; d < dim; ++d)
              for (unsigned int a = 0; a < dpc; ++a)
                b[(k * dim + d) * dpc + a] = d_ptr[(1 + d) * dpc + a];
            inverse.apply(d_ptr, d_ptr);
            for (unsigned int i = 0; i < n_dofs; ++i)
              rates[k * n_dofs + i] = d_ptr[i] * Number(mu);
          }

        // state and rho' at the x-quadrature points
        evaluate_modes(phi, src, W);
        for (unsigned int k = 0; k < n_modes; ++k)
          {
            for (unsigned int i = 0; i < n_dofs; ++i)
              phi.begin_dof_values()[i] = rates[k * n_dofs + i];
            phi.evaluate(EvaluationFlags::values);
            for (unsigned int q = 0; q < n_q; ++q)
              rho_t[k * n_q + q] = phi.get_value(q)[0];
          }
        // density perturbation rho'(x_q, z_n) at the nodes of rule 2 (the
        // far-field part rho_inf is the separate, exact term rho_inf/mu u'_i)
        for (unsigned int q = 0; q < n_q; ++q)
          for (unsigned int n = 0; n < Q; ++n)
            {
              VA s = Number(0.);
              for (unsigned int j = 0; j < n_modes; ++j)
                s += Number(rule.phi[j * Q + n]) * W[j * n_q + q][0];
              rho_z[q * Q + n] = s;
            }

        // b -= (A_i^{rho', u}, phi)
        //   A_i^{rho', u} = u_inf rho'_i / mu
        //                   + sum_n weight_n rho''(z_n) u'(z_n) L_i(z_n)
        for (unsigned int q = 0; q < n_q; ++q)
          {
            for (unsigned int i = 0; i < n_modes; ++i)
              for (unsigned int d = 0; d < dim; ++d)
                acc[i * n_q + q][d] =
                  w_inf[1 + d] * rho_t[i * n_q + q] / Number(mu);
            for (unsigned int n = 0; n < Q; ++n)
              {
                VA       rt = Number(0.);
                Velocity u_n;
                for (unsigned int j = 0; j < n_modes; ++j)
                  {
                    const Number phi_j = rule.phi[j * Q + n];
                    rt += phi_j * rho_t[j * n_q + q];
                    for (unsigned int d = 0; d < dim; ++d)
                      u_n[d] += phi_j * W[j * n_q + q][1 + d];
                  }
                u_n *= Number(rule.weight[n]) * rt;
                for (unsigned int i = 0; i < n_modes; ++i)
                  acc[i * n_q + q] += Number(rule.phi[i * Q + n]) * u_n;
              }
          }
        integrate_velocity(Ap.data()); // Ap as temporary
        for (unsigned int i = 0; i < n_vel; ++i)
          b[i] -= Ap[i];

        // collocated preconditioner: assemble and factorize M_rho(x_q) at
        // the p+1 Gauss points
        for (unsigned int k = 0; k < N; ++k)
          {
            phi_mass.read_dof_values(src.block(k));
            phi_mass.evaluate(EvaluationFlags::values);
            for (unsigned int q = 0; q < n_q1; ++q)
              rho1[k * n_q1 + q] = phi_mass.get_value(q)[0];
          }
        const VA diag_inf = rho_inf / Number(mu);
        for (unsigned int q = 0; q < n_q1; ++q)
          {
            // c_n = weight_n rho'(x_q, z_n), a(i, n) = c_n L_i(z_n), and
            // M_rho(i,k) = sum_n a(i,n) L_k(z_n) as dot products over the
            // z-nodes (contiguous in n, accumulated in registers)
            for (unsigned int n = 0; n < Q; ++n)
              {
                const Number *phi_n = &phi_t[n * N];
                VA            c     = Number(0.);
                for (unsigned int j = 0; j < N; ++j)
                  c += phi_n[j] * rho1[j * n_q1 + q];
                c_z[n] = c * Number(rule.weight[n]);
              }
            for (unsigned int i = 0; i < N; ++i)
              for (unsigned int n = 0; n < Q; ++n)
                a_iz[i * Q + n] = c_z[n] * phi_pad[i * Q + n];

            // lower triangle in 4x4 register blocks (16 independent
            // accumulators); rows >= N of a_iz and phi_pad are zero
            VA *L = &chol[q * N * N];
            for (unsigned int ib = 0; ib < N_pad; ib += 4)
              for (unsigned int kb = 0; kb <= ib; kb += 4)
                {
                  VA acc[4][4];
                  for (unsigned int r = 0; r < 4; ++r)
                    for (unsigned int c = 0; c < 4; ++c)
                      acc[r][c] = Number(0.);
                  const VA     *a0 = &a_iz[ib * Q];
                  const Number *p0 = &phi_pad[kb * Q];
                  for (unsigned int n = 0; n < Q; ++n)
                    {
                      const VA     a[4] = {a0[n],
                                           a0[Q + n],
                                           a0[2 * Q + n],
                                           a0[3 * Q + n]};
                      const Number p[4] = {p0[n],
                                           p0[Q + n],
                                           p0[2 * Q + n],
                                           p0[3 * Q + n]};
                      for (unsigned int r = 0; r < 4; ++r)
                        for (unsigned int c = 0; c < 4; ++c)
                          acc[r][c] += a[r] * p[c];
                    }
                  for (unsigned int r = 0; r < 4 && ib + r < N; ++r)
                    for (unsigned int c = 0; c < 4 && kb + c <= ib + r; ++c)
                      L[(ib + r) * N + kb + c] = acc[r][c];
                }
            for (unsigned int i = 0; i < N; ++i)
              L[i * N + i] += diag_inf;

            // in-place Cholesky (lower triangle); the pivot is bounded from
            // below so that non-physical states (rho~ <= 0) cannot break the
            // factorization of the preconditioner
            for (unsigned int j = 0; j < N; ++j)
              {
                VA s = L[j * N + j];
                for (unsigned int k = 0; k < j; ++k)
                  s -= L[j * N + k] * L[j * N + k];
                s              = std::max(s, Number(1e-12) * diag_inf);
                L[j * N + j]   = std::sqrt(s);
                const VA inv   = Number(1.) / L[j * N + j];
                diag_inv[q * N + j] = inv;
                for (unsigned int i = j + 1; i < N; ++i)
                  {
                    VA t = L[i * N + j];
                    for (unsigned int k = 0; k < j; ++k)
                      t -= L[i * N + k] * L[j * N + k];
                    L[i * N + j] = t * inv;
                  }
              }
          }

        // preconditioned CG for the rho-weighted mass matrix
        for (unsigned int i = 0; i < n_vel; ++i)
          {
            x[i] = Number(0.);
            r[i] = b[i];
          }
        precondition(r.data(), z.data());
        p             = z;
        VA       rz   = dot(r, z);
        const VA bb   = dot(b, b);
        unsigned int it = 0;
        for (; it < 200; ++it)
          {
            apply_mass(p.data(), Ap.data());
            const VA alpha = safe_divide(rz, dot(p, Ap));
            for (unsigned int i = 0; i < n_vel; ++i)
              {
                x[i] += alpha * p[i];
                r[i] -= alpha * Ap[i];
              }
            const VA rr        = dot(r, r);
            bool     converged = true;
            for (unsigned int v = 0; v < n_lanes; ++v)
              if (rr[v] > 1e-26 * bb[v])
                converged = false;
            if (converged)
              break;
            precondition(r.data(), z.data());
            const VA rz_new = dot(r, z);
            const VA beta   = safe_divide(rz_new, rz);
            rz              = rz_new;
            for (unsigned int i = 0; i < n_vel; ++i)
              p[i] = z[i] + beta * p[i];
          }
        max_cg_iterations = std::max(max_cg_iterations, it + 1);

        // write (rho', u', E')
        for (unsigned int k = 0; k < n_modes; ++k)
          {
            VA *d_ptr = phi_mass.begin_dof_values();
            for (unsigned int i = 0; i < n_dofs; ++i)
              d_ptr[i] = rates[k * n_dofs + i];
            for (unsigned int d = 0; d < dim; ++d)
              for (unsigned int a = 0; a < dpc; ++a)
                d_ptr[(1 + d) * dpc + a] = x[(k * dim + d) * dpc + a];
            phi_mass.set_dof_values(dst.block(k));
          }
      }
  }

  static Tensor<1, dim, VA> embed_normal(const Tensor<1, dim_x, VA> &n_x)
  {
    Tensor<1, dim, VA> n;
    for (unsigned int d = 0; d < dim_x; ++d)
      n[d] = n_x[d];
    return n; // n[dim-1] = 0: lateral faces are parallel to the z-axis
  }

  std::shared_ptr<const MatrixFree<dim_x, Number>> matrix_free;
  const LaguerreProductRules                      *rules     = nullptr;
  const EulerTestCase<dim>                        *test_case = nullptr;
  unsigned int                                     n_modes = 0, n_fine = 0;
  double                                           mu = 1., lambda_z = 0.;
  std::vector<double>                              integral_phi;
  std::vector<Number>                              fine_phi;
  // projection rule (initial and boundary data)
  unsigned int        n_proj = 0;
  std::vector<double> proj_z;
  std::vector<Number> proj_w_phi;
  // source term
  const SourceParameters<dim> *source      = nullptr;
  bool                         has_gravity = false, has_damping = false;
  std::vector<Number>          reaction_matrix, cubic_phi, cubic_weight;
  unsigned int                 Q_cubic = 0;
  mutable AlignedVector<std::array<VA, n_comp>> buffer_source;
  mutable AlignedVector<State>                  buffer_linear;
  State                                            w_inf;
  AlignedVector<VA>                                inverse_h;
  mutable AlignedVector<std::array<VA, 2 * n_comp>> buffer_m, buffer_p;
  mutable double                                   time              = 0.;
  mutable unsigned int                             max_cg_iterations = 0;
};


// ===========================================================================
//  Low-storage explicit Runge-Kutta (as in step-67), on BlockVectors
// ===========================================================================

class LowStorageRungeKuttaIntegrator
{
public:
  LowStorageRungeKuttaIntegrator()
  {
    TimeStepping::LowStorageRungeKutta<
      LinearAlgebra::distributed::Vector<double>>
      rk(TimeStepping::LOW_STORAGE_RK_STAGE5_ORDER4);
    rk.get_coefficients(ai, bi, ci);
  }

  unsigned int n_stages() const { return bi.size(); }

  // r_1 = y;  k_i = y'(t + c_i dt, r_i);
  // r_{i+1} = y + a_i dt k_i;  y <- y + b_i dt k_i
  template <typename Operator, typename VectorType>
  void perform_time_step(const Operator &op,
                         const double    t,
                         const double    dt,
                         VectorType     &solution,
                         VectorType     &vec_ri,
                         VectorType     &vec_ki) const
  {
    for (unsigned int stage = 0; stage < bi.size(); ++stage)
      {
        op.apply(t + ci[stage] * dt, stage == 0 ? solution : vec_ri, vec_ki);
        if (stage + 1 < bi.size())
          {
            vec_ri = solution;
            vec_ri.add(ai[stage] * dt, vec_ki);
          }
        solution.add(bi[stage] * dt, vec_ki);
      }
  }

private:
  std::vector<double> ai, bi, ci;
};


// ===========================================================================
//  Driver
// ===========================================================================

// Monitored quantities at the output times (damped vortex):
//   perturbation ||w - w_inf|| in Omega_b x (0, inf) (density, velocity)
//   ||w - w_undamped|| in Omega_b x (0, z_window): deviation from the
//   undamped vortex below the damping layer (reflections)
struct MonitorEntry
{
  double time, pert_rho, pert_u, window_rho, window_u;
};

struct RunResult
{
  unsigned int              n_cells_x, pz, n_steps, max_cg_iterations;
  types::global_dof_index   n_dofs_total;
  std::array<double, 3>     errors; // rho, u, E (exact solution only)
  double                    mass_initial, mass_final, wall_time;
  std::vector<MonitorEntry> monitor;
  bool                      blew_up = false;
};

template <int dim, int fe_degree, typename Number = double>
class LaguerreEulerProblem
{
public:
  static constexpr int          dim_x  = dim - 1;
  static constexpr unsigned int n_comp = dim + 2;
  static constexpr int          n_q_1d = fe_degree + 2; // over-integration

  using BlockVectorType = LinearAlgebra::distributed::BlockVector<Number>;

  LaguerreEulerProblem(const unsigned int           n_refinements,
                       const unsigned int           pz,
                       const double                 mu,
                       const TestCase               test,
                       const double                 cfl,
                       const SourceParameters<dim> &source =
                         SourceParameters<dim>(),
                       const std::string &output_label = "")
    : n_refinements(n_refinements)
    , cfl(cfl)
    , output_label(output_label)
    , test_case(test, source)
    , fe(FE_DGQ<dim_x>(fe_degree), n_comp)
    , dof_handler(triangulation)
    , mapping(1)
    , basis(pz, mu, std::min(300u, 2 * pz + 20))
    , basis_error(pz, mu, std::min(300u, 2 * pz + 60))
    , rules(pz, mu, projection_extra_points)
  {}

  // extra points of the z-projection rule of initial/boundary data, for all
  // problems constructed afterwards. 0 gives the Q = pz + 1 of the notes,
  // which limits the accuracy of the vortex test to ~2e-3 (the nodes in
  // y = mu z / 2 are spread over twice the z-range of the basis, so the data
  // are under-sampled); with 40 extra points the error converges
  // spectrally in pz down to the x-error.
  static inline unsigned int projection_extra_points = 40;

  // Checks of the z-quadrature: Proposition-2 rules and source integrals
  void verify()
  {
    setup();
    rules.verify(std::cout);
    op.verify_source_integrals(std::cout);
  }

  // If n_outputs > 0, the solution is written n_outputs + 1 times
  // (including t = 0) to laguerre_systems_<case><label>_<dim>d_<NNN>.vtu,
  // together with the vertical profile of the vortex (see output_results),
  // and the monitored quantities are recorded.
  RunResult run(const double       final_time,
                const bool         verbose,
                const unsigned int n_outputs = 0)
  {
    setup();
    if (verbose)
      {
        rules.verify(std::cout);
        op.verify_source_integrals(std::cout);
      }

    BlockVectorType solution, vec_ri, vec_ki;
    op.initialize_vector(solution);
    op.initialize_vector(vec_ri);
    op.initialize_vector(vec_ki);
    op.project_initial_condition(solution);

    RunResult r;
    r.mass_initial = op.compute_mass_perturbation(solution);

    const LowStorageRungeKuttaIntegrator integrator;
    double       time = 0, time_step = op.compute_time_step(solution, cfl);
    unsigned int step = 0, output_index = 0;
    if (n_outputs > 0)
      {
        output_results(solution, time, output_index++);
        r.monitor.push_back(monitor(solution, time));
      }

    Timer timer;
    while (time < final_time - 1e-12 * final_time)
      {
        if (step % 10 == 0 && step > 0)
          time_step = op.compute_time_step(solution, cfl);
        // land exactly on the output times / final time
        double next_stop = final_time;
        if (n_outputs > 0)
          next_stop =
            std::min(final_time, output_index * final_time / n_outputs);
        const double dt = std::min(time_step, next_stop - time);

        integrator.perform_time_step(op, time, dt, solution, vec_ri, vec_ki);
        time += dt;
        ++step;

        if (n_outputs > 0 &&
            std::abs(time - output_index * final_time / n_outputs) <
              1e-12 * final_time)
          {
            output_results(solution, time, output_index++);
            r.monitor.push_back(monitor(solution, time));
          }

        if (!std::isfinite(solution.l2_norm()))
          {
            std::cout << "  *** solution blew up at t = " << time
                      << ", run stopped" << std::endl;
            r.blew_up = true;
            break;
          }
      }
    r.wall_time = timer.wall_time();

    r.n_cells_x         = triangulation.n_active_cells();
    r.pz                = basis.get_pz();
    r.n_steps           = step;
    r.max_cg_iterations = op.get_and_reset_max_cg_iterations();
    r.n_dofs_total      = solution.size();
    r.mass_final        = op.compute_mass_perturbation(solution);
    r.errors            = test_case.has_exact_solution() ?
                            op.compute_errors(solution, basis_error, time) :
                            std::array<double, 3>{{0., 0., 0.}};
    if (verbose)
      std::cout << "  " << step << " time steps, last dt = " << time_step
                << ", max CG iterations (momentum mass) = "
                << r.max_cg_iterations << std::endl;
    return r;
  }

private:
  // Minimal DataOutInterface writing hand-built dim-dimensional patches
  struct PatchWriter : public DataOutInterface<dim, dim>
  {
    std::vector<DataOutBase::Patch<dim, dim>> patches;
    std::vector<std::string>                  names;

  protected:
    const std::vector<DataOutBase::Patch<dim, dim>> &
    get_patches() const override
    {
      return patches;
    }
    std::vector<std::string> get_dataset_names() const override
    {
      return names;
    }
    std::vector<
      std::tuple<unsigned int,
                 unsigned int,
                 std::string,
                 DataComponentInterpretation::DataComponentInterpretation>>
    get_nonscalar_data_ranges() const override
    {
      return {std::make_tuple(
        1u,
        static_cast<unsigned int>(dim),
        std::string("velocity"),
        DataComponentInterpretation::component_is_part_of_vector)};
    }
  };

  MonitorEntry monitor(const BlockVectorType &solution, const double time)
  {
    MonitorEntry e;
    e.time = time;
    // whole half-space (fine Gauss-Laguerre rule), reference w_inf
    std::vector<double> z(basis_error.n_quadrature_points()), wz(z.size());
    for (unsigned int p = 0; p < z.size(); ++p)
      {
        z[p]  = basis_error.quadrature_point(p);
        wz[p] = basis_error.quadrature_weight(p);
      }
    auto n     = op.compute_l2_norms(solution, z, wz, basis, time, false);
    e.pert_rho = n[0];
    e.pert_u   = n[1];
    // window (0, z_window) below the layer, Gauss-Legendre, reference the
    // undamped vortex
    const double    z_window = 3.;
    const QGauss<1> gauss(60);
    z.resize(gauss.size());
    wz.resize(gauss.size());
    for (unsigned int p = 0; p < gauss.size(); ++p)
      {
        z[p]  = z_window * gauss.point(p)[0];
        wz[p] = z_window * gauss.weight(p);
      }
    n            = op.compute_l2_norms(solution, z, wz, basis, time, true);
    e.window_rho = n[0];
    e.window_u   = n[1];
    return e;
  }

  void setup()
  {
    if (is_setup)
      return;
    is_setup = true;
    GridGenerator::subdivided_hyper_rectangle(triangulation,
                                              test_case.repetitions,
                                              test_case.lower,
                                              test_case.upper);
    triangulation.refine_global(n_refinements);
    dof_handler.distribute_dofs(fe);
    constraints.close();

    typename MatrixFree<dim_x, Number>::AdditionalData data;
    data.tasks_parallel_scheme =
      MatrixFree<dim_x, Number>::AdditionalData::none;
    data.mapping_update_flags =
      update_gradients | update_JxW_values | update_quadrature_points;
    data.mapping_update_flags_inner_faces =
      update_JxW_values | update_normal_vectors;
    data.mapping_update_flags_boundary_faces =
      update_JxW_values | update_normal_vectors | update_quadrature_points;

    // 0: nonlinear terms (Gauss, p+2), 1: mass matrix (Gauss, p+1),
    // 2: rho-weighted mass matrix and error (Gauss, p+3)
    const std::vector<Quadrature<1>> quads = {QGauss<1>(n_q_1d),
                                              QGauss<1>(fe_degree + 1),
                                              QGauss<1>(fe_degree + 3)};
    const std::vector<const DoFHandler<dim_x> *> dofs = {&dof_handler};
    const std::vector<const AffineConstraints<double> *> cons = {
      &constraints};
    auto mf = std::make_shared<MatrixFree<dim_x, Number>>();
    mf->reinit(mapping, dofs, cons, quads, data);
    matrix_free = mf;

    op.initialize(matrix_free, basis, rules, test_case);
  }

  // Output on Omega_b x (0, Lz): each x-cell is extruded into slabs in z,
  // every slab is a patch subdivided fe_degree+1 times per direction (see
  // laguerre_poisson.cc).
  void output_results(BlockVectorType   &solution,
                      const double       time,
                      const unsigned int index) const
  {
    const double       Lz        = test_case.output_height();
    const unsigned int n_z_slabs = 30;
    const unsigned int n_sub     = fe_degree + 1;
    const unsigned int n_pts_1d  = n_sub + 1;
    const unsigned int n_modes   = basis.n_modes();
    const bool         exact     = test_case.has_exact_solution();

    std::vector<Point<dim_x>> ref_points;
    for (unsigned int i = 0; i < Utilities::fixed_power<dim_x>(n_pts_1d); ++i)
      {
        Point<dim_x> p;
        unsigned int idx = i;
        for (unsigned int d = 0; d < dim_x; ++d, idx /= n_pts_1d)
          p[d] = static_cast<double>(idx % n_pts_1d) / n_sub;
        ref_points.push_back(p);
      }
    const unsigned int n_x = ref_points.size();
    FEValues<dim_x>    fe_values(mapping,
                              fe,
                              Quadrature<dim_x>(ref_points),
                              update_values | update_quadrature_points);

    const unsigned int               n_z_points = n_z_slabs * n_sub + 1;
    std::vector<double>              z_points(n_z_points);
    std::vector<std::vector<double>> phi_z(n_z_points);
    for (unsigned int i = 0; i < n_z_points; ++i)
      {
        z_points[i] = Lz * i / (n_z_points - 1);
        basis.values(z_points[i], phi_z[i]);
      }

    PatchWriter writer;
    writer.names = {"density"};
    for (unsigned int d = 0; d < dim; ++d)
      writer.names.emplace_back("velocity");
    writer.names.emplace_back("pressure");
    if (exact)
      writer.names.emplace_back("density_error");
    const bool damping = test_case.source.has_damping();
    if (damping)
      writer.names.emplace_back("damping_gamma_rho");
    const unsigned int n_fields = writer.names.size();

    // vertical profile of the vortex: max over x of |rho - rho_inf| and of
    // |u - u_inf| at every output height z
    std::vector<double> profile_rho(n_z_points, 0.), profile_u(n_z_points, 0.);
    writer.patches.reserve(triangulation.n_active_cells() * n_z_slabs);

    const Tensor<1, n_comp> w_inf =
      to_rho_u_E<dim>(test_case.far_field_state());
    solution.update_ghost_values();
    std::vector<std::vector<Vector<double>>> U(
      n_modes, std::vector<Vector<double>>(n_x, Vector<double>(n_comp)));

    for (const auto &cell : dof_handler.active_cell_iterators())
      {
        fe_values.reinit(cell);
        for (unsigned int k = 0; k < n_modes; ++k)
          fe_values.get_function_values(solution.block(k), U[k]);

        for (unsigned int slab = 0; slab < n_z_slabs; ++slab)
          {
            DataOutBase::Patch<dim, dim> patch;
            patch.reference_cell = ReferenceCells::get_hypercube<dim>();
            patch.n_subdivisions = n_sub;
            patch.patch_index    = writer.patches.size();

            const unsigned int n_vx = GeometryInfo<dim_x>::vertices_per_cell;
            for (unsigned int v = 0; v < GeometryInfo<dim>::vertices_per_cell;
                 ++v)
              {
                const Point<dim_x> xv = cell->vertex(v % n_vx);
                for (unsigned int d = 0; d < dim_x; ++d)
                  patch.vertices[v][d] = xv[d];
                patch.vertices[v][dim - 1] =
                  z_points[(slab + v / n_vx) * n_sub];
              }

            patch.data.reinit(n_fields, n_x * n_pts_1d);
            for (unsigned int iz = 0; iz < n_pts_1d; ++iz)
              {
                const unsigned int zi = slab * n_sub + iz;
                for (unsigned int q = 0; q < n_x; ++q)
                  {
                    Tensor<1, n_comp> w = w_inf;
                    for (unsigned int k = 0; k < n_modes; ++k)
                      for (unsigned int c = 0; c < n_comp; ++c)
                        w[c] += U[k][q][c] * phi_z[zi][k];
                    const unsigned int i = iz * n_x + q;
                    patch.data(0, i)     = w[0];
                    for (unsigned int d = 0; d < dim; ++d)
                      patch.data(1 + d, i) = w[1 + d]; // velocity
                    patch.data(dim + 1, i) = pressure_from_rho_u_E<dim>(w);
                    if (exact)
                      {
                        Point<dim> p;
                        for (unsigned int d = 0; d < dim_x; ++d)
                          p[d] = fe_values.quadrature_point(q)[d];
                        p[dim - 1] = z_points[zi];
                        patch.data(dim + 2, i) =
                          w[0] - test_case.state(p, time)[0];
                      }
                    if (damping)
                      patch.data(n_fields - 1, i) =
                        test_case.source.gamma(0, z_points[zi]);
                    profile_rho[zi] =
                      std::max(profile_rho[zi], std::abs(w[0] - w_inf[0]));
                    double du = 0;
                    for (unsigned int d = 0; d < dim; ++d)
                      du += (w[1 + d] - w_inf[1 + d]) * (w[1 + d] - w_inf[1 + d]);
                    profile_u[zi] = std::max(profile_u[zi], std::sqrt(du));
                  }
              }
            writer.patches.push_back(std::move(patch));
          }
      }
    solution.zero_out_ghost_values();

    const std::string base = "laguerre_systems_" + test_case.name() +
                             output_label + "_" + std::to_string(dim) + "d_" +
                             Utilities::int_to_string(index, 3);
    const std::string filename = base + ".vtu";
    std::ofstream     out(filename);
    writer.write_vtu(out);

    std::ofstream profile(base + "_profile.txt");
    profile << "# t = " << time
            << "\n# z  max_x|rho - rho_inf|  max_x|u - u_inf|  "
               "gamma_rho(z)\n"
            << std::scientific << std::setprecision(8);
    for (unsigned int zi = 0; zi < n_z_points; ++zi)
      profile << z_points[zi] << ' ' << profile_rho[zi] << ' '
              << profile_u[zi] << ' '
              << test_case.source.gamma(0, z_points[zi]) << '\n';
    std::cout << "  t = " << std::setw(6) << std::fixed << std::setprecision(3)
              << time << ": solution written to " << filename << std::endl;
  }

  const unsigned int                               n_refinements;
  const double                                     cfl;
  const std::string                                output_label;
  bool                                             is_setup = false;
  const EulerTestCase<dim>                         test_case;
  Triangulation<dim_x>                             triangulation;
  FESystem<dim_x>                                  fe;
  DoFHandler<dim_x>                                dof_handler;
  MappingQ<dim_x>                                  mapping;
  AffineConstraints<double>                        constraints;
  std::shared_ptr<const MatrixFree<dim_x, Number>> matrix_free;
  ClassicalLaguerreBasis                           basis, basis_error;
  LaguerreProductRules                             rules;
  LaguerreDGEulerOperator<dim, fe_degree, n_q_1d, Number> op;
};


// ===========================================================================

void print_header()
{
  std::cout << "  cells_x  pz    n_dofs  steps  CG   err(rho)  ratio"
               "      err(u)     err(E)    time [s]"
            << std::endl;
}

void print_row(const RunResult &r, const double previous_error)
{
  std::cout << std::setw(9) << r.n_cells_x << std::setw(4) << r.pz
            << std::setw(10) << r.n_dofs_total << std::setw(7) << r.n_steps
            << std::setw(4) << r.max_cg_iterations << std::scientific
            << std::setprecision(3) << std::setw(11) << r.errors[0];
  if (previous_error > 0)
    std::cout << std::setw(7) << std::fixed << std::setprecision(2)
              << previous_error / r.errors[0];
  else
    std::cout << std::setw(7) << "-";
  std::cout << std::scientific << std::setprecision(3) << std::setw(13)
            << r.errors[1] << std::setw(11) << r.errors[2] << std::setw(11)
            << std::setprecision(2) << r.wall_time << std::endl;
}


void print_monitor(const std::vector<MonitorEntry> &undamped,
                   const std::vector<MonitorEntry> &damped)
{
  std::cout << "\n  ||w - w_inf|| over Omega_b x (0,inf)      "
               "||w - w_undamped_exact|| over Omega_b x (0,3)\n"
            << "      t   rho(no damp)  rho(damp)   u(no damp)   u(damp)"
               "   | rho(no damp)  rho(damp)   u(no damp)   u(damp)\n";
  const double nan = std::numeric_limits<double>::quiet_NaN();
  for (unsigned int i = 0; i < damped.size(); ++i)
    {
      // missing entries (undamped run stopped) are printed as nan
      const MonitorEntry  missing{damped[i].time, nan, nan, nan, nan};
      const MonitorEntry &a = i < undamped.size() ? undamped[i] : missing;
      const MonitorEntry &b = damped[i];
      std::cout << std::fixed << std::setprecision(2) << std::setw(7)
                << b.time << std::scientific << std::setprecision(3)
                << std::setw(12) << a.pert_rho << std::setw(12) << b.pert_rho
                << std::setw(12) << a.pert_u << std::setw(12) << b.pert_u
                << "   |" << std::setw(12) << a.window_rho << std::setw(12)
                << b.window_rho << std::setw(12) << a.window_u
                << std::setw(12) << b.window_u << std::endl;
    }
}


int main(int argc, char **argv)
{
  try
    {
      Utilities::MPI::MPI_InitFinalize mpi(argc, argv, 1);

      // usage: laguerre_systems [verify|convergence|damping|all] [n_extra]
      // n_extra: extra points of the projection rule (default 40, 0 = the
      // Q = pz + 1 of the notes)
      const std::string mode = argc > 1 ? argv[1] : "all";
      const bool        all  = mode == "all";

      constexpr int dim       = 2; // full dimension: Omega_b subset R^(dim-1)
      constexpr int fe_degree = 3;
      if (argc > 2)
        LaguerreEulerProblem<dim, fe_degree>::projection_extra_points =
          std::stoi(argv[2]);
      // Laguerre scaling mu (equal for all components and for the test
      // functions): L_j(z) = exp(-mu z/2) L_j(mu z). The vortex lives at
      // z ~ 2 with width ~ 1, for which mu = 10 is a good choice.
      const double mu = 10.;
      // relative to the bound in compute_time_step() (see laguerre_euler.cc)
      const double cfl = 4.;

      std::cout << "Euler with source term on Omega_b x [0,inf), dim = " << dim
                << ", DG Q" << fe_degree << " in x, Laguerre in z, mu = " << mu
                << "\nSIMD width: " << VectorizedArray<double>::size()
                << " doubles\n";

      // ---- checks of the z-integrals (with gravity and damping) -----------
      if (all || mode == "verify")
        {
          std::cout << "\n========== verification of the z-integrals "
                       "==========\n";
          SourceParameters<dim> source;
          source.gravity[0]       = 0.3;
          source.gravity[dim - 1] = -1.;
          source.delta_gamma      = {{1., 2., 2.5, 3.}};
          source.alpha            = {{1.5, 2., 2.5, 2.}};
          source.sigma            = {{0.3, 0.5, 0.4, 0.6}};
          for (const unsigned int pz : {8u, 16u, 32u, 48u})
            {
              LaguerreEulerProblem<dim, fe_degree> problem(
                1, pz, mu, TestCase::damped_vortex, cfl, source);
              problem.verify();
            }
        }

      // ---- isentropic vortex without source: convergence study ------------
      if (all || mode == "convergence")
        {
          const double final_time = 1.;
          std::cout << "\nprojection rule of initial/boundary data: pz + 1 + "
                    << LaguerreEulerProblem<dim, fe_degree>::projection_extra_points
                    << " Gauss-Laguerre points\n";
          std::cout << "\n========== isentropic vortex, no source term "
                       "(exact Dirichlet data), t_end = "
                    << final_time << " ==========\n";

          const unsigned int ref_spectral = (dim == 2 ? 3 : 2);
          std::cout << "\n--- Spectral convergence in z ---\n";
          print_header();
          double prev = -1;
          for (const unsigned int pz : {12u, 16u, 24u, 32u, 40u, 48u})
            {
              LaguerreEulerProblem<dim, fe_degree> problem(
                ref_spectral, pz, mu, TestCase::isentropic_vortex, cfl);
              const RunResult r = problem.run(final_time, pz == 32u);
              print_row(r, prev);
              prev = r.errors[0];
            }

          std::cout << "\n--- h-convergence in x (pz = 64), expected ratio "
                       "2^(p+1) = "
                    << (1 << (fe_degree + 1)) << " ---\n";
          print_header();
          prev = -1;
          for (unsigned int ref = 0; ref <= (dim == 2 ? 3u : 2u); ++ref)
            {
              LaguerreEulerProblem<dim, fe_degree> problem(
                ref, 64, mu, TestCase::isentropic_vortex, cfl);
              const RunResult r = problem.run(final_time, false);
              print_row(r, prev);
              prev = r.errors[0];
            }
        }

      // ---- vortex moving along +z into a damping layer --------------------
      if (all || mode == "damping")
        {
          const double       final_time = 6.;
          const unsigned int n_outputs  = 12;
          const unsigned int pz         = 48;
          const double       dgamma = 3., alpha = 4.5, sigma = 0.4;
          std::cout << "\n========== vortex advected along +z, damping layer "
                       "dgamma = "
                    << dgamma << ", alpha = " << alpha << ", sigma = " << sigma
                    << ", pz = " << pz << ", t_end = " << final_time
                    << " ==========\n";

          std::cout << "\n--- without damping ---\n";
          LaguerreEulerProblem<dim, fe_degree> undamped(
            3, pz, mu, TestCase::damped_vortex, cfl, {}, "_nodamping");
          const RunResult r0 = undamped.run(final_time, false, n_outputs);

          std::cout << "\n--- with damping ---\n";
          LaguerreEulerProblem<dim, fe_degree> damped(
            3,
            pz,
            mu,
            TestCase::damped_vortex,
            cfl,
            SourceParameters<dim>::uniform_damping(dgamma, alpha, sigma));
          const RunResult r1 = damped.run(final_time, true, n_outputs);

          print_monitor(r0.monitor, r1.monitor);
        }
    }
  catch (std::exception &exc)
    {
      std::cerr << "Exception: " << exc.what() << std::endl;
      return 1;
    }
  return 0;
}
