// ---------------------------------------------------------------------
//
// Copyright (C) 2016 - 2022 by the deal.II authors
//
// This file is part of the deal.II library.
//
// The deal.II library is free software; you can use it, redistribute
// it, and/or modify it under the terms of the GNU Lesser General
// Public License as published by the Free Software Foundation; either
// version 2.1 of the License, or (at your option) any later version.
// The full text of the license can be found in the file LICENSE at
// the top level of the deal.II distribution.
//
// ---------------------------------------------------------------------


// Verify that the maximum number of indices that can be used to record tapes is
// indeed that stored within the HelperBase class
//
// AD number type: ADOL-C taped

#include <deal.II/base/logstream.h>
#include <deal.II/base/symmetric_tensor.h>
#include <deal.II/base/tensor.h>

#include <deal.II/differentiation/ad.h>

#include <deal.II/fe/fe_values_extractors.h>

#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/vector.h>

#include <iostream>

#include "../tests.h"

namespace AD = dealii::Differentiation::AD;

// Function and its derivatives
template <int dim, typename NumberType>
struct FunctionsTestScalarScalarCoupled
{
  static NumberType
  psi(const NumberType &s1, const NumberType &s2)
  {
    return 2.0 * pow(s1, 4) * pow(s2, 3);
  };

  static NumberType
  dpsi_ds1(const NumberType &s1, const NumberType &s2)
  {
    return 8.0 * pow(s1, 3) * pow(s2, 3);
  };

  static NumberType
  dpsi_ds2(const NumberType &s1, const NumberType &s2)
  {
    return 6.0 * pow(s1, 4) * pow(s2, 2);
  };

  static NumberType
  d2psi_ds1_ds1(const NumberType &s1, const NumberType &s2)
  {
    return 24.0 * pow(s1, 2) * pow(s2, 3);
  };

  static NumberType
  d2psi_ds2_ds1(const NumberType &s1, const NumberType &s2)
  {
    return 24.0 * pow(s1, 3) * pow(s2, 2);
  };

  static NumberType
  d2psi_ds1_ds2(const NumberType &s1, const NumberType &s2)
  {
    return d2psi_ds2_ds1(s1, s2);
  };

  static NumberType
  d2psi_ds2_ds2(const NumberType &s1, const NumberType &s2)
  {
    return 12.0 * pow(s1, 4) * pow(s2, 1);
  };
};

template <int dim,
          typename number_t,
          enum AD::NumberTypes ad_type_code,
          typename tape_index_t>
void
test_scalar_scalar_coupled_one_run(const number_t     s1,
                                   const number_t     s2,
                                   const tape_index_t tape_no)
{
  using ADHelper         = AD::ScalarFunction<dim, ad_type_code, number_t>;
  using ADNumberType     = typename ADHelper::ad_type;
  using ScalarNumberType = typename ADHelper::scalar_type;

  std::cout
    //      << "*** Test variables: Scalar + Scalar (coupled), "
    //      << "dim = " << Utilities::to_string(dim) << ", "
    //      << "Type code: " << static_cast<int>(ad_type_code) << ". "
    << "Tape number: " << tape_no << std::endl;

  // Values computed from the AD energy function
  ScalarNumberType             psi;
  Vector<ScalarNumberType>     Dpsi;
  FullMatrix<ScalarNumberType> D2psi;

  // Function and its derivatives
  using func_ad = FunctionsTestScalarScalarCoupled<dim, ADNumberType>;

  // Setup the variable components and choose a value at which to
  // evaluate the tape
  const FEValuesExtractors::Scalar s1_dof(0);
  const FEValuesExtractors::Scalar s2_dof(1);
  const unsigned int               n_AD_components = 2;
  ADHelper                         ad_helper(n_AD_components);
  ad_helper.set_tape_buffer_sizes();     // Increase the buffer size from the
                                         // default values
  const bool write_tapes_to_file = true; // This flag will help check whether
                                         // tape indices are universally valid

  // Configure tape
  const bool is_recording =
    ad_helper.start_recording_operations(tape_no,
                                         false /*overwrite_tape*/,
                                         true /*keep*/);
  if (is_recording == true)
    {
      ad_helper.register_independent_variable(s1, s1_dof);
      ad_helper.register_independent_variable(s2, s2_dof);

      const ADNumberType s1_ad = ad_helper.get_sensitive_variables(s1_dof);
      const ADNumberType s2_ad = ad_helper.get_sensitive_variables(s2_dof);

      const ADNumberType psi(func_ad::psi(s1_ad, s2_ad));

      ad_helper.register_dependent_variable(psi);
      ad_helper.stop_recording_operations(write_tapes_to_file);

      //      std::cout << "Taped data..." << std::endl;
      //      std::cout << "independent variable values: " << std::flush;
      //      ad_helper.print_values(std::cout);
      //      std::cout << "s1_ad: " << s1_ad << std::endl;
      //      std::cout << "s2_ad: " << s2_ad << std::endl;
      //      std::cout << "psi: " << psi << std::endl;
      //      std::cout << std::endl;
    }
  else
    {
      //      std::cout << "Using tape with different values for independent
      //      variables..." << std::endl;
      // Set a new evaluation point
      ad_helper.activate_recorded_tape(tape_no);
      ad_helper.set_independent_variable(s1, s1_dof);
      ad_helper.set_independent_variable(s2, s2_dof);
    }

  //  std::cout << "independent variable values: " << std::flush;
  //  ad_helper.print_values(std::cout);

  // Compute the function value, gradient and hessian for the new evaluation
  // point
  psi = ad_helper.compute_value();
  ad_helper.compute_gradient(Dpsi);
  if (AD::ADNumberTraits<ADNumberType>::n_supported_derivative_levels >= 2)
    {
      ad_helper.compute_hessian(D2psi);
    }

  // Output the full stored function, gradient vector and hessian matrix
  //  std::cout << "psi: " << psi << std::endl;
  //  std::cout << "Dpsi: \n";
  //  Dpsi.print(std::cout);
  //  if (AD::ADNumberTraits<ADNumberType>::n_supported_derivative_levels >= 2)
  //    {
  //      std::cout << "D2psi: \n";
  //      D2psi.print_formatted(std::cout,3,true,0,"0.0");
  //    }

  // Extract components of the solution
  const ScalarNumberType dpsi_ds1 =
    ad_helper.extract_gradient_component(Dpsi, s1_dof);
  const ScalarNumberType dpsi_ds2 =
    ad_helper.extract_gradient_component(Dpsi, s2_dof);
  //  std::cout
  //      << "extracted Dpsi (s1): " << dpsi_ds1 << "\n"
  //      << "extracted Dpsi (s2): " << dpsi_ds2 << "\n";

  // Verify the result
  using func = FunctionsTestScalarScalarCoupled<dim, ScalarNumberType>;
  static const ScalarNumberType tol = 1.e-7;
  //  std::cout << "dpsi_ds1:            " << dpsi_ds1 << std::endl;
  //  std::cout << "func::dpsi_ds1(s1,s2): " << func::dpsi_ds1(s1,s2) <<
  //  std::endl; std::cout << "diff: " << std::abs(dpsi_ds1 -
  //  func::dpsi_ds1(s1,s2)) << std::endl; std::cout << "dpsi_ds2:            "
  //  << dpsi_ds2 << std::endl; std::cout << "func::dpsi_ds2(s1,s2): " <<
  //  func::dpsi_ds2(s1,s2) << std::endl; std::cout << "diff: " <<
  //  std::abs(dpsi_ds2 - func::dpsi_ds2(s1,s2)) << std::endl;
  Assert(std::abs(psi - func::psi(s1, s2)) < tol,
         ExcMessage("No match for function value."));
  Assert(std::abs(dpsi_ds1 - func::dpsi_ds1(s1, s2)) < tol,
         ExcMessage("No match for first derivative."));
  Assert(std::abs(dpsi_ds2 - func::dpsi_ds2(s1, s2)) < tol,
         ExcMessage("No match for first derivative."));
  if (AD::ADNumberTraits<ADNumberType>::n_supported_derivative_levels >= 2)
    {
      const ScalarNumberType d2psi_ds1_ds1 =
        ad_helper.extract_hessian_component(D2psi, s1_dof, s1_dof);
      const ScalarNumberType d2psi_ds2_ds1 =
        ad_helper.extract_hessian_component(D2psi, s1_dof, s2_dof);
      const ScalarNumberType d2psi_ds1_ds2 =
        ad_helper.extract_hessian_component(D2psi, s2_dof, s1_dof);
      const ScalarNumberType d2psi_ds2_ds2 =
        ad_helper.extract_hessian_component(D2psi, s2_dof, s2_dof);
      //      std::cout
      //          << "extracted D2psi (s1,s1): " << d2psi_ds1_ds1 << "\n"
      //          << "extracted D2psi (s1,s2): " << d2psi_ds2_ds1 << "\n"
      //          << "extracted D2psi (s2,s1): " << d2psi_ds1_ds2 << "\n"
      //          << "extracted D2psi (s2,s2): " << d2psi_ds2_ds2 << "\n"
      //          << std::endl;
      //      std::cout << "d2psi_ds1_ds1:            " << d2psi_ds1_ds1 <<
      //      std::endl; std::cout << "func::d2psi_ds1_ds1(s1,s2): " <<
      //      func::d2psi_ds1_ds1(s1,s2) << std::endl; std::cout << "diff: " <<
      //      std::abs(d2psi_ds1_ds1 - func::d2psi_ds1_ds1(s1,s2)) << std::endl;
      //      std::cout << "d2psi_ds2_ds1:            " << d2psi_ds2_ds1 <<
      //      std::endl; std::cout << "func::d2psi_ds2_ds1(s1,s2): " <<
      //      func::d2psi_ds2_ds1(s1,s2) << std::endl; std::cout << "diff: " <<
      //      std::abs(d2psi_ds2_ds1 - func::d2psi_ds2_ds1(s1,s2)) << std::endl;
      //      std::cout << "d2psi_ds1_ds2:            " << d2psi_ds1_ds2 <<
      //      std::endl; std::cout << "func::d2psi_ds1_ds2(s1,s2): " <<
      //      func::d2psi_ds1_ds2(s1,s2) << std::endl; std::cout << "diff: " <<
      //      std::abs(d2psi_ds1_ds2 - func::d2psi_ds1_ds2(s1,s2)) << std::endl;
      //      std::cout << "d2psi_ds2_ds2:            " << d2psi_ds2_ds2 <<
      //      std::endl; std::cout << "func::d2psi_ds2_ds2(s1,s2): " <<
      //      func::d2psi_ds2_ds2(s1,s2) << std::endl; std::cout << "diff: " <<
      //      std::abs(d2psi_ds2_ds2 - func::d2psi_ds2_ds2(s1,s2)) << std::endl;
      Assert(std::abs(d2psi_ds1_ds1 - func::d2psi_ds1_ds1(s1, s2)) < tol,
             ExcMessage("No match for second derivative."));
      Assert(std::abs(d2psi_ds2_ds1 - func::d2psi_ds2_ds1(s1, s2)) < tol,
             ExcMessage("No match for second derivative."));
      Assert(std::abs(d2psi_ds1_ds2 - func::d2psi_ds1_ds2(s1, s2)) < tol,
             ExcMessage("No match for second derivative."));
      Assert(std::abs(d2psi_ds2_ds2 - func::d2psi_ds2_ds2(s1, s2)) < tol,
             ExcMessage("No match for second derivative."));
    }
}



template <int dim,
          typename number_t,
          enum AD::NumberTypes ad_type_code,
          typename tape_index_t>
void
test_scalar_scalar_coupled(const tape_index_t tape_index_begin,
                           const tape_index_t tape_index_end)
{
  for (tape_index_t tape_no = tape_index_begin; tape_no < tape_index_end;
       ++tape_no)
    {
      const number_t s1 = 3.1 * (1.0 + tape_no / 10.0);
      const number_t s2 = 5.9 * (1.0 + tape_no / 10.0);
      test_scalar_scalar_coupled_one_run<3, number_t, ad_type_code>(s1,
                                                                    s2,
                                                                    tape_no);
    }
}


int
main()
{
  initlog();

  using number_t                     = double;
  const AD::NumberTypes ad_type_code = AD::NumberTypes::adolc_taped;
  using ADNumberType = typename AD::HelperBase<ad_type_code, number_t>::ad_type;
  using tape_index_t = typename AD::Types<ADNumberType>::tape_index;

  const tape_index_t invalid_tape_index =
    AD::Numbers<ADNumberType>::invalid_tape_index;
  const tape_index_t max_tape_index = AD::Numbers<ADNumberType>::max_tape_index;

  deallog.push("Valid range");
  {
    test_scalar_scalar_coupled<2, number_t, ad_type_code>(
      static_cast<tape_index_t>(invalid_tape_index + 1), max_tape_index);
    test_scalar_scalar_coupled<3, number_t, ad_type_code>(
      static_cast<tape_index_t>(invalid_tape_index + 1), max_tape_index);
    deallog << "OK" << std::endl;
  }
  deallog.pop();

  // We expect the following two checks to fail
  deal_II_exceptions::disable_abort_on_exception();

  deallog.push("Invalid index");
  {
    try
      {
        test_scalar_scalar_coupled<2, number_t, ad_type_code>(
          invalid_tape_index,
          static_cast<tape_index_t>(invalid_tape_index + 1));
        test_scalar_scalar_coupled<3, number_t, ad_type_code>(
          invalid_tape_index,
          static_cast<tape_index_t>(invalid_tape_index + 1));
        deallog << "Expected failure" << std::endl;
      }
    catch (...)
      {
        deallog << "OK" << std::endl;
      }
  }
  deallog.pop();

  deallog.push("Max index");
  {
    try
      {
        test_scalar_scalar_coupled<2, number_t, ad_type_code>(
          max_tape_index, static_cast<tape_index_t>(max_tape_index + 1));
        test_scalar_scalar_coupled<3, number_t, ad_type_code>(
          max_tape_index, static_cast<tape_index_t>(max_tape_index + 1));
        deallog << "Expected failure" << std::endl;
      }
    catch (...)
      {
        deallog << "OK" << std::endl;
      }
  }
  deallog.pop();

  deallog << "OK" << std::endl;
}
