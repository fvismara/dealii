// ---------------------------------------------------------------------
//
// Copyright (C) 2014 - 2022 by the deal.II authors
//
// This file is part of the deal.II library.
//
// The deal.II library is free software; you can use it, redistribute
// it, and/or modify it under the terms of the GNU Lesser General
// Public License as published by the Free Software Foundation; either
// version 2.1 of the License, or (at your option) any later version.
// The full text of the license can be found in the file LICENSE.md at
// the top level directory of deal.II.
//
// ---------------------------------------------------------------------


// check initialiser (aka do_reinit). .
//
// PETScWrappers::FullMatrix::reinit ()

#include <deal.II/lac/petsc_full_matrix.h>

#include <iostream>

#include "../tests.h"

// Simply take an already initialized full matrix, fill in some of the
// elements, reinitialise it to a different size and fill in some
// elements again. To be sure this generates the correct results the
// test is verbose (and uses tiny matrices).
void
test(PETScWrappers::FullMatrix &m)
{
  // assign some matrix elements
  for (unsigned int i = 0; i < m.m(); ++i)
    for (unsigned int j = 0; j < m.n(); ++j)
      m.set(i, j, i + 2 * j);

  m.compress(VectorOperation::insert);

  // things we know
  AssertThrow(m.m() == 3, ExcInternalError());
  AssertThrow(m.n() == 3, ExcInternalError());

  // Generate some output
  deallog << "initial matrix: " << std::endl;
  for (unsigned int i = 0; i < m.m(); ++i)
    {
      for (unsigned int j = 0; j < m.n(); ++j)
        deallog << m(i, j) << ' ';
      deallog << std::endl;
    }
  deallog << std::endl;

  // test reinit, this time a different size matrix
  m.reinit(5, 5);

  // set some entries
  for (unsigned int i = 0; i < m.m(); ++i)
    for (unsigned int j = 0; j < m.n(); ++j)
      m.set(i, j, j + 2 * i);
  m.compress(VectorOperation::insert);

  // things we know
  AssertThrow(m.m() == 5, ExcInternalError());
  AssertThrow(m.n() == 5, ExcInternalError());

  // Generate some output
  deallog << "after reinit: " << std::endl;
  for (unsigned int i = 0; i < m.m(); ++i)
    {
      for (unsigned int j = 0; j < m.n(); ++j)
        deallog << m(i, j) << ' ';
      deallog << std::endl;
    }
  deallog << std::endl;

  // done
  deallog << "OK" << std::endl;
}

int
main(int argc, char **argv)
{
  initlog();

  try
    {
      Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, 1);
      {
        // Standard initialiser...
        PETScWrappers::FullMatrix m(3, 3);
        test(m);
      }
    }
  catch (const std::exception &exc)
    {
      std::cerr << std::endl
                << std::endl
                << "----------------------------------------------------"
                << std::endl;
      std::cerr << "Exception on processing: " << std::endl
                << exc.what() << std::endl
                << "Aborting!" << std::endl
                << "----------------------------------------------------"
                << std::endl;

      return 1;
    }
  catch (...)
    {
      std::cerr << std::endl
                << std::endl
                << "----------------------------------------------------"
                << std::endl;
      std::cerr << "Unknown exception!" << std::endl
                << "Aborting!" << std::endl
                << "----------------------------------------------------"
                << std::endl;
      return 1;
    };
}
