// ---------------------------------------------------------------------
//
// Copyright (C) 2004 - 2018 by the deal.II authors
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



// this is equivalent to the petsc vector_assign_01 test, except that we use
// operator+= instead of operator=. Now, this does not present a problem,
// since the compiler does not automatically generate a version of this
// operator, but simply performs the conversion to PetscScalar, i.e. the
// argument to the user-defined operator+=. This is not exciting, but since I
// wrote the test to make sure it works this way, let's keep it then...

#include <deal.II/lac/petsc_vector.h>

#include <iostream>
#include <vector>

#include "../tests.h"


void
test(PETScWrappers::MPI::Vector &v, PETScWrappers::MPI::Vector &w)
{
  // set the first vector
  for (unsigned int i = 0; i < v.size(); ++i)
    v(i) = i;

  // add elements by reference
  for (unsigned int i = 0; i < v.size(); ++i)
    w(i) += v(i);

  // check that they're equal
  AssertThrow(v == w, ExcInternalError());

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
        IndexSet indices(100);
        indices.add_range(0, 100);
        PETScWrappers::MPI::Vector v(indices, MPI_COMM_WORLD);
        PETScWrappers::MPI::Vector w(indices, MPI_COMM_WORLD);
        test(v, w);
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
