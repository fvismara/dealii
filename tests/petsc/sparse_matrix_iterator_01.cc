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



// test PETScWrappers::MatrixBase::const_iterator

#include <deal.II/lac/petsc_sparse_matrix.h>

#include <iostream>

#include "../tests.h"


void
test()
{
  PETScWrappers::SparseMatrix m(5, 5, 5);
  m.set(0, 0, 1);
  m.set(1, 1, 2);
  m.set(1, 2, 3);
  m.compress(VectorOperation::insert);
  PETScWrappers::SparseMatrix::const_iterator i = m.begin();
  deallog << i->row() << ' ' << i->column() << ' ' << i->value() << std::endl;
  ++i;
  deallog << i->row() << ' ' << i->column() << ' ' << i->value() << std::endl;
  i++;
  deallog << i->row() << ' ' << i->column() << ' ' << i->value() << std::endl;

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
        test();
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
