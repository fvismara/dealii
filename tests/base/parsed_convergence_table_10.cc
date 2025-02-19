// ---------------------------------------------------------------------
//
// Copyright (C) 2019 - 2022 by the deal.II authors
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

// Test the functionality of the ParsedConvergenceTable class for
// custom error computations, writing to files

#include <deal.II/base/function_lib.h>
#include <deal.II/base/parameter_handler.h>
#include <deal.II/base/parsed_convergence_table.h>

#include <deal.II/dofs/dof_handler.h>

#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_system.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/tria.h>

#include <deal.II/lac/vector.h>

#include <map>

#include "../tests.h"

int
main()
{
  initlog();

  ParsedConvergenceTable table({"u"}, {{}});

  ParameterHandler prm;
  table.add_parameters(prm);

  std::string input = "set Extra columns = dofs, cells\n"
                      "set Rate key = dt\n";

  prm.parse_input_from_string(input);

  Triangulation<2> tria;
  GridGenerator::hyper_cube(tria);

  FESystem<2>   fe(FE_Q<2>(1), 1);
  DoFHandler<2> dh(tria);

  Functions::CosineFunction<2> exact(1);

  for (unsigned int i = 0; i < 5; ++i)
    {
      tria.refine_global(1);
      dh.distribute_dofs(fe);
      Vector<double> sol(dh.n_dofs());

      auto cycle = [&]() { return (i + 1) * 1.0; };
      auto dt    = [&]() { return i + 1.0; };

      table.add_extra_column("cycle", cycle);
      table.add_extra_column("dt", dt, false);
      table.error_from_exact(dh, sol, exact);
    }

  input = "set Error file name = error.txt\n";
  prm.parse_input_from_string(input);
  table.output_table();
  cat_file("error.txt");

  input = "set Error file name = error.org\n";
  prm.parse_input_from_string(input);
  table.output_table();
  cat_file("error.org");

  input = "set Error file name = error.tex\n";
  prm.parse_input_from_string(input);
  table.output_table();
  cat_file("error.tex");
}
