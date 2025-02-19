// ---------------------------------------------------------------------
//
// Copyright (C) 2005 - 2018 by the deal.II authors
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


// test Utilities::generate_random_normal_number. this function is
// supposed to be thread-safe, so check that a standard measure of
// normality holds for the numbers produced on different threads

#include <deal.II/base/utilities.h>

#include "../tests.h"


std::pair<double, double>
test()
{
  const double mu    = 13;
  const double sigma = 3;

  unsigned int N          = 1000000;
  double       sum        = 0;
  double       sum_square = 0;
  for (unsigned int i = 0; i < N; ++i)
    {
      const double x = Utilities::generate_normal_random_number(mu, sigma);

      sum += x;
      sum_square += x * x;
    }

  const double mean   = sum / N;
  const double stddev = std::sqrt(sum_square / N - sum / N * sum / N);

  return std::make_pair(mean, stddev);
}



int
main()
{
  initlog();

  // create 4 threads, run the test function on them
  std::vector<std::future<std::pair<double, double>>> tg;
  tg.push_back(std::async(&test));
  tg.push_back(std::async(&test));
  tg.push_back(std::async(&test));
  tg.push_back(std::async(&test));

  // the random number generator is thread-local, so we should get the
  // same result every time
  auto t0_result = tg[0].get();
  deallog << t0_result.first << ' ' << t0_result.second << std::endl;
  AssertThrow(tg[1].get() == t0_result, ExcInternalError());
  AssertThrow(tg[2].get() == t0_result, ExcInternalError());
  AssertThrow(tg[3].get() == t0_result, ExcInternalError());
}
