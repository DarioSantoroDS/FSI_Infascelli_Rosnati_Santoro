#include <fstream>
#include <iostream>
#include <vector>

#include "FluidStructureProblem.hpp"

int
main(int argc, char *argv[])

{
  try
  {
#ifdef DEBUG
  // Utilities::MPI::MPI_InitFinalize mpi_init(argc, argv, 1); to use for GDB on VScode
  std::cout << "im in debug mode" << std::endl;
#endif
  // #else
  Utilities::MPI::MPI_InitFinalize mpi_init(argc, argv);
  // #endif
  // initializing the parameter handler and reading the parameter file
  ParameterHandler prm;
  ParameterReader  param(prm);
  param.read_parameters("../config.prm");
  prm.enter_subsection("Refinement");
  const unsigned int n_refinement = prm.get_integer("Refinement cycles");
  prm.leave_subsection();
  FluidStructureProblem flow_problem(prm);
  flow_problem.make_grid();
  for (unsigned int refinement_cycle = 0; refinement_cycle < n_refinement + 1;
       ++refinement_cycle)
    {
      if (refinement_cycle > 0)
        flow_problem.refine_mesh(refinement_cycle);
      flow_problem.setup_dofs();
      flow_problem.assemble_system();
#ifndef DEBUG
      flow_problem.assemble_preconditioners();
#endif
#ifdef DEBUG
      flow_problem.output_matrix(); // output matrix not anymore implemented, we used it to debug when PETSc was implemented and
                                    // we could output the matrix to check it on matlab
#endif
#ifdef ITERATIVE_SOLVER
      flow_problem.solve_iterative();
#endif
#ifdef DIRECT_SOLVER
      flow_problem.solve(); // direct solver not anymore implemented, we used it when the matrix was not a block matrix
#endif
      flow_problem.output_results(refinement_cycle);
      flow_problem.timer.print_summary();
      flow_problem.timer.reset();
    }
  flow_problem.pcout << std::endl;
  }

  catch (std::exception& exc)
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
  }

  return 0;
}