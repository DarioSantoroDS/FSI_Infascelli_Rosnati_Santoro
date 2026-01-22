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
      flow_problem.assemble_preconditioners();
#ifdef ITERATIVE_SOLVER
      flow_problem.solve_iterative();
#endif
      flow_problem.output_results(refinement_cycle);
#ifdef EXACT
      flow_problem.compute_error(refinement_cycle);
#endif
      flow_problem.timer.print_summary();
      flow_problem.timer.reset();
    }
  flow_problem.pcout << std::endl;
#ifdef EXACT_TABLE
  flow_problem.output_table();
#endif
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