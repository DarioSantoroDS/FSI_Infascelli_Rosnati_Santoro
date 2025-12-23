#include "FluidStructureProblem.hpp"

// functions for the ParameterReader class
void
ParameterReader::declare_parameters()
{
  prm.enter_subsection("Geometry");
  {
    prm.declare_entry("Number of cells per edge",
                      "8", // Default value in file
                      Patterns::Integer(1),
                      "The number of cells per edge of the domain");
    prm.declare_entry("Stokes degree",
                      "2", // Default value in file
                      Patterns::Integer(1),
                      "The polynomial degree for the Stokes problem");
    prm.declare_entry("Elasticity degree",
                      "1", // Default value in file
                      Patterns::Integer(1),
                      "The polynomial degree for the Elasticity problem");
    prm.declare_entry("Fluid weight",
                      "1", // Default is same for
                      Patterns::Integer(1),
                      "Weight given to the fluid cell");
    prm.declare_entry("Solid weight",
                      "1", // Default is same for
                      Patterns::Integer(1),
                      "Weight given to the solid cell");
  }
  prm.leave_subsection();
  prm.enter_subsection("Refinement");
  {
    prm.declare_entry("Refinement cycles",
                      "1", // Default value in file
                      Patterns::Integer(0),
                      "Number of refinement cycles to be performed");
  }
  prm.leave_subsection();
  prm.enter_subsection("Physics");
  {
    prm.declare_entry("Viscosity",
                      "1.0", // Default value in file
                      Patterns::Double(0.0),
                      "The viscosity of the fluid");
    prm.declare_entry("lambda",
                      "1.0", // Default value in file
                      Patterns::Double(0.0),
                      "The first Lamé parameter of the solid");
    prm.declare_entry("mu",
                      "1.0", // Default value in file
                      Patterns::Double(0.0),
                      "The second Lamé parameter of the solid");
  }
  prm.leave_subsection();
}

void
ParameterReader::read_parameters(const std::string &parameter_file)
{
  declare_parameters();
  prm.parse_input(parameter_file);
}

void
FluidStructureProblem::set_boundary_ids(
  parallel::distributed::Triangulation<dim> & /*triangulation*/) const
{
  // for (const auto &cell : triangulation.active_cell_iterators())
  //   {
  //     for (const auto &face : cell->face_iterators())
  //       if (face->at_boundary() && (face->center()[dim - 1] == 1))
  //         face->set_all_boundary_ids(1);
  //   }
  // for (const auto &cell : triangulation.active_cell_iterators())
  //   {
  //     if (((std::fabs(cell->center()[0]) < 0.25) &&
  //          (cell->center()[dim - 1] > 0.5)) ||
  //         ((std::fabs(cell->center()[0]) >= 0.25) &&
  //          (cell->center()[dim - 1] > -0.5)))
  //       cell->set_material_id(fluid_domain_id);
  //     else
  //       cell->set_material_id(solid_domain_id);
  //   }
}

void
FluidStructureProblem::create_coarse_mesh(
  parallel::distributed::Triangulation<dim> & /*coarse_grid*/) const
{
  // GridGenerator::subdivided_hyper_cube(coarse_grid, problemsize, -1, 1);
  // set_boundary_ids(coarse_grid);

  // coarse_grid.signals.post_refinement.connect(
  //   [this, &coarse_grid]() { this->set_boundary_ids(coarse_grid); });
}


void
FluidStructureProblem::make_grid()
{
  TimerOutput::Scope t(timer, "make_grid");
  pcout << "   Generating the mesh..." << std::endl;
  // useful if we want to set different weights to the cells in different physics domains
  // prm.enter_subsection("Geometry");
  // const int fluid_weight = prm.get_integer("Fluid weight");
  // const int solid_weight = prm.get_integer("Solid weight");
  // prm.leave_subsection();
  GridGenerator::subdivided_hyper_cube(triangulation, problemsize, -1, 1);

  for (const auto &cell : triangulation.active_cell_iterators())
    {
      for (const auto &face : cell->face_iterators())
        if (face->at_boundary() && (face->center()[dim - 1] == 1))
          face->set_all_boundary_ids(1);
    }
  for (const auto &cell : triangulation.active_cell_iterators())
    {
      if (((std::fabs(cell->center()[0]) < 0.25) &&
           (cell->center()[dim - 1] > 0.5)) ||
          ((std::fabs(cell->center()[0]) >= 0.25) &&
           (cell->center()[dim - 1] > -0.5)))
        cell->set_material_id(fluid_domain_id);
      else
        cell->set_material_id(solid_domain_id);
    }
//  functions necessary to use for the 
//  parallel::distributed::Triangulation::::execute_coarsening_and_refinement()	
//  method when giving weights to the cells in different physics domains.

  // triangulation.signals.cell_weight.connect(
  //   [&](const typename
  //   parallel::distributed::Triangulation<dim>::cell_iterator
  //         &cell,
  //       const typename parallel::distributed::Triangulation<dim>::CellStatus
  //         status) -> unsigned int {
  //     // If the cell is in the fluid domain, make it "heavy".
  //     // This forces the partitioner to put FEWER fluid cells on a process.
  //     if (cell->material_id() == fluid_domain_id)
  //       {
  //         // Adjust this ratio based on actual cost (e.g., 5x, 10x expensive)
  //         return fluid_weight;
  //       }
  //     else
  //       {
  //         // Solid cells are "light", so a process can handle many of them.
  //         return solid_weight;
  //       }
  //   });

  // // Force a repartition now that weights and IDs are defined
  // triangulation.repartition();
#ifdef VERBOSE
  pcout << "Mesh generated!" << std::endl;
#endif
  pcout << "  Number of elements = " << triangulation.n_global_active_cells()
        << std::endl;
}

void
FluidStructureProblem::set_active_fe_indices()
{
  for (const auto &cell : dof_handler.active_cell_iterators())
    {
      if (!cell->is_locally_owned())
        continue;
      if (cell_is_in_fluid_domain(cell))
        cell->set_active_fe_index(0);
      else if (cell_is_in_solid_domain(cell))
        cell->set_active_fe_index(1);
      else
        Assert(false, ExcNotImplemented());
    }
}

void
FluidStructureProblem::setup_dofs()
{
  TimerOutput::Scope t(timer, "setup_dofs");
  pcout << "   Initializing dofs..." << std::endl;

  set_active_fe_indices();
  dof_handler.distribute_dofs(fe_collection);
  // DoFRenumbering::Cuthill_McKee(dof_handler); // we tried to use this, but didn't seem to help
  std::vector<unsigned int> block_component(dim + 1 + dim, 0);
  block_component[dim] = 1;
  for (unsigned int i = dim + 1; i < dim + dim + 1; ++i)
    block_component[i] = 2;
  DoFRenumbering::component_wise(dof_handler, block_component);

  locally_owned_dofs = dof_handler.locally_owned_dofs();
  DoFTools::extract_locally_relevant_dofs(dof_handler, locally_relevant_dofs);
// set up for the block structure
  std::vector<types::global_dof_index> dofs_per_block =
    DoFTools::count_dofs_per_fe_block(dof_handler, block_component);
  const unsigned int n_u = dofs_per_block[0];
  const unsigned int n_p = dofs_per_block[1];
  const unsigned int n_d = dofs_per_block[2];

  block_owned_dofs.resize(3);
  block_relevant_dofs.resize(3);
  block_owned_dofs[0] = locally_owned_dofs.get_view(0, n_u);
  block_owned_dofs[1] = locally_owned_dofs.get_view(n_u, n_u + n_p);
  block_owned_dofs[2] = locally_owned_dofs.get_view(n_u + n_p, n_u + n_p + n_d);
  block_relevant_dofs[0] = locally_relevant_dofs.get_view(0, n_u);
  block_relevant_dofs[1] = locally_relevant_dofs.get_view(n_u, n_u + n_p);
  block_relevant_dofs[2] =
    locally_relevant_dofs.get_view(n_u + n_p, n_u + n_p + n_d);
#ifdef VERBOSE
  pcout << "  Number of DoFs: " << std::endl;
  pcout << "    velocity = " << n_u << std::endl;
  pcout << "    pressure = " << n_p << std::endl;
  pcout << "    displacement = " << n_d << std::endl;
#endif
  pcout << "    total  Dofs  = " << n_u + n_p + n_d << std::endl;

  locally_relevant_solution.reinit(block_owned_dofs,
                                   block_relevant_dofs,
                                   MPI_COMM_WORLD);

#ifdef DEBUG
  pcout << "Locally owned" << std::endl;
  std::cout << locally_relevant_dofs.n_elements() << " locally relevant dofs."
            << mpi_rank << std::endl;
  std::cout << locally_owned_dofs.n_elements() << " locally owned dofs."
            << mpi_rank << std::endl;
#endif
  constraints.clear();
  constraints.reinit(locally_relevant_dofs);
  DoFTools::make_hanging_node_constraints(dof_handler, constraints);

  const FEValuesExtractors::Vector velocities(0);
  VectorTools::interpolate_boundary_values(dof_handler,
                                           1,
                                           StokesBoundaryValues(),
                                           constraints,
                                           fe_collection.component_mask(
                                             velocities));

  const FEValuesExtractors::Vector displacements(dim + 1);
  VectorTools::interpolate_boundary_values(
    dof_handler,
    0,
    Functions::ZeroFunction<dim>(dim + 1 + dim),
    constraints,
    fe_collection.component_mask(displacements));
  // There are more constraints we have to handle, though: we have to make
  // sure that the velocity is zero at the interface between fluid and
  // solid. The following piece of code was already presented in the
  // introduction:
  {
    std::vector<types::global_dof_index> local_face_dof_indices(
      stokes_fe.n_dofs_per_face());
    for (const auto &cell : dof_handler.active_cell_iterators())
      {
        if (!cell->is_locally_owned())
          continue;

        if (cell_is_in_fluid_domain(cell))
          for (const auto face_no : cell->face_indices())
            if (cell->face(face_no)->at_boundary() == false)
              {
                bool face_is_on_interface = false;

                if ((cell->neighbor(face_no)->has_children() == false) &&
                    (cell_is_in_solid_domain(cell->neighbor(face_no))))
                  face_is_on_interface = true;
                else if (cell->neighbor(face_no)->has_children() == true)
                  {
                    for (unsigned int sf = 0;
                         sf < cell->face(face_no)->n_children();
                         ++sf)
                      if (cell_is_in_solid_domain(
                            cell->neighbor_child_on_subface(face_no, sf)))
                        {
                          face_is_on_interface = true;
                          break;
                        }
                  }

                if (face_is_on_interface)
                  {
                    cell->face(face_no)->get_dof_indices(local_face_dof_indices,
                                                         0);
                    for (unsigned int i = 0; i < local_face_dof_indices.size();
                         ++i)
                      if (stokes_fe.face_system_to_component_index(i).first <
                          dim)
                        constraints.add_line(local_face_dof_indices[i]);
                  }
              }
      }
  }

  // At the end of all this, we can declare to the constraints object that
  // we now have all constraints ready to go and that the object can rebuild
  // its internal data structures for better efficiency:
  constraints.close();
#ifdef VERBOSE
  pcout << "   Number of active cells: " << triangulation.n_active_cells()
        << std::endl
        << "   Number of degrees of freedom: " << dof_handler.n_dofs()
        << std::endl;
#endif
  // In the rest of this function we create a sparsity pattern as discussed
  // extensively in the introduction, and use it to initialize the matrix;
  // then also set vectors to their correct sizes:
#ifdef FORCE_USE_OF_TRILINOS
#  ifndef ALTERNATIVE_PATTERN
  // Right now this pattern is not working, but we tested it in the past and didn't yelded 
  // different results from the alternative one, so we keep it commented out for possible future use.
  TrilinosWrappers::BlockSparsityPattern dsp(block_owned_dofs,
                                             block_owned_dofs,
                                             block_relevant_dofs,
                                             MPI_COMM_WORLD);
  // #endif
  // for (unsigned int i = 0; i < fe_collection.n_blocks(); ++i)
  //   for (unsigned int j = 0; j < fe_collection.n_blocks(); ++j)
  //     dsp.block(i, j).reinit(dofs_per_block[i], dofs_per_block[j]);
  // dsp.collect_sizes();

  Table<2, DoFTools::Coupling> cell_coupling(fe_collection.n_components(),
                                             fe_collection.n_components());
  Table<2, DoFTools::Coupling> face_coupling(fe_collection.n_components(),
                                             fe_collection.n_components());

  for (unsigned int c = 0; c < fe_collection.n_components(); ++c)
    for (unsigned int d = 0; d < fe_collection.n_components(); ++d)
      {
        if (((c < dim + 1) && (d < dim + 1) && !((c == dim) && (d == dim))) ||
            ((c >= dim + 1) && (d >= dim + 1)))
          cell_coupling[c][d] = DoFTools::always;

        if ((c >= dim + 1) && (d < dim + 1))
          face_coupling[c][d] = DoFTools::always;
      }
  // constraints.condense(dsp);
  DoFTools::make_flux_sparsity_pattern(dof_handler,
                                       dsp,
                                       constraints,
                                       false,
                                       cell_coupling,
                                       face_coupling,
                                       mpi_rank);
  // SparsityTools::distribute_sparsity_pattern(dsp,
  //                                            locally_owned_dofs,
  //                                            MPI_COMM_WORLD,
  //                                            locally_relevant_dofs);

  dsp.compress(); // useless ? not present in step 40

  system_matrix.reinit(dsp);
  system_rhs.reinit(block_owned_dofs, MPI_COMM_WORLD);
  Table<2, DoFTools::Coupling> coupling_pressure(fe_collection.n_components(),
                                                 fe_collection.n_components());

  for (unsigned int c = 0; c < dim + 1; ++c)
    {
      for (unsigned int d = 0; d < dim + 1; ++d)
        {
          if (c == dim && d == dim) // pressure-pressure term
            coupling_pressure[c][d] = DoFTools::always;
          else // other combinations
            coupling_pressure[c][d] = DoFTools::none;
        }
    }
  TrilinosWrappers::BlockSparsityPattern sparsity_pressure_mass(
    block_owned_dofs, MPI_COMM_WORLD);
  DoFTools::make_sparsity_pattern(dof_handler,
                                  coupling_pressure,
                                  sparsity_pressure_mass);
  sparsity_pressure_mass.compress(); // useless ? not present in step 40

  pressure_mass.reinit(sparsity_pressure_mass);
#  endif
#endif
#ifdef ALTERNATIVE_PATTERN
// resetting the matrix and pressure mass in case they were already initialized
  system_matrix.clear();
  pressure_mass.clear();
  BlockDynamicSparsityPattern dsp(dofs_per_block, dofs_per_block);

  Table<2, DoFTools::Coupling> cell_coupling(fe_collection.n_components(),
                                             fe_collection.n_components());
  Table<2, DoFTools::Coupling> face_coupling(fe_collection.n_components(),
                                             fe_collection.n_components());

  for (unsigned int c = 0; c < fe_collection.n_components(); ++c)
    for (unsigned int d = 0; d < fe_collection.n_components(); ++d)
      {
        if (((c < dim + 1) && (d < dim + 1) && !((c == dim) && (d == dim))) ||
            ((c >= dim + 1) && (d >= dim + 1)))
          cell_coupling[c][d] = DoFTools::always;

        if ((c >= dim + 1) && (d < dim + 1))
          face_coupling[c][d] = DoFTools::always;
      }

  DoFTools::make_flux_sparsity_pattern(dof_handler,
                                       dsp,
                                       constraints,
                                       false,
                                       cell_coupling,
                                       face_coupling,
                                       mpi_rank);
  SparsityTools::distribute_sparsity_pattern(dsp,
                                             locally_owned_dofs,
                                             MPI_COMM_WORLD,
                                             locally_relevant_dofs);


  constraints.condense(dsp);

  dsp.compress(); 
  system_matrix.reinit(block_owned_dofs, dsp, MPI_COMM_WORLD);
  system_rhs.reinit(block_owned_dofs, MPI_COMM_WORLD);

  Table<2, DoFTools::Coupling> coupling_pressure(fe_collection.n_components(),
                                                 fe_collection.n_components());

  for (unsigned int c = 0; c < dim + 1; ++c)
    {
      for (unsigned int d = 0; d < dim + 1; ++d)
        {
          if (c == dim && d == dim) // pressure-pressure term only
            coupling_pressure[c][d] = DoFTools::always;
          else
            coupling_pressure[c][d] = DoFTools::none;
        }
    }

  BlockDynamicSparsityPattern dsp_pressure(dofs_per_block, dofs_per_block);

  DoFTools::make_sparsity_pattern(dof_handler, coupling_pressure, dsp_pressure);

  SparsityTools::distribute_sparsity_pattern(dsp_pressure,
                                             locally_owned_dofs,
                                             MPI_COMM_WORLD,
                                             locally_relevant_dofs);
  constraints.condense(dsp_pressure);
  pressure_mass.reinit(block_owned_dofs, dsp_pressure, MPI_COMM_WORLD);
#endif
#ifdef VERBOSE
  pcout << "Dofs initialized!" << std::endl;
#endif
}

void
FluidStructureProblem::assemble_system()
{
  TimerOutput::Scope t(timer, "assembly_system");

  pcout << "   Assembling the system..." << std::endl;
  system_matrix = 0.0;
  system_rhs    = 0.0;
  pressure_mass = 0.0;
// setting all the objects needed for the assembly process
  const QGauss<dim> stokes_quadrature(stokes_degree + 2);
  const QGauss<dim> elasticity_quadrature(elasticity_degree + 2);

  hp::QCollection<dim> q_collection;
  q_collection.push_back(stokes_quadrature);
  q_collection.push_back(elasticity_quadrature);

  hp::FEValues<dim> hp_fe_values(fe_collection,
                                 q_collection,
                                 update_values | update_quadrature_points |
                                   update_JxW_values | update_gradients);

  const QGauss<dim - 1> common_face_quadrature(
    std::max(stokes_degree + 2, elasticity_degree + 2));

  FEFaceValues<dim>    stokes_fe_face_values(stokes_fe,
                                          common_face_quadrature,
                                          update_JxW_values | update_gradients |
                                            update_values);
  FEFaceValues<dim>    elasticity_fe_face_values(elasticity_fe,
                                              common_face_quadrature,
                                              update_normal_vectors |
                                                update_values);
  FESubfaceValues<dim> stokes_fe_subface_values(stokes_fe,
                                                common_face_quadrature,
                                                update_JxW_values |
                                                  update_gradients |
                                                  update_values);
  FESubfaceValues<dim> elasticity_fe_subface_values(elasticity_fe,
                                                    common_face_quadrature,
                                                    update_normal_vectors |
                                                      update_values);

  // ...to objects that are needed to describe the local contributions to
  // the global linear system...
  const unsigned int stokes_dofs_per_cell     = stokes_fe.n_dofs_per_cell();
  const unsigned int elasticity_dofs_per_cell = elasticity_fe.n_dofs_per_cell();

  FullMatrix<double> local_matrix;
  FullMatrix<double> local_interface_matrix(elasticity_dofs_per_cell,
                                            stokes_dofs_per_cell);
  // FullMatrix<double> cell_pressure_mass_matrix;

  Vector<double> local_rhs;

  std::vector<types::global_dof_index> local_dof_indices;
  std::vector<types::global_dof_index> neighbor_dof_indices(
    stokes_dofs_per_cell);

  const Functions::ZeroFunction<dim> right_hand_side(dim + 1);

  // ...to variables that allow us to extract certain components of the
  // shape functions and cache their values rather than having to recompute
  // them at every quadrature point:
  const FEValuesExtractors::Vector velocities(0);
  const FEValuesExtractors::Scalar pressure(dim);
  const FEValuesExtractors::Vector displacements(dim + 1);

  std::vector<SymmetricTensor<2, dim>> stokes_symgrad_phi_u(
    stokes_dofs_per_cell);
  std::vector<double> stokes_div_phi_u(stokes_dofs_per_cell);
  std::vector<double> stokes_phi_p(stokes_dofs_per_cell);

  std::vector<Tensor<2, dim>> elasticity_grad_phi(elasticity_dofs_per_cell);
  std::vector<double>         elasticity_div_phi(elasticity_dofs_per_cell);
  std::vector<Tensor<1, dim>> elasticity_phi(elasticity_dofs_per_cell);
  // Then comes the main loop over all cells and, as in step-27, the
  // initialization of the hp::FEValues object for the current cell and the
  // extraction of a FEValues object that is appropriate for the current
  // cell:
  for (const auto &cell : dof_handler.active_cell_iterators())
    {
      if (!cell->is_locally_owned())
        continue;
      hp_fe_values.reinit(cell);

      const FEValues<dim> &fe_values = hp_fe_values.get_present_fe_values();

      std::vector<types::global_dof_index> dof_indices(
        cell->get_fe().n_dofs_per_cell());
      cell->get_dof_indices(dof_indices);


      local_matrix.reinit(cell->get_fe().n_dofs_per_cell(),
                          cell->get_fe().n_dofs_per_cell());
      local_rhs.reinit(cell->get_fe().n_dofs_per_cell());

      // With all of this done, we continue to assemble the cell terms for
      // cells that are part of the Stokes and elastic regions. While we
      // could in principle do this in one formula, in effect implementing
      // the one bilinear form stated in the introduction, we realize that
      // our finite element spaces are chosen in such a way that on each
      // cell, one set of variables (either velocities and pressure, or
      // displacements) are always zero, and consequently a more efficient
      // way of computing local integrals is to do only what's necessary
      // based on an <code>if</code> clause that tests which part of the
      // domain we are in.
      //
      // The actual computation of the local matrix is the same as in
      // step-22 as well as that given in the @ref vector_valued
      // documentation module for the elasticity equations:
      if (cell_is_in_fluid_domain(cell))
        {
          const unsigned int dofs_per_cell = cell->get_fe().n_dofs_per_cell();
          Assert(dofs_per_cell == stokes_dofs_per_cell, ExcInternalError());

          for (unsigned int q = 0; q < fe_values.n_quadrature_points; ++q)
            {
              for (unsigned int k = 0; k < dofs_per_cell; ++k)
                {
                  stokes_symgrad_phi_u[k] =
                    fe_values[velocities].symmetric_gradient(k, q);
                  stokes_div_phi_u[k] = fe_values[velocities].divergence(k, q);
                  stokes_phi_p[k]     = fe_values[pressure].value(k, q);
                }

              for (unsigned int i = 0; i < dofs_per_cell; ++i)
                for (unsigned int j = 0; j < dofs_per_cell; ++j)
                  local_matrix(i, j) +=
                    (2 * viscosity * stokes_symgrad_phi_u[i] *
                       stokes_symgrad_phi_u[j] -
                     stokes_div_phi_u[i] * stokes_phi_p[j] -
                     stokes_phi_p[i] * stokes_div_phi_u[j]) *
                    fe_values.JxW(q);
            }
        }
      else
        {
          const unsigned int dofs_per_cell = cell->get_fe().n_dofs_per_cell();
          Assert(dofs_per_cell == elasticity_dofs_per_cell, ExcInternalError());

          for (unsigned int q = 0; q < fe_values.n_quadrature_points; ++q)
            {
              for (unsigned int k = 0; k < dofs_per_cell; ++k)
                {
                  elasticity_grad_phi[k] =
                    fe_values[displacements].gradient(k, q);
                  elasticity_div_phi[k] =
                    fe_values[displacements].divergence(k, q);
                }

              for (unsigned int i = 0; i < dofs_per_cell; ++i)
                for (unsigned int j = 0; j < dofs_per_cell; ++j)
                  {
                    local_matrix(i, j) +=
                      (lambda * elasticity_div_phi[i] * elasticity_div_phi[j] +
                       mu * scalar_product(elasticity_grad_phi[i],
                                           elasticity_grad_phi[j]) +
                       mu * scalar_product(elasticity_grad_phi[i],
                                           transpose(elasticity_grad_phi[j]))) *
                      fe_values.JxW(q);
                  }
            }
        }
      // Once we have the contributions from cell integrals, we copy them
      // into the global matrix (taking care of constraints right away,
      // through the AffineConstraints::distribute_local_to_global
      // function). Note that we have not written anything into the
      // <code>local_rhs</code> variable, though we still need to pass it
      // along since the elimination of nonzero boundary values requires the
      // modification of local and consequently also global right hand side
      // values:
      local_dof_indices.resize(cell->get_fe().n_dofs_per_cell());
      cell->get_dof_indices(local_dof_indices);
      constraints.distribute_local_to_global(
        local_matrix, local_rhs, local_dof_indices, system_matrix, system_rhs);
      // We now assemble the pressure mass matrix.
      std::vector<unsigned int> pressure_local_indices;
      pressure_local_indices.reserve(cell->get_fe().n_dofs_per_cell());

      for (unsigned int i = 0; i < cell->get_fe().n_dofs_per_cell(); ++i)
        {
          // Check if this DoF belongs to the pressure component (index 'dim')
          const unsigned int component_index =
            cell->get_fe().system_to_component_index(i).first;

          if (component_index == dim) // dim is always the pressure index
            {
              pressure_local_indices.push_back(i);
            }
        }

      // Resize the small structures
      const unsigned int n_pressure_dofs = pressure_local_indices.size();
      FullMatrix<double> local_pressure_matrix(n_pressure_dofs, n_pressure_dofs);
      std::vector<types::global_dof_index> pressure_global_dof_indices(
        n_pressure_dofs);

      // Fill the small matrix and indices
      for (unsigned int i = 0; i < n_pressure_dofs; ++i)
        {
          // Get the original local index (e.g., 5) and map to global
          const unsigned int original_i  = pressure_local_indices[i];
          pressure_global_dof_indices[i] = dof_indices[original_i];

          for (unsigned int j = 0; j < n_pressure_dofs; ++j)
            {
              const unsigned int original_j = pressure_local_indices[j];

              // Compute the integral directly here
              // (No need to compute the full matrix first)
              double value = 0.0;
              for (unsigned int q = 0; q < fe_values.n_quadrature_points; ++q)
                {
                  value += fe_values[pressure].value(original_i, q) *
                           fe_values[pressure].value(original_j, q) /
                           viscosity * fe_values.JxW(q);
                }
              local_pressure_matrix(i, j) = value;
            }
        }

      // Distribute ONLY the pressure part
      // This is safe because pressure_mass ONLY has pressure rows allocated.
      constraints.distribute_local_to_global(
        local_pressure_matrix,
        pressure_global_dof_indices,
        pressure_mass); 
      // The more interesting part of this function is where
      // we see about
      // face terms along the interface between the two subdomains. To this
      // end, we first have to make sure that we only assemble them once
      // even though a loop over all faces of all cells would encounter each
      // part of the interface twice. We arbitrarily make the decision that
      // we will only evaluate interface terms if the current cell is part
      // of the solid subdomain and if, consequently, a face is not at the
      // boundary and the potential neighbor behind it is part of the fluid
      // domain. Let's start with these conditions:
      if (cell_is_in_solid_domain(cell))
        for (const auto f : cell->face_indices())
          if (cell->face(f)->at_boundary() == false)
            {
              // At this point we know that the current cell is a candidate
              // for integration and that a neighbor behind face
              // <code>f</code> exists. There are now three possibilities:
              //
              // - The neighbor is at the same refinement level and has no
              //   children.
              // - The neighbor has children.
              // - The neighbor is coarser.
              //
              // In all three cases, we are only interested in it if it is
              // part of the fluid subdomain. So let us start with the first
              // and simplest case: if the neighbor is at the same level,
              // has no children, and is a fluid cell, then the two cells
              // share a boundary that is part of the interface along which
              // we want to integrate interface terms. All we have to do is
              // initialize two FEFaceValues object with the current face
              // and the face of the neighboring cell (note how we find out
              // which face of the neighboring cell borders on the current
              // cell) and pass things off to the function that evaluates
              // the interface terms (the third through fifth arguìments to
              // this function provide it with scratch arrays). The result
              // is then again copied into the global matrix, using a
              // function that knows that the DoF indices of rows and
              // columns of the local matrix result from different cells:
              if ((cell->neighbor(f)->level() == cell->level()) &&
                  (cell->neighbor(f)->has_children() == false) &&
                  cell_is_in_fluid_domain(cell->neighbor(f)))
                {
                  elasticity_fe_face_values.reinit(cell, f);
                  stokes_fe_face_values.reinit(cell->neighbor(f),
                                               cell->neighbor_of_neighbor(f));

                  assemble_interface_term(elasticity_fe_face_values,
                                          stokes_fe_face_values,
                                          elasticity_phi,
                                          stokes_symgrad_phi_u,
                                          stokes_phi_p,
                                          local_interface_matrix);

                  cell->neighbor(f)->get_dof_indices(neighbor_dof_indices);
                  constraints.distribute_local_to_global(local_interface_matrix,
                                                         local_dof_indices,
                                                         neighbor_dof_indices,
                                                         system_matrix);
                }

              // The second case is if the neighbor has further children. In
              // that case, we have to loop over all the children of the
              // neighbor to see if they are part of the fluid subdomain. If
              // they are, then we integrate over the common interface,
              // which is a face for the neighbor and a subface of the
              // current cell, requiring us to use an FEFaceValues for the
              // neighbor and an FESubfaceValues for the current cell:
              else if ((cell->neighbor(f)->level() == cell->level()) &&
                       (cell->neighbor(f)->has_children() == true))
                {
                  for (unsigned int subface = 0;
                       subface < cell->face(f)->n_children();
                       ++subface)
                    if (cell_is_in_fluid_domain(
                          cell->neighbor_child_on_subface(f, subface)))
                      {
                        elasticity_fe_subface_values.reinit(cell, f, subface);
                        stokes_fe_face_values.reinit(
                          cell->neighbor_child_on_subface(f, subface),
                          cell->neighbor_of_neighbor(f));

                        assemble_interface_term(elasticity_fe_subface_values,
                                                stokes_fe_face_values,
                                                elasticity_phi,
                                                stokes_symgrad_phi_u,
                                                stokes_phi_p,
                                                local_interface_matrix);
                        cell->neighbor_child_on_subface(f, subface)
                          ->get_dof_indices(neighbor_dof_indices);
                        constraints.distribute_local_to_global(
                          local_interface_matrix,
                          local_dof_indices,
                          neighbor_dof_indices,
                          system_matrix);
                      }
                }

              // The last option is that the neighbor is coarser. In that
              // case we have to use an FESubfaceValues object for the
              // neighbor and a FEFaceValues for the current cell; the rest
              // is the same as before:
              else if (cell->neighbor_is_coarser(f) &&
                       cell_is_in_fluid_domain(cell->neighbor(f)))
                {
                  elasticity_fe_face_values.reinit(cell, f);
                  stokes_fe_subface_values.reinit(
                    cell->neighbor(f),
                    cell->neighbor_of_coarser_neighbor(f).first,
                    cell->neighbor_of_coarser_neighbor(f).second);

                  assemble_interface_term(elasticity_fe_face_values,
                                          stokes_fe_subface_values,
                                          elasticity_phi,
                                          stokes_symgrad_phi_u,
                                          stokes_phi_p,
                                          local_interface_matrix);
                  cell->neighbor(f)->get_dof_indices(neighbor_dof_indices);
                  constraints.distribute_local_to_global(local_interface_matrix,
                                                         local_dof_indices,
                                                         neighbor_dof_indices,
                                                         system_matrix);
                }
            }

    }

  system_matrix.compress(VectorOperation::add);
  system_rhs.compress(VectorOperation::add);
  pressure_mass.compress(VectorOperation::add);
#ifdef VERBOSE
  pcout << "Assembly complete!" << std::endl;
#endif
}


void
FluidStructureProblem::assemble_interface_term(
  const FEFaceValuesBase<dim>          &elasticity_fe_face_values,
  const FEFaceValuesBase<dim>          &stokes_fe_face_values,
  std::vector<Tensor<1, dim>>          &elasticity_phi,
  std::vector<SymmetricTensor<2, dim>> &stokes_symgrad_phi_u,
  std::vector<double>                  &stokes_phi_p,
  FullMatrix<double>                   &local_interface_matrix) const
{
  Assert(stokes_fe_face_values.n_quadrature_points ==
           elasticity_fe_face_values.n_quadrature_points,
         ExcInternalError());
  const unsigned int n_face_quadrature_points =
    elasticity_fe_face_values.n_quadrature_points;

  const FEValuesExtractors::Vector velocities(0);
  const FEValuesExtractors::Scalar pressure(dim);
  const FEValuesExtractors::Vector displacements(dim + 1);

  local_interface_matrix = 0;
  for (unsigned int q = 0; q < n_face_quadrature_points; ++q)
    {
      const Tensor<1, dim> normal_vector =
        elasticity_fe_face_values.normal_vector(q);

      for (unsigned int k = 0; k < stokes_fe_face_values.dofs_per_cell; ++k)
        {
          stokes_symgrad_phi_u[k] =
            stokes_fe_face_values[velocities].symmetric_gradient(k, q);
          stokes_phi_p[k] = stokes_fe_face_values[pressure].value(k, q);
        }
      for (unsigned int k = 0; k < elasticity_fe_face_values.dofs_per_cell; ++k)
        elasticity_phi[k] =
          elasticity_fe_face_values[displacements].value(k, q);

      for (unsigned int i = 0; i < elasticity_fe_face_values.dofs_per_cell; ++i)
        for (unsigned int j = 0; j < stokes_fe_face_values.dofs_per_cell; ++j)
          local_interface_matrix(i, j) +=
            -((2 * viscosity * (stokes_symgrad_phi_u[j] * normal_vector) -
               stokes_phi_p[j] * normal_vector) *
              elasticity_phi[i] * stokes_fe_face_values.JxW(q));
    }
}


// Here we set up the Algebraic Multigrid preconditioners for the inside
// iterative solvers.
// Part of the AMG additional data is taken from the literature.
// This is the function giving the problems described in the beginning
// of the hpp file when using the DEBUG flag.
// As said, in Release mode the AssertDimension doesn't happen, and the
// AMG preconditioner still works as expected, given the constant number
// of outside iterations GMRES requests and speed.
// We noticed a Issue on the Deal.ii github page regarding this problem,
// an answer and fix has been issued when dealing with FE_Q and FE_Nothing
// elements, but in our case we are using FE_System, so the problem is still
// open. Issue #12854 in deal.ii repository.
// When used in parallel, this function may output some warnings, nothing that
// seems serious.
void
FluidStructureProblem::assemble_preconditioners()
{
  TimerOutput::Scope t(timer, "assemble_preconditioners");

  pcout << "   Building preconditioners..." << std::endl;
  stokes_preconditioner = std::make_shared<TrilinosWrappers::PreconditionAMG>();
  const FEValuesExtractors::Vector velocity_components(0);
  std::vector<std::vector<bool>>   stokes_constant_modes;
  DoFTools::extract_constant_modes(dof_handler,
                                   fe_collection.component_mask(velocity_components),
                                   stokes_constant_modes);
  TrilinosWrappers::PreconditionAMG::AdditionalData stokes_amg_data;
  stokes_amg_data.constant_modes = stokes_constant_modes;

  stokes_amg_data.elliptic              = true;
  stokes_amg_data.higher_order_elements = true;
  stokes_amg_data.smoother_sweeps       = 2;
  stokes_amg_data.aggregation_threshold = 0.02;
  stokes_preconditioner->initialize(system_matrix.block(0, 0), stokes_amg_data);
  mp_preconditioner = std::make_shared<TrilinosWrappers::PreconditionAMG>();
  mp_preconditioner->initialize(pressure_mass.block(1, 1));

  elasticity_preconditioner =
    std::make_shared<TrilinosWrappers::PreconditionAMG>();
  const FEValuesExtractors::Vector elasticity_components(dim + 1);
  std::vector<std::vector<bool>>   elasticity_constant_modes;
  DoFTools::extract_constant_modes(dof_handler,
                                   fe_collection.component_mask(elasticity_components),
                                   elasticity_constant_modes);
  TrilinosWrappers::PreconditionAMG::AdditionalData elasticity_amg_data;
  elasticity_amg_data.constant_modes = elasticity_constant_modes;

  elasticity_amg_data.elliptic              = true;
  elasticity_amg_data.higher_order_elements = true;
  elasticity_amg_data.smoother_sweeps       = 2;
  elasticity_amg_data.aggregation_threshold = 0.02;
  elasticity_preconditioner->initialize(system_matrix.block(2, 2),
                                        elasticity_amg_data);
#ifdef VERBOSE
  pcout << "Preconditioners assembled!" << std::endl;
#endif
}
#ifdef DEBUG
// not anymore implemented
void
FluidStructureProblem::output_matrix() const
{
#  ifdef USE_PETSC_LA
  PetscViewer mat_viewer;
  // Create an ASCII viewer that writes to "system_matrix.m"
  PetscViewerASCIIOpen(system_matrix.get_mpi_communicator(),
                       "system_matrix.m",
                       &mat_viewer);
  // Set format to MATLAB (this produces a coordinate list)
  PetscViewerPushFormat(mat_viewer, PETSC_VIEWER_ASCII_MATLAB);
  // View the matrix (deal.II object converts to PETSc Mat automatically)
  MatView(system_matrix, mat_viewer);
  PetscViewerPopFormat(mat_viewer);
  PetscViewerDestroy(&mat_viewer);
  PetscViewer vec_viewer;
  PetscViewerASCIIOpen(system_rhs.get_mpi_communicator(),
                       "system_rhs.m",
                       &vec_viewer);

  // Set format to MATLAB (prints entries one by one)
  PetscViewerPushFormat(vec_viewer, PETSC_VIEWER_ASCII_MATLAB);

  // View the vector
  VecView(system_rhs, vec_viewer);
  PetscViewerPopFormat(vec_viewer);
  PetscViewerDestroy(&vec_viewer);
#  endif
}
#endif
#ifdef DIRECT_SOLVER
// not anymore implemented
void
FluidStructureProblem::solve()
{
  TimerOutput::Scope t(timer, "solve");

  pcout << "solvingthissutff" << std::endl;
  LA::MPI::BlockVector completely_distributed_solution(block_owned_dofs,
                                                       MPI_COMM_WORLD);
#  ifdef FORCE_USE_OF_TRILINOS
  SolverControl                  solver_control(1, 0);
  TrilinosWrappers::SolverDirect direct(solver_control);
  direct.solve(system_matrix, completely_distributed_solution, system_rhs);
#  else
  SolverControl                    cn;
  PETScWrappers::SparseDirectMUMPS solver(cn, MPI_COMM_WORLD);
  solver.set_symmetric_mode(false);
  solver.solve(system_matrix, completely_distributed_solution, system_rhs);
#  endif
  constraints.distribute(completely_distributed_solution);
  locally_relevant_solution = completely_distributed_solution;
}
#endif

#ifdef ITERATIVE_SOLVER
void
FluidStructureProblem::solve_iterative()
{
  TimerOutput::Scope t(timer, "solve_iterative");

  pcout << "   Solving iterative..." << std::endl;
  LA::MPI::BlockVector completely_distributed_solution(block_owned_dofs,
                                                       MPI_COMM_WORLD);
  // completely_distributed_solution = 1.0;
  SolverControl solver_control(100000, 1e-6 * system_rhs.l2_norm());
#  ifdef FORCE_USE_OF_TRILINOS

#ifndef DEBUG
  PreconditionBlockTriangularAMG preconditioner;
  preconditioner.initialize(system_matrix.block(0, 0),
                            pressure_mass.block(1, 1),
                            system_matrix.block(1, 0),
                            system_matrix.block(2, 0),
                            system_matrix.block(2, 1),
                            system_matrix.block(2, 2),
                            stokes_preconditioner,
                            mp_preconditioner,
                            elasticity_preconditioner);
#else
  PreconditionBlockTriangular preconditioner;
  preconditioner.initialize(system_matrix.block(0, 0),
                            pressure_mass.block(1, 1),
                            system_matrix.block(1, 0),
                            system_matrix.block(2, 0),
                            system_matrix.block(2, 1),
                            system_matrix.block(2, 2));
#endif
  SolverFGMRES<TrilinosWrappers::MPI::BlockVector> solver(solver_control);
  solver.solve(system_matrix,
               completely_distributed_solution,
               system_rhs,
               preconditioner);
  pcout << "  " << solver_control.last_step() << " FGMRES iterations"
        << std::endl;
#  else
  Assert(false, ExcNotImplemented());

  // // PETScWrappers::PreconditionBlockJacobi::AdditionalData data;
  // // // This tells Block Jacobi to use ILU on the local blocks
  // // data.internal_preconditioner_type = "ilu";
  // PETScWrappers::PreconditionBoomerAMG preconditioner(MPI_COMM_WORLD);
  // preconditioner.initialize(system_matrix);
  // PETScWrappers::SolverGMRES solver(solver_control, MPI_COMM_WORLD);
  // solver.solve(system_matrix,
  //              completely_distributed_solution,
  //              system_rhs,
  //              preconditioner);
  // pcout << "  " << solver_control.last_step() << " GMRES iterations"
  //       << std::endl;
#  endif
  constraints.distribute(completely_distributed_solution);
  locally_relevant_solution = completely_distributed_solution;
}
#endif
void
FluidStructureProblem::output_results(const unsigned int refinement_cycle) const
{
#ifdef VERBOSE
  pcout << "   Output results..." << std::endl;
#endif
  std::vector<std::string> solution_names(dim, "velocity");
  solution_names.emplace_back("pressure");
  for (unsigned int d = 0; d < dim; ++d)
    solution_names.emplace_back("displacement");

  std::vector<DataComponentInterpretation::DataComponentInterpretation>
    data_component_interpretation(
      dim, DataComponentInterpretation::component_is_part_of_vector);
  data_component_interpretation.push_back(
    DataComponentInterpretation::component_is_scalar);
  for (unsigned int d = 0; d < dim; ++d)
    data_component_interpretation.push_back(
      DataComponentInterpretation::component_is_part_of_vector);

  DataOut<dim> data_out;
  data_out.attach_dof_handler(dof_handler);

  data_out.add_data_vector(locally_relevant_solution,
                           solution_names,
                           DataOut<dim>::type_dof_data,
                           data_component_interpretation);

  Vector<float> subdomain(triangulation.n_active_cells());
  for (unsigned int i = 0; i < subdomain.size(); ++i)
    subdomain(i) = triangulation.locally_owned_subdomain();
  data_out.add_data_vector(subdomain, "subdomain");

  Vector<float> material(triangulation.n_active_cells());
  unsigned int  i = 0;
  for (const auto &cell : triangulation.active_cell_iterators())
    {
      material(i) = static_cast<float>(cell->material_id());
      ++i;
    }
  data_out.add_data_vector(material,
                           "material_id",
                           DataOut<dim>::type_cell_data);

  data_out.build_patches();

  data_out.write_vtu_with_pvtu_record(
    "./", "solution", refinement_cycle, MPI_COMM_WORLD, 2, 8);
#ifdef VERBOSE
  pcout << "Done!" << std::endl;
  pcout << "   Written solution_0" << refinement_cycle << ".pvtu" << std::endl;
#endif
}
// The mesh refinement is done by estimating the errors in the fluid and solid
// domains separately, normalizing them, scaling the fluid error indicators
// by a factor of 4, and adding them together. The resulting indicators are
// then used to refine the mesh in the standard way.
// That factor of 4 is taken from literature.
void
FluidStructureProblem::refine_mesh(const unsigned int n_cycle)
{
  TimerOutput::Scope t(timer, "refining");

  pcout << "   Refining mesh, cycle number " << n_cycle << std::endl;

  Vector<float> stokes_estimated_error_per_cell(triangulation.n_active_cells());
  Vector<float> elasticity_estimated_error_per_cell(
    triangulation.n_active_cells());

  const QGauss<dim - 1> stokes_face_quadrature(stokes_degree + 2);
  const QGauss<dim - 1> elasticity_face_quadrature(elasticity_degree + 2);

  hp::QCollection<dim - 1> face_q_collection;
  face_q_collection.push_back(stokes_face_quadrature);
  face_q_collection.push_back(elasticity_face_quadrature);

  const FEValuesExtractors::Vector velocities(0);
  KellyErrorEstimator<dim>::estimate(
    dof_handler,
    face_q_collection,
    std::map<types::boundary_id, const Function<dim> *>(),
    locally_relevant_solution,
    stokes_estimated_error_per_cell,
    fe_collection.component_mask(velocities));

  const FEValuesExtractors::Vector displacements(dim + 1);
  KellyErrorEstimator<dim>::estimate(
    dof_handler,
    face_q_collection,
    std::map<types::boundary_id, const Function<dim> *>(),
    locally_relevant_solution,
    elasticity_estimated_error_per_cell,
    fe_collection.component_mask(displacements));

  // We then normalize error estimates by dividing by their norm and scale
  // the fluid error indicators by a factor of 4 as discussed in the
  // introduction. The results are then added together into a vector that
  // contains error indicators for all cells:
  // This is to compute the L2 norm of the error indicators in a parallel way.
  float stokes_local_sum = 0.0;

  for (unsigned int i = 0; i < stokes_estimated_error_per_cell.size(); ++i)
    stokes_local_sum +=
      stokes_estimated_error_per_cell[i] * stokes_estimated_error_per_cell[i];
  const float stokes_global_sum =
    Utilities::MPI::sum(stokes_local_sum, MPI_COMM_WORLD);
  const float stokes_l2_norm = std::sqrt(stokes_global_sum);

  float elasticity_local_sum = 0.0;
  for (unsigned int i = 0; i < elasticity_estimated_error_per_cell.size(); ++i)
    elasticity_local_sum += elasticity_estimated_error_per_cell[i] *
                            elasticity_estimated_error_per_cell[i];
  const float elasticity_global_sum =
    Utilities::MPI::sum(elasticity_local_sum, MPI_COMM_WORLD);
  const float elasticity_l2_norm = std::sqrt(elasticity_global_sum);

  stokes_estimated_error_per_cell *= 4.0f / stokes_l2_norm;
  elasticity_estimated_error_per_cell *= 1.0f / elasticity_l2_norm;

  Vector<float> estimated_error_per_cell(triangulation.n_active_cells());

  estimated_error_per_cell += stokes_estimated_error_per_cell;

  estimated_error_per_cell += elasticity_estimated_error_per_cell;

  // The second to last part of the function, before actually refining the
  // mesh, involves a heuristic that we have already mentioned in the
  // introduction: because the solution is discontinuous, the
  // KellyErrorEstimator class gets all confused about cells that sit at
  // the
  // boundary between subdomains: it believes that the error is large there
  // because the jump in the gradient is large, even though this is
  // entirely
  // expected and a feature that is in fact present in the exact solution
  // as
  // well and therefore not indicative of any numerical error.
  //
  // Consequently, we set the error indicators to zero for all cells at the
  // interface; the conditions determining which cells this affects are
  // slightly awkward because we have to account for the possibility of
  // adaptively refined meshes, meaning that the neighboring cell can be
  // coarser than the current one, or could in fact be refined some
  // more. The structure of these nested conditions is much the same as we
  // encountered when assembling interface terms in
  // <code>assemble_system</code>.
  for (const auto &cell : dof_handler.active_cell_iterators())
    {
      if (!cell->is_locally_owned())
        continue;

      for (const auto f : cell->face_indices())
        if (cell_is_in_solid_domain(cell))
          {
            if ((cell->at_boundary(f) == false) &&
                (((cell->neighbor(f)->level() == cell->level()) &&
                  (cell->neighbor(f)->has_children() == false) &&
                  cell_is_in_fluid_domain(cell->neighbor(f))) ||
                 ((cell->neighbor(f)->level() == cell->level()) &&
                  (cell->neighbor(f)->has_children() == true) &&
                  (cell_is_in_fluid_domain(
                    cell->neighbor_child_on_subface(f, 0)))) ||
                 (cell->neighbor_is_coarser(f) &&
                  cell_is_in_fluid_domain(cell->neighbor(f)))))
              estimated_error_per_cell(cell->active_cell_index()) = 0;
          }
        else
          {
            if ((cell->at_boundary(f) == false) &&
                (((cell->neighbor(f)->level() == cell->level()) &&
                  (cell->neighbor(f)->has_children() == false) &&
                  cell_is_in_solid_domain(cell->neighbor(f))) ||
                 ((cell->neighbor(f)->level() == cell->level()) &&
                  (cell->neighbor(f)->has_children() == true) &&
                  (cell_is_in_solid_domain(
                    cell->neighbor_child_on_subface(f, 0)))) ||
                 (cell->neighbor_is_coarser(f) &&
                  cell_is_in_solid_domain(cell->neighbor(f)))))
              estimated_error_per_cell(cell->active_cell_index()) = 0;
          }
    }

  parallel::distributed::GridRefinement::refine_and_coarsen_fixed_number(
    triangulation, estimated_error_per_cell, 0.3, 0.0);
  triangulation.execute_coarsening_and_refinement();
#ifdef VERBOSE
  pcout << "Refinement done!" << std::endl;
#endif
  pcout << "  Number of elements = " << triangulation.n_global_active_cells()
        << std::endl;
}