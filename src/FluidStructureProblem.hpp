#ifndef FluidStructureInteractionProblem
#define FluidStructureInteractionProblem

#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/convergence_table.h>
#include <deal.II/base/function.h>
#include <deal.II/base/index_set.h>
#include <deal.II/base/logstream.h>
#include <deal.II/base/parameter_handler.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/timer.h>
#include <deal.II/base/utilities.h>

#include <deal.II/distributed/fully_distributed_tria.h>
#include <deal.II/distributed/grid_refinement.h>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_renumbering.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_nothing.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_simplex_p.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/fe/fe_values_extractors.h>
#include <deal.II/fe/mapping_fe.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_in.h>
#include <deal.II/grid/grid_refinement.h>
#include <deal.II/grid/tria.h>

#include <deal.II/hp/fe_collection.h>
#include <deal.II/hp/fe_values.h>

#include <deal.II/lac/affine_constraints.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/generic_linear_algebra.h>
#include <deal.II/lac/petsc_solver.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/solver_gmres.h>
#include <deal.II/lac/sparse_direct.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/sparsity_tools.h>
#include <deal.II/lac/trilinos_block_sparse_matrix.h>
#include <deal.II/lac/trilinos_parallel_block_vector.h>
#include <deal.II/lac/trilinos_precondition.h>
#include <deal.II/lac/trilinos_solver.h>
#include <deal.II/lac/trilinos_sparse_matrix.h>
#include <deal.II/lac/vector.h>

#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/error_estimator.h>
#include <deal.II/numerics/matrix_tools.h>
#include <deal.II/numerics/vector_tools.h>
#include <deal.II/numerics/vector_tools_integrate_difference.h>

#include <petscviewer.h> // PETSc viewer

#define FORCE_USE_OF_TRILINOS
#define ALTERNATIVE_PATTERN
#define ITERATIVE_SOLVER
// #define DIRECT_SOLVER
#define VERBOSE  //define to have more output
// #define EXACT
// #define EXACT_TABLE
#define USE_CONSTANT_MODES
// When compiled in deal.II DEBUG mode, there is a problem:
// depending on the version an assert inside the
//  DoFTools::extract_constant_modes()
// function fails, used in the assemble_preconditioners method.
// If deal.II is in version lower than 9.4, the Assert
// is a ExcNotImplemented, while in version 9.4 or higher it is an
// AssertDimension that fails. Nevertheless, in Release mode, where the
// Asserts are not active, the code works fine.
// To avoid this problem, when in DEBUG mode
// we do not use the block triangular preconditioner AMG, that needs the
// constant modes to be extracted. Instead, we use a simpler block with AMG
// preconditioner that does not use the constant modes.
// It has to be said that that preconditioner is less efficient, but at least
// it works.

#include <fstream>
#include <iostream>

#if !defined(ITERATIVE_SOLVER) && !defined(DIRECT_SOLVER)
#  error Either ITERATIVE_SOLVER or DIRECT_SOLVER must be defined.
#endif

namespace LA
{
#if defined(DEAL_II_WITH_PETSC) && !defined(DEAL_II_PETSC_WITH_COMPLEX) && \
  !(defined(DEAL_II_WITH_TRILINOS) && defined(FORCE_USE_OF_TRILINOS))
  using namespace dealii::LinearAlgebraPETSc;
#  define USE_PETSC_LA
#  error PETSC not anymore supported
#elif defined(DEAL_II_WITH_TRILINOS)
  using namespace dealii::LinearAlgebraTrilinos;
#else
#  error DEAL_II_WITH_PETSC or DEAL_II_WITH_TRILINOS required
#endif
} // namespace LA


using namespace dealii;


// Class to read parameters from a parameter file
class ParameterReader : public Subscriptor
{
public:
  ParameterReader(ParameterHandler &paramhandler)
    : prm(paramhandler)
  {}
  void
  read_parameters(const std::string &);

private:
  void
                    declare_parameters();
  ParameterHandler &prm;
};


class FluidStructureProblem
{
public:
  static constexpr unsigned int dim = 2;

  template <int dim>
  class ExactSolution_FSI : public Function<dim>
  {
  public:
    ExactSolution_FSI()
      : Function<dim>(dim + 1 + dim) // 5 componenti: u(2), p(1), d(2)
    {}

    // ======================
    // valore singola componente
    // ======================
    virtual double
    value(const Point<dim> &p, const unsigned int component = 0) const override
    {
      const double x = p[0];
      const double y = p[1];

      // -------- velocity u --------
      if (component == 0) // u0
        return (1.0 / 256.0) * std::pow(1.0 - 16.0 * x * x, 2) * y *
               (-1.0 + 4.0 * y * y);

      if (component == 1) // u1
        return -(1.0 / 64.0) * x * (-1.0 + 16.0 * x * x) *
               std::pow(1.0 - 4.0 * y * y, 2);

      // -------- pressure p --------
      if (component == 2)
        return (x - 1.0 / 4.0) * (x + 1.0 / 4.0) * (y - 1.0 / 2.0) *
               (y + 1.0 / 2.0);

      // -------- displacement d --------
      if (component == 3) // d0
        return (1.0 / 256.0) * std::pow(1.0 - 16.0 * x * x, 2) * y *
               (-1.0 + 4.0 * y * y);

      if (component == 4) // d1
        return -(1.0 / 64.0) * x * (-1.0 + 16.0 * x * x) *
               std::pow(1.0 - 4.0 * y * y, 2);

      return 0.0;
    }

    // ======================
    // valore vettoriale
    // ======================
    virtual void
    vector_value(const Point<dim> &p, Vector<double> &values) const override
    {
      AssertDimension(values.size(), this->n_components);
      for (unsigned int c = 0; c < this->n_components; ++c)
        values[c] = value(p, c);
    }

    // ======================
    // gradiente vettoriale
    // grad[i][j] = ∂(componente i)/∂x_j
    // ======================
    virtual void
    vector_gradient(const Point<dim>            &p,
                    std::vector<Tensor<1, dim>> &grad) const override
    {
      AssertDimension(grad.size(), this->n_components);

      for (auto &g : grad)
        g = 0.0;

      const double x = p[0];
      const double y = p[1];

      // ---- grad u0 ----
      grad[0][0] =
        (1.0 / 4.0) * x * (-1.0 + 16.0 * x * x) * y * (-1.0 + 4.0 * y * y);

      grad[0][1] =
        (1.0 / 256.0) * std::pow(1.0 - 16.0 * x * x, 2) * (-1.0 + 12.0 * y * y);

      // ---- grad u1 ----
      grad[1][0] =
        (1.0 / 64.0) * (1.0 - 48.0 * x * x) * std::pow(1.0 - 4.0 * y * y, 2);

      grad[1][1] =
        (1.0 / 4.0) * x * (-1.0 + 16.0 * x * x) * y * (1.0 - 4.0 * y * y);

      // ---- grad p ----
      grad[2][0] = (x - 1.0 / 4.0) * (y - 1.0 / 2.0) * (y + 1.0 / 2.0) +
                   (x + 1.0 / 4.0) * (y - 1.0 / 2.0) * (y + 1.0 / 2.0);

      grad[2][1] = (x - 1.0 / 4.0) * (x + 1.0 / 4.0) * (y - 1.0 / 2.0) +
                   (x - 1.0 / 4.0) * (x + 1.0 / 4.0) * (y + 1.0 / 2.0);

      // ---- grad d0 ----
      grad[3][0] = grad[0][0];
      grad[3][1] = grad[0][1];

      // ---- grad d1 ----
      grad[4][0] = grad[1][0];
      grad[4][1] = grad[1][1];
    }
  };


  class ExactSolution_u : public Function<dim>
  {
  public:
    ExactSolution_u()
      : Function<dim>(dim + 1 + dim) // 5 componenti
    {}

    // ======================
    // valore singola componente
    // ======================
    virtual double
    value(const Point<dim> &p, const unsigned int component = 0) const override
    {
      const double x = p[0];
      const double y = p[1];

      if (component == 0)
        return (1.0 / 256.0) * std::pow(1.0 - 16.0 * x * x, 2) * y *
               (-1.0 + 4.0 * y * y);

      if (component == 1)
        return -(1.0 / 64.0) * x * (-1.0 + 16.0 * x * x) *
               std::pow(1.0 - 4.0 * y * y, 2);

      return 0.0;
    }

    // ======================
    // valore vettoriale
    // ======================
    virtual void
    vector_value(const Point<dim> &p, Vector<double> &values) const override
    {
      AssertDimension(values.size(), this->n_components);

      for (unsigned int c = 0; c < this->n_components; ++c)
        values[c] = value(p, c);
    }

    // ======================
    // gradiente
    // grad[i][j] = ∂ u_i / ∂ x_j
    // ======================
    virtual void
    vector_gradient(const Point<dim>            &p,
                    std::vector<Tensor<1, dim>> &grad) const override
    {
      AssertDimension(grad.size(), dim);

      const double x = p[0];
      const double y = p[1];

      // ∂u0/∂x
      grad[0][0] =
        (1.0 / 4.0) * x * (-1.0 + 16.0 * x * x) * y * (-1.0 + 4.0 * y * y);

      // ∂u0/∂y
      grad[0][1] =
        (1.0 / 256.0) * std::pow(1.0 - 16.0 * x * x, 2) * (-1.0 + 12.0 * y * y);

      // ∂u1/∂x
      grad[1][0] =
        (1.0 / 64.0) * (1.0 - 48.0 * x * x) * std::pow(1.0 - 4.0 * y * y, 2);

      // ∂u1/∂y
      grad[1][1] =
        (1.0 / 4.0) * x * (-1.0 + 16.0 * x * x) * y * (1.0 - 4.0 * y * y);
    }
  };

  class ExactSolution_onlyu : public Function<dim>
  {
  public:
    ExactSolution_onlyu()
      : Function<dim>(dim) // 2 componenti
    {}

    // ======================
    // valore singola componente
    // ======================
    virtual double
    value(const Point<dim> &p, const unsigned int component = 0) const override
    {
      const double x = p[0];
      const double y = p[1];

      if (component == 0)
        return (1.0 / 256.0) * std::pow(1.0 - 16.0 * x * x, 2) * y *
               (-1.0 + 4.0 * y * y);

      if (component == 1)
        return -(1.0 / 64.0) * x * (-1.0 + 16.0 * x * x) *
               std::pow(1.0 - 4.0 * y * y, 2);

      return 0.0;
    }

    // ======================
    // valore vettoriale
    // ======================
    virtual void
    vector_value(const Point<dim> &p, Vector<double> &values) const override
    {
      AssertDimension(values.size(), this->n_components);

      for (unsigned int c = 0; c < this->n_components; ++c)
        values[c] = value(p, c);
    }

    virtual void
    vector_gradient(const Point<dim>            &p,
                    std::vector<Tensor<1, dim>> &grad) const override
    {
      AssertDimension(grad.size(), dim);

      const double x = p[0];
      const double y = p[1];

      // ∂u0/∂x
      grad[0][0] =
        (1.0 / 4.0) * x * (-1.0 + 16.0 * x * x) * y * (-1.0 + 4.0 * y * y);

      // ∂u0/∂y
      grad[0][1] =
        (1.0 / 256.0) * std::pow(1.0 - 16.0 * x * x, 2) * (-1.0 + 12.0 * y * y);

      // ∂u1/∂x
      grad[1][0] =
        (1.0 / 64.0) * (1.0 - 48.0 * x * x) * std::pow(1.0 - 4.0 * y * y, 2);

      // ∂u1/∂y
      grad[1][1] =
        (1.0 / 4.0) * x * (-1.0 + 16.0 * x * x) * y * (1.0 - 4.0 * y * y);
    }
  };

  class ExactSolution_d : public Function<dim>
  {
  public:
    ExactSolution_d()
      : Function<dim>(dim + 1 + dim) // 5 componenti
    {}

    // ======================
    // valore singola componente
    // ======================
    virtual double
    value(const Point<dim> &p, const unsigned int component = 0) const override
    {
      const double x = p[0];
      const double y = p[1];

      if (component == 3) // d0
        return (1.0 / 256.0) * std::pow(1.0 - 16.0 * x * x, 2) * y *
               (-1.0 + 4.0 * y * y);

      if (component == 4) // d1
        return -(1.0 / 64.0) * x * (-1.0 + 16.0 * x * x) *
               std::pow(1.0 - 4.0 * y * y, 2);

      return 0.0;
    }

    // ======================
    // valore vettoriale
    // ======================
    virtual void
    vector_value(const Point<dim> &p, Vector<double> &values) const override
    {
      for (unsigned int c = 0; c < this->n_components; ++c)
        values[c] = value(p, c);
    }

    // ======================
    // gradiente
    // grad[i][j] = ∂(componente i)/∂x_j
    // ======================
    virtual void
    vector_gradient(const Point<dim>            &p,
                    std::vector<Tensor<1, dim>> &grad) const override
    {
      // inizializza tutto a zero
      for (auto &g : grad)
        g = 0.0;

      const double x = p[0];
      const double y = p[1];

      // ∂d0/∂x
      grad[3][0] =
        (1.0 / 4.0) * x * (-1.0 + 16.0 * x * x) * y * (-1.0 + 4.0 * y * y);

      // ∂d0/∂y
      grad[3][1] =
        (1.0 / 256.0) * std::pow(1.0 - 16.0 * x * x, 2) * (-1.0 + 12.0 * y * y);

      // ∂d1/∂x
      grad[4][0] =
        (1.0 / 64.0) * (1.0 - 48.0 * x * x) * std::pow(1.0 - 4.0 * y * y, 2);

      // ∂d1/∂y
      grad[4][1] =
        (1.0 / 4.0) * x * (-1.0 + 16.0 * x * x) * y * (1.0 - 4.0 * y * y);
    }
  };

  class ExactSolution_p : public Function<dim>
  {
  public:
    ExactSolution_p()
      : Function<dim>(dim + 1 + dim) // 5 componenti
    {}

    // ======================
    // valore singola componente
    // ======================
    virtual double
    value(const Point<dim> &p, const unsigned int component = 0) const override
    {
      const double x = p[0];
      const double y = p[1];

      if (component == 2) // pressione
        return (x - 1.0 / 4.0) * (x + 1.0 / 4.0) * (y - 1.0 / 2.0) *
               (y + 1.0 / 2.0);

      return 0.0;
    }

    // ======================
    // valore vettoriale
    // ======================
    virtual void
    vector_value(const Point<dim> &p, Vector<double> &values) const override
    {
      for (unsigned int c = 0; c < this->n_components; ++c)
        values[c] = value(p, c);
    }

    // ======================
    // gradiente vettoriale
    // grad[i][j] = ∂(componente i)/∂x_j
    // ======================
    virtual void
    vector_gradient(const Point<dim>            &p,
                    std::vector<Tensor<1, dim>> &grad) const override
    {
      // tutto a zero
      for (auto &g : grad)
        g = 0.0;

      const double x = p[0];
      const double y = p[1];

      // ∂p/∂x  → componente 2
      grad[2][0] = (x - 1.0 / 4.0) * (y - 1.0 / 2.0) * (y + 1.0 / 2.0) +
                   (x + 1.0 / 4.0) * (y - 1.0 / 2.0) * (y + 1.0 / 2.0);

      // ∂p/∂y  → componente 2
      grad[2][1] = (x - 1.0 / 4.0) * (x + 1.0 / 4.0) * (y - 1.0 / 2.0) +
                   (x - 1.0 / 4.0) * (x + 1.0 / 4.0) * (y + 1.0 / 2.0);
    }
  };

  class ExactForce_f : public Function<dim>
  {
  public:
    ExactForce_f(const double viscosity)
      : Function<dim>(dim + 1 + dim)
      , nu(viscosity)
    {}

    // ======================
    // valore singola componente
    // ======================
    virtual double
    value(const Point<dim> &p, const unsigned int component) const override
    {
      const double x = p[0];
      const double y = p[1];

      if (component == 0)
        return x * (-0.5 + 2.0 * y * y) - (11.0 / 32.0) * y * nu -
               24.0 * std::pow(x, 4) * y * nu + std::pow(y, 3) * nu +
               3.0 * x * x * y * (5.0 - 16.0 * y * y) * nu;

      if (component == 1)
        return (-1.0 / 8.0 + 2.0 * x * x) * y + (7.0 / 4.0) * x * nu -
               4.0 * std::pow(x, 3) * nu +
               3.0 * x * (-5.0 + 16.0 * x * x) * y * y * nu +
               24.0 * x * std::pow(y, 4) * nu;

      return 0.0;
    }

    // ======================
    // valore vettoriale
    // ======================
    virtual void
    vector_value(const Point<dim> &p, Vector<double> &values) const override
    {
      for (unsigned int c = 0; c < this->n_components; ++c)
        values[c] = value(p, c);
    }

  private:
    const double nu;
  };



  class ExactForce_g : public Function<dim>
  {
  public:
    ExactForce_g(const double mu_)
      : Function<dim>(dim + 1 + dim) // 5 componenti
      , mu(mu_)
    {}

    virtual double
    value(const Point<dim> &p, const unsigned int component = 0) const override
    {
      const double x = p[0];
      const double y = p[1];

      if (component == 3)
        return (-(11.0 / 32.0) + 15.0 * x * x - 24.0 * std::pow(x, 4)) * y *
                 mu +
               std::pow(y, 3) * (mu - 48.0 * x * x * mu);

      if (component == 4)
        return 1.0 / 4.0 * x *
               (7.0 - 60.0 * y * y + 96.0 * std::pow(y, 4) +
                16.0 * x * x * (-1.0 + 12.0 * y * y)) *
               mu;

      return 0.0;
    }

    virtual void
    vector_value(const Point<dim> &p, Vector<double> &values) const override
    {
      AssertDimension(values.size(), this->n_components);

      for (unsigned int c = 0; c < this->n_components; ++c)
        values[c] = value(p, c);
    }

  private:
    const double mu;
  };



  class ExactNeumann_hRight : public Function<dim>
  {
  public:
    ExactNeumann_hRight(const double viscosity_)
      : Function<dim>(dim + 1 + dim) // 5 componenti
      , nu(viscosity_)
    {}

    virtual double
    value(const Point<dim> &p, const unsigned int component = 0) const override
    {
      const double x = p[0];
      const double y = p[1];

      if (component == 0)
        return 1.0 / 64.0 * (-1.0 + 16.0 * x * x) * (-1.0 + 4.0 * y * y) *
               (-1.0 + 32.0 * x * y * nu);

      if (component == 1)
        return 1.0 / 256.0 *
               (4.0 * (1.0 - 48.0 * x * x) * std::pow(1.0 - 4.0 * y * y, 2) +
                std::pow(1.0 - 16.0 * x * x, 2) * (-1.0 + 12.0 * y * y)) *
               nu;

      return 0.0;
    }

    virtual void
    vector_value(const Point<dim> &p, Vector<double> &values) const override
    {
      AssertDimension(values.size(), this->n_components);

      for (unsigned int c = 0; c < this->n_components; ++c)
        values[c] = value(p, c);
    }

  private:
    const double nu;
  };

  class ExactNeumann_hLeft : public Function<dim>
  {
  public:
    ExactNeumann_hLeft(const double viscosity_)
      : Function<dim>(dim + 1 + dim) // 5 componenti
      , nu(viscosity_)
    {}

    virtual double
    value(const Point<dim> &p, const unsigned int component = 0) const override
    {
      const double x = p[0];
      const double y = p[1];


      if (component == 0)
        return -(1.0 / 64.0) * (-1.0 + 16.0 * x * x) * (-1.0 + 4.0 * y * y) *
               (-1.0 + 32.0 * x * y * nu);

      if (component == 1)
        return -(1.0 / 256.0) *
               (4.0 * (1.0 - 48.0 * x * x) * std::pow(1.0 - 4.0 * y * y, 2) +
                std::pow(1.0 - 16.0 * x * x, 2) * (-1.0 + 12.0 * y * y)) *
               nu;

      return 0.0;
    }

    virtual void
    vector_value(const Point<dim> &p, Vector<double> &values) const override
    {
      AssertDimension(values.size(), this->n_components);

      for (unsigned int c = 0; c < this->n_components; ++c)
        values[c] = value(p, c);
    }

  private:
    const double nu;
  };


  // template <int dim>
  // class ExactSolution_FSI : public Function<dim>
  // {
  // public:
  //   ExactSolution_FSI()
  //     : Function<dim>(dim + 1 + dim) // 5 components: u(2), p(1), d(2)
  //   {}

  //   virtual double
  //   value(const Point<dim> &p, const unsigned int component = 0) const override
  //   {
  //     const double x  = p[0];
  //     const double y  = p[1];
  //     const double pi = 3.14159265358979323846;

  //     // -------- velocity u (components 0, 1) --------
  //     if (component == 0) // u0
  //       return (std::pow(1.0 - 16.0 * x * x, 2) * (-1.0 + 2.0 * y) *
  //               (1.0 + 2.0 * y) * std::sin(pi * x) *
  //               (-16.0 * y * std::cos(pi * y) +
  //                pi * (-1.0 + 4.0 * y * y) * std::sin(pi * y))) /
  //              4096.0;

  //     if (component == 1) // u1
  //       return -(((-1.0 + 4.0 * x) * (1.0 + 4.0 * x) *
  //                 std::pow(1.0 - 4.0 * y * y, 2) * std::cos(pi * y) *
  //                 (pi * (-1.0 + 16.0 * x * x) * std::cos(pi * x) +
  //                  64.0 * x * std::sin(pi * x))) /
  //                4096.0);

  //     // -------- pressure p (component 2) --------
  //     if (component == 2)
  //       return (-0.25 + x) * (0.25 + x) * (-0.5 + y) * (0.5 + y) *
  //              std::sin(pi * x) * std::sin(pi * y);

  //     // -------- displacement d (components 3, 4) --------
  //     if (component == 3) // d0
  //       return (std::pow(1.0 - 16.0 * x * x, 2) * (-1.0 + 2.0 * y) *
  //               (1.0 + 2.0 * y) * std::sin(pi * x) *
  //               (-16.0 * y * std::cos(pi * y) +
  //                pi * (-1.0 + 4.0 * y * y) * std::sin(pi * y))) /
  //              4096.0;

  //     if (component == 4) // d1
  //       return -(((-1.0 + 4.0 * x) * (1.0 + 4.0 * x) *
  //                 std::pow(1.0 - 4.0 * y * y, 2) * std::cos(pi * y) *
  //                 (pi * (-1.0 + 16.0 * x * x) * std::cos(pi * x) +
  //                  64.0 * x * std::sin(pi * x))) /
  //                4096.0);

  //     return 0.0;
  //   }

  //   virtual void
  //   vector_value(const Point<dim> &p, Vector<double> &values) const override
  //   {
  //     AssertDimension(values.size(), this->n_components);
  //     for (unsigned int c = 0; c < this->n_components; ++c)
  //       values[c] = value(p, c);
  //   }

  //   // ======================
  //   // GRADIENT IMPLEMENTATION
  //   // ======================
  //   virtual void
  //   vector_gradient(const Point<dim>            &p,
  //                   std::vector<Tensor<1, dim>> &grad) const override
  //   {
  //     AssertDimension(grad.size(), this->n_components);

  //     const double x  = p[0];
  //     const double y  = p[1];
  //     const double pi = 3.14159265358979323846;

  //     // Reset all to zero
  //     for (auto &g : grad)
  //       g = 0.0;

  //     // --- GRAD U (Components 0 and 1) ---

  //     // grad[0][0] = du0/dx
  //     grad[0][0] =
  //       -(((-1.0 + 16.0 * x * x) * (-1.0 + 2.0 * y) * (1.0 + 2.0 * y) *
  //          (pi * (-1.0 + 16.0 * x * x) * std::cos(pi * x) +
  //           64.0 * x * std::sin(pi * x)) *
  //          (-16.0 * y * std::cos(pi * y) +
  //           pi * (-1.0 + 4.0 * y * y) * std::sin(pi * y))) /
  //         4096.0);

  //     // grad[0][1] = du0/dy
  //     grad[0][1] =
  //       -((std::pow(1.0 - 16.0 * x * x, 2) * std::sin(pi * x) *
  //          ((16.0 - 192.0 * y * y +
  //            std::pow(pi, 2) * std::pow(1.0 - 4.0 * y * y, 2)) *
  //             std::cos(pi * y) +
  //           32.0 * pi * y * (-1.0 + 4.0 * y * y) * std::sin(pi * y))) /
  //         4096.0);

  //     // grad[1][0] = du1/dx
  //     grad[1][0] =
  //       (std::pow(1.0 - 4.0 * y * y, 2) * std::cos(pi * y) *
  //        (-128.0 * pi * x * (-1.0 + 16.0 * x * x) * std::cos(pi * x) +
  //         (64.0 - 3072.0 * x * x +
  //          std::pow(pi, 2) * std::pow(1.0 - 16.0 * x * x, 2)) *
  //           std::sin(pi * x))) /
  //       4096.0;

  //     // grad[1][1] = du1/dy
  //     grad[1][1] = ((-1.0 + 4.0 * x) * (1.0 + 4.0 * x) * (1.0 - 4.0 * y * y) *
  //                   (pi * (-1.0 + 16.0 * x * x) * std::cos(pi * x) +
  //                    64.0 * x * std::sin(pi * x)) *
  //                   (16.0 * y * std::cos(pi * y) +
  //                    pi * (1.0 - 4.0 * y * y) * std::sin(pi * y))) /
  //                  4096.0;


  //     // --- GRAD P (Component 2) ---

  //     // grad[2][0] = dp/dx
  //     grad[2][0] = pi * (-0.25 + x) * (0.25 + x) * (-0.5 + y) * (0.5 + y) *
  //                    std::cos(pi * x) * std::sin(pi * y) +
  //                  (-0.25 + x) * (-0.5 + y) * (0.5 + y) * std::sin(pi * x) *
  //                    std::sin(pi * y) +
  //                  (0.25 + x) * (-0.5 + y) * (0.5 + y) * std::sin(pi * x) *
  //                    std::sin(pi * y);

  //     // grad[2][1] = dp/dy
  //     grad[2][1] = pi * (-0.25 + x) * (0.25 + x) * (-0.5 + y) * (0.5 + y) *
  //                    std::cos(pi * y) * std::sin(pi * x) +
  //                  (-0.25 + x) * (0.25 + x) * (-0.5 + y) * std::sin(pi * x) *
  //                    std::sin(pi * y) +
  //                  (-0.25 + x) * (0.25 + x) * (0.5 + y) * std::sin(pi * x) *
  //                    std::sin(pi * y);


  //     // --- GRAD D (Components 3 and 4) ---

  //     // grad[3][0] = dd0/dx
  //     grad[3][0] =
  //       -(((-1.0 + 16.0 * x * x) * (-1.0 + 2.0 * y) * (1.0 + 2.0 * y) *
  //          (pi * (-1.0 + 16.0 * x * x) * std::cos(pi * x) +
  //           64.0 * x * std::sin(pi * x)) *
  //          (-16.0 * y * std::cos(pi * y) +
  //           pi * (-1.0 + 4.0 * y * y) * std::sin(pi * y))) /
  //         4096.0);

  //     // grad[3][1] = dd0/dy
  //     grad[3][1] =
  //       (std::pow(1.0 - 4.0 * y * y, 2) * std::cos(pi * y) *
  //          (-128.0 * pi * x * (-1.0 + 16.0 * x * x) * std::cos(pi * x) +
  //           (64.0 - 3072.0 * x * x +
  //            std::pow(pi, 2) * std::pow(1.0 - 16.0 * x * x, 2)) *
  //             std::sin(pi * x)) -
  //        std::pow(1.0 - 16.0 * x * x, 2) * std::sin(pi * x) *
  //          ((16.0 - 192.0 * y * y +
  //            std::pow(pi, 2) * std::pow(1.0 - 4.0 * y * y, 2)) *
  //             std::cos(pi * y) +
  //           32.0 * pi * y * (-1.0 + 4.0 * y * y) * std::sin(pi * y))) /
  //       8192.0;

  //     // grad[4][0] = dd1/dx
  //     grad[4][0] =
  //       (std::pow(1.0 - 4.0 * y * y, 2) * std::cos(pi * y) *
  //          (-128.0 * pi * x * (-1.0 + 16.0 * x * x) * std::cos(pi * x) +
  //           (64.0 - 3072.0 * x * x +
  //            std::pow(pi, 2) * std::pow(1.0 - 16.0 * x * x, 2)) *
  //             std::sin(pi * x)) -
  //        std::pow(1.0 - 16.0 * x * x, 2) * std::sin(pi * x) *
  //          ((16.0 - 192.0 * y * y +
  //            std::pow(pi, 2) * std::pow(1.0 - 4.0 * y * y, 2)) *
  //             std::cos(pi * y) +
  //           32.0 * pi * y * (-1.0 + 4.0 * y * y) * std::sin(pi * y))) /
  //       8192.0;

  //     // grad[4][1] = dd1/dy
  //     grad[4][1] = ((-1.0 + 4.0 * x) * (1.0 + 4.0 * x) * (1.0 - 4.0 * y * y) *
  //                   (pi * (-1.0 + 16.0 * x * x) * std::cos(pi * x) +
  //                    64.0 * x * std::sin(pi * x)) *
  //                   (16.0 * y * std::cos(pi * y) +
  //                    pi * (1.0 - 4.0 * y * y) * std::sin(pi * y))) /
  //                  4096.0;
  //   }
  // };


  // class ExactSolution_u : public Function<dim>
  // {
  // public:
  //   ExactSolution_u()
  //     : Function<dim>(dim + 1 + dim) // 5 components
  //   {}

  //   virtual double
  //   value(const Point<dim> &p, const unsigned int component = 0) const override
  //   {
  //     const double x  = p[0];
  //     const double y  = p[1];
  //     const double pi = 3.14159265358979323846;

  //     if (component == 0)
  //       return (std::pow(1.0 - 16.0 * x * x, 2) * (-1.0 + 2.0 * y) *
  //               (1.0 + 2.0 * y) * std::sin(pi * x) *
  //               (-16.0 * y * std::cos(pi * y) +
  //                pi * (-1.0 + 4.0 * y * y) * std::sin(pi * y))) /
  //              4096.0;

  //     if (component == 1)
  //       return -(((-1.0 + 4.0 * x) * (1.0 + 4.0 * x) *
  //                 std::pow(1.0 - 4.0 * y * y, 2) * std::cos(pi * y) *
  //                 (pi * (-1.0 + 16.0 * x * x) * std::cos(pi * x) +
  //                  64.0 * x * std::sin(pi * x))) /
  //                4096.0);

  //     return 0.0;
  //   }

  //   virtual void
  //   vector_value(const Point<dim> &p, Vector<double> &values) const override
  //   {
  //     AssertDimension(values.size(), this->n_components);
  //     for (unsigned int c = 0; c < this->n_components; ++c)
  //       values[c] = value(p, c);
  //   }

  //   virtual void
  //   vector_gradient(const Point<dim>            &p,
  //                   std::vector<Tensor<1, dim>> &grad) const override
  //   {
  //     AssertDimension(grad.size(),
  //                     dim); // This class usually only requests gradients for
  //                           // its own components if size matches

  //     // Note: If vector_gradient is called with a vector of size 5, we should
  //     // fill 0 and 1. If it is called with size 2 (for just u), we fill 0
  //     // and 1. Standard deal.II usage for an ExactSolution_u often implies we
  //     // just want the u-block. However, to be safe regarding indices, we fill
  //     // indices 0 and 1 relative to the object.

  //     const double x  = p[0];
  //     const double y  = p[1];
  //     const double pi = 3.14159265358979323846;

  //     for (auto &g : grad)
  //       g = 0.0;

  //     // grad[0][0] = du0/dx
  //     grad[0][0] =
  //       -(((-1.0 + 16.0 * x * x) * (-1.0 + 2.0 * y) * (1.0 + 2.0 * y) *
  //          (pi * (-1.0 + 16.0 * x * x) * std::cos(pi * x) +
  //           64.0 * x * std::sin(pi * x)) *
  //          (-16.0 * y * std::cos(pi * y) +
  //           pi * (-1.0 + 4.0 * y * y) * std::sin(pi * y))) /
  //         4096.0);

  //     // grad[0][1] = du0/dy
  //     grad[0][1] =
  //       -((std::pow(1.0 - 16.0 * x * x, 2) * std::sin(pi * x) *
  //          ((16.0 - 192.0 * y * y +
  //            std::pow(pi, 2) * std::pow(1.0 - 4.0 * y * y, 2)) *
  //             std::cos(pi * y) +
  //           32.0 * pi * y * (-1.0 + 4.0 * y * y) * std::sin(pi * y))) /
  //         4096.0);

  //     // grad[1][0] = du1/dx
  //     grad[1][0] =
  //       (std::pow(1.0 - 4.0 * y * y, 2) * std::cos(pi * y) *
  //        (-128.0 * pi * x * (-1.0 + 16.0 * x * x) * std::cos(pi * x) +
  //         (64.0 - 3072.0 * x * x +
  //          std::pow(pi, 2) * std::pow(1.0 - 16.0 * x * x, 2)) *
  //           std::sin(pi * x))) /
  //       4096.0;

  //     // grad[1][1] = du1/dy
  //     grad[1][1] = ((-1.0 + 4.0 * x) * (1.0 + 4.0 * x) * (1.0 - 4.0 * y * y) *
  //                   (pi * (-1.0 + 16.0 * x * x) * std::cos(pi * x) +
  //                    64.0 * x * std::sin(pi * x)) *
  //                   (16.0 * y * std::cos(pi * y) +
  //                    pi * (1.0 - 4.0 * y * y) * std::sin(pi * y))) /
  //                  4096.0;
  //   }
  // };


  // class ExactSolution_onlyu : public Function<dim>
  // {
  // public:
  //   ExactSolution_onlyu()
  //     : Function<dim>(dim) // 2 components
  //   {}

  //   virtual double
  //   value(const Point<dim> &p, const unsigned int component = 0) const override
  //   {
  //     const double x  = p[0];
  //     const double y  = p[1];
  //     const double pi = 3.14159265358979323846;

  //     if (component == 0)
  //       return (std::pow(1.0 - 16.0 * x * x, 2) * (-1.0 + 2.0 * y) *
  //               (1.0 + 2.0 * y) * std::sin(pi * x) *
  //               (-16.0 * y * std::cos(pi * y) +
  //                pi * (-1.0 + 4.0 * y * y) * std::sin(pi * y))) /
  //              4096.0;

  //     if (component == 1)
  //       return -(((-1.0 + 4.0 * x) * (1.0 + 4.0 * x) *
  //                 std::pow(1.0 - 4.0 * y * y, 2) * std::cos(pi * y) *
  //                 (pi * (-1.0 + 16.0 * x * x) * std::cos(pi * x) +
  //                  64.0 * x * std::sin(pi * x))) /
  //                4096.0);

  //     return 0.0;
  //   }

  //   virtual void
  //   vector_value(const Point<dim> &p, Vector<double> &values) const override
  //   {
  //     AssertDimension(values.size(), this->n_components);
  //     for (unsigned int c = 0; c < this->n_components; ++c)
  //       values[c] = value(p, c);
  //   }

  //   virtual void
  //   vector_gradient(const Point<dim>            &p,
  //                   std::vector<Tensor<1, dim>> &grad) const override
  //   {
  //     AssertDimension(grad.size(), dim);

  //     const double x  = p[0];
  //     const double y  = p[1];
  //     const double pi = 3.14159265358979323846;

  //     for (auto &g : grad)
  //       g = 0.0;

  //     // grad[0][0] = du0/dx
  //     grad[0][0] =
  //       -(((-1.0 + 16.0 * x * x) * (-1.0 + 2.0 * y) * (1.0 + 2.0 * y) *
  //          (pi * (-1.0 + 16.0 * x * x) * std::cos(pi * x) +
  //           64.0 * x * std::sin(pi * x)) *
  //          (-16.0 * y * std::cos(pi * y) +
  //           pi * (-1.0 + 4.0 * y * y) * std::sin(pi * y))) /
  //         4096.0);

  //     // grad[0][1] = du0/dy
  //     grad[0][1] =
  //       -((std::pow(1.0 - 16.0 * x * x, 2) * std::sin(pi * x) *
  //          ((16.0 - 192.0 * y * y +
  //            std::pow(pi, 2) * std::pow(1.0 - 4.0 * y * y, 2)) *
  //             std::cos(pi * y) +
  //           32.0 * pi * y * (-1.0 + 4.0 * y * y) * std::sin(pi * y))) /
  //         4096.0);

  //     // grad[1][0] = du1/dx
  //     grad[1][0] =
  //       (std::pow(1.0 - 4.0 * y * y, 2) * std::cos(pi * y) *
  //        (-128.0 * pi * x * (-1.0 + 16.0 * x * x) * std::cos(pi * x) +
  //         (64.0 - 3072.0 * x * x +
  //          std::pow(pi, 2) * std::pow(1.0 - 16.0 * x * x, 2)) *
  //           std::sin(pi * x))) /
  //       4096.0;

  //     // grad[1][1] = du1/dy
  //     grad[1][1] = ((-1.0 + 4.0 * x) * (1.0 + 4.0 * x) * (1.0 - 4.0 * y * y) *
  //                   (pi * (-1.0 + 16.0 * x * x) * std::cos(pi * x) +
  //                    64.0 * x * std::sin(pi * x)) *
  //                   (16.0 * y * std::cos(pi * y) +
  //                    pi * (1.0 - 4.0 * y * y) * std::sin(pi * y))) /
  //                  4096.0;
  //   }
  // };


  // class ExactSolution_d : public Function<dim>
  // {
  // public:
  //   ExactSolution_d()
  //     : Function<dim>(dim + 1 + dim) // 5 components
  //   {}

  //   virtual double
  //   value(const Point<dim> &p, const unsigned int component = 0) const override
  //   {
  //     const double x  = p[0];
  //     const double y  = p[1];
  //     const double pi = 3.14159265358979323846;

  //     if (component == 3) // d0
  //       return (std::pow(1.0 - 16.0 * x * x, 2) * (-1.0 + 2.0 * y) *
  //               (1.0 + 2.0 * y) * std::sin(pi * x) *
  //               (-16.0 * y * std::cos(pi * y) +
  //                pi * (-1.0 + 4.0 * y * y) * std::sin(pi * y))) /
  //              4096.0;

  //     if (component == 4) // d1
  //       return -(((-1.0 + 4.0 * x) * (1.0 + 4.0 * x) *
  //                 std::pow(1.0 - 4.0 * y * y, 2) * std::cos(pi * y) *
  //                 (pi * (-1.0 + 16.0 * x * x) * std::cos(pi * x) +
  //                  64.0 * x * std::sin(pi * x))) /
  //                4096.0);

  //     return 0.0;
  //   }

  //   virtual void
  //   vector_value(const Point<dim> &p, Vector<double> &values) const override
  //   {
  //     for (unsigned int c = 0; c < this->n_components; ++c)
  //       values[c] = value(p, c);
  //   }

  //   virtual void
  //   vector_gradient(const Point<dim>            &p,
  //                   std::vector<Tensor<1, dim>> &grad) const override
  //   {
  //     const double x  = p[0];
  //     const double y  = p[1];
  //     const double pi = 3.14159265358979323846;

  //     for (auto &g : grad)
  //       g = 0.0;

  //     // The logic for indices 3 and 4 matches the ExactSolution_FSI class, but
  //     // mapped to the corresponding components for this specific function
  //     // object. If this object is used to represent the full solution vector of
  //     // size 5:

  //     // grad[3][0] = dd0/dx
  //     grad[3][0] =
  //       -(((-1.0 + 16.0 * x * x) * (-1.0 + 2.0 * y) * (1.0 + 2.0 * y) *
  //          (pi * (-1.0 + 16.0 * x * x) * std::cos(pi * x) +
  //           64.0 * x * std::sin(pi * x)) *
  //          (-16.0 * y * std::cos(pi * y) +
  //           pi * (-1.0 + 4.0 * y * y) * std::sin(pi * y))) /
  //         4096.0);

  //     // grad[3][1] = dd0/dy
  //     grad[3][1] =
  //       (std::pow(1.0 - 4.0 * y * y, 2) * std::cos(pi * y) *
  //          (-128.0 * pi * x * (-1.0 + 16.0 * x * x) * std::cos(pi * x) +
  //           (64.0 - 3072.0 * x * x +
  //            std::pow(pi, 2) * std::pow(1.0 - 16.0 * x * x, 2)) *
  //             std::sin(pi * x)) -
  //        std::pow(1.0 - 16.0 * x * x, 2) * std::sin(pi * x) *
  //          ((16.0 - 192.0 * y * y +
  //            std::pow(pi, 2) * std::pow(1.0 - 4.0 * y * y, 2)) *
  //             std::cos(pi * y) +
  //           32.0 * pi * y * (-1.0 + 4.0 * y * y) * std::sin(pi * y))) /
  //       8192.0;

  //     // grad[4][0] = dd1/dx
  //     grad[4][0] =
  //       (std::pow(1.0 - 4.0 * y * y, 2) * std::cos(pi * y) *
  //          (-128.0 * pi * x * (-1.0 + 16.0 * x * x) * std::cos(pi * x) +
  //           (64.0 - 3072.0 * x * x +
  //            std::pow(pi, 2) * std::pow(1.0 - 16.0 * x * x, 2)) *
  //             std::sin(pi * x)) -
  //        std::pow(1.0 - 16.0 * x * x, 2) * std::sin(pi * x) *
  //          ((16.0 - 192.0 * y * y +
  //            std::pow(pi, 2) * std::pow(1.0 - 4.0 * y * y, 2)) *
  //             std::cos(pi * y) +
  //           32.0 * pi * y * (-1.0 + 4.0 * y * y) * std::sin(pi * y))) /
  //       8192.0;

  //     // grad[4][1] = dd1/dy
  //     grad[4][1] = ((-1.0 + 4.0 * x) * (1.0 + 4.0 * x) * (1.0 - 4.0 * y * y) *
  //                   (pi * (-1.0 + 16.0 * x * x) * std::cos(pi * x) +
  //                    64.0 * x * std::sin(pi * x)) *
  //                   (16.0 * y * std::cos(pi * y) +
  //                    pi * (1.0 - 4.0 * y * y) * std::sin(pi * y))) /
  //                  4096.0;
  //   }
  // };


  // class ExactSolution_p : public Function<dim>
  // {
  // public:
  //   ExactSolution_p()
  //     : Function<dim>(dim + 1 + dim) // 5 components
  //   {}

  //   virtual double
  //   value(const Point<dim> &p, const unsigned int component = 0) const override
  //   {
  //     const double x  = p[0];
  //     const double y  = p[1];
  //     const double pi = 3.14159265358979323846;

  //     if (component == 2) // pressure
  //       {
  //         return (-0.25 + x) * (0.25 + x) * (-0.5 + y) * (0.5 + y) *
  //                std::sin(pi * x) * std::sin(pi * y);
  //       }
  //     return 0.0;
  //   }

  //   virtual void
  //   vector_value(const Point<dim> &p, Vector<double> &values) const override
  //   {
  //     for (unsigned int c = 0; c < this->n_components; ++c)
  //       values[c] = value(p, c);
  //   }

  //   virtual void
  //   vector_gradient(const Point<dim>            &p,
  //                   std::vector<Tensor<1, dim>> &grad) const override
  //   {
  //     const double x  = p[0];
  //     const double y  = p[1];
  //     const double pi = 3.14159265358979323846;

  //     for (auto &g : grad)
  //       g = 0.0;

  //     // grad[2][0] = dp/dx
  //     grad[2][0] = pi * (-0.25 + x) * (0.25 + x) * (-0.5 + y) * (0.5 + y) *
  //                    std::cos(pi * x) * std::sin(pi * y) +
  //                  (-0.25 + x) * (-0.5 + y) * (0.5 + y) * std::sin(pi * x) *
  //                    std::sin(pi * y) +
  //                  (0.25 + x) * (-0.5 + y) * (0.5 + y) * std::sin(pi * x) *
  //                    std::sin(pi * y);

  //     // grad[2][1] = dp/dy
  //     grad[2][1] = pi * (-0.25 + x) * (0.25 + x) * (-0.5 + y) * (0.5 + y) *
  //                    std::cos(pi * y) * std::sin(pi * x) +
  //                  (-0.25 + x) * (0.25 + x) * (-0.5 + y) * std::sin(pi * x) *
  //                    std::sin(pi * y) +
  //                  (-0.25 + x) * (0.25 + x) * (0.5 + y) * std::sin(pi * x) *
  //                    std::sin(pi * y);
  //   }
  // };

  // class ExactForce_f : public Function<dim>
  // {
  // public:
  //   ExactForce_f(const double viscosity)
  //     : Function<dim>(dim + 1 + dim)
  //     , nu(viscosity)
  //   {}

  //   virtual double
  //   value(const Point<dim> &p, const unsigned int component) const override
  //   {
  //     const double x  = p[0];
  //     const double y  = p[1];
  //     const double pi = 3.14159265358979323846;

  //     if (component == 0)
  //       {
  //         // First component of f
  //         double term1 = pi * (-0.25 + x) * (0.25 + x) * (-0.5 + y) *
  //                        (0.5 + y) * std::cos(pi * x) * std::sin(pi * y);
  //         double term2 = (-0.25 + x) * (-0.5 + y) * (0.5 + y) *
  //                        std::sin(pi * x) * std::sin(pi * y);
  //         double term3 = (0.25 + x) * (-0.5 + y) * (0.5 + y) *
  //                        std::sin(pi * x) * std::sin(pi * y);
  //         double term4 = (1.0 / 16.0) * pi * x * (-1.0 + 16.0 * x * x) *
  //                        (-1.0 + 4.0 * y * y) * nu * std::cos(pi * x) *
  //                        (-16.0 * y * std::cos(pi * y) +
  //                         pi * (-1.0 + 4.0 * y * y) * std::sin(pi * y));

  //         double complex_term_inside =
  //           -32.0 * y *
  //             (-22.0 - 1536.0 * std::pow(x, 4) + 64.0 * y * y +
  //              std::pow(pi, 2) * std::pow(1.0 - 16.0 * x * x, 2) *
  //                (-1.0 + 4.0 * y * y) -
  //              192.0 * x * x * (-5.0 + 16.0 * y * y)) *
  //             std::cos(pi * y) +
  //           pi *
  //             (std::pow(pi, 2) * std::pow(1.0 - 16.0 * x * x, 2) *
  //                std::pow(1.0 - 4.0 * y * y, 2) -
  //              8.0 * (-7.0 + 68.0 * y * y - 64.0 * std::pow(y, 4) +
  //                     768.0 * std::pow(x, 4) * (-1.0 + 12.0 * y * y) +
  //                     96.0 * x * x *
  //                       (3.0 - 28.0 * y * y + 32.0 * std::pow(y, 4)))) *
  //             std::sin(pi * y);

  //         double term5 =
  //           -(1.0 / 1024.0) * nu * std::sin(pi * x) * complex_term_inside;

  //         return term1 + term2 + term3 + term4 + term5;
  //       }

  //     if (component == 1)
  //       {
  //         // Second component of f
  //         double term1 = pi * (-0.25 + x) * (0.25 + x) * (-0.5 + y) *
  //                        (0.5 + y) * std::cos(pi * y) * std::sin(pi * x);

  //         double term2 = (1.0 / 64.0) * std::pow(1.0 - 4.0 * y * y, 2) * nu *
  //                        std::cos(pi * y) *
  //                        (pi * (-1.0 + 16.0 * x * x) * std::cos(pi * x) +
  //                         64.0 * x * std::sin(pi * x));

  //         double term3 =
  //           -(pi * (-1.0 + 4.0 * x) * (1.0 + 4.0 * x) *
  //             std::pow(1.0 - 4.0 * y * y, 2) * nu * std::cos(pi * y) *
  //             ((-160.0 + std::pow(pi, 2) * (-1.0 + 16.0 * x * x)) *
  //                std::cos(pi * x) +
  //              128.0 * pi * x * std::sin(pi * x))) /
  //           2048.0;

  //         double term4 = -(1.0 / 32.0) * x * std::pow(1.0 - 4.0 * y * y, 2) *
  //                        nu * std::cos(pi * y) *
  //                        (-96.0 * pi * x * std::cos(pi * x) +
  //                         (-64.0 + std::pow(pi, 2) * (-1.0 + 16.0 * x * x)) *
  //                           std::sin(pi * x));

  //         double term5 = (-0.25 + x) * (0.25 + x) * (-0.5 + y) *
  //                        std::sin(pi * x) * std::sin(pi * y);
  //         double term6 = (-0.25 + x) * (0.25 + x) * (0.5 + y) *
  //                        std::sin(pi * x) * std::sin(pi * y);

  //         double term7 =
  //           -((-1.0 + 4.0 * x) * (1.0 + 4.0 * x) * nu *
  //             (pi * (-1.0 + 16.0 * x * x) * std::cos(pi * x) +
  //              64.0 * x * std::sin(pi * x)) *
  //             ((16.0 - 192.0 * y * y +
  //               std::pow(pi, 2) * std::pow(1.0 - 4.0 * y * y, 2)) *
  //                std::cos(pi * y) +
  //              32.0 * pi * y * (-1.0 + 4.0 * y * y) * std::sin(pi * y))) /
  //           2048.0;

  //         return term1 + term2 + term3 + term4 + term5 + term6 + term7;
  //       }

  //     return 0.0;
  //   }

  //   virtual void
  //   vector_value(const Point<dim> &p, Vector<double> &values) const override
  //   {
  //     for (unsigned int c = 0; c < this->n_components; ++c)
  //       values[c] = value(p, c);
  //   }

  // private:
  //   const double nu;
  // };


  // class ExactForce_g : public Function<dim>
  // {
  // public:
  //   ExactForce_g(const double mu_)
  //     : Function<dim>(dim + 1 + dim) // 5 components
  //     , mu(mu_)
  //   {}

  //   virtual double
  //   value(const Point<dim> &p, const unsigned int component = 0) const override
  //   {
  //     const double x  = p[0];
  //     const double y  = p[1];
  //     const double pi = 3.14159265358979323846;

  //     if (component == 3)
  //       {
  //         double term1 = 64.0 * pi * x * (-1.0 + 16.0 * x * x) *
  //                        (-1.0 + 4.0 * y * y) * mu * std::cos(pi * x) *
  //                        (-16.0 * y * std::cos(pi * y) +
  //                         pi * (-1.0 + 4.0 * y * y) * std::sin(pi * y));

  //         double inside_term =
  //           -32.0 * y *
  //             (-22.0 - 1536.0 * std::pow(x, 4) + 64.0 * y * y +
  //              std::pow(pi, 2) * std::pow(1.0 - 16.0 * x * x, 2) *
  //                (-1.0 + 4.0 * y * y) -
  //              192.0 * x * x * (-5.0 + 16.0 * y * y)) *
  //             std::cos(pi * y) +
  //           pi *
  //             (std::pow(pi, 2) * std::pow(1.0 - 16.0 * x * x, 2) *
  //                std::pow(1.0 - 4.0 * y * y, 2) -
  //              8.0 * (-7.0 + 68.0 * y * y - 64.0 * std::pow(y, 4) +
  //                     768.0 * std::pow(x, 4) * (-1.0 + 12.0 * y * y) +
  //                     96.0 * x * x *
  //                       (3.0 - 28.0 * y * y + 32.0 * std::pow(y, 4)))) *
  //             std::sin(pi * y);

  //         return (1.0 / 2048.0) * (term1 - mu * std::sin(pi * x) * inside_term);
  //       }

  //     if (component == 4)
  //       {
  //         double term1 =
  //           pi * std::cos(pi * x) *
  //           ((std::pow(pi, 2) * std::pow(1.0 - 16.0 * x * x, 2) *
  //               std::pow(1.0 - 4.0 * y * y, 2) -
  //             8.0 * (-13.0 + 108.0 * y * y - 192.0 * std::pow(y, 4) +
  //                    256.0 * std::pow(x, 4) * (-1.0 + 12.0 * y * y) +
  //                    32.0 * x * x *
  //                      (19.0 - 156.0 * y * y + 288.0 * std::pow(y, 4)))) *
  //              std::cos(pi * y) +
  //            16.0 * pi * std::pow(1.0 - 16.0 * x * x, 2) * y *
  //              (-1.0 + 4.0 * y * y) * std::sin(pi * y));

  //         double term2 = 128.0 * x * std::sin(pi * x) *
  //                        ((std::pow(pi, 2) * (-1.0 + 16.0 * x * x) *
  //                            std::pow(1.0 - 4.0 * y * y, 2) -
  //                          4.0 * (7.0 - 60.0 * y * y + 96.0 * std::pow(y, 4) +
  //                                 16.0 * x * x * (-1.0 + 12.0 * y * y))) *
  //                           std::cos(pi * y) +
  //                         8.0 * pi * (-1.0 + 16.0 * x * x) * y *
  //                           (-1.0 + 4.0 * y * y) * std::sin(pi * y));

  //         return -(1.0 / 2048.0) * mu * (term1 + term2);
  //       }

  //     return 0.0;
  //   }

  //   virtual void
  //   vector_value(const Point<dim> &p, Vector<double> &values) const override
  //   {
  //     AssertDimension(values.size(), this->n_components);
  //     for (unsigned int c = 0; c < this->n_components; ++c)
  //       values[c] = value(p, c);
  //   }

  // private:
  //   const double mu;
  // };


  // class ExactNeumann_hRight : public Function<dim>
  // {
  // public:
  //   ExactNeumann_hRight(const double viscosity_)
  //     : Function<dim>(dim + 1 + dim) // 5 components
  //     , nu(viscosity_)
  //   {}

  //   virtual double
  //   value(const Point<dim> &p, const unsigned int component = 0) const override
  //   {
  //     const double x  = p[0];
  //     const double y  = p[1];
  //     const double pi = 3.14159265358979323846;

  //     if (component == 0)
  //       {
  //         double term1 = -((-0.25 + x) * (0.25 + x) * (-0.5 + y) * (0.5 + y) *
  //                          std::sin(pi * x) * std::sin(pi * y));
  //         double term2 =
  //           ((-1.0 + 16.0 * x * x) * (-1.0 + 2.0 * y) * (1.0 + 2.0 * y) * nu *
  //            (pi * (-1.0 + 16.0 * x * x) * std::cos(pi * x) +
  //             64.0 * x * std::sin(pi * x)) *
  //            (-16.0 * y * std::cos(pi * y) +
  //             pi * (-1.0 + 4.0 * y * y) * std::sin(pi * y))) /
  //           2048.0;
  //         return term1 - term2;
  //       }

  //     if (component == 1)
  //       {
  //         double term1 =
  //           std::pow(1.0 - 4.0 * y * y, 2) * std::cos(pi * y) *
  //           (-128.0 * pi * x * (-1.0 + 16.0 * x * x) * std::cos(pi * x) +
  //            (64.0 - 3072.0 * x * x +
  //             std::pow(pi, 2) * std::pow(1.0 - 16.0 * x * x, 2)) *
  //              std::sin(pi * x));

  //         double term2 =
  //           std::pow(1.0 - 16.0 * x * x, 2) * std::sin(pi * x) *
  //           ((16.0 - 192.0 * y * y +
  //             std::pow(pi, 2) * std::pow(1.0 - 4.0 * y * y, 2)) *
  //              std::cos(pi * y) +
  //            32.0 * pi * y * (-1.0 + 4.0 * y * y) * std::sin(pi * y));

  //         return (nu * (term1 - term2)) / 4096.0;
  //       }

  //     return 0.0;
  //   }

  //   virtual void
  //   vector_value(const Point<dim> &p, Vector<double> &values) const override
  //   {
  //     AssertDimension(values.size(), this->n_components);
  //     for (unsigned int c = 0; c < this->n_components; ++c)
  //       values[c] = value(p, c);
  //   }

  // private:
  //   const double nu;
  // };

  // class ExactNeumann_hLeft : public Function<dim>
  // {
  // public:
  //   ExactNeumann_hLeft(const double viscosity_)
  //     : Function<dim>(dim + 1 + dim) // 5 components
  //     , nu(viscosity_)
  //   {}

  //   virtual double
  //   value(const Point<dim> &p, const unsigned int component = 0) const override
  //   {
  //     const double x  = p[0];
  //     const double y  = p[1];
  //     const double pi = 3.14159265358979323846;

  //     if (component == 0)
  //       {
  //         double term1 = (-0.25 + x) * (0.25 + x) * (-0.5 + y) * (0.5 + y) *
  //                        std::sin(pi * x) * std::sin(pi * y);
  //         double term2 =
  //           ((-1.0 + 16.0 * x * x) * (-1.0 + 2.0 * y) * (1.0 + 2.0 * y) * nu *
  //            (pi * (-1.0 + 16.0 * x * x) * std::cos(pi * x) +
  //             64.0 * x * std::sin(pi * x)) *
  //            (-16.0 * y * std::cos(pi * y) +
  //             pi * (-1.0 + 4.0 * y * y) * std::sin(pi * y))) /
  //           2048.0;
  //         return term1 + term2;
  //       }

  //     if (component == 1)
  //       {
  //         double term1 =
  //           std::pow(1.0 - 4.0 * y * y, 2) * std::cos(pi * y) *
  //           (-128.0 * pi * x * (-1.0 + 16.0 * x * x) * std::cos(pi * x) +
  //            (64.0 - 3072.0 * x * x +
  //             std::pow(pi, 2) * std::pow(1.0 - 16.0 * x * x, 2)) *
  //              std::sin(pi * x));

  //         double term2 =
  //           std::pow(1.0 - 16.0 * x * x, 2) * std::sin(pi * x) *
  //           ((16.0 - 192.0 * y * y +
  //             std::pow(pi, 2) * std::pow(1.0 - 4.0 * y * y, 2)) *
  //              std::cos(pi * y) +
  //            32.0 * pi * y * (-1.0 + 4.0 * y * y) * std::sin(pi * y));

  //         return -((nu * (term1 - term2)) / 4096.0);
  //       }

  //     return 0.0;
  //   }

  //   virtual void
  //   vector_value(const Point<dim> &p, Vector<double> &values) const override
  //   {
  //     AssertDimension(values.size(), this->n_components);
  //     for (unsigned int c = 0; c < this->n_components; ++c)
  //       values[c] = value(p, c);
  //   }

  // private:
  //   const double nu;
  // };

  // constructor
  FluidStructureProblem(ParameterHandler &param)
    : prm(param)
    , problemsize(param.get_integer(std::vector<std::string>{"Geometry"},
                                    "Number of cells per edge"))
    , stokes_degree(param.get_integer(std::vector<std::string>{"Geometry"},
                                      "Stokes degree"))
    , elasticity_degree(param.get_integer(std::vector<std::string>{"Geometry"},
                                          "Elasticity degree"))
    , triangulation(MPI_COMM_WORLD,
                    typename Triangulation<dim>::MeshSmoothing(
                      Triangulation<dim>::smoothing_on_refinement |
                      Triangulation<dim>::smoothing_on_coarsening)
                    // , parallel::distributed::Triangulation<
                    //   dim>::Settings::no_automatic_repartitioning
                    ) // to use when we want to avoid automatic repartitioning
    , stokes_fe(FE_Q<dim>(stokes_degree + 1),
                dim,
                FE_Q<dim>(stokes_degree),
                1,
                FE_Nothing<dim>(),
                dim)
    , mpi_size(Utilities::MPI::n_mpi_processes(MPI_COMM_WORLD))
    , mpi_rank(Utilities::MPI::this_mpi_process(MPI_COMM_WORLD))
    , elasticity_fe(FE_Nothing<dim>(),
                    dim,
                    FE_Nothing<dim>(),
                    1,
                    FE_Q<dim>(elasticity_degree),
                    dim)
    , dof_handler(triangulation)
    , viscosity(
        param.get_double(std::vector<std::string>{"Physics"}, "Viscosity"))
    , lambda(param.get_double(std::vector<std::string>{"Physics"}, "lambda"))
    , mu(param.get_double(std::vector<std::string>{"Physics"}, "mu"))
    , pcout(std::cout, mpi_rank == 0)
    , timer(MPI_COMM_WORLD, pcout, TimerOutput::never, TimerOutput::wall_times)

  {
    fe_collection.push_back(stokes_fe);
    fe_collection.push_back(elasticity_fe);
  }
  void
  set_boundary_ids(
    parallel::distributed::Triangulation<dim> &triangulation) const;
  void
  create_coarse_mesh(
    parallel::distributed::Triangulation<dim> &coarse_grid) const;
  void
  make_grid();
  void
  setup_dofs();
  void
  assemble_system();
  void
  assemble_preconditioners();
#ifdef DIRECT_SOLVER
  void
  solve(); // not anymore implemented
#endif
#ifdef ITERATIVE_SOLVER
  void
  solve_iterative();
#endif
  void
  output_results(const unsigned int refinement_cycle) const;
#ifdef DEBUG
  void
  output_matrix() const; // not anymore implemented
#endif
  void
  compute_error(const unsigned int refinement_cycle);
  double
  compute_velocity_error(const VectorTools::NormType &norm_type) const;
  double
  compute_pressure_error(const VectorTools::NormType &norm_type) const;
  double
  compute_displacement_error(const VectorTools::NormType &norm_type) const;
  void
  refine_mesh(const unsigned int n_cycle);
  void
  output_table();


  // Boundary values for the Stokes equations
  class StokesBoundaryValues : public Function<dim>
  {
  public:
    StokesBoundaryValues()
      : Function<dim>(dim + 1 + dim)
    {}

    virtual double
    value(const Point<dim> &p, const unsigned int component = 0) const override
    {
      Assert(component < this->n_components,
             ExcIndexRange(component, 0, this->n_components));

      if (component == dim - 1)
        switch (dim)
          {
            case 2:
              return std::sin(numbers::PI * p[0]);
            case 3:
              return std::sin(numbers::PI * p[0]) *
                     std::sin(numbers::PI * p[1]);
            default:
              Assert(false, ExcNotImplemented());
          }

      return 0;
    }

    virtual void
    vector_value(const Point<dim> &p, Vector<double> &values) const override
    {
      for (unsigned int c = 0; c < this->n_components; ++c)
        values(c) = StokesBoundaryValues::value(p, c);
    }
  };


//   // Older preconditioner class for the block triangular preconditioner
//   // It can be used if DEBUG is defined, because of a problem happening
//   // inside deal.ii extract_constant_modes function when used with
//   // the fe_collection made by two different finite elements, one of which
//   // is FE_Nothing.
//   // It is less efficient than the new one implemented below,
//   // due to a problem happening with the AMG preconditioner when not correctly
//   // set up.
//   class PreconditionBlockTriangular
//   {
//   public:
//     // Initialize the preconditioner, given the velocity stiffness matrix, the
//     // pressure mass matrix.
//     void
//     initialize(
//       const LA::MPI::SparseMatrix &velocity_stiffness_, // A(0,0)
//       const LA::MPI::SparseMatrix &pressure_mass_,      // pressurematrix(1,1)
//       const LA::MPI::SparseMatrix &B_,                  // A(1,0)
//       const LA::MPI::SparseMatrix &D1_,                 // A(2,0)
//       const LA::MPI::SparseMatrix &D2_,                 // A(2,1)
//       const LA::MPI::SparseMatrix &solid_matrix_        // A(2,2)
//     )
//     {
//       velocity_stiffness = &velocity_stiffness_;
//       pressure_mass      = &pressure_mass_;
//       B                  = &B_;
//       D1                 = &D1_;
//       D2                 = &D2_;
//       solid_matrix       = &solid_matrix_;

//       preconditioner_velocity.initialize(velocity_stiffness_);
//       preconditioner_pressure.initialize(pressure_mass_);
//       preconditioner_solid.initialize(solid_matrix_);
//     }

//     // Application of the preconditioner.
//     void
//     vmult(TrilinosWrappers::MPI::BlockVector       &dst,
//           const TrilinosWrappers::MPI::BlockVector &src) const
//     {
//       SolverControl                           solver_control_velocity(1000,
//                                             1e-2 * src.block(0).l2_norm());
//       SolverCG<TrilinosWrappers::MPI::Vector> solver_cg_velocity(
//         solver_control_velocity);
//       solver_cg_velocity.solve(*velocity_stiffness,
//                                dst.block(0),
//                                src.block(0),
//                                preconditioner_velocity);
// // #ifdef DEBUG
// //       std::cout << "  " << solver_control_velocity.last_step()
// //                 << " CG1 iterations" << std::endl;
// // #endif
//       tmpStokes.reinit(src.block(1));
//       B->vmult(tmpStokes, dst.block(0));
//       tmpStokes.sadd(-1.0, src.block(1));

//       SolverControl solver_control_pressure(1000, 1e-2 * tmpStokes.l2_norm());
//       SolverCG<TrilinosWrappers::MPI::Vector> solver_cg_pressure(
//         solver_control_pressure);
//       solver_cg_pressure.solve(*pressure_mass,
//                                dst.block(1),
//                                tmpStokes,
//                                preconditioner_pressure);
// // #ifdef DEBUG
// //       std::cout << "  " << solver_control_pressure.last_step()
// //                 << " CG2 iterations" << std::endl;
// // #endif

//       tmpStokes.reinit(src.block(2));
//       D1->vmult(tmpStokes, dst.block(0));
//       D2->vmult_add(tmpStokes, dst.block(1));
//       tmpStokes.sadd(-1.0, src.block(2));


//       // other way to do it, after testing we found it to be extremely slow.
//       // preconditioner_solid.vmult(dst.block(2), tmpStokes);

//       SolverControl solver_control_solid(1000, 1e-2 * tmpStokes.l2_norm());
//       SolverCG<TrilinosWrappers::MPI::Vector> solver_cg_solid(
//         solver_control_solid);

//       solver_cg_solid.solve(*solid_matrix,
//                             dst.block(2),
//                             tmpStokes,
//                             preconditioner_solid);
// // #ifdef DEBUG
// //       std::cout << "  " << solver_control_solid.last_step() << " CG3 iterations"
// //                 << std::endl;
// // #endif
//     }

//   protected:
//     // Velocity stiffness matrix.
//     const LA::MPI::SparseMatrix *velocity_stiffness;

//     // Preconditioner used for the velocity block.
//     TrilinosWrappers::PreconditionAMG preconditioner_velocity;

//     // Pressure mass matrix.
//     const LA::MPI::SparseMatrix *pressure_mass;

//     // Preconditioner used for the pressure block.
//     TrilinosWrappers::PreconditionAMG preconditioner_pressure;

//     // B matrix.
//     const LA::MPI::SparseMatrix *B;

//     // D1 matrix.
//     const LA::MPI::SparseMatrix *D1;

//     // D2 matrix.
//     const LA::MPI::SparseMatrix *D2;

//     // Preconditioner used for the pressure block.
//     TrilinosWrappers::PreconditionAMG preconditioner_solid;

//     // Solid matrix.
//     const LA::MPI::SparseMatrix *solid_matrix;

//     // Temporary vector stokes
//     mutable LA::MPI::Vector tmpStokes;
//   };

  // New preconditioner class for the block triangular preconditioner.
  // This one uses shared pointers for the AMG preconditioners, and is
  // initialized correctly given the necessity for the Trilinos ML AMG
  // preconditioner to be set up with the constant modes when dealing with a
  // vector-valued problem.

  class PreconditionBlockTriangularAMG
  {
  public:
    // Initialize the preconditioner, given the velocity stiffness matrix, the
    // pressure mass matrix.
    void
    initialize(
      const LA::MPI::SparseMatrix &velocity_stiffness_, // A(0,0)
      const LA::MPI::SparseMatrix &pressure_mass_,      // pressurematrix(1,1)
      const LA::MPI::SparseMatrix &B_,                  // A(1,0)
      const LA::MPI::SparseMatrix &D1_,                 // A(2,0)
      const LA::MPI::SparseMatrix &D2_,                 // A(2,1)
      const LA::MPI::SparseMatrix &solid_matrix_,       // A(2,2)
      std::shared_ptr<TrilinosWrappers::PreconditionAMG> precond_velocity_,
      std::shared_ptr<TrilinosWrappers::PreconditionAMG> precond_pressure_,
      std::shared_ptr<TrilinosWrappers::PreconditionAMG> precond_solid_

    )
    {
      velocity_stiffness      = &velocity_stiffness_;
      pressure_mass           = &pressure_mass_;
      B                       = &B_;
      D1                      = &D1_;
      D2                      = &D2_;
      solid_matrix            = &solid_matrix_;
      preconditioner_velocity = precond_velocity_;
      preconditioner_pressure = precond_pressure_;
      preconditioner_solid    = precond_solid_;
    }

    // Application of the preconditioner.
    void
    vmult(TrilinosWrappers::MPI::BlockVector       &dst,
          const TrilinosWrappers::MPI::BlockVector &src) const
    {
      SolverControl                           solver_control_velocity(1000,
                                            1e-2 * src.block(0).l2_norm());
      SolverCG<TrilinosWrappers::MPI::Vector> solver_cg_velocity(
        solver_control_velocity);
      solver_cg_velocity.solve(*velocity_stiffness,
                               dst.block(0),
                               src.block(0),
                               *preconditioner_velocity);
#ifdef VERBOSE
      std::cout << "  " << solver_control_velocity.last_step()
                << " CG1 iterations" << std::endl;
#endif
      tmpStokes.reinit(src.block(1));
      B->vmult(tmpStokes, dst.block(0));
      tmpStokes.sadd(-1.0, src.block(1));

      SolverControl solver_control_pressure(1000, 1e-2 * tmpStokes.l2_norm());
      SolverCG<TrilinosWrappers::MPI::Vector> solver_cg_pressure(
        solver_control_pressure);
      solver_cg_pressure.solve(*pressure_mass,
                               dst.block(1),
                               tmpStokes,
                               *preconditioner_pressure);
#ifdef VERBOSE
      std::cout << "  " << solver_control_pressure.last_step()
                << " CG2 iterations" << std::endl;
#endif
      tmpStokes.reinit(src.block(2));
      D1->vmult(tmpStokes, dst.block(0));
      D2->vmult_add(tmpStokes, dst.block(1));
      tmpStokes.sadd(-1.0, src.block(2));


      // other way to do it, after testing we found it to be extremely slow.
      // preconditioner_solid.vmult(dst.block(2), tmpStokes);

      SolverControl solver_control_solid(1000, 1e-2 * tmpStokes.l2_norm());
      SolverCG<TrilinosWrappers::MPI::Vector> solver_cg_solid(
        solver_control_solid);

      solver_cg_solid.solve(*solid_matrix,
                            dst.block(2),
                            tmpStokes,
                            *preconditioner_solid);
#ifdef VERBOSE
      std::cout << "  " << solver_control_solid.last_step() << " CG3 iterations"
                << std::endl;
#endif
    }

  protected:
    // Velocity stiffness matrix.
    const LA::MPI::SparseMatrix *velocity_stiffness;

    // Preconditioner used for the velocity block.
    std::shared_ptr<TrilinosWrappers::PreconditionAMG> preconditioner_velocity;

    // Pressure mass matrix.
    const LA::MPI::SparseMatrix *pressure_mass;

    // Preconditioner used for the pressure block.
    std::shared_ptr<TrilinosWrappers::PreconditionAMG> preconditioner_pressure;

    // B matrix.
    const LA::MPI::SparseMatrix *B;

    // D1 matrix.
    const LA::MPI::SparseMatrix *D1;

    // D2 matrix.
    const LA::MPI::SparseMatrix *D2;

    // Preconditioner used for the pressure block.
    std::shared_ptr<TrilinosWrappers::PreconditionAMG> preconditioner_solid;

    // Solid matrix.
    const LA::MPI::SparseMatrix *solid_matrix;

    // Temporary vector stokes
    mutable LA::MPI::Vector tmpStokes;
  };

//   // Searching for other preconditioner to implement, we also tried a sort of
//   // SIMPLE preconditioner used for Navier-Stokes equations. However, the
//   // performance was not satisfactory, so we decided not to use it. The code is
//   // left here commented for possible future reference. 
//   class PreconditionBlockTriangularSimple
//   {
//   public:
//     // Initialize the preconditioner, given the velocity stiffness matrix,
//     // the
//     // pressure mass matrix.
//     void
//     initialize(
//       const LA::MPI::SparseMatrix &velocity_stiffness_, // A(0,0)
//       const LA::MPI::SparseMatrix &pressure_mass_,      // pressurematrix(1,1)
//       const LA::MPI::SparseMatrix &B_,                  // A(1,0) const
//       LA::MPI::SparseMatrix       &D1_,                 // A(2,0) const
//       LA::MPI::SparseMatrix       &D2_,                 // A(2,1) const
//       LA::MPI::SparseMatrix       &solid_matrix_,       // A(2,2)
//       std::shared_ptr<TrilinosWrappers::PreconditionAMG> precond_velocity_,
//       std::shared_ptr<TrilinosWrappers::PreconditionAMG> precond_pressure_,
//       std::shared_ptr<TrilinosWrappers::PreconditionAMG> precond_solid_
//     )
//     {
//       velocity_stiffness = &velocity_stiffness_;
//       pressure_mass      = &pressure_mass_;
//       B                  = &B_;
//       D1                 = &D1_;
//       D2                 = &D2_;
//       solid_matrix       = &solid_matrix_;

//       preconditioner_velocity = precond_velocity_;
//       preconditioner_pressure = precond_pressure_;
//       preconditioner_solid    = precond_solid_;
//     }

//     // Application of the preconditioner.
//     void
//     vmult(TrilinosWrappers::MPI::BlockVector       &dst,
//           const TrilinosWrappers::MPI::BlockVector &src) const
//     {
//       SolverControl solver_control_u(1000, 1e-2 * src.block(0).l2_norm());
//       SolverCG<TrilinosWrappers::MPI::Vector> solver_u(solver_control_u);

//       solver_u.solve(*velocity_stiffness,
//                      dst.block(0),
//                      src.block(0),
//                      *preconditioner_velocity);
// #ifdef VERBOSE
//       std::cout << "  " << solver_control_u.last_step()
//                 << " CG1 iterations" << std::endl;
// #endif

//       // u* is now stored in dst.block(0)

//       tmpStokes.reinit(src.block(1));
//       B->vmult(tmpStokes, dst.block(0));  // tmpStokes = B*u*
//       tmpStokes.sadd(-1.0, src.block(1)); // tmpStokes = B*u* - rhs_p

//       // We approximate S^{-1} ≈ (pressure_mass)^{-1}
//       SolverControl solver_control_S(1000, 1e-2 * tmpStokes.l2_norm());
//       SolverCG<TrilinosWrappers::MPI::Vector> solver_S(solver_control_S);

//       solver_S.solve(*pressure_mass,
//                      dst.block(1),
//                      tmpStokes,
//                      *preconditioner_pressure);
// #ifdef VERBOSE
//       std::cout << "  " << solver_control_S.last_step() << " CG2 iterations"
//                 << std::endl;
// #endif
//       // dst.block(1) = p

//       tmpStokes.reinit(src.block(0));
//       B->Tvmult(tmpStokes, dst.block(1)); // tmpStokes = B^T p

//       TrilinosWrappers::MPI::Vector AuInv(tmpStokes);
//       SolverControl solver_control_Acorr(1000, 1e-2 * tmpStokes.l2_norm());
//       SolverCG<TrilinosWrappers::MPI::Vector> solver_Acorr(
//         solver_control_Acorr);

//       solver_Acorr.solve(*velocity_stiffness,
//                          AuInv,
//                          tmpStokes,
//                          *preconditioner_velocity);
// #ifdef VERBOSE
//       std::cout << "  " << solver_control_Acorr.last_step() << " CG3 iterations"
//                 << std::endl;
// #endif

//       dst.block(0).sadd(1.0, -1.0, AuInv); // u = u* − A^{-1} B^T p

//       tmpStokes.reinit(src.block(2));
//       D1->vmult(tmpStokes, dst.block(0));
//       D2->vmult_add(tmpStokes, dst.block(1));
//       tmpStokes.sadd(-1.0, src.block(2));
//       SolverControl solver_control_solid(1000, 1e-2 * tmpStokes.l2_norm());
//       SolverCG<TrilinosWrappers::MPI::Vector> solver_solid(
//         solver_control_solid);

//       solver_solid.solve(*solid_matrix,
//                          dst.block(2),
//                          tmpStokes,
//                          *preconditioner_solid);
// #ifdef VERBOSE
//       std::cout << "  " << solver_control_solid.last_step() << " CG4 iterations"
//                 << std::endl;
// #endif
//     }


//   protected:
//     // Velocity stiffness matrix.
//     const LA::MPI::SparseMatrix *velocity_stiffness;

//     // Preconditioner used for the velocity block.
//     std::shared_ptr<TrilinosWrappers::PreconditionAMG> preconditioner_velocity;

//     // Pressure mass matrix.
//     const LA::MPI::SparseMatrix *pressure_mass;

//     // Preconditioner used for the pressure block.
//     std::shared_ptr<TrilinosWrappers::PreconditionAMG> preconditioner_pressure;

//     // B matrix.
//     const LA::MPI::SparseMatrix *B;

//     // D1 matrix.
//     const LA::MPI::SparseMatrix *D1;

//     // D2 matrix.
//     const LA::MPI::SparseMatrix *D2;

//     // Preconditioner used for the pressure block.
//     std::shared_ptr<TrilinosWrappers::PreconditionAMG> preconditioner_solid;

//     // Solid matrix.
//     const LA::MPI::SparseMatrix *solid_matrix;

//     // Temporary vector stokes
//     mutable LA::MPI::Vector tmpStokes;
//   };

  // Searching for other preconditioner to implement, we also tried a sort of
  // SIMPLE preconditioner used for Navier-Stokes equations. However, the
  // performance was not satisfactory, so we decided not to use it. The code is
  // left here commented for possible future reference.
  class PreconditionBlockTriangularSimple
  {
  public:
    // Initialize the preconditioner, given the velocity stiffness matrix,
    // the
    // pressure mass matrix.
    void
    initialize(
      const LA::MPI::SparseMatrix &velocity_stiffness_, // A(0,0)
      const LA::MPI::SparseMatrix &pressure_mass_,      // pressurematrix(1,1)
      const LA::MPI::SparseMatrix &B_,                  // A(1,0) const
      LA::MPI::SparseMatrix       &D1_,                 // A(2,0) const
      LA::MPI::SparseMatrix       &D2_,                 // A(2,1) const
      LA::MPI::SparseMatrix       &solid_matrix_,       // A(2,2)
      std::shared_ptr<TrilinosWrappers::PreconditionAMG> precond_velocity_,
      std::shared_ptr<TrilinosWrappers::PreconditionAMG> precond_pressure_,
      std::shared_ptr<TrilinosWrappers::PreconditionAMG> precond_solid_)
    {
      velocity_stiffness = &velocity_stiffness_;
      pressure_mass      = &pressure_mass_;
      B                  = &B_;
      D1                 = &D1_;
      D2                 = &D2_;
      solid_matrix       = &solid_matrix_;

      preconditioner_velocity = precond_velocity_;
      preconditioner_pressure = precond_pressure_;
      preconditioner_solid    = precond_solid_;
    }

    // Application of the preconditioner.
    void
    vmult(TrilinosWrappers::MPI::BlockVector       &dst,
          const TrilinosWrappers::MPI::BlockVector &src) const
    {
      SolverControl solver_control_u(1000, 1e-2 * src.block(0).l2_norm());
      SolverCG<TrilinosWrappers::MPI::Vector> solver_u(solver_control_u);
      solver_u.solve(*velocity_stiffness, dst.block(0), src.block(0), *preconditioner_velocity);

      // 2. Form RHS for Pressure: B u* - r_p
      tmpStokes.reinit(src.block(1));
      B->vmult(tmpStokes, dst.block(0));
      tmpStokes.sadd(-1.0, src.block(1));

      // --- FIX 1: Project out the Nullspace from RHS ---
      // The pressure Laplacian is singular. We must ensure the RHS sums to
      // zero.
      const double rhs_mean = tmpStokes.mean_value();
      tmpStokes.add(-rhs_mean);
      // -------------------------------------------------

      // 3. Solve Pressure: S p = RHS
      SolverControl solver_control_S(
        1000, 1e-4 * tmpStokes.l2_norm()); // Tighter tol might help
      SolverCG<TrilinosWrappers::MPI::Vector> solver_S(solver_control_S);

      // Note: The variable is named 'pressure_mass' in your class,
      // but you initialized it with 'pressure_laplacian'. That is correct
      // logic.
      solver_S.solve(*pressure_mass,
                     dst.block(1),
                     tmpStokes,
                     *preconditioner_pressure);

      // --- FIX 2: Project out the Nullspace from Solution ---
      // Prevent pressure drift by removing the mean value
      const double p_mean = dst.block(1).mean_value();
      dst.block(1).add(-p_mean);
      // -----------------------------------------------------
      // 4. Correct Velocity: u = u* - diag(A)^-1 * B^T * p
      // CRITICAL CHANGE: Do NOT do a full solve here.
      // Use a diagonal approximation or just apply the inverse diagonal.
      // If you don't have the inverse diagonal stored, you might have to assume
      // A is roughly Identity/Mass scaled (approximate).

      tmpStokes.reinit(src.block(0));
      B->Tvmult(tmpStokes, dst.block(1)); // B^T * p
      
      // For standard SIMPLE, we assume A^-1 is effectively a scalar scaling factor
      // or a diagonal inverse. Doing a full solve here is "Block Triangular", not "SIMPLE".
      // If you keep the full solve, it is accurate but slow.
      
      TrilinosWrappers::MPI::Vector AuInv(tmpStokes);
      
      // OPTION A (True SIMPLE - Fast):
      // AuInv.scale( 1.0 / approximate_diagonal_value ); 
      
      // OPTION B (Your current "Exact" approach - Slow but robust):
      SolverControl solver_control_Acorr(1000, 1e-2 * tmpStokes.l2_norm());
      SolverCG<TrilinosWrappers::MPI::Vector> solver_Acorr(solver_control_Acorr);
      solver_Acorr.solve(*velocity_stiffness, AuInv, tmpStokes, *preconditioner_velocity);

      dst.block(0).sadd(1.0, -1.0, AuInv);

      tmpStokes.reinit(src.block(2));
      D1->vmult(tmpStokes, dst.block(0));
      D2->vmult_add(tmpStokes, dst.block(1));
      tmpStokes.sadd(-1.0, src.block(2));
      SolverControl solver_control_solid(1000, 1e-2 * tmpStokes.l2_norm());
      SolverCG<TrilinosWrappers::MPI::Vector> solver_solid(
        solver_control_solid);

      solver_solid.solve(*solid_matrix,
                         dst.block(2),
                         tmpStokes,
                         *preconditioner_solid);
#ifdef VERBOSE
      std::cout << "  " << solver_control_solid.last_step() << " CG4 iterations"
                << std::endl;
#endif
    }


  protected:
    // Velocity stiffness matrix.
    const LA::MPI::SparseMatrix *velocity_stiffness;

    // Preconditioner used for the velocity block.
    std::shared_ptr<TrilinosWrappers::PreconditionAMG> preconditioner_velocity;

    // Pressure mass matrix.
    const LA::MPI::SparseMatrix *pressure_mass;

    // Preconditioner used for the pressure block.
    std::shared_ptr<TrilinosWrappers::PreconditionAMG> preconditioner_pressure;

    // B matrix.
    const LA::MPI::SparseMatrix *B;

    // D1 matrix.
    const LA::MPI::SparseMatrix *D1;

    // D2 matrix.
    const LA::MPI::SparseMatrix *D2;

    // Preconditioner used for the pressure block.
    std::shared_ptr<TrilinosWrappers::PreconditionAMG> preconditioner_solid;

    // Solid matrix.
    const LA::MPI::SparseMatrix *solid_matrix;

    // Temporary vector stokes
    mutable LA::MPI::Vector tmpStokes;
  };


private:
  enum
  {
    fluid_domain_id,
    solid_domain_id
  };

  static bool
  cell_is_in_fluid_domain(const typename DoFHandler<dim>::cell_iterator &cell)
  {
    return (cell->material_id() == fluid_domain_id);
  }

  static bool
  cell_is_in_solid_domain(const typename DoFHandler<dim>::cell_iterator &cell)
  {
    return (cell->material_id() == solid_domain_id);
  }

  void
  set_active_fe_indices();
  void
  assemble_interface_term(
    const FEFaceValuesBase<dim>          &elasticity_fe_face_values,
    const FEFaceValuesBase<dim>          &stokes_fe_face_values,
    std::vector<Tensor<1, dim>>          &elasticity_phi,
    std::vector<SymmetricTensor<2, dim>> &stokes_symgrad_phi_u,
    std::vector<double>                  &stokes_phi_p,
    FullMatrix<double>                   &local_interface_matrix) const;

  ParameterHandler  &prm;
  const int          problemsize;
  const unsigned int stokes_degree;
  const unsigned int elasticity_degree;

  // Number of MPI processes.
  // parallel::fullydistributed::Triangulation<dim> triangulation; doesn't
  // work with GridGenerator directly
  parallel::distributed::Triangulation<dim> triangulation;

  FESystem<dim> stokes_fe;

  const unsigned int mpi_size;
  // This MPI process.
  const unsigned int    mpi_rank;
  FESystem<dim>         elasticity_fe;
  hp::FECollection<dim> fe_collection;
  DoFHandler<dim>       dof_handler;
  const double          viscosity;
  const double          lambda;
  const double          mu;

public:
  ConditionalOStream pcout;
  TimerOutput        timer;

private:
  // class which handles constraints, hanging nodes, boundary conditions and
  // interface u=0 condition
  AffineConstraints<double>  constraints;
  LA::MPI::BlockSparseMatrix system_matrix;
  LA::MPI::BlockSparseMatrix pressure_mass;
    LA::MPI::BlockSparseMatrix pressure_laplacian;

  LA::MPI::BlockVector       solution;
  LA::MPI::BlockVector       locally_relevant_solution;
  LA::MPI::BlockVector       system_rhs;

  // shared pointers to the preconditioners
  std::shared_ptr<TrilinosWrappers::PreconditionAMG> stokes_preconditioner;
  std::shared_ptr<TrilinosWrappers::PreconditionAMG> mp_preconditioner;
  std::shared_ptr<TrilinosWrappers::PreconditionAMG> elasticity_preconditioner;


  IndexSet              locally_owned_dofs;
  IndexSet              locally_relevant_dofs;
  std::vector<IndexSet> block_owned_dofs;
  std::vector<IndexSet> block_relevant_dofs;
#ifdef EXACT_TABLE
  ConvergenceTable velocity_convergence_table;
  ConvergenceTable pressure_convergence_table;
  ConvergenceTable displacement_convergence_table;
#endif
};

#endif
