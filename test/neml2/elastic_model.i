[Models]
  [Erate]
    type = SR2ForceRate
    force = 'E'
  []
  [elasticity]
    type = LinearIsotropicElasticity
    youngs_modulus = 1e5
    poisson_ratio = 0.3
    rate_form = true
    strain = 'forces/E'
  []
  [integrate_stress]
    type = SR2BackwardEulerTimeIntegration
    variable = 'S'
  []
  [implicit_rate]
    type = ComposedModel
    models = 'Erate elasticity integrate_stress'
  []
[]
