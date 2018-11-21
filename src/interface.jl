"""
    AdaFVF(params, η = 0.001; 
    	κ::Real=1e-10 , α::Real=2.0 , β::Real=1e-10 , ϕᵅ₁::Real=9.5 , ϕᵝ₁::Real=.5 , ϕᵅ₂::Real=99.0 , ϕᵝ₂::Real=1., βᵅ::Real = 20.0, βᵝ::Real = 20.0,
    	ϵ = 1e-08, decay = 0)
[AdaFVF](https://arxiv.org/pdf/1805.05703) optimiser.
"""
AdaFVF(ps, η = 0.001; 
	κ::Real=1e-10 , α::Real=2.0 , β::Real=1e-10 , ϕᵅ₁::Real=9.5 , ϕᵝ₁::Real=.5 , ϕᵅ₂::Real=99.0 , ϕᵝ₂::Real=1., βᵅ::Real = 20.0, βᵝ::Real = 20.0,
       ϵ = 1e-08, decay = 0) =
  optimiser(ps, p->adafvf(p; η=η, 
			  κ = κ, α = α, β = β, ϕᵅ₁ = ϕᵅ₁, ϕᵝ₁ = ϕᵝ₁, ϕᵅ₂ = ϕᵅ₂, ϕᵝ₂ = ϕᵝ₂, βᵅ = βᵅ, βᵝ = βᵝ,
			  ϵ=ϵ), p->invdecay(p,decay), p->descent(p,1))


