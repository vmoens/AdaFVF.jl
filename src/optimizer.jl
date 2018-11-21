function adafvf(p::Param; η::Real = 0.001, 
		κ::Real=1e-10 , α::Real=2.0 , β::Real=1e-10 , ϕᵅ₁::Real=9.5 , ϕᵝ₁::Real=.5 , ϕᵅ₂::Real=99.0 , ϕᵝ₂::Real=1., βᵅ::Real = 20.0, βᵝ::Real = 20.0,
		ϵ::Real = 1e-8)
  
  
  μt = zero(p.x)
  βt = ones(p.x) * β
  αt = α
  κt = κ

  ϕᵅ₁t,ϕᵝ₁t,ϕᵅ₂t,ϕᵝ₂t,βᵅt,βᵝt = ϕᵅ₁,ϕᵝ₁,ϕᵅ₂,ϕᵝ₂,βᵅ,βᵝ
    
  lim = 200
  curr_elbo = Inf
  n = length(μt)

  function ()
    
    μtm1,βtm1 = copy.((μt,βt))
    κtm1,αtm1 = αt,κt
    ϕᵅ₁tm1,ϕᵝ₁tm1,ϕᵅ₂tm1,ϕᵝ₂tm1,βᵅtm1,βᵝtm1 = ϕᵅ₁t,ϕᵝ₁t,ϕᵅ₂t,ϕᵝ₂t,βᵅt,βᵝt
    
    βᵅt = 0.999 * (βᵅt - 1.0) + 1.0
    βᵝt = 0.999 * (βᵝt - 1.0) + 1.0


    d = Inf
    i = 0
    c = .3
    old_elbo = curr_elbo
    
    v = [ϕᵅ₁t,ϕᵝ₁t,ϕᵅ₂t,ϕᵝ₂t;βᵅt,βᵝt] # beta parameters

    grad = p.Δ
    while true
        αt, κt = updNIGloop!(grad,μt,βt,
		    μtm1,κtm1,αtm1,βtm1,
		    κ,α,β,
		    [ϕᵅ₁t,ϕᵝ₁t,ϕᵅ₂t,ϕᵝ₂t])
	elbo = ncvmp_α!(v,
			μtm1,κtm1,αtm1,βtm1,
			μtm1,κtm1,αtm1,βtm1,
			μtm1,κtm1,αtm1,βtm1,

    end






    curr_elbo = elbo
    @. mt = β1 * mt + (1 - β1) * p.Δ
    @. vt = β2 * vt + (1 - β2) * p.Δ^2
    @. p.Δ =  mt / (1 - β1p) / √(vt / (1 - β2p) + ϵ) * η
    β1p *= β1
    β2p *= β2
  end
end

