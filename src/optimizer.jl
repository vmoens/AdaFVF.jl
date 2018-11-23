import Flux.Optimise: Param
function adafvf(p::Param; η::Real = 0.001, 
		κ::Real=1e-10 , α::Real=2.0 , β::Real=1e-10 , ϕᵅ₁::Real=9.5 , ϕᵝ₁::Real=.5 , ϕᵅ₂::Real=99.0 , ϕᵝ₂::Real=1., βᵅ::Real = 20.0, βᵝ::Real = 20.0,
		ϵ::Real = 1e-8)
   
  μt = zeros(size(p.x))
  βt = ones(size(p.x)) * β
  αt = α
  κt = κ

  ϕᵅ₁t,ϕᵝ₁t,ϕᵅ₂t,ϕᵝ₂t,βᵅt,βᵝt = ϕᵅ₁,ϕᵝ₁,ϕᵅ₂,ϕᵝ₂,βᵅ,βᵝ
    
  lim = 200
  curr_elbo = Inf
  n = length(μt)
  cc = 0
  Count = 1
  function ()
    grad = p.Δ
    if NAN_CHECK
        if checkNaN(grad)
		grad .= 0.0
		return grad
	end
    end
    μtm1,βtm1 = copy.((μt,βt))
    κtm1,αtm1 = αt,κt
    ϕᵅ₁tm1,ϕᵝ₁tm1,ϕᵅ₂tm1,ϕᵝ₂tm1,βᵅtm1,βᵝtm1 = ϕᵅ₁t,ϕᵝ₁t,ϕᵅ₂t,ϕᵝ₂t,βᵅt,βᵝt
    
    βᵅt = 0.999 * (βᵅt - 1.0) + 1.0
    βᵝt = 0.999 * (βᵝt - 1.0) + 1.0


    d = Inf
    i = 0
    c = 0.3
    old_elbo = curr_elbo
    
    v = [ϕᵅ₁t,ϕᵝ₁t,ϕᵅ₂t,ϕᵝ₂t,βᵅt,βᵝt] # beta parameters

    while true
        αt, κt = updNIGloop!(grad,
		    μt,βt,
		    μtm1,κtm1,αtm1,βtm1,
		    κ,α,β,
		    ϕᵅ₁t,ϕᵝ₁t,ϕᵅ₂t,ϕᵝ₂t)
	elbo = ncvmp_α!(v,
			μt  , κt,   αt,   βt,
			μtm1, κtm1, αtm1, βtm1, ϕᵅ₁tm1, ϕᵝ₁tm1, ϕᵅ₂tm1, ϕᵝ₂tm1, βᵅtm1, βᵝtm1,
			      κ   , α   , β,    ϕᵅ₁   , ϕᵝ₁   , ϕᵅ₂   , ϕᵝ₂   , βᵅ   , βᵝ,
			      c)
	d = abs(elbo-old_elbo)/(NORMALIZED_DIFF ? 1 : n) # normalized ELBO

	old_elbo = elbo

	i += 1
	if d<1e-3 || i==lim
		break
	else
		c = 0.4+rand()*.2
	end
    end
    
    ϕᵅ₁t,ϕᵝ₁t,ϕᵅ₂t,ϕᵝ₂t,βᵅt,βᵝt = v

    if (i==lim && d>one(d)) || checkNaN((μt, βt, αt, κt, ϕᵅ₁t, ϕᵝ₁t, ϕᵅ₂t, ϕᵝ₂t, βᵅt, βᵝt))
	print("Resetting params, d = ",round(d;digits=6),"\tC=",Count,"\nDiagnostic:")


	if !isfinite(d) # do nothing
		print("Skip!\n")
		Count=1
		μt .= μτm1*.1
		βt .= βτm1*.1
		ϕᵅ₁t,ϕᵝ₁t,ϕᵅ₂t,ϕᵝ₂t,βᵅt,βᵝt = ϕᵅ₁,ϕᵝ₁,ϕᵅ₂,ϕᵝ₂,βᵅ,βᵝ
	else
		@show d
		μt .= μτm1
		βt .= βτm1
		cc += 1
		if cc < 50
			@warn "call f() once more"
		end
	end
    end
    @. grad = η * randNIG(μt, κt, αt, βt)
    end
end



function make_ncvmp_α()
    ∇α    = zeros(4)
    ∇β    = zeros(2)
    ∇ab   = zeros(6)
    Z     = zeros(6,6)
    Λα¹   = view(Z,1:2,1:2)
    Λα²   = view(Z,3:4,3:4)
    Λβ    = view(Z,5:6,5:6)

    function ncvmp_α!(v,
			μt  , κt,   αt,   βt,
			μtm1, κtm1, αtm1, βtm1, ϕᵅ₁tm1, ϕᵝ₁tm1, ϕᵅ₂tm1, ϕᵝ₂tm1, βᵅtm1, βᵝtm1,
			      κ   , α   , β,    ϕᵅ₁   , ϕᵝ₁   , ϕᵅ₂   , ϕᵝ₂   , βᵅ   , βᵝ,
			      c)

		elbo = ∇ELBOtotal!(∇ab,μt  , κt,   αt,   βt ,v...,
				   μtm1, κtm1, αtm1, βtm1, ϕᵅ₁tm1, ϕᵝ₁tm1, ϕᵅ₂tm1, ϕᵝ₂tm1, βᵅtm1, βᵝtm1,
				   κ   , α   , β,    ϕᵅ₁   , ϕᵝ₁   , ϕᵅ₂   , ϕᵝ₂)
		
		map((a,x,y)->iCov!(a,x,y),(Λα¹,Λα²,Λβ),view(v,[1,3,5]),view(v,[2,4,6]))
		∇ab .= Z*∇ab .+ 1
		copyto!(v,dampit!(∇ab,v,c))
		elbo
	end
end
ncvmp_α! = make_ncvmp_α()

function subDamp(x,y,c)
	x>0 ? (c==one(c) ? x : (x^c * y^(1-c))) : y^0.99
end
function dampit!(A,A0,c)
	@. A = subDamp(A,A0,c)
	A
end


include("grad.jl")

@inline mval(Eα::Float64,v1::Float64,v2::Float64)::Float64 = Eα*(v1-v2)+v2
@inline function updNIG1(Er::Float64,Eα::Float64,μτm1::Float64,κτm1::Float64,κ₀::Float64,κₓ::Float64)::Float64
	(Eα*κτm1*μτm1 + Er)/κₓ
end
@inline function updNIG2(Er::Float64,Er2::Float64,Eα1::Float64,Eα2::Float64,μτm1::Float64,κτm1::Float64,βτm1::Float64,κ₀::Float64,β₀::Float64,μₓ::Float64)::Float64
	mval(Eα2,βτm1,β₀) + .5  * (mval(Eα1 , κτm1 * (μₓ-μτm1)^2 , κ₀ * μₓ^2) + μₓ^2 + Er2 - 2μₓ*Er)
end
@inline function Εα(aa::Float64,ba::Float64)
        aa/(aa+ba)
end
@inline function Εα(aa::Float64,ba::Float64,lag::Int64)::Float64
	exp(lgamma(ba)+lgamma(aa+lag)-lbeta(aa,ba)-lgamma(aa+ba+lag))
end
@noinline function updNIGloop!(grad::AbstractArray{Float64,N},
				  μₓ::Array{Float64,N},βₓ::Array{Float64,N},
				  μτm1::Array{Float64,N},κτm1,ατm1,βτm1::Array{Float64,N},κ₀,α₀,β₀,
				  αᵅ¹,βᵅ¹,αᵅ²,βᵅ²) where N

	Eα1::Float64 = Εα(αᵅ¹,βᵅ¹)# : Εα(αᵅ¹,βᵅ¹,Count))
	Eα2::Float64 = Εα(αᵅ²,βᵅ²)# : Εα(αᵅ²,βᵅ²,Count))
	Eα1=Eα1*Eα2
	αₓ::Float64 = mval(Eα2,ατm1,α₀)+.5
	κₓ::Float64 = mval(Eα1,κτm1,κ₀)+1.
	for k in eachindex(μₓ)
		Er = grad[k]
		Er2 = Er*Er
		isfinite(Er2) || (Er2=realmax(Er2))
		
		ms = μτm1[k]
		mx = μₓ[k] = updNIG1(Er,Eα1,ms,κτm1,κ₀,κₓ)
		βₓ[k] = updNIG2(Er,Er2,Eα1,Eα2,ms,κτm1,βτm1[k],κ₀,β₀,mx)
		isfinite(βₓ[k]) || (βₓ[k]=realmax(βₓ[k]))
	#	if -lgamma(ατm1)+lgamma(α₀)+ατm1*log(βτm1[k])-α₀*log(β₀)-(ατm1+1-α₀-1)*(log(βₓ[k])-digamma(αₓ))-(βτm1[k]-β₀)*αₓ/βₓ[k]<0
	#		@show αₓ,βₓ[k],ατm1,βτm1[k],α₀,β₀
	#	end
		if !(βₓ[k]>0)
			@show Er,Er2,Eα1,Eα2,ms,κτm1,βτm1[k],κ₀,β₀,mx
			error(string("βₓ[k]=",βₓ[k]))
		end
	end
	αₓ,κₓ
end
function iCov!(Λα,αᵅ,βᵅ)
	poly_a,poly_b,poly_ab = polygamma.(1,[αᵅ;βᵅ;αᵅ+βᵅ])
	
	A = poly_a-poly_ab
	C = poly_b-poly_ab
	B = -poly_ab

	Λα[1,1] = C
	Λα[1,2] = -B
	Λα[2,1] = -B
	Λα[2,2] = A
	@. Λα /= (-B*B+A*C)
	Λα
end
function iCov(αᵅ,βᵅ)
	poly_a,poly_b,poly_ab = polygamma.(1,[αᵅ;βᵅ;αᵅ+βᵅ])
	
	A = poly_a-poly_ab
	C = poly_b-poly_ab
	B = -poly_ab

	Λα = [C -B;-B A]/(-B*B+A*C)
end

if GRAD_SAMPLING
	@inline function randNIG(m,k,a,b)
		s = rand(Distributions.InverseGamma(a,b))
		m2 = m+sqrt(s/k)*randn()
		m2 / sqrt(s+m2^2)
	end
else
	@inline function randNIG(m,k,a,b)
		a>1 ? m/sqrt(m^2 +b/(a-1)/k + b/(a-1)) : m/sqrt(m^2 +b/a/k + b/a)
	end
end


checkNaN(x::Float64)=!isfinite(x)
checkNaN(x::AbstractArray)=any(!isfinite,x)
checkNaN(x::Tuple)=any(checkNaN,x)

