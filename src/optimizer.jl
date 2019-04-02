abstract type AbstractADAFVF  end
mutable struct ADAFVF <: AbstractADAFVF
    η
    κ
    α
    β
    ϕᵅ₁
    ϕᵝ₁
    ϕᵅ₂
    ϕᵝ₂
    βᵅ
    βᵝ
    μt
    βt
    αt
    κt
    ϕᵅ₁t
    ϕᵝ₁t
    ϕᵅ₂t
    ϕᵝ₂t
    βᵅt
    βᵝt
    ϵ
    ncvmp_α!
    cumcount
    elbo
    function ADAFVF(;
        η::R = 0.0001,
        κ::R=1e-10 , α::R=2.0 , β::R=1e-6 , ϕᵅ₁::R=0.9 , ϕᵝ₁::R=.1 , ϕᵅ₂::R=9.9 , ϕᵝ₂::R=0.1, βᵅ::R = 20.0, βᵝ::R = 20.0,
        ϵ::R = 1e-8,
        T::Type=Float32) where R <: Real

        new(T(η),    T(κ),    T(α),    T(β),    T(ϕᵅ₁),    T(ϕᵝ₁),    T(ϕᵅ₂),    T(ϕᵝ₂),    T(βᵅ),    T(βᵝ),
            IdDict(),    IdDict(),    IdDict(),    IdDict(),    IdDict(),    IdDict(),    IdDict(),    IdDict(),    IdDict(),     IdDict(),
            T(ϵ),
            IdDict(),
            IdDict(),
            IdDict())
    end
end
mutable struct ADAFVFHD <: AbstractADAFVF
    η
    γ
    init_params
    modif_params
    μt
    βt
    αt
    κt
    ϕᵅ₁t
    ϕᵝ₁t
    ϕᵅ₂t
    ϕᵝ₂t
    βᵅt
    βᵝt
    ϵ
    ncvmp_α!
    cumcount
    elbo
    function ADAFVFHD(;
        η::R = 0.0001,
        γ::R = 0.0001,
        κ::R=1e-10 , α::R=2.0 , β::R=1e-6 , ϕᵅ₁::R=0.9 , ϕᵝ₁::R=.1 , ϕᵅ₂::R=9.9 , ϕᵝ₂::R=0.1, βᵅ::R = 20.0, βᵝ::R = 20.0,
        ϵ::R = 1e-8,
        T::Type=Float32) where R <: Real

        new(T(η),    T(γ),
            [T(κ), T(α), T(β), T(ϕᵅ₁), T(ϕᵝ₁), T(ϕᵅ₂), T(ϕᵝ₂), T(βᵅ), T(βᵝ)],
            IdDict(),
            IdDict(),    IdDict(),    IdDict(),    IdDict(),    IdDict(),    IdDict(),    IdDict(),    IdDict(),    IdDict(),     IdDict(),
            T(ϵ),
            IdDict(),
            IdDict(),
            IdDict())
    end
end

function getHP(o::ADAFVF, x)
    o.κ   , o.α   , o.β,    o.ϕᵅ₁   , o.ϕᵝ₁   , o.ϕᵅ₂   , o.ϕᵝ₂   , o.βᵅ   , o.βᵝ
end

function getHP(o::ADAFVFHD, x, getdata=true)
    init_params = o.init_params
    modif_params = get!(o.modif_params, x , log.(init_params))
    if getdata
        return exp.(modif_params)
    else
        return modif_params
    end
end

@generated function apply!(o::A, x::AbstractArray{T,N}, Δ) where {A <: AbstractADAFVF,T,N}
    expr = quote
        lim = 200
        κ   , α   , β,    ϕᵅ₁   , ϕᵝ₁   , ϕᵅ₂   , ϕᵝ₂   , βᵅ   , βᵝ  = getHP(o,x)
        μt = get!(o.μt, x, zero(x))
        n = length(μt)
        βt = get!(o.βt, x, fill!(similar(x), β))
        αt = get!(o.αt, x, α)
        κt = get!(o.κt, x, κ)
        ϕᵅ₁t = get!(o.ϕᵅ₁t, x, ϕᵅ₁)
        ϕᵝ₁t = get!(o.ϕᵝ₁t, x, ϕᵅ₁)
        ϕᵅ₂t = get!(o.ϕᵅ₂t, x, ϕᵅ₂)
        ϕᵝ₂t = get!(o.ϕᵝ₂t, x, ϕᵝ₂)
        βᵅt = get!(o.βᵅt, x, βᵅ)
        βᵝt = get!(o.βᵝt, x, βᵝ)
        cumcount = get!(o.cumcount, x, [0])
        curr_elbo = get!(o.elbo, x, T(Inf))

        ncvmp_α! = get!(o.ncvmp_α!, x, make_ncvmp_α(A,T))


        grad = Δ
        if NAN_CHECK
            if checkNaN(grad)
            grad .= 0.0
            return grad
        end
        end
        μtm1,βtm1 = copy.((μt,βt))
        κtm1,αtm1 = κt,αt
        ϕᵅ₁tm1,ϕᵝ₁tm1,ϕᵅ₂tm1,ϕᵝ₂tm1,βᵅtm1,βᵝtm1 = ϕᵅ₁t,ϕᵝ₁t,ϕᵅ₂t,ϕᵝ₂t,βᵅt,βᵝt

        βᵅt = T(0.999) * (βᵅt - T(1.0)) + T(1.0)
        βᵝt = T(0.999) * (βᵝt - T(1.0)) + T(1.0)

        d = Inf
        i = 0
        c = T(0.6)
        old_elbo = curr_elbo

        v = [ϕᵅ₁t, ϕᵝ₁t, ϕᵅ₂t, ϕᵝ₂t, βᵅt, βᵝt] # beta parameters

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
            d = abs(elbo - old_elbo) / (NORMALIZED_DIFF ? 1 : n) # normalized ELBO

            old_elbo = elbo

            i += 1
            if d < T(1e-3) || i == lim
                break
            else
                c = T(0.4) + rand(T) * T(0.2)
            end
        end
    end
    if A <: ADAFVFHD
        expr = quote
            $expr
            loghp = getHP(o,x, false)
            ∇loghp = ∇ELBOhp(loghp,
                μt  , κt,   αt,   βt , v,
                μtm1, κtm1, αtm1, βtm1, ϕᵅ₁tm1, ϕᵝ₁tm1, ϕᵅ₂tm1, ϕᵝ₂tm1, βᵅtm1, βᵝtm1)
            loghp .+= o.γ .* ∇loghp
        end
    end
    expr = quote
        $expr
        o.elbo[x] = old_elbo
        o.ϕᵅ₁t[x], o.ϕᵝ₁t[x], o.ϕᵅ₂t[x], o.ϕᵝ₂t[x] = v
        o.κt[x], o.αt[x] = κt,αt

        if (i==lim && d>one(d)) || checkNaN((μt, βt, αt, κt, ϕᵅ₁t, ϕᵝ₁t, ϕᵅ₂t, ϕᵝ₂t, βᵅt, βᵝt))
        print("Resetting params, d = ",round(d;digits=6),"\tC=",cumcount,"\nDiagnostic:")


        if !isfinite(d) # do nothing
            print("Skip!\n")
            Count=1
            μt .= μtm1*T(0.1)
            βt .= βtm1*T(0.1)
            o.ϕᵅ₁t, o.ϕᵝ₁t, o.ϕᵅ₂t, o.ϕᵝ₂t, o.βᵅt, o.βᵝt = ϕᵅ₁,ϕᵝ₁,ϕᵅ₂,ϕᵝ₂,βᵅ,βᵝ
        else
            μt .= μtm1
            βt .= βtm1
        end
        cumcount[1] += 1
        if cumcount[1] < 50
            @warn "call f() once more"
            return apply!(o, x, Δ)
        end
        end
        @. grad = o.η * randNIG(μt, κt, αt, βt)
    end
end



function make_ncvmp_α(A=ADAFVF, T=Float32)
    ∇α = zeros(T,4)
    ∇β = zeros(T,2)
    ∇ab = zeros(T,6)
    Z = zeros(T,6,6)
    Λα¹ = view(Z,1:2,1:2)
    Λα² = view(Z,3:4,3:4)
    Λβ = view(Z,5:6,5:6)

    function ncvmp_α!(v,
            μt  , κt, αt, βt,
            μtm1, κtm1, αtm1, βtm1, ϕᵅ₁tm1, ϕᵝ₁tm1, ϕᵅ₂tm1, ϕᵝ₂tm1, βᵅtm1, βᵝtm1,
                  κ   , α   , β, ϕᵅ₁   , ϕᵝ₁   , ϕᵅ₂   , ϕᵝ₂   , βᵅ   , βᵝ,
                  c)

        elbo = ∇ELBOtotal!(∇ab, μt  , κt,   αt,   βt ,v...,
                   μtm1, κtm1, αtm1, βtm1, ϕᵅ₁tm1, ϕᵝ₁tm1, ϕᵅ₂tm1, ϕᵝ₂tm1, βᵅtm1, βᵝtm1,
                   κ   , α   , β,    ϕᵅ₁   , ϕᵝ₁   , ϕᵅ₂   , ϕᵝ₂)
        broadcast((a, x, y)->iCov!(a,x,y),(Λα¹,Λα²,Λβ),view(v,[1, 3, 5]),view(v,[2, 4, 6]))
        ∇ab .= Z*∇ab
        ∇ab .+= 1
        copyto!(v,dampit!(∇ab,v,c))
        elbo
    end
end

function subDamp(x::T, y::T, c::T) where T
    x > zero(x) ? (c == one(c) ? x : (x^c * y^(1 - c))) : y^T(0.99)
end
function dampit!(A, A0, c)
    @. A = subDamp(A,A0,c)
    A
end



@inline function mval(Eα::R, v1::R, v2::R)::R where R
    Eα * (v1 - v2) + v2
end
@inline function updNIG1(Er::R, Eα::R, μtm1::R, κtm1::R, κ₀::R, κₓ::R)::R where R
    (Eα*κtm1*μtm1 + Er) / κₓ
end
@inline function updNIG2(Er::R, Er2::R, Eα1::R, Eα2::R, μtm1::R, κtm1::R, βtm1::R, κ₀::R, β₀::R, μₓ::R)::R where R
    mval(Eα2,βtm1,β₀) + R(0.5) * (mval(Eα1 , κtm1 * (μₓ - μtm1)^2 , κ₀ * μₓ^2) + μₓ^2 + Er2 - 2μₓ * Er)
end
@inline function Εα(aa::Real, ba::Real)
        aa / (aa + ba)
end
@inline function Εα(aa::Real, ba::Real, lag::Int64)::Real
    exp(lgamma(ba) + lgamma(aa + lag) - lbeta(aa,ba) - lgamma(aa + ba + lag))
end
@noinline function updNIGloop!(grad::AbstractArray{R,N},
                  μₓ::AbstractArray{R,N}, βₓ::AbstractArray{R,N},
                  μtm1::AbstractArray{R,N}, κtm1, αtm1, βtm1::AbstractArray{R,N}, κ₀, α₀, β₀,
                  αᵅ¹, βᵅ¹, αᵅ², βᵅ²) where {N,R <: Real}

    Eα1::R = Εα(αᵅ¹,βᵅ¹)# : Εα(αᵅ¹,βᵅ¹,Count))
    Eα2::R = Εα(αᵅ²,βᵅ²)# : Εα(αᵅ²,βᵅ²,Count))
    Eα1 = Eα1*Eα2
    αₓ::R = mval(Eα2,αtm1,α₀) + R(0.5)
    κₓ::R = mval(Eα1,κtm1,κ₀) + one(R)
    @inbounds for k in eachindex(μₓ)
        Er = grad[k]
        Er2 = Er*Er
        isfinite(Er2) || (Er2 = floatmax(Er2))

        ms::R = μtm1[k]
        mx::R = μₓ[k] = updNIG1(Er,Eα1,ms,κtm1,κ₀,κₓ)
        βₓ[k] = updNIG2(Er,Er2,Eα1,Eα2,ms,κtm1,βtm1[k],κ₀,β₀,mx)
        isfinite(βₓ[k]) || (βₓ[k] = floatmax(βₓ[k]))
    #	if -lgamma(αtm1)+lgamma(α₀)+αtm1*log(βtm1[k])-α₀*log(β₀)-(αtm1+1-α₀-1)*(log(βₓ[k])-digamma(αₓ))-(βtm1[k]-β₀)*αₓ/βₓ[k]<0
    #		@show αₓ,βₓ[k],αtm1,βtm1[k],α₀,β₀
    #	end
        if zero(R) > βₓ[k] > R(-1e-4)
            βₓ[k] = R(1e-5)
        end
        if !(βₓ[k] > zero(βₓ[k]))
            @show Er,Er2,Eα1,Eα2,ms,κtm1,βtm1[k],κ₀,β₀,mx
            error(string("βₓ[k]=",βₓ[k]))
        end
    end
    αₓ,κₓ
end
function iCov!(Λα, αᵅ, βᵅ)
    poly_a,poly_b,poly_ab = polygamma.(1,[αᵅ; βᵅ; αᵅ + βᵅ])

    A = poly_a - poly_ab
    C = poly_b - poly_ab
    B = -poly_ab

    Λα[1, 1] = C
    Λα[1, 2] = -B
    Λα[2, 1] = -B
    Λα[2, 2] = A
    @. Λα /= (-B*B+A*C)
    Λα
end
function iCov(αᵅ, βᵅ)
    poly_a,poly_b,poly_ab = polygamma.(1,[αᵅ; βᵅ; αᵅ + βᵅ])

    A = poly_a - poly_ab
    C = poly_b - poly_ab
    B = -poly_ab

    Λα = [C - B; -B A] / (-B*B + A*C)
end

if GRAD_SAMPLING
    @inline function randNIG(m, k, a, b)
        s = rand(Distributions.InverseGamma(a,b))
        m2 = m + sqrt(s / k) * randn()
        m2 / sqrt(s + m2^2)
    end
else
    @inline function randNIG(m, k, a, b)
        a > one(a) ? m / sqrt(m^2 + b / (a - 1) / k + b / (a - 1)) : m / sqrt(m^2 + b / a / k + b / a)
    end
end


checkNaN(x::Real) = !isfinite(x)
checkNaN(x::AbstractArray) = any(!isfinite,x)
checkNaN(x::Tuple) = any(checkNaN,x)

