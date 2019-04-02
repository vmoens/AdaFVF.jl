polygamma(x::Int64, z::Tracker.TrackedReal{Float32}) = polygamma(Float32(x),z)
polygamma(x::AbstractFloat, z::AbstractFloat) = polygamma(Int(x),z)

function ∇ELBOhp(loghp::AbstractArray{T},
    μt  , κt, αt, βt , v,
    μtm1, κtm1, αtm1, βtm1, ϕᵅ₁tm1, ϕᵝ₁tm1, ϕᵅ₂tm1, ϕᵝ₂tm1, βᵅtm1, βᵝtm1) where T

    loghp = loghp[1:end-2]
    a,b,c,d,e,f,g = loghp
    G = Zygote.gradient((a,b,c,d,e,f,g)->ELBOtotal(
                    μt  , κt,   αt,   βt ,v[1], v[2], v[3], v[4], v[5], v[6],
                    μtm1, κtm1, αtm1, βtm1, ϕᵅ₁tm1, ϕᵝ₁tm1, ϕᵅ₂tm1, ϕᵝ₂tm1, βᵅtm1, βᵝtm1,
                    exp(a),exp(b),exp(c),exp(d),exp(e),exp(f),exp(g)),
                    a,b,c,d,e,f,g)

    return vcat(G...,zeros(T,2))
end

function ELBOtotal(μₓ, κₓ, αₓ, βₓ, αᵅ¹, βᵅ¹, αᵅ², βᵅ², αᵝ, βᵝ,
                          μτm1, κτm1, ατm1, βτm1, αᵅ¹τm1, βᵅ¹τm1, αᵅ²τm1, βᵅ²τm1, αᵝτm1, βᵝτm1,
              κ₀::T, α₀, β₀, αᵅ¹₀, βᵅ¹₀, αᵅ²₀, βᵅ²₀)::T where T

            Eα1,Vα1 = EαVα1(αᵅ¹,  βᵅ¹)
            Eα2,Vα2 = EαVα1(αᵅ²,  βᵅ²)
            N = length(μₓ)
            L::T = sub_ELBO(Eα1, Vα1, Eα2, Vα2, μₓ,κₓ, αₓ, βₓ, μτm1, κτm1, ατm1, βτm1, κ₀, α₀, β₀, digamma(αₓ))
            L = L+logpart_cst1(N,Eα1,Vα1,Eα2,Vα2,ατm1,α₀,κτm1,κ₀)::T
            if NORMALIZED_DIFF
                 L = L/N
            end
            L = L+ELBO_priors1(αᵅ¹,βᵅ¹,αᵅ²,βᵅ²,αᵝ,βᵝ,αᵅ¹τm1,βᵅ¹τm1,αᵅ²τm1,βᵅ²τm1,αᵝτm1,βᵝτm1,αᵅ¹₀, βᵅ¹₀, αᵅ²₀, βᵅ²₀)::T
            return L
end

function sub_ELBO(
                Eα1, Vα1, Eα2, Vα2,
                   μₓ, κₓ, αₓ, βₓ,
                  μτm1, κτm1, ατm1, βτm1,
                  κ₀::T, α₀, β₀, digamma_a)::T where T
    N = length(μₓ)
    L::T = ELBO_lower1(Eα1, Vα1, Eα2, Vα2,μₓ[1],  κₓ, αₓ, βₓ[1], μτm1[1], κτm1, ατm1, βτm1[1], κ₀, α₀, β₀, digamma_a)::T
    @inbounds for k in 2:length(μₓ)
            L = L+ELBO_lower1(Eα1, Vα1, Eα2, Vα2,μₓ[k],  κₓ, αₓ, βₓ[k], μτm1[k], κτm1, ατm1, βτm1[k], κ₀, α₀, β₀, digamma_a)::T
   end
   L
end


function ELBO_lower1(a277,b958,c35,d261,e858,f456,g433,h993,i688,j256,k386,l59,m493,n791,o989::T,p208::R)::T where {R <: Real,T}
    g129 = a277 * c35
    g133 = j256 - m493
    g119 = g133 * g129 + m493
    g118 = 1 / g119
    g140 = g118 * g118
    g137 = i688 * j256
    g117 = e858 - g137 * g129 * g118
    g127 = g137 * m493 * i688
    g130 = g127 * g118
    g125 = g130 * a277
    g132 = g125 * c35
    g135 = l59 - o989
    g168 = R(1/2)*g129
    g142 = R(1/2) - g168
    g157 = g142 * g132
    g124 = g157 + g135 * c35 + o989
    g123 = 1 / g124
    g141 = g123 * g123
    g126 = g129 * g127
    g131 = g127 * c35
    g134 = k386 - n791
    g136 = (g433 * 1) / h993
    g144 = g133 * g140
    g145 = g133 * a277
    g146 = g144 * c35
    g153 = g142 - g168
    g154 = g142 * g118
    g155 = g134 * c35 + n791
    g158 = g142 * g140 * g126
    g165 = (g153 * g125 - g158 * g145) + g135
    g166 = 2 * (g130 - g144 * g126) * g142
    g167 = R(1/2)*g132
    l = (((g155 * log(g124) + g119 * R(1/2) * (-1 / f456 - g136 * g117 * g117)) - R(1/2) * (d261 * (g123 * (-g165 * g134 - g155 * a277 * (g133 * (-a277 * (g154 * g130 + g131 * g140 * (-R(1/2)*a277 - 2 * g154 * g145)) - g153 * g140 * g127) - g125)) + (g165 * g155 * g141 - g134 * g123) * (g135 + a277 * (2 * g142 * g127 * R(1/2) * (g118 - g146) - g167))) - g155 * b958 * c35 * (g123 * ((-g130 * c35 - g153 * g144 * g131) + g133 * (g157 * g146 - g118 * c35 * (g166 - g167))) - g141 * c35 * (g153 * g130 - g158 * g133) * (g166 - g132)))) - g124 * g136) - (log(h993) - p208) * (g155 + R(1.5))
end
    function EαVα1(αᵅ::R, βᵅ) where R # /Users/OldVince/Dropbox/Julia/adawhatever/../NDiff/NDiff.jl, line 376:
            tmp5 = one(R) + αᵅ + βᵅ
            tmp8 = one(R) / tmp5
            tmp2 = αᵅ + βᵅ
            tmp1 = αᵅ / tmp2
            tmp7 = -(tmp1 / tmp2)
            tmp6 = one(R) / tmp2 + tmp7
            tmp4 = one(R)  - tmp1
            tmp3 = (tmp1 * tmp4) / tmp5
            tmp9 = -(tmp3 / tmp5)
            tmp1, tmp3
    end
    function EαVα1(αᵅ, βᵅ, Count) # /Users/OldVince/Dropbox/Julia/adawhatever/../NDiff/NDiff.jl, line 376:
            tmp16 = digamma(βᵅ)
            tmp12 = digamma(αᵅ + βᵅ)
            tmp15 = -((tmp16 - tmp12))
            tmp11 = -((digamma(αᵅ) - tmp12))
            tmp8 = 2Count
            tmp9 = tmp8 + αᵅ + βᵅ
            tmp18 = -(digamma(tmp9))
            tmp7 = tmp8 + αᵅ
            tmp5 = Count + αᵅ + βᵅ
            tmp13 = -(digamma(tmp5))
            tmp14 = tmp13 + tmp15 + tmp16
            tmp4 = lbeta(αᵅ, βᵅ)
            tmp3 = lgamma(βᵅ)
            tmp6 = exp(((tmp3 + lgamma(tmp7)) - tmp4) - lgamma(tmp9))
            tmp2 = Count + αᵅ
            tmp10 = tmp11 + tmp13 + digamma(tmp2)
            tmp1 = exp(((tmp3 + lgamma(tmp2)) - tmp4) - lgamma(tmp5))
            tmp17 = tmp1
            tmp1, tmp6 - tmp1 ^ 2
    end

    function logpart_cst1(N, Eα1::R, Vα1, Eα2, Vα2, at, a0, kt, k0::T)::T where {R,T}
            tmp14 = kt - k0
            tmp23 = Eα2 * tmp14
            tmp22 = tmp23 ^ 2
            tmp21 = tmp22 /R(2)
            tmp15 = R(2)*tmp14
            tmp13 = Eα1 * tmp14
            tmp20 = tmp13 ^ 2
            tmp19 = tmp20 /R(2)
            tmp10 = at - a0
            tmp11 = Eα2 * tmp10 + a0
            tmp18 = Eα2 * tmp13 + k0
            tmp17 = one(R) / tmp18
            tmp38 = -(tmp17 ^ 2)
            tmp27 = tmp17 /R(2)
            tmp37 = tmp17 * tmp38
            tmp7 = tmp15 * tmp38
            tmp26 = -(tmp19 * tmp38) + (tmp10 * trigamma(tmp11)) * tmp10
            tmp30 = -(tmp21 * tmp38)
            tmp32 = -(2 * tmp13 * tmp37)
            tmp31 = -(Eα2 * tmp15 * tmp37)
        L::T = (((-(lgamma(tmp11)) - one(R) /R(2) * (Vα1 * tmp30 + Vα2 * tmp26)) + one(R) /R(2) * log(tmp18)) - R(0.918938533204673)) * N
        return L
    end
function ELBO_priors1(αupperalphaupperone::R, βupperalphaupperone, αupperalphauppertwo, βupperalphauppertwo, αupperbeta, βupperbeta, αupperalphaupperonetm1, βupperalphaupperonetm1, αupperalphauppertwotm1, βupperalphauppertwotm1, αupperbetatm1, βupperbetatm1, αupperalphaupperonelower0::T, βupperalphaupperonelower0, αupperalphauppertwolower0, βupperalphauppertwolower0)::T where {R,T}
    tmp61 = βupperbetatm1 - one(R)
    tmp60 = αupperbetatm1 - one(R)
    tmp59 = lbeta(αupperalphauppertwotm1, βupperalphauppertwotm1)
    tmp58 = βupperalphauppertwotm1 - one(R)
    tmp57 = αupperalphauppertwotm1 - one(R)
    tmp55 = lbeta(αupperalphaupperonetm1, βupperalphaupperonetm1)
    tmp54 = βupperalphaupperonetm1 - one(R)
    tmp53 = αupperalphaupperonetm1 - one(R)
    tmp51 = lbeta(αupperalphaupperonelower0, βupperalphaupperonelower0)
    tmp50 = βupperalphaupperonelower0 - one(R)
    tmp48 = αupperalphaupperonelower0 - one(R)
    tmp47 = αupperalphaupperone + βupperalphaupperone
    tmp97 = digamma(tmp47)
    tmp49 = digamma(βupperalphaupperone) - tmp97
    tmp46 = digamma(αupperalphaupperone) - tmp97
    tmp45 = (tmp46 * tmp48 + tmp49 * tmp50) - tmp51
    tmp52 = ((tmp46 * tmp53 + tmp49 * tmp54) - tmp55) - tmp45
    tmp44 = lbeta(αupperalphauppertwolower0, βupperalphauppertwolower0)
    tmp43 = βupperalphauppertwolower0 - one(R)
    tmp41 = αupperalphauppertwolower0 - one(R)
    tmp40 = αupperalphauppertwo + βupperalphauppertwo
    tmp101 = digamma(tmp40)
    tmp42 = digamma(βupperalphauppertwo) - tmp101
    tmp39 = digamma(αupperalphauppertwo) - tmp101
    tmp38 = (tmp39 * tmp41 + tmp42 * tmp43) - tmp44
    tmp56 = ((tmp39 * tmp57 + tmp42 * tmp58) - tmp59) - tmp38
    tmp35 = αupperalphaupperonetm1 - αupperalphaupperonelower0
    tmp37 = tmp35 ^ 2
    tmp29 = βupperalphaupperonetm1 - βupperalphaupperonelower0
    tmp28 = tmp29 ^ 2
    tmp25 = αupperalphaupperonelower0 + βupperalphaupperonelower0
    tmp26 = (αupperalphaupperonetm1 + βupperalphaupperonetm1) - tmp25
    tmp27 = tmp26 ^ 2
    tmp24 = αupperalphauppertwotm1 - αupperalphauppertwolower0
    tmp23 = tmp24 ^ 2
    tmp21 = βupperalphauppertwotm1 - βupperalphauppertwolower0
    tmp20 = tmp21 ^ 2
    tmp18 = αupperbeta + βupperbeta
    tmp112 = one(R) / tmp18
    tmp122 = tmp112 * αupperbeta
    tmp107 = digamma(tmp18)
    tmp125 = tmp122 * tmp26 + tmp25
    tmp127 = one(R) /R(2) * polygamma(1, tmp125)
    tmp114 = tmp122 * tmp35 + αupperalphaupperonelower0
    tmp126 = one(R) /R(2) * polygamma(1, tmp114)
    tmp113 = tmp122 * tmp24 + αupperalphauppertwolower0
    tmp124 = one(R) /R(2) * polygamma(1, tmp113)
    tmp32 = one(R) - tmp122
    tmp30 = tmp122 * tmp29 + βupperalphaupperonelower0
    tmp22 = tmp122 * tmp21 + βupperalphauppertwolower0
    tmp14 = αupperalphauppertwolower0 + βupperalphauppertwolower0
    tmp15 = (αupperalphauppertwotm1 + βupperalphauppertwotm1) - tmp14
    tmp115 = tmp122 * tmp15 + tmp14
    tmp123 = one(R) /R(2) * polygamma(1, tmp115)
    tmp19 = tmp15 ^ 2
    tmp8 = tmp123 * tmp19
    tmp7 = one(R) /R(2) * tmp20 * polygamma(1, tmp22)
    tmp6 = tmp124 * tmp23
    tmp4 = tmp127 * tmp27
    tmp3 = one(R) /R(2) * tmp28 * polygamma(1, tmp30)
    tmp33 = one(R) / (one(R) + tmp18)
    tmp31 = tmp122 * tmp32 * tmp33
    tmp1 = tmp126 * tmp37
    L = ((((tmp122 * tmp52 + tmp45) - ((-((tmp122 * tmp55 + tmp32 * tmp51)) + tmp1 * tmp31 + tmp3 * tmp31 + lgamma(tmp114) + lgamma(tmp30)) - (                                                  tmp31 * tmp4 + lgamma(tmp125)))) + tmp122 * tmp56 + tmp38) - ((                                                                                                                                  -(                                                                                                                                  (                                                                                                                                              tmp122 * tmp59 + tmp32 * tmp44)) + tmp31 * tmp6 + tmp31 * tmp7 + lgamma(                                                                                                                              tmp113) + lgamma(                                                                                                                              tmp22)) - (                                                                                                                                  tmp31 * tmp8 + lgamma(                                                                                                                              tmp115)))) + (                (tmp60 * (digamma(αupperbeta) - tmp107) + tmp61 * (digamma(βupperbeta) - tmp107)) - lbeta(αupperbetatm1, βupperbetatm1))
end
