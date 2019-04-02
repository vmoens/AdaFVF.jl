function sub_∇ELBO!(∇α::AbstractArray{R,1},Eα1::R, Vα1::R, Eα2::R, Vα2::R,
			       μₓ::AbstractArray{R,S},κₓ::R, αₓ::R, βₓ::AbstractArray{R,S}, 
			      μτm1::AbstractArray{R,S}, κτm1::R, ατm1::R, βτm1::AbstractArray{R,S}, 
			      κ₀::R, α₀::R, β₀::R, digamma_a::R) where {S,R<:Real}
	L::R = zero(R)
	@inbounds for k in eachindex(μₓ)
	        L += ∇ELBO_lower1!(∇α,Eα1, Vα1, Eα2, Vα2,μₓ[k],  κₓ, αₓ, βₓ[k], μτm1[k], κτm1, ατm1, βτm1[k], κ₀, α₀, β₀, digamma_a)
	end
	L
end

function ∇ELBOtotal!(∇₆::AbstractArray{R,1},
			μₓ::AbstractArray{R,S}, κₓ::R, αₓ::R, βₓ::AbstractArray{R,S}, αᵅ¹::R,  βᵅ¹::R, αᵅ²::R,  βᵅ²::R, αᵝ::R,  βᵝ::R,
                          μτm1::AbstractArray{R,S}, κτm1::R, ατm1::R, βτm1::AbstractArray{R,S}, αᵅ¹τm1::R,βᵅ¹τm1::R,αᵅ²τm1::R,βᵅ²τm1::R,αᵝτm1::R,βᵝτm1::R,
			  κ₀::R, α₀::R, β₀::R,    αᵅ¹₀::R, βᵅ¹₀::R, αᵅ²₀::R, βᵅ²₀::R, Count::Int64=1)::R where {S,R<:Real}
		
		∇EVα1 = zeros(R,2,2)
		∇EVα2 = zeros(R,2,2)
		∇α = zeros(R,4)
		∇ = zeros(R,4)

		fill!(∇₆,zero(R))
	        Eα1,Vα1 = Count==1 ? ∇EαVα1!(∇EVα1,αᵅ¹,  βᵅ¹) : ∇EαVα1!(∇EVα1,αᵅ¹,  βᵅ¹,Count)
	        Eα2,Vα2 = Count==1 ? ∇EαVα1!(∇EVα2,αᵅ²,  βᵅ²) : ∇EαVα1!(∇EVα2,αᵅ²,  βᵅ²,Count)
	
	        N = length(μₓ)
	        digamma_a = digamma(αₓ)
		L = zero(R)
	        L::R = sub_∇ELBO!(∇α,Eα1, Vα1, Eα2, Vα2, μₓ,κₓ, αₓ, βₓ, μτm1, κτm1, ατm1, βτm1, κ₀, α₀, β₀, digamma_a)
		∇α[1:2] .= ∇EVα1*∇α[1:2]
		∇α[3:4] .= ∇EVα2*∇α[3:4]
	        L::R += ∇logpart_cst1!(∇,N,Eα1,Vα1,Eα2,Vα2,ατm1,α₀,κτm1,κ₀)
		∇α[1:2] .+= ∇EVα1*∇[1:2]
		∇α[3:4] .+= ∇EVα2*∇[3:4]
	        if NORMALIZED_DIFF
	                ∇α ./= N
	                L::R /= N
	        end
	        L::R += ∇ELBO_priors1!(∇₆,
	                          αᵅ¹,βᵅ¹,αᵅ²,βᵅ²,αᵝ,βᵝ,
	                          αᵅ¹τm1,βᵅ¹τm1,αᵅ²τm1,βᵅ²τm1,αᵝτm1,βᵝτm1,
				  αᵅ¹₀, βᵅ¹₀, αᵅ²₀, βᵅ²₀)
	        ∇₆[1:4] .+= ∇α
	        L
end
  function ∇ELBO_lower1!(D1::AbstractArray{R,1},
			   Eα1::R, Vα1::R,  Eα2::R,  Vα2::R, m::R, k::R, a::R, b::R, mt::R, kt::R, at::R, bt::R, k0::R, a0::R, b0::R, digamma_a::R)::R where R<:Real
            _tmp201::R = -Eα2
            _tmp200::R = -Eα1
            _tmp198::R = 2_tmp200
            _tmp186::R = 2_tmp201
            _tmp169::R = Base.Math.log(b) - digamma_a
            _tmp158::R = bt - b0
            _tmp159::R = Eα2 * _tmp158 + b0
            _tmp153::R = at - a0
            _tmp210::R = -_tmp153
            _tmp152::R = Eα2 * _tmp153 + a0
            _tmp202::R = -_tmp152
            _tmp171::R = Vα1 * _tmp202
            _tmp151::R = Eα1 * _tmp202
	    _tmp149::R = one(R)/2 * Eα1
	    _tmp137::R = one(R)/2 * _tmp200
            _tmp134::R = (1 / b) * a 
            _tmp125::R = Eα1 * Eα2 
            _tmp216::R = k0 * kt * mt ^ 2 
	    _tmp245::R = one(R)/2 * _tmp216
            _tmp121::R = kt - k0
            _tmp139::R = Eα2 * _tmp121
            _tmp123::R = Eα1 * _tmp121
            _tmp120::R = 2_tmp121
            _tmp118::R = 1 - _tmp125
	    _tmp234::R,= one(R)/2 * _tmp118
            _tmp144::R = Eα1 * _tmp201 + _tmp118
            _tmp138::R = Eα2 * _tmp200 + _tmp118
	    _tmp117::R = one(R) * Eα2
            _tmp97::R = _tmp117 * _tmp118
            _tmp94::R = Eα2 * _tmp120
            _tmp132::R = _tmp121 * _tmp216
            _tmp235::R = Eα2 * _tmp245
            _tmp90::R = Eα2 * _tmp123
            _tmp127::R = _tmp90 + k0
            _tmp217::R = 1 / _tmp127
            _tmp250::R = _tmp217 * kt * mt
            _tmp218::R = _tmp217 ^ 2 
            _tmp243::R = -_tmp218
            _tmp249::R = _tmp217 + _tmp243 * _tmp90
            _tmp248::R = _tmp123 * _tmp243
            _tmp247::R = _tmp132 * _tmp243
            _tmp255::R = Eα2 * _tmp247
            _tmp258::R = 2_tmp255
	    _tmp254::R = one(R) * _tmp247
            _tmp246::R = _tmp121 * _tmp243
            _tmp253::R = Eα2 * _tmp246
            _tmp257::R = _tmp217 + _tmp253
            _tmp191::R = 2_tmp217
            _tmp146::R = Eα1 * _tmp217
            _tmp145::R = _tmp201 * _tmp217
            _tmp136::R = _tmp118 * _tmp217
            _tmp133::R = _tmp217 
            _tmp147::R = Eα1 * _tmp243
            _tmp142::R = _tmp118 * _tmp243
            _tmp197::R = _tmp118 * _tmp248 + _tmp200 * _tmp217
            _tmp99::R = _tmp235 * _tmp249
            _tmp185::R = _tmp118 * _tmp253 + _tmp145
            _tmp78::R = _tmp216 * _tmp217
            _tmp76::R = _tmp133 * _tmp94
            _tmp75::R = Eα1 * _tmp120 * _tmp133
            _tmp74::R = Eα1 * _tmp216
            _tmp222::R = Eα1 * _tmp245
            _tmp220::R = _tmp217 * _tmp245
            _tmp214::R = _tmp144 * _tmp220
            _tmp72::R = _tmp216 * _tmp218
            _tmp236::R = _tmp125 * _tmp250
            _tmp194::R = -(_tmp217 * _tmp236) * _tmp121 + _tmp250
            _tmp135::R = m - _tmp236
	    _tmp221::R = -one(R)/2 * (1 / k + _tmp134 * _tmp135 ^ 2)
	    _tmp69::R = -one(R)/2 * 2 * _tmp127 * _tmp134 * _tmp135
	    _tmp67::R = one(R)/2 * _tmp72
            _tmp66::R = _tmp245 * _tmp249
            _tmp160::R = Eα1 * _tmp66
            _tmp224::R = Eα2 * (_tmp136 * _tmp248 + _tmp137 * _tmp218) * _tmp216
            _tmp64::R = _tmp138 * _tmp245
            _tmp205::R = -(_tmp243 * _tmp75)
            _tmp140::R = _tmp139 * _tmp205 + _tmp246
            _tmp206::R = -(_tmp243 * _tmp76)
            _tmp232::R = Eα1 * _tmp206 + _tmp243
            _tmp60::R = Eα2 * _tmp132
	    _tmp256::R = one(R)/2 * _tmp60
            _tmp58::R = 2_tmp78
            _tmp223::R = (_tmp217 + _tmp248) * _tmp235
            _tmp54::R = Eα1 * _tmp247
            _tmp231::R = _tmp246 * _tmp247
            _tmp50::R = _tmp217 * _tmp222
            _tmp42::R = 2_tmp222
	    _tmp39::R = one(R) / 2 * _tmp58
            _tmp38::R = _tmp206 * _tmp60
            _tmp37::R = _tmp132 * _tmp146
	    _tmp36::R = one(R)/2 * _tmp132 * _tmp138
            _tmp35::R = Eα1 * _tmp235
            _tmp225::R = _tmp118 * _tmp35
            _tmp238::R = _tmp217 * _tmp225
            _tmp34::R = _tmp118 * _tmp222
            _tmp33::R = _tmp222 * _tmp257
            _tmp27::R = _tmp149 * _tmp257 * _tmp74
            _tmp116::R = (Eα2 * _tmp205 + _tmp147 + _tmp243) * _tmp121 * _tmp34 + _tmp200 * _tmp27 + _tmp200 * _tmp33
            _tmp25::R = _tmp243 * _tmp37
            _tmp24::R = Eα2 * _tmp223
            _tmp23::R = Eα1 * _tmp38
            _tmp251::R = _tmp200 * _tmp39 + _tmp243 * _tmp36
            _tmp199::R = (-(((_tmp142 * _tmp75 + _tmp200 * _tmp218) * _tmp35 + _tmp218 * _tmp34)) * _tmp121 + _tmp251) * Eα1
            _tmp215::R = -(Eα1 * (_tmp224 + _tmp234 * _tmp72)) * _tmp121 + _tmp251
            _tmp155::R = Eα1 * _tmp255 + _tmp78
            _tmp105::R = _tmp117 * (_tmp118 * _tmp155 + _tmp145 * _tmp74)
            _tmp98::R = _tmp117 * _tmp155
	    _tmp174::R = one(R)/2 * (_tmp144 * _tmp255 + _tmp201 * _tmp58)
            _tmp244::R = _tmp125 * _tmp243 * _tmp256
            _tmp207::R = -(_tmp217 * _tmp238)
            _tmp112::R = Eα1 * _tmp138 * _tmp220 + _tmp123 * _tmp207 + _tmp158
            _tmp181::R = Eα2 * _tmp214 + _tmp139 * _tmp207
            _tmp103::R = _tmp159 + _tmp238
            _tmp213::R = Base.Math.log(_tmp103)
            _tmp180::R = (1 / _tmp103) * _tmp152
            _tmp13::R = _tmp217 * _tmp35
            _tmp208::R = -_tmp13
            _tmp88::R = (Eα2 * (_tmp140 * _tmp74 + _tmp54) + _tmp155) * _tmp234 + _tmp160 * _tmp201 + _tmp200 * _tmp99 + _tmp208
            _tmp85::R = (Eα1 * (_tmp139 * _tmp206 + _tmp253) + _tmp257) * _tmp118 * _tmp245 + _tmp200 * _tmp24 + _tmp201 * _tmp33 + _tmp208
            _tmp161::R = _tmp13 * _tmp200 + _tmp158
            _tmp114::R = _tmp118 * _tmp27 + _tmp161
            _tmp113::R = _tmp118 * _tmp160 + _tmp161
            _tmp100::R = _tmp118 * _tmp33 + _tmp161
            _tmp209::R = -(_tmp218 * _tmp225)
            _tmp162::R = _tmp121 * _tmp209
            _tmp183::R = Eα2 * (_tmp162 + _tmp214)
            _tmp102::R = _tmp123 * _tmp209 + _tmp146 * _tmp64 + _tmp158
            _tmp178::R = _tmp102 * _tmp210
            _tmp226::R = _tmp151 * _tmp215 + _tmp178
            _tmp84::R = -((Eα1 * (_tmp142 * _tmp76 + _tmp201 * _tmp218) + _tmp118 * _tmp218) * _tmp235) * _tmp123 + _tmp138 * _tmp66 + _tmp145 * _tmp42 + _tmp162
            _tmp9::R = _tmp13 * _tmp201
            _tmp190::R = _tmp118 * _tmp24 + _tmp9
            _tmp182::R = _tmp118 * _tmp223 + _tmp9
            _tmp175::R = _tmp118 * _tmp98 + _tmp9
            _tmp164::R = 1 / (_tmp118 * _tmp13 + _tmp159)
            _tmp233::R = -(_tmp164 ^ 2)
            _tmp241::R = Eα2 * _tmp233
            _tmp239::R = _tmp100 * _tmp233
            _tmp228::R = _tmp102 * _tmp233
            _tmp187::R = _tmp183 * _tmp233
            _tmp189::R = _tmp226 * _tmp233
            _tmp240::R = 2 * _tmp164 * _tmp233
            _tmp252::R = _tmp100 * (_tmp164 * _tmp210 + _tmp202 * _tmp228) + _tmp164 * _tmp226
            _tmp196::R = -((Eα1 * Eα2 ^ 2 * _tmp118 * _tmp254 + _tmp155 * _tmp97 + _tmp9) * _tmp217) * _tmp121
            _tmp195::R = Eα2 * (_tmp174 + _tmp196)
            _tmp5::R = _tmp137 * _tmp258
            _tmp184::R = _tmp217 * _tmp235 + _tmp244
            _tmp173::R = (-((_tmp136 * _tmp244 + _tmp175 * _tmp217)) * _tmp121 + _tmp174) * Eα2
            _tmp172::R = (-(_tmp13 * _tmp136) * _tmp121 + _tmp214) * _tmp105 * _tmp241 + _tmp164 * _tmp173
	    v::R = (((_tmp127 * _tmp221 + _tmp152 * _tmp213) - (R(1.5) + _tmp152) * _tmp169) - _tmp103 * _tmp134) - one(R)/2 * (Vα2 * _tmp252 + _tmp171 * _tmp172)
            D1tmp1::R = -((((((-(((-_tmp67 + _tmp118 * (_tmp125 * _tmp231 + _tmp132 * _tmp217 * _tmp232) + _tmp133 * _tmp5 + _tmp201 * _tmp25) * _tmp125 + _tmp136 * _tmp66 + _tmp185 * _tmp50 + _tmp224)) * _tmp121 + -_tmp245 * _tmp191 + _tmp186 * _tmp254 + _tmp206 * _tmp36 + _tmp5) * Eα1 + _tmp215) * _tmp202 + _tmp210 * _tmp84) * _tmp164 + ((-(_tmp190 * _tmp240) * _tmp102 + _tmp233 * _tmp84) * _tmp100 + _tmp228 * _tmp85) * _tmp202 + (_tmp164 * _tmp85 + _tmp190 * _tmp239) * _tmp210 + _tmp189 * _tmp190) * Vα2 + (((-((_tmp13 * _tmp185 + _tmp136 * _tmp184)) * _tmp121 + _tmp144 * _tmp254 + _tmp201 * _tmp39) * _tmp241 + -(_tmp182 * _tmp240) * _tmp183) * _tmp105 + (-((((_tmp23 + _tmp258) * _tmp97 + _tmp184 * _tmp201 + _tmp201 * _tmp98) * _tmp217 + _tmp117 * _tmp136 * (_tmp23 + _tmp255) + _tmp175 * _tmp253 + _tmp185 * _tmp244)) * _tmp121 + one(R)/2 * (_tmp144 * _tmp38 + _tmp186 * _tmp255 + _tmp201 * _tmp258)) * Eα2 * _tmp164 + _tmp173 * _tmp182 * _tmp233 + _tmp187 * (Eα2 * _tmp118 * (_tmp232 + _tmp243) * _tmp256 + _tmp201 * _tmp223 + _tmp201 * _tmp99)) * _tmp171) * one(R)/2) + -(Eα2 * _tmp194) * _tmp69 + -(_tmp134 * _tmp181) + _tmp139 * _tmp221 + _tmp180 * _tmp181
	    D1tmp2::R = -(one(R)/2 * _tmp172 * _tmp202)
	    D1tmp3::R = -((((((-((_tmp118 * (_tmp149 * _tmp255 + _tmp220) + _tmp197 * _tmp235) * _tmp146) * _tmp139 + _tmp144 * _tmp66 + _tmp162 + _tmp191 * _tmp200 * _tmp235) * _tmp233 + -(_tmp113 * _tmp240) * _tmp183) * _tmp105 + ((-_tmp191 + _tmp140 * _tmp144 + _tmp186 * _tmp248 + _tmp200 * _tmp243 * _tmp94) * _tmp235 + (_tmp144 * _tmp253 + _tmp186 * _tmp217) * _tmp245 + -(((_tmp140 + _tmp246) * _tmp238 + _tmp105 * _tmp248 + _tmp197 * _tmp244 + _tmp217 * _tmp88)) * _tmp139 + _tmp196) * _tmp164 + _tmp113 * _tmp195 * _tmp233 + _tmp187 * _tmp88) * _tmp202 + (_tmp105 * _tmp187 + _tmp164 * _tmp195) * _tmp210) * Vα1 + (((-((((Eα1 ^ 2 * _tmp231 + _tmp205 * _tmp37) * _tmp118 + one(R)/2 * _tmp133 * _tmp198 * _tmp54 + _tmp200 * _tmp25) * _tmp125 + Eα1 * (_tmp118 * _tmp25 + _tmp200 * _tmp67) + _tmp147 * _tmp234 * _tmp37 + _tmp197 * _tmp50)) + _tmp198 * _tmp243 * _tmp245 + _tmp200 * _tmp243 * _tmp42 + _tmp205 * _tmp64) * _tmp121 * _tmp151 + Eα1 * _tmp210 * _tmp215 + _tmp199 * _tmp210) * _tmp164 + ((-(_tmp114 * _tmp240) * _tmp102 + _tmp199 * _tmp233) * _tmp100 + _tmp116 * _tmp228) * _tmp202 + (_tmp114 * _tmp239 + _tmp116 * _tmp164) * _tmp210 + _tmp114 * _tmp189 + _tmp178 * _tmp239) * Vα2) * one(R)/2) + -(Eα1 * _tmp194) * _tmp69 + -(_tmp112 * _tmp134) + -(_tmp153 * _tmp169) + _tmp112 * _tmp180 + _tmp123 * _tmp221 + _tmp153 * _tmp213
	    D1tmp4::R = -(_tmp252/2)

	    D1[1]+=D1tmp1
	    D1[2]+=D1tmp2
	    D1[3]+=D1tmp3
	    D1[4]+=D1tmp4
        v
    end

    function ∇EαVα1!(D1, αᵅ::R, βᵅ) where R # /Users/OldVince/Dropbox/Julia/adawhatever/../NDiff/NDiff.jl, line 376:
        begin  # /Users/OldVince/Dropbox/Julia/adawhatever/../NDiff/simplify.jl, line 330:
            _tmp5 = one(R) + αᵅ + βᵅ
            _tmp8 = 1 / _tmp5
            _tmp2 = αᵅ + βᵅ
            _tmp1 = αᵅ / _tmp2
            _tmp7 = -(_tmp1 / _tmp2)
            _tmp6 = 1 / _tmp2 + _tmp7
            _tmp4 = 1 - _tmp1
            _tmp3 = (_tmp1 * _tmp4) / _tmp5
            _tmp9 = -(_tmp3 / _tmp5)
            v = vcat(_tmp1, _tmp3)
            D1 .= begin  # /Users/OldVince/Dropbox/Julia/adawhatever/../NDiff/NDiff.jl, line 204:
                    reshape([_tmp6, _tmp7, (-_tmp6 * _tmp1 + _tmp4 * _tmp6) * _tmp8 + _tmp9, (-_tmp7 * _tmp1 + _tmp4 * _tmp7) * _tmp8 + _tmp9], (2, 2))
                end
        end # /Users/OldVince/Dropbox/Julia/adawhatever/../NDiff/NDiff.jl, line 377:
        v
    end
    function ∇EαVα1!(D1, αᵅ, βᵅ, Count) # /Users/OldVince/Dropbox/Julia/adawhatever/../NDiff/NDiff.jl, line 376:
        begin  # /Users/OldVince/Dropbox/Julia/adawhatever/../NDiff/simplify.jl, line 330:
            _tmp16 = digamma(βᵅ)
            _tmp12 = digamma(αᵅ + βᵅ)
            _tmp15 = -((_tmp16 - _tmp12))
            _tmp11 = -((digamma(αᵅ) - _tmp12))
            _tmp8 = 2Count
            _tmp9 = _tmp8 + αᵅ + βᵅ
            _tmp18 = -(digamma(_tmp9))
            _tmp7 = _tmp8 + αᵅ
            _tmp5 = Count + αᵅ + βᵅ
            _tmp13 = -(digamma(_tmp5))
            _tmp14 = _tmp13 + _tmp15 + _tmp16
            _tmp4 = lbeta(αᵅ, βᵅ)
            _tmp3 = lgamma(βᵅ)
            _tmp6 = exp(((_tmp3 + lgamma(_tmp7)) - _tmp4) - lgamma(_tmp9))
            _tmp2 = Count + αᵅ
            _tmp10 = _tmp11 + _tmp13 + digamma(_tmp2)
            _tmp1 = exp(((_tmp3 + lgamma(_tmp2)) - _tmp4) - lgamma(_tmp5))
            _tmp17 = _tmp1 
            v = vcat(_tmp1, _tmp6 - _tmp1 ^ 2)
            D1 .= begin  # /Users/OldVince/Dropbox/Julia/adawhatever/../NDiff/NDiff.jl, line 204:
                    reshape([_tmp1 * _tmp10, _tmp1 * _tmp14, (_tmp11 + _tmp18 + digamma(_tmp7)) * _tmp6 + -(2 * _tmp1 * _tmp10 * _tmp17), (_tmp15 + _tmp16 + _tmp18) * _tmp6 + -(2 * _tmp1 * _tmp14 * _tmp17)], (2, 2))
                end
        end # /Users/OldVince/Dropbox/Julia/adawhatever/../NDiff/NDiff.jl, line 377:
        v
    end

    function ∇logpart_cst1!(D1, N, Eα1, Vα1, Eα2, Vα2, at, a0, kt, k0::R) where R
            _tmp14 = kt - k0
            _tmp23 = Eα2 * _tmp14
            _tmp22 = _tmp23 ^ 2
            _tmp21 = _tmp22/2
            _tmp15 = 2_tmp14
            _tmp13 = Eα1 * _tmp14
            _tmp20 = _tmp13 ^ 2
            _tmp19 = _tmp20/2
            _tmp10 = at - a0
            _tmp11 = Eα2 * _tmp10 + a0
            _tmp18 = Eα2 * _tmp13 + k0
            _tmp17 = 1 / _tmp18
            _tmp38 = -(_tmp17 ^ 2)
            _tmp27 = _tmp17/2
            _tmp37 = _tmp17 * _tmp38
            _tmp7 = _tmp15 * _tmp38
            _tmp26 = -(_tmp19 * _tmp38) + (_tmp10 * trigamma(_tmp11)) * _tmp10
            _tmp30 = -(_tmp21 * _tmp38)
            _tmp32 = -(2 * _tmp13 * _tmp37)
            _tmp31 = -(Eα2 * _tmp15 * _tmp37)
	    v = (((-(lgamma(_tmp11)) - one(R)/2 * (Vα1 * _tmp30 + Vα2 * _tmp26)) + one(R)/2 * log(_tmp18)) - R(0.918938533204673)) * N
            D1 .= reshape([(-((-(one(R)/2 * (_tmp13 * _tmp7 + _tmp20 * _tmp31)) * Vα2 + -(_tmp21 * _tmp31) * Vα1) * one(R)/2) + _tmp23 * _tmp27) * N, -(_tmp30/2) * N, (-(one(R)/2 * (-(one(R)/2 * (_tmp22 * _tmp32 + _tmp23 * _tmp7)) * Vα1 + (-(_tmp19 * _tmp32) + ((_tmp10 * polygamma(2, _tmp11)) * _tmp10) * _tmp10) * Vα2)) + -(_tmp10 * digamma(_tmp11)) + _tmp13 * _tmp27) * N, -(_tmp26/2) * N], (4,))
        v
    end

    function ∇ELBO_priors1!(D1, αᵅ¹::R, βᵅ¹, αᵅ², βᵅ², αᵝ, βᵝ, αᵅ¹tm1, βᵅ¹tm1, αᵅ²tm1, βᵅ²tm1, αᵝtm1, βᵝtm1, αᵅ¹₀, βᵅ¹₀, αᵅ²₀, βᵅ²₀)  where R
            _tmp61 = βᵝtm1 - one(R)
            _tmp60 = αᵝtm1 - one(R)
            _tmp59 = lbeta(αᵅ²tm1, βᵅ²tm1)
            _tmp58 = βᵅ²tm1 - one(R)
            _tmp57 = αᵅ²tm1 - one(R)
            _tmp55 = lbeta(αᵅ¹tm1, βᵅ¹tm1)
            _tmp54 = βᵅ¹tm1 - one(R)
            _tmp53 = αᵅ¹tm1 - one(R)
            _tmp51 = lbeta(αᵅ¹₀, βᵅ¹₀)
            _tmp50 = βᵅ¹₀ - one(R)
            _tmp48 = αᵅ¹₀ - one(R)
            _tmp47 = αᵅ¹ + βᵅ¹
            _tmp97 = digamma(_tmp47)
            _tmp117 = -(trigamma(_tmp47))
            _tmp120 = _tmp117 + trigamma(αᵅ¹)
            _tmp65 = _tmp117 + trigamma(βᵅ¹)
            _tmp64 = _tmp117 * _tmp48 + _tmp50 * _tmp65
            _tmp62 = _tmp117 * _tmp50 + _tmp120 * _tmp48
            _tmp49 = digamma(βᵅ¹) - _tmp97
            _tmp46 = digamma(αᵅ¹) - _tmp97
            _tmp45 = (_tmp46 * _tmp48 + _tmp49 * _tmp50) - _tmp51
            _tmp52 = ((_tmp46 * _tmp53 + _tmp49 * _tmp54) - _tmp55) - _tmp45
            _tmp44 = lbeta(αᵅ²₀, βᵅ²₀)
            _tmp43 = βᵅ²₀ - one(R)
            _tmp41 = αᵅ²₀ - one(R)
            _tmp40 = αᵅ² + βᵅ²
            _tmp101 = digamma(_tmp40)
            _tmp118 = -(trigamma(_tmp40))
            _tmp121 = _tmp118 + trigamma(αᵅ²)
            _tmp69 = _tmp118 + trigamma(βᵅ²)
            _tmp68 = _tmp118 * _tmp41 + _tmp43 * _tmp69
            _tmp66 = _tmp118 * _tmp43 + _tmp121 * _tmp41
            _tmp42 = digamma(βᵅ²) - _tmp101
            _tmp39 = digamma(αᵅ²) - _tmp101
            _tmp38 = (_tmp39 * _tmp41 + _tmp42 * _tmp43) - _tmp44
            _tmp56 = ((_tmp39 * _tmp57 + _tmp42 * _tmp58) - _tmp59) - _tmp38
            _tmp35 = αᵅ¹tm1 - αᵅ¹₀
            _tmp37 = _tmp35 ^ 2
            _tmp29 = βᵅ¹tm1 - βᵅ¹₀
            _tmp28 = _tmp29 ^ 2
            _tmp25 = αᵅ¹₀ + βᵅ¹₀
            _tmp26 = (αᵅ¹tm1 + βᵅ¹tm1) - _tmp25
            _tmp27 = _tmp26 ^ 2
            _tmp24 = αᵅ²tm1 - αᵅ²₀
            _tmp23 = _tmp24 ^ 2
            _tmp21 = βᵅ²tm1 - βᵅ²₀
            _tmp20 = _tmp21 ^ 2
            _tmp18 = αᵝ + βᵝ
            _tmp112 = 1 / _tmp18
            _tmp122 = _tmp112 * αᵝ
            _tmp107 = digamma(_tmp18)
            _tmp119 = -(trigamma(_tmp18))
            _tmp125 = _tmp122 * _tmp26 + _tmp25
            _tmp127 = one(R)/2 * polygamma(1, _tmp125)
            _tmp114 = _tmp122 * _tmp35 + αᵅ¹₀
            _tmp126 = one(R)/2 * polygamma(1, _tmp114)
            _tmp113 = _tmp122 * _tmp24 + αᵅ²₀
            _tmp124 = one(R)/2 * polygamma(1, _tmp113)
            _tmp103 = -(_tmp112 * _tmp122)
            _tmp104 = -_tmp103
            _tmp86 = _tmp103/2
            _tmp70 = _tmp103 + _tmp112
            _tmp106 = -_tmp70
            _tmp82 = _tmp70/2
            _tmp79 = _tmp24 * digamma(_tmp113)
            _tmp84 = _tmp35 * digamma(_tmp114)
            _tmp32 = one(R) - _tmp122
            _tmp30 = _tmp122 * _tmp29 + βᵅ¹₀
            _tmp85 = _tmp29 * digamma(_tmp30)
            _tmp22 = _tmp122 * _tmp21 + βᵅ²₀
            _tmp80 = _tmp21 * digamma(_tmp22)
            _tmp14 = αᵅ²₀ + βᵅ²₀
            _tmp15 = (αᵅ²tm1 + βᵅ²tm1) - _tmp14
            _tmp115 = _tmp122 * _tmp15 + _tmp14
            _tmp123 = one(R)/2 * polygamma(1, _tmp115)
            _tmp19 = _tmp15 ^ 2
            _tmp78 = _tmp15 * digamma(_tmp115)
            _tmp8 = _tmp123 * _tmp19
            _tmp7 = one(R)/2 * _tmp20 * polygamma(1, _tmp22)
            _tmp6 = _tmp124 * _tmp23
            _tmp83 = _tmp26 * digamma(_tmp125)
            _tmp4 = _tmp127 * _tmp27
            _tmp3 = one(R)/2 * _tmp28 * polygamma(1, _tmp30)
            _tmp33 = 1 / (one(R) + _tmp18)
            _tmp105 = -(_tmp122 * _tmp32 * _tmp33 ^ 2)
            _tmp93 = (_tmp103 * _tmp32 + _tmp104 * _tmp122) * _tmp33 + _tmp105
            _tmp76 = (_tmp106 * _tmp122 + _tmp32 * _tmp70) * _tmp33 + _tmp105
            _tmp31 = _tmp122 * _tmp32 * _tmp33
            _tmp88 = _tmp26 * _tmp27 * _tmp31 * polygamma(2, _tmp125)
            _tmp87 = _tmp15 * _tmp19 * _tmp31 * polygamma(2, _tmp115)
            _tmp92 = _tmp28 * _tmp29 * _tmp31 * polygamma(2, _tmp30)
            _tmp91 = _tmp31 * _tmp35 * _tmp37 * polygamma(2, _tmp114)
            _tmp90 = _tmp20 * _tmp21 * _tmp31 * polygamma(2, _tmp22)
            _tmp89 = _tmp23 * _tmp24 * _tmp31 * polygamma(2, _tmp113)
            _tmp1 = _tmp126 * _tmp37
            v = ((((_tmp122 * _tmp52 + _tmp45) - ((-((_tmp122 * _tmp55 + _tmp32 * _tmp51)) + _tmp1 * _tmp31 + _tmp3 * _tmp31 + lgamma(_tmp114) + lgamma(_tmp30)) - (_tmp31 * _tmp4 + lgamma(_tmp125)))) + _tmp122 * _tmp56 + _tmp38) - ((-((_tmp122 * _tmp59 + _tmp32 * _tmp44)) + _tmp31 * _tmp6 + _tmp31 * _tmp7 + lgamma(_tmp113) + lgamma(_tmp22)) - (_tmp31 * _tmp8 + lgamma(_tmp115)))) + ((_tmp60 * (digamma(αᵝ) - _tmp107) + _tmp61 * (digamma(βᵝ) - _tmp107)) - lbeta(αᵝtm1, βᵝtm1))
            D1 .= begin  # /Users/OldVince/Dropbox/Julia/adawhatever/../NDiff/NDiff.jl, line 204:
                    reshape([(-_tmp62 + _tmp117 * _tmp54 + _tmp120 * _tmp53) * _tmp122 + _tmp62, (-_tmp64 + _tmp117 * _tmp53 + _tmp54 * _tmp65) * _tmp122 + _tmp64, (-_tmp66 + _tmp118 * _tmp58 + _tmp121 * _tmp57) * _tmp122 + _tmp66, (-_tmp68 + _tmp118 * _tmp57 + _tmp58 * _tmp69) * _tmp122 + _tmp68, (_tmp119 + trigamma(αᵝ)) * _tmp60 + -((-((_tmp106 * _tmp44 + _tmp59 * _tmp70)) + -((_tmp123 * _tmp19 * _tmp76 + _tmp70 * _tmp78 + _tmp82 * _tmp87)) + _tmp124 * _tmp23 * _tmp76 + _tmp7 * _tmp76 + _tmp70 * _tmp79 + _tmp70 * _tmp80 + _tmp82 * _tmp89 + _tmp82 * _tmp90)) + -((-((_tmp106 * _tmp51 + _tmp55 * _tmp70)) + -((_tmp127 * _tmp27 * _tmp76 + _tmp70 * _tmp83 + _tmp82 * _tmp88)) + _tmp126 * _tmp37 * _tmp76 + _tmp3 * _tmp76 + _tmp70 * _tmp84 + _tmp70 * _tmp85 + _tmp82 * _tmp91 + _tmp82 * _tmp92)) + _tmp119 * _tmp61 + _tmp52 * _tmp70 + _tmp56 * _tmp70, (_tmp119 + trigamma(βᵝ)) * _tmp61 + -((-((_tmp103 * _tmp55 + _tmp104 * _tmp51)) + -((_tmp103 * _tmp83 + _tmp4 * _tmp93 + _tmp86 * _tmp88)) + _tmp1 * _tmp93 + _tmp103 * _tmp84 + _tmp103 * _tmp85 + _tmp3 * _tmp93 + _tmp86 * _tmp91 + _tmp86 * _tmp92)) + -((-((_tmp103 * _tmp59 + _tmp104 * _tmp44)) + -((_tmp103 * _tmp78 + _tmp8 * _tmp93 + _tmp86 * _tmp87)) + _tmp103 * _tmp79 + _tmp103 * _tmp80 + _tmp6 * _tmp93 + _tmp7 * _tmp93 + _tmp86 * _tmp89 + _tmp86 * _tmp90)) + _tmp103 * _tmp52 + _tmp103 * _tmp56 + _tmp119 * _tmp60], (6,))
                end
        v
    end

