using AdaFVF,Flux,Random,Profile, Statistics, Random
import Flux: Tracker, Optimise

function zero_grads!(P)
	for p in P
		p.grad .= 0
	end
	nothing
end
println("Testing grad update...")

MyModel = Chain(Dense(30,128,elu),Dense(128,128,elu),Dense(128,128,elu),Dense(128,1))
P = params(MyModel) 
adafvf = AdaFVF.Adafvf(P)

f(x) = begin
Random.seed!(Random.MersenneTwister(x))
X = randn(30,32)
loss = sum(MyModel(X))/32
Tracker.back!(loss)
end

zero_grads!(P)
f(1)
adafvf()

println("Update time for 4-layer ANN...")

zero_grads!(P)
f(2)
t = @elapsed adafvf()
println("done (took $t seconds).")

println("Comparing with Adam:")
adam = ADAM(P)

zero_grads!(P)
f(1)
adam()

zero_grads!(P)
f(2)
t = @elapsed adam()
println("done (took $t seconds).")


f(2)
Profile.@profile adafvf()
Profile.print(format=:flat)



function g(ms,x)
	m = ms[1]
	s = exp(ms[2])
	(x - m)^2 / 2 / s + ms[2] / 2
end
nX = 10000
X = -2.0 .+ 10 .* randn(nX)
function testopt(opt,ms,l)
for k in 1:10000
#	k % 1000 == 0 && (@show k, Tracker.data(ms))
	data = X[randperm(nX)[1:l]]
	loss = mean(map(data->g(ms,data),data))
	Tracker.back!(loss)
	opt()
	ms.grad .= 0
end
return ms[1],exp(ms[2])
end

ms0 = randn(2)
for k in (3,5,10,20)
for s in (0.1, 0.01, 0.001, 0.0001)
	@show s,k
	ms = param(ms0)
	adafvf = AdaFVF.Adafvf([ms],s)
	@show testopt(adafvf,ms,k)
	ms = param(ms0)
	adam = ADAM([ms],s)
	@show testopt(adam,ms,k)
end
end
