using AdaFVF,Flux,Random,Profile
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
