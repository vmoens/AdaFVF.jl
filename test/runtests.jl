using AdaFVF,Flux
import Flux: Tracker, Optimise

println("Testing grad update...")

MyModel = Chain(Dense(30,128,elu),Dense(128,128,elu),Dense(128,128,elu),Dense(128,1))
P = params(MyModel) 
adafvf = AdaFVF.Adafvf(P)

X = randn(30,32)
loss = sum(MyModel(X))/32
Tracker.back!(loss)

adafvf()

println("Update time for 4-layer ANN...")
t = @elapsed adafvf()
println("done (took $t seconds).")

println("Comparing with Adam:")
adam = ADAM(P)
adam()
t = @elapsed adam()
println("done (took $t seconds).")

