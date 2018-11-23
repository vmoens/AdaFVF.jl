using AdaFVF,Flux
import Flux: Tracker, Optimise

println("Testing grad update...")

MyModel = Chain(Dense(30,30,elu),Dense(30,10))
P = params(MyModel) 
adafvf = AdaFVF.Adafvf(P)

X = randn(30,100)
loss = sum(MyModel(X))
Tracker.back!(loss)

adafvf()

println("Update time for 2-layer ANN...")
t = @elapsed adafvf()
println("done (took $t seconds).")

println("Comparing with Adam:")
adam = ADAM(P)
adam()
t = @elapsed adam()
println("done (took $t seconds).")

