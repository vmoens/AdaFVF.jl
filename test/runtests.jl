using AdaFVF,Flux
import Flux: Tracker, Optimise

println("Testing grad update...")

MyModel = Chain(Dense(30,30,σ),Dense(30,10))
P = parameters(MyModel) 
adafvf = AdaFVF.Adafvf(P)

loss = sum(MyModel(randn(30,100))
Flux.Tracker.back!(loss)

adafvf()

println("Update time for 2-layer ANN...")
t = @elapsed adafvf()
println("done (took $t seconds).")

println("Comparing with Adam:")
adam = ADAM(P)
adam()
t = @elapsed adam()
println("done (took $t seconds).")

