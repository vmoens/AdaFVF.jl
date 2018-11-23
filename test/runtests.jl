using AdaFVF,Flux
import Flux: Tracker, Optimise

MyModel = Chain(Dense(30,30,σ),Dense(30,10))
P = parameters(MyModel) 
opt = AdaFVF.Adafvf(P)

loss = sum(MyModel(randn(30,100))
Flux.Tracker.back!(loss)

opt()
opt()
opt()

