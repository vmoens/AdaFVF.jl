using Pkg
Pkg.activate("./")
using Flux, AdaFVF
P = [param(randn(30,30))]
opt = AdaFVF.Adafvf(P)

L = sum(P[1] .* P[1])
Flux.Tracker.back!(L)

opt()
opt()
opt()

