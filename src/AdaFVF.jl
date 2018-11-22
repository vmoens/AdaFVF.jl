module AdaFVF

using Flux, SpecialFunctions
using Flux: Tracker, Optimise
import Flux.Optimise: optimiser, invdecay, descent
GRAD_SAMPLING = false
NORMALIZED_DIFF = true
include("interface.jl")
include("optimizer.jl")

end # module
