module AdaFVF

using Flux, SpecialFunctions
using Flux: Tracker, Optimise
import Flux.Optimise: optimiser, invdecay, descent

export Adafvf
GRAD_SAMPLING = false
NORMALIZED_DIFF = true
NAN_CHECK = true
include("interface.jl")
include("optimizer.jl")

end # module
