module AdaFVF


using Flux, SpecialFunctions
using Distributions
using Flux: Tracker, Optimise
import SpecialFunctions: polygamma
import Flux.Optimise: apply!
using Zygote

export ADAFVF, ADAFVFHD

include("system_vars.jl")
include("grad.jl")
include("elbo.jl")
include("optimizer.jl")
end # module
