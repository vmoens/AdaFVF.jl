module AdaFVF

macro mainDef(x)
    """
        Variables defined using @mainDef macro are overwritten in module if they exist in the Main environment.
        When this occurs, @mainDef warns the user about this.
    """
        xn = string(x.args[1])
        y = isdefined(Main,x.args[1]) ? :(Printf.@printf("Warning: Overwriting %s to %s\n",$xn,Main.$(x.args[1]));const $(x.args[1]) = Main.$(x.args[1])) : Expr(:const,x)
        esc(y)
end

using Flux, SpecialFunctions
using Flux: Tracker, Optimise
import SpecialFunctions: polygamma
import Flux.Optimise: apply!
using Zygote

export ADAFVF, ADAFVFHD

@mainDef GRAD_SAMPLING = false
@mainDef NORMALIZED_DIFF = true
@mainDef NAN_CHECK = true

include("grad.jl")
include("elbo.jl")
include("optimizer.jl")
end # module
