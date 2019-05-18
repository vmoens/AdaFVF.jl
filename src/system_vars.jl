GRAD_SAMPLING = Ref(true)
NORMALIZED_DIFF = Ref(true)
NAN_CHECK = Ref(true)

function set_grad_sampling!(value::Bool=true)
global GRAD_SAMPLING
GRAD_SAMPLING[] = value
end
function set_normalized_diff!(value::Bool=true)
global NORMALIZED_DIFF
NORMALIZED_DIFF[] = value
end
function set_nan_check!(value::Bool=true)
global NAN_CHECK
NAN_CHECK[] = value
end
