
function rbf(x, z, h)
    exp(-((x - z)/h)^2)
end

function silu(x)
    x / (1 + exp(-x))
end

