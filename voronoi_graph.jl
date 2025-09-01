using Random
using Statistics
using DelaunayTriangulation
using Graphs
using RandomizedQuasiMonteCarlo
using QuasiMonteCarlo


struct Bounds
    xmin::Float64
    xmax::Float64
    ymin::Float64
    ymax::Float64
end

"""
TODO: Add more sample methods - later it will be done by LLM
"""
function generate_start_sample(N,bounds::Bounds; type = "random")
    xmin = bounds.xmin
    xmax = bounds.xmax
    ymin = bounds.ymin
    ymax = bounds.ymax
    points = Vector{Tuple{Float64,Float64}}(undef, N)
    if type == "random"
        points = [(rand()*(xmax-xmin)+xmin, 
                 rand()*(ymax-ymin)+ymin) for _ in 1:N]
    end
    return points
end

function uniform_points(N::Int, bounds; epsilon = 1)
    xmin = bounds[1] - epsilon 
    xmax = bounds[2] + epsilon 
    ymin = bounds[3] + epsilon 
    ymax = bounds[4] - epsilon 
    
    return [(round(rand()*(xmax-xmin)+xmin,digits=5), 
            round((rand()*(ymax-ymin)+ymin),digits=5)) for _ in 1:N]
end

function sobol_points(N::Int, bounds; epsilon = 1)
    xmax = bounds[2] - epsilon
    xmin = bounds[1] + epsilon
    ymin = bounds[3] + epsilon
    ymax = bounds[4] - epsilon
    points = QuasiMonteCarlo.sample(N, [xmin, ymin], [xmax, ymax], SobolSample())'
     return [(round(x; digits=5), round(y; digits=5)) for (x,y) in eachrow(points)]
end

const SAMPLERS::Dict{Symbol,Function} = Dict(
    :uniform => uniform_points,
    :sobol => sobol_points
)

function min_spacing_from_points(bounds; frac=0.01)
    xmin,xmax,ymin,ymax = bounds
    return frac * min(xmax - xmin, ymax - ymin)
end

@inline function far_enough(p::Tuple{Float64,Float64},
                            pts::Vector{Tuple{Float64,Float64}},
                            r::Float64)
    @inbounds for q in pts
        dx = p[1] - q[1]; dy = p[2] - q[2]
        if dx*dx + dy*dy < r
            return false
        end
    end
    return true
end


"""
type = :uniform,:sobol
"""
function generate_sample(N,poly;oversample=2,type=:uniform)
    xs = first.(poly)   
    ys = last.(poly)
    xmin,xmax,ymin,ymax = minimum(xs), maximum(xs), minimum(ys), maximum(ys)
    bounds = (xmin,xmax,ymin,ymax)
    ring = Ring([Meshes.Point(x,y) for (x,y) in poly])
    dom  = PolyArea(ring)
    sample = Vector{Tuple{Float64,Float64}}()
    to_generate = N
    sampler = SAMPLERS[type]
    r = min_spacing_from_points(bounds,frac=0.01)
    while to_generate > 0
        rand_points = sampler(N*oversample,bounds)
        for point in rand_points
            if Meshes.in(Meshes.Point(point[1],point[2]), dom) && 
                                                    far_enough(point, sample, r)
                push!(sample,point)
                to_generate -= 1
            end
            to_generate == 0 && break
        end
        oversample *= 2
    end
    return sample
end


function generate_rect_vorn_diagram(points::Vector{Tuple{Float64,Float64}},bounds::Bounds)
    tri = triangulate(points)

    rect_xy = [(bounds.xmin,bounds.ymin), 
                (bounds.xmax,bounds.ymin), 
                (bounds.xmax,bounds.ymax), 
                (bounds.xmin,bounds.ymax), 
                (bounds.xmin,bounds.ymin)]

    boundary_nodes, poly_pts = convert_boundary_points_to_indices(rect_xy)
    vorn = voronoi(tri; clip=true, clip_polygon=(poly_pts, boundary_nodes))
    return vorn
end

function signed_area(poly::Vector{Tuple{Float64,Float64}})
    s = 0.0
    @inbounds for i in 1:length(poly)-1
        (x1,y1) = poly[i]; (x2,y2) = poly[i+1]
        s += x1*y2 - x2*y1
    end
    return 0.5*s
end

function generate_vorn_diagram(seeds::Vector{Tuple{Float64,Float64}},
                            poly::Vector{Tuple{Float64,Float64}})
    ring = copy(poly)
    if ring[end] != ring[1]
        push!(ring, ring[1])
    end
    if signed_area(ring) < 0
        reverse!(ring)
    end

    tri = triangulate(seeds)
    boundary_nodes, poly_pts = convert_boundary_points_to_indices(ring)
    vorn = voronoi(tri; clip=true, clip_polygon=(poly_pts, boundary_nodes))
    return vorn
end


function extract_graph(vorn;DIGITS=9)
    roundpt(p::NTuple{2,Number}) = (round(Float64(p[1]); digits=DIGITS),
                                    round(Float64(p[2]); digits=DIGITS))
    node_id = Dict{Tuple{Float64,Float64}, Int}()
    nodes = Tuple{Float64,Float64}[]
    getid(p) = get!(node_id, p) do
        push!(nodes, p)
        length(nodes)
    end

    edges_set = Set{Tuple{Int,Int}}()
    polys_bounds = Dict()
    polys = DelaunayTriangulation.get_polygons(vorn)
    for (ind, vs) in enumerate(values(polys))
        polys_bounds[ind] = []
        L = length(vs)
        L < 2 && continue
        for k in 1:(L-1)
            i = vs[k]; j = vs[k+1]
            push!(polys_bounds[ind],get_polygon_point(vorn, i))
            pi = roundpt(get_polygon_point(vorn, i))
            pj = roundpt(get_polygon_point(vorn, j))
            ui = getid(pi); vj = getid(pj)
            if ui != vj
                a, b = min(ui, vj), max(ui, vj)
                push!(edges_set, (a, b))
            end
        end
    end
    g = SimpleGraph(length(nodes))
    for (u,v) in edges_set
        add_edge!(g, u, v)
    end

    euclid(a::Tuple{Float64,Float64}, b::Tuple{Float64,Float64}) = hypot(a[1]-b[1], a[2]-b[2])

    edge_list = collect(edges_set)
    lengths = [euclid(nodes[u], nodes[v]) for (u,v) in edge_list]
    return g, nodes, edge_list, polys_bounds
end
