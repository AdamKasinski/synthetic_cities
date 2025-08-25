using Random
using Statistics
using DelaunayTriangulation
using Graphs

struct Bounds
    xmin::Float64
    xmax::Float64
    ymin::Float64
    ymax::Float64
end

"""
TODO: Add more sample methods - later it will be done by LLM
"""
function generate_sample(N,bounds::Bounds; type = "random")
    xmin = bounds.xmin
    xmax = bounds.xmax
    ymin = bounds.ymin
    ymax = bounds.ymax
    points = Vector{Tuple{Float64,Float64}}(undef, N)
    if type == "random"
        points = [(rand()*(xmax-xmin)+xmin, rand()*(ymax-ymin)+ymin) for _ in 1:N]
    end
    return points
end


function generate_vorn_diagram(points::Vector{Tuple{Float64,Float64}},bounds::Bounds)
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

    polys = DelaunayTriangulation.get_polygons(vorn)
    for vs in values(polys)
        L = length(vs)
        L < 2 && continue
        for k in 1:(L-1)
            i = vs[k]; j = vs[k+1]
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
    return g, nodes, edge_list
end
