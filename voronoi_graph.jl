module voronoi_graph


    export Bounds, generate_sample, generate_vorn_diagram, extract_graph, generate_city, save_to_json, load_city,edge_lengths,ecdf_distance,compare_cities, wasserstein_1d, normalize, ccw, check_planarity,segments_intersect

    using Random
    using Statistics
    using DelaunayTriangulation: triangulate, voronoi,
    convert_boundary_points_to_indices, get_polygon_point, get_polygons
    using Graphs
    using RandomizedQuasiMonteCarlo
    using QuasiMonteCarlo
    using Meshes
    using JSON

    struct Bounds
        xmin::Float64
        xmax::Float64
        ymin::Float64
        ymax::Float64
    end

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
        ring = Meshes.Ring([Meshes.Point(x,y) for (x,y) in poly])
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
        roundpt(p::NTuple{2,Number}) = (round(Float64(p[1])),
                                        round(Float64(p[2])))
        node_id = Dict{Tuple{Float64,Float64}, Int}()
        nodes = Tuple{Float64,Float64}[]
        getid(p) = get!(node_id, p) do
            push!(nodes, p)
            length(nodes)
        end

        edges_set = Set{Tuple{Int,Int}}()
        polys_bounds = Dict()
        polys = get_polygons(vorn)
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

    """
        smallest_faces(nodes, edges) -> Vector{Vector{Int}}

    Compute all **bounded (smallest) faces** of a planar graph.

    Arguments
    ---------
    - `nodes :: Vector{Tuple{Float64,Float64}}`:
        Node coordinates, 1-based indexing.
    - `edges :: Vector{Tuple{Int,Int}}`:
        Undirected edges (u,v).

    Returns
    -------
    - A vector of faces; each face is a `Vector{Int}` with the
    indices of the vertices along its boundary in order.
    These are all the *bounded* faces; the outer face is removed.
    """
    function smallest_faces(nodes, edges)
        n = length(nodes)
        adj = [Int[] for _ in 1:n]
        for (u, v) in edges
            push!(adj[u], v)
            push!(adj[v], u)
        end

        for v in 1:n
            x_v, y_v = nodes[v]
            nbrs = adj[v]
            sort!(nbrs, by = u -> begin
                x_u, y_u = nodes[u]
                atan(y_u - y_v, x_u - x_v)
            end)
            adj[v] = nbrs
        end

        next_edge = Dict{Tuple{Int,Int},Tuple{Int,Int}}()
        for v in 1:n
            nbrs = adj[v]
            L = length(nbrs)
            for i in 1:L
                u = nbrs[i]
                w = nbrs[mod1(i - 1, L)]
                next_edge[(u, v)] = (v, w)
            end
        end

        visited = Set{Tuple{Int,Int}}()
        faces_idx = Vector{Vector{Int}}()

        for (e_start, _) in next_edge
            e_start in visited && continue

            face = Int[]
            e = e_start
            while true
                push!(visited, e)
                push!(face, e[1])
                e = next_edge[e]
                if e == e_start
                    break
                elseif e in visited && e != e_start
                    break
                end
            end

            length(face) ≥ 3 && push!(faces_idx, face)
        end

        polygon_area(face) = begin
            s = 0.0
            L = length(face)
            @inbounds for i in 1:L
                i1 = face[i]
                i2 = face[mod1(i + 1, L)]
                x1, y1 = nodes[i1]
                x2, y2 = nodes[i2]
                s += x1*y2 - x2*y1
            end
            0.5 * s
        end

        areas = [polygon_area(f) for f in faces_idx]
        isempty(areas) && return Vector{Vector{Tuple{Float64,Float64}}}()

        outer_idx = argmax(abs.(areas))

        faces_coords = Vector{Vector{Tuple{Float64,Float64}}}()
        for (i, face) in enumerate(faces_idx)
            i == outer_idx && continue
            abs(areas[i]) ≤ 0.00001 && continue
            push!(faces_coords, [nodes[v] for v in face])
        end

        return faces_coords
    end

    function generate_first_level(nodes_dict,edges_dict)
        node_ids = sort(collect(keys(nodes_dict)))  # assume 1:N, but sort to be safe
        nodes = [(Float64(nodes_dict[i][1]), Float64(nodes_dict[i][2])) for i in node_ids]

        edge_list = [edges_dict[k] for k in sort(collect(keys(edges_dict)))]

        g = SimpleGraph(length(nodes))
        for (u, v) in edge_list
            add_edge!(g, u, v)
        end
        polys_vec = smallest_faces(nodes,edge_list)
        polys = Dict{Int,Vector{Tuple{Float64,Float64}}}()
        for (idx, poly) in enumerate(polys_vec)
            polys[idx] = [(Float64(p[1]), Float64(p[2])) for p in poly]
        end

        return g, nodes, edge_list, polys
    end


    function save_to_json(lvls)
        open("city_levels.json", "w") do f
            JSON.print(f, lvls)
        end

    end

    """
        generate_city(N, xmin, xmax, ymin, ymax, nodes_dict, edges_dict)

    Generate a hierarchical, multi-level city graph using recursive Voronoi
    subdivision.

    This function constructs a **three-level spatial graph** representing a city:
    - **Level 1**: A coarse city partition over the global bounding box
    - **Level 2**: Subdivision of each level-1 polygon using Sobol sampling
    - **Level 3**: Further subdivision of level-2 polygons using uniform sampling

    Each level produces nodes and edges extracted from Voronoi diagrams, forming
    nested graph structures.

    # Arguments
    - `N` : Number of samples used at the top level (currently passed through but
    not directly used inside this function).
    - `xmin`, `xmax`, `ymin`, `ymax` : Float values defining the global bounding box
    of the city.
    - `nodes_dict` : Dictionary containing node metadata for the first-level graph
    generation.
    - `edges_dict` : Dictionary containing edge metadata for the first-level graph
    generation.
    THE nodes_dict and edges_dict inputs MUSN'T BE EMPTY
    # Returns
    A `Dict{String, Any}` with the following structure:

    - `"level1"` => `[nodes, edges]`  
    Nodes and edges of the first-level (coarse) city graph.

    - `"level2"` => `[nodes_list, edges_list]`  
    Lists of nodes and edges for each second-level subdivision, one per
    level-1 polygon.

    - `"level3"` => `[nodes_list_2, edges_list_2]`  
    Lists of nodes and edges for each third-level subdivision, one per
    level-2 polygon.

    Each `nodes` object represents spatial graph nodes, and each `edges` object
    represents connectivity extracted from Voronoi diagrams.


    # Notes
    - Level-2 sampling uses Sobol sequences for low-discrepancy subdivision.
    - Level-3 sampling uses uniform random sampling.
    - Polygon geometry is propagated explicitly between levels.
    - The function assumes helper methods such as `generate_first_level`,
    `generate_sample`, `generate_vorn_diagram`, and `extract_graph` are defined
    elsewhere.

    # Intended Use
    This function is designed for procedural city generation, hierarchical
    spatial modeling, or multiscale network simulations where geometry-aware
    graph structure is required.
    """
    function generate_city(N, xmin, xmax, ymin, ymax, nodes_dict,edges_dict)

        bounds = Bounds(xmin, xmax, ymin, ymax)
        g, nodes, edge_list, polys = generate_first_level(nodes_dict,edges_dict)

        nodes_list = []
        edges_list = []
        polys_2 = []
        for i in values(polys)
            coords::Vector{Tuple{Float64,Float64}} = collect(i) 
            smpl = generate_sample(5, i, type = :sobol, oversample=2)
            vorn_diagram_2 = generate_vorn_diagram(smpl, coords)
            gs, ndes, edge_lst, pols = extract_graph(vorn_diagram_2)
            push!(polys_2, pols)
            push!(nodes_list, ndes)
            push!(edges_list, edge_lst)
        end

        third_polys = [poly for d in polys_2 for poly in values(d)]

        nodes_list_2 = []
        edges_list_2 = []

        for i in third_polys
            coords::Vector{Tuple{Float64,Float64}} = collect(i) 
            smpl = generate_sample(5, i, type = :uniform, oversample=2)
            vorn_diagram_3 = generate_vorn_diagram(smpl, coords)
            gs, ndes, edge_lst, pols = extract_graph(vorn_diagram_3)
            push!(nodes_list_2, ndes)
            push!(edges_list_2, edge_lst)
        end

        return Dict(
            "level1" => [nodes, edge_list],
            "level2" => [nodes_list, edges_list],
            "level3" => [nodes_list_2, edges_list_2])
    end


    function vecs_to_tuples(x)
        if x isa AbstractVector
            if length(x) == 2 && all(e -> e isa Number, x)
                return (x[1], x[2])                 
            else
                return [vecs_to_tuples(e) for e in x]
            end
        elseif x isa Dict
            return Dict(k => vecs_to_tuples(v) for (k, v) in x)
        else
            return x
        end
    end

    function load_city(filename)
        raw = JSON.parsefile(filename)      
        return vecs_to_tuples(raw)          
    end

    function edge_lengths(nodes, edges)
        euclid(a, b) = hypot(a[1] - b[1], a[2] - b[2])
        return [euclid(nodes[u], nodes[v]) for (u, v) in edges]
    end


    normalize(x) = x ./ mean(x)

    function wasserstein_1d(a, b)

        a = normalize(a)
        b = normalize(b)

        a = sort(a)
        b = sort(b)
        n = min(length(a), length(b))
        return mean(abs.(a[1:n] .- b[1:n]))
    end


    """
        ccw(A, B, C)

    Check if three points are in counter-clockwise order.
    """
    function ccw(A::Tuple{Float64,Float64}, B::Tuple{Float64,Float64}, C::Tuple{Float64,Float64})::Bool
        return (C[2] - A[2]) * (B[1] - A[1]) > (B[2] - A[2]) * (C[1] - A[1])
    end

    """
        segments_intersect(A, B, C, D)

    Check if segment AB properly intersects segment CD.
    Returns false if they only share an endpoint.
    """
    function segments_intersect(A::Tuple{Float64,Float64}, B::Tuple{Float64,Float64},
                                C::Tuple{Float64,Float64}, D::Tuple{Float64,Float64})::Bool
        if A == C || A == D || B == C || B == D
            return false
        end
        return ccw(A, C, D) != ccw(B, C, D) && ccw(A, B, C) != ccw(A, B, D)
    end

    """
        check_planarity(nodes_dict, edges_dict)

    Check if the graph defined by nodes_dict and edges_dict is planar (no edge crossings).

    # Arguments
    - `nodes_dict`: Dict mapping node IDs (Int) to [x, y] coordinates
    - `edges_dict`: Dict mapping edge IDs (Int) to (source, target) tuples

    # Returns
    Dict with:
    - `is_planar`: Bool
    - `num_crossings`: Int
    - `crossings`: Vector of crossing edge pairs with details
    """
    function check_planarity(nodes_dict::Dict, edges_dict::Dict)
        nodes = Dict{Int, Tuple{Float64, Float64}}()
        for (k, v) in nodes_dict
            kid = isa(k, Int) ? k : parse(Int, string(k))
            nodes[kid] = (Float64(v[1]), Float64(v[2]))
        end
        
        edges = Dict{Int, Tuple{Int, Int}}()
        for (k, v) in edges_dict
            kid = isa(k, Int) ? k : parse(Int, string(k))
            edges[kid] = (Int(v[1]), Int(v[2]))
        end
        
        edge_list = collect(values(edges))
        
        crossings = Vector{Dict{String, Any}}()
        
        for i in 1:length(edge_list)
            for j in (i+1):length(edge_list)
                e1 = edge_list[i]
                e2 = edge_list[j]
                
                A = nodes[e1[1]]
                B = nodes[e1[2]]
                C = nodes[e2[1]]
                D = nodes[e2[2]]
                
                if segments_intersect(A, B, C, D)
                    push!(crossings, Dict(
                        "edge1" => [e1[1], e1[2]],
                        "edge2" => [e2[1], e2[2]],
                        "edge1_coords" => [[A[1], A[2]], [B[1], B[2]]],
                        "edge2_coords" => [[C[1], C[2]], [D[1], D[2]]]
                    ))
                end
            end
        end
        
        return Dict(
            "is_planar" => length(crossings) == 0,
            "num_crossings" => length(crossings),
            "crossings" => crossings
        )
    end



end