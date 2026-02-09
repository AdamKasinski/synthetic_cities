module GeometryAPI
include("voronoi_graph.jl")
using HTTP, JSON3, Sockets, Logging
using .voronoi_graph: generate_city, check_planarity, segments_intersect, ccw


function fix_crossing_impl(nodes_dict, edges_dict, edge_to_remove)
    nodes = Dict{Int, Tuple{Float64, Float64}}()
    for (k, v) in nodes_dict
        kid = isa(k, Integer) ? Int(k) : parse(Int, string(k))
        nodes[kid] = (Float64(v[1]), Float64(v[2]))
    end
    
    edges = Dict{Int, Tuple{Int, Int}}()
    for (k, v) in edges_dict
        kid = isa(k, Integer) ? Int(k) : parse(Int, string(k))
        edges[kid] = (Int(v[1]), Int(v[2]))
    end
    
    remove_u = Int(edge_to_remove[1])
    remove_v = Int(edge_to_remove[2])
    
    new_edges = Dict{Int, Tuple{Int, Int}}()
    edge_id = 1
    removed = false
    
    for (eid, (u, v)) in edges
        if (u == remove_u && v == remove_v) || (v == remove_u && u == remove_v)
            removed = true
            continue
        end
        new_edges[edge_id] = (u, v)
        edge_id += 1
    end
    
    if !removed
        return Dict("error" => "Edge [$remove_u, $remove_v] not found")
    end
    
    existing_edges = Set((min(u,v), max(u,v)) for (u, v) in values(new_edges))
    node_ids = sort(collect(keys(nodes)))
    safe_alternatives = Vector{Dict{String, Any}}()
    
    for i in 1:length(node_ids)
        for j in (i+1):length(node_ids)
            u, v = node_ids[i], node_ids[j]
            if (min(u,v), max(u,v)) in existing_edges
                continue
            end
            A = nodes[u]
            B = nodes[v]
            crosses = false
            for (eu, ev) in values(new_edges)
                if segments_intersect(A, B, nodes[eu], nodes[ev])
                    crosses = true
                    break
                end
            end
            if !crosses
                len = hypot(A[1] - B[1], A[2] - B[2])
                push!(safe_alternatives, Dict("edge" => [u, v], "length" => round(len, digits=2)))
            end
        end
    end
    
    sort!(safe_alternatives, by = x -> x["length"])
    new_edges_json = Dict(string(k) => [u, v] for (k, (u, v)) in new_edges)
    
    return Dict(
        "success" => true,
        "new_edges_dict" => new_edges_json,
        "safe_alternatives" => safe_alternatives[1:min(10, length(safe_alternatives))]
    )
end

"""
    handle(req::HTTP.Request)
HTTP handler for the Geometry API.
Currently supports:
- `POST /generate_city`
- `POST /check_planarity`
- `POST /fix_crossing`
"""
function handle(req::HTTP.Request)
    try
        if req.method == "POST" && req.target == "/generate_city"
            body = JSON3.read(String(req.body))
            xmin = Float64(body["xmin"]); xmax = Float64(body["xmax"])
            ymin = Float64(body["ymin"]); ymax = Float64(body["ymax"])
            N    = Int(body["N"])
            nodes_dict = haskey(body, "nodes_dict") ? body["nodes_dict"] : nothing
            edges_dict = haskey(body, "edges_dict") ? body["edges_dict"] : nothing
            result = generate_city(xmin, xmax, ymin, ymax, N, nodes_dict, edges_dict)
            return HTTP.Response(200, JSON3.write(result))
        
        elseif req.method == "POST" && req.target == "/check_planarity"
            body = JSON3.read(String(req.body))
            nodes_dict = Dict{Any,Any}()
            for (k, v) in pairs(body["nodes_dict"])
                nodes_dict[k] = [v[1], v[2]]
            end
            edges_dict = Dict{Any,Any}()
            for (k, v) in pairs(body["edges_dict"])
                edges_dict[k] = (v[1], v[2])
            end
            result = check_planarity(nodes_dict, edges_dict)
            return HTTP.Response(200, JSON3.write(result))
        
        elseif req.method == "POST" && req.target == "/fix_crossing"
            body = JSON3.read(String(req.body))
            nodes_dict = Dict{Any,Any}()
            for (k, v) in pairs(body["nodes_dict"])
                nodes_dict[k] = [v[1], v[2]]
            end
            edges_dict = Dict{Any,Any}()
            for (k, v) in pairs(body["edges_dict"])
                edges_dict[k] = (v[1], v[2])
            end
            edge_to_remove = body["edge_to_remove"]
            result = fix_crossing_impl(nodes_dict, edges_dict, edge_to_remove)
            return HTTP.Response(200, JSON3.write(result))
        
        else
            return HTTP.Response(404, "not found")
        end
    catch e
        bt = catch_backtrace()
        @error "API failed" exception=(e, bt) req_body=String(req.body)
        err_json = Dict(
            "error" => sprint(showerror, e),
            "stacktrace" => sprint(Base.show_backtrace, bt)
        )
        return HTTP.Response(500, JSON3.write(err_json))
    end
end

"""
    start_server(host::String="127.0.0.1", port::Int=8080)
Start the Geometry API server.
"""
function start_server(host::String="127.0.0.1", port::Int=8080)
    HTTP.serve(handle, IPv4(host), port)
end
end

if abspath(PROGRAM_FILE) == @__FILE__
    using .GeometryAPI
    GeometryAPI.start_server()
end