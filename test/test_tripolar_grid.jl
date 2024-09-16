include("dependencies_for_runtests.jl")

@testset "Orthogonality of family of ellipses and hyperbolae..." begin
    # Test the grid?
    grid = TripolarGrid(size = (50, 50, 1); radius = 1)


    # Get the cartesian nodes on the faces
    cartesian_nodes, _ = get_cartesian_nodes_and_vertices(grid, Face(), Face(), Center())
    xF, yF, zF = cartesian_nodes

    o = zeros(49, 49)

    for i in 1:49, j in 1:49
        x⁻ = xF[i, j]
        y⁻ = yF[i, j]

        x⁺¹ = xF[i + 1, j]
        y⁺¹ = yF[i + 1, j]
        x⁺² = xF[i, j + 1]
        y⁺² = yF[i, j + 1]

        v1 = (x⁺¹ - x⁻, y⁺¹ - y⁻)
        v2 = (x⁺² - x⁻, y⁺² - y⁻)

        # Check orthogonality by computing scalar products:
        o[i, j] = dot(v1, v2)
    end
end