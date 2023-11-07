def greedy_graph_coloring(graph):
    colors = {}
    available_colors = ["red", "blue", "green", "yellow"]  # Define the available color names

    for node in graph:
        used_colors = set(colors.get(neighbor, None) for neighbor in graph[node])
        for color in available_colors:
            if color not in used_colors:
                colors[node] = color
                break

    return colors

# Example usage:
graph = {
    'A': ['B', 'C'],
    'B': ['A', 'C', 'D'],
    'C': ['A', 'B', 'D'],
    'D': ['B', 'C'],
}

coloring = greedy_graph_coloring(graph)
print(coloring)