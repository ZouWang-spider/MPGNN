arcs = [1, 3, 3, 3, 12, 12, 8, 8, 12, 8, 12, 12, 3, 12, 3, 18, 18, 18, 3, 20, 18, 22, 20, 24, 22, 24, 27, 25, 27, 27, 27, 33, 33, 27, 27, 36, 27, 3]
edges = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37]

print(len(arcs))
print(len(edges))

# Iterate over arcs and edges to remove common elements
i = 0
while i < len(arcs):
    j = 0
    while j < len(edges):
        if arcs[j] == edges[j]:
            # Remove the common element from both arcs and edges
            arcs.pop(j)
            edges.pop(j)
        else:
            j += 1
    i += 1

# Print the updated lists
print(len(arcs))
print(len(edges))
