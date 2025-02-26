#ifndef CONSTRAINT_GRAPH_HEADER_FILE
#define CONSTRAINT_GRAPH_HEADER_FILE

#include <vector>
#include <unordered_map>
#include <queue>
#include "constraints.h"

// Class to manage constraint relationships and detect islands
class ConstraintGraph {
private:
    // Graph representation: maps mesh indices to connected meshes via constraints
    std::unordered_map<int, std::vector<int>> adjacencyList;
    // The list of detected islands (each island is a list of mesh indices)
    std::vector<std::vector<int>> islands;
    
    // For each mesh, this maps to the island it belongs to
    std::unordered_map<int, int> meshToIsland;

public:
    // Build the constraint graph from the given constraints
    void buildGraph(const std::vector<Constraint>& constraints) {
        // Clear previous state
        adjacencyList.clear();
        islands.clear();
        meshToIsland.clear();
        
        // Add edges to the graph for each constraint
        for (const Constraint& constraint : constraints) {
            int m1 = constraint.m1;
            int m2 = constraint.m2;
            
            // Add bidirectional edge
            adjacencyList[m1].push_back(m2);
            adjacencyList[m2].push_back(m1);
        }
        
        // Identify islands using BFS
        std::unordered_map<int, bool> visited;
        int islandIndex = 0;
        
        for (const auto& entry : adjacencyList) {
            int meshIndex = entry.first;
            
            if (visited.find(meshIndex) == visited.end()) {
                // Start a new island
                std::vector<int> island;
                std::queue<int> queue;
                
                queue.push(meshIndex);
                visited[meshIndex] = true;
                
                while (!queue.empty()) {
                    int current = queue.front();
                    queue.pop();
                    
                    island.push_back(current);
                    meshToIsland[current] = islandIndex;
                    
                    for (int neighbor : adjacencyList[current]) {
                        if (visited.find(neighbor) == visited.end()) {
                            queue.push(neighbor);
                            visited[neighbor] = true;
                        }
                    }
                }
                
                islands.push_back(island);
                islandIndex++;
            }
        }
    }
    
    // Get all islands
    const std::vector<std::vector<int>>& getIslands() const {
        return islands;
    }
    
    // Get the island index for a given mesh
    int getIslandForMesh(int meshIndex) const {
        auto it = meshToIsland.find(meshIndex);
        if (it != meshToIsland.end()) {
            return it->second;
        }
        return -1; // Not in any island
    }
    
    // Get constraints related to a specific island
    std::vector<int> getConstraintsForIsland(int islandIndex, const std::vector<Constraint>& allConstraints) const {
        std::vector<int> result;
        
        if (islandIndex < 0 || islandIndex >= islands.size()) {
            return result;
        }
        
        std::unordered_map<int, bool> islandMeshes;
        for (int meshIndex : islands[islandIndex]) {
            islandMeshes[meshIndex] = true;
        }
        
        for (size_t i = 0; i < allConstraints.size(); i++) {
            const Constraint& constraint = allConstraints[i];
            // If both meshes are in this island
            if (islandMeshes.find(constraint.m1) != islandMeshes.end() && 
                islandMeshes.find(constraint.m2) != islandMeshes.end()) {
                result.push_back(i);
            }
        }
        
        return result;
    }
    
    // Get all meshes that are in some island
    std::vector<int> getAllConnectedMeshes() const {
        std::vector<int> result;
        for (const auto& island : islands) {
            result.insert(result.end(), island.begin(), island.end());
        }
        return result;
    }
};

#endif