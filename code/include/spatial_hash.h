#ifndef SPATIAL_HASH_HEADER_FILE
#define SPATIAL_HASH_HEADER_FILE

#include <unordered_map>
#include <vector>
#include <set>
#include <Eigen/Dense>
#include "mesh.h"

// A simple spatial hash grid for broad-phase collision detection
class SpatialHash {
private:
    float cellSize;
    // Hash table: maps cell indices to lists of objects in that cell
    std::unordered_map<size_t, std::vector<int>> grid;
    
    // Hash function to convert 3D cell coordinates to a single index
    size_t hashFunction(int x, int y, int z) const {
        return ((size_t)x * 73856093) ^ ((size_t)y * 19349663) ^ ((size_t)z * 83492791);
    }
    
    // Get the cell index for a 3D position
    size_t getHashIndex(const Eigen::RowVector3d& position) const {
        int x = static_cast<int>(floor(position(0) / cellSize));
        int y = static_cast<int>(floor(position(1) / cellSize));
        int z = static_cast<int>(floor(position(2) / cellSize));
        return hashFunction(x, y, z);
    }

public:
    SpatialHash(float cellSize = 2.0f) : cellSize(cellSize) {}
    
    // Set the cell size
    void setCellSize(float size) {
        cellSize = size;
    }
    
    // Get current cell size
    float getCellSize() const {
        return cellSize;
    }
    
    // Clear the grid
    void clear() {
        grid.clear();
    }
    
    // Insert a mesh into the grid using its bounding box
    void insert(int meshIndex, const Eigen::MatrixXd& vertices) {
        if (vertices.rows() == 0) return;
        
        // Find min and max coordinates of the mesh
        Eigen::RowVector3d minCoords = vertices.colwise().minCoeff();
        Eigen::RowVector3d maxCoords = vertices.colwise().maxCoeff();
        
        // Determine the range of cells the mesh spans
        int minX = static_cast<int>(floor(minCoords(0) / cellSize));
        int minY = static_cast<int>(floor(minCoords(1) / cellSize));
        int minZ = static_cast<int>(floor(minCoords(2) / cellSize));
        int maxX = static_cast<int>(floor(maxCoords(0) / cellSize));
        int maxY = static_cast<int>(floor(maxCoords(1) / cellSize));
        int maxZ = static_cast<int>(floor(maxCoords(2) / cellSize));
        
        // Insert the mesh into all cells it overlaps with
        for (int x = minX; x <= maxX; x++) {
            for (int y = minY; y <= maxY; y++) {
                for (int z = minZ; z <= maxZ; z++) {
                    size_t hash = hashFunction(x, y, z);
                    grid[hash].push_back(meshIndex);
                }
            }
        }
    }
    
    // Get potential collision pairs
    std::vector<std::pair<int, int>> getPotentialCollisionPairs() const {
        std::vector<std::pair<int, int>> pairs;
        // Use a set to avoid duplicate pairs
        std::set<std::pair<int, int>> uniquePairs;
        
        // For each cell in the grid
        for (const auto& cell : grid) {
            const std::vector<int>& objects = cell.second;
            
            // For each pair of objects in the cell
            for (size_t i = 0; i < objects.size(); i++) {
                for (size_t j = i + 1; j < objects.size(); j++) {
                    int a = objects[i];
                    int b = objects[j];
                    // Ensure consistent ordering
                    if (a > b) std::swap(a, b);
                    uniquePairs.insert(std::make_pair(a, b));
                }
            }
        }
        
        // Convert set to vector
        pairs.assign(uniquePairs.begin(), uniquePairs.end());
        return pairs;
    }
    
    // Calculate and adjust cell size based on scene objects
    void optimizeCellSize(const std::vector<Mesh>& meshes) {
        if (meshes.empty()) return;
        
        // Calculate average object size
        float totalSize = 0.0f;
        for (const Mesh& mesh : meshes) {
            Eigen::RowVector3d minCoords = mesh.currV.colwise().minCoeff();
            Eigen::RowVector3d maxCoords = mesh.currV.colwise().maxCoeff();
            Eigen::RowVector3d size = maxCoords - minCoords;
            totalSize += size.norm() / 3.0f;  // Average across dimensions
        }
        
        float avgSize = totalSize / meshes.size();
        cellSize = avgSize * 1.5f;  // 1.5 times the average size works well in practice
    }
};

#endif