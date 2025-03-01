#include "scene.h"
#include <iostream>
#include <fstream>
#include <string>
#include <chrono>

int main(int argc, char* argv[]) {
    if (argc < 6) {
        std::cerr << "Usage: " << argv[0] << " <scene-file> <constraint-file> <num-frames> "
                  << "<collision-method> <constraint-method> [thread-count] [output-file]" << std::endl;
        std::cerr << "Collision methods: 0=Brute, 1=SpatialHash" << std::endl;
        std::cerr << "Constraint methods: 0=Sequential, 1=Islands, 2=ParallelIslands, 3=PropagationIslands" << std::endl;
        return 1;
    }

    const std::string sceneFile = argv[1];
    const std::string constraintFile = argv[2];
    const int numFrames = std::stoi(argv[3]);
    const int collisionMethod = std::stoi(argv[4]);
    const int constraintMethod = std::stoi(argv[5]);
    // Default to 4 threads if not specified
    const int threadCount = (argc > 6) ? std::stoi(argv[6]) : 4;
    
    // Output file is now the 7th argument (or empty if not provided)
    const std::string outputFile = (argc > 7) ? argv[7] : "";

    Scene scene;
    
    // Configure scene
    scene.numThreads = threadCount;
    scene.enableMetrics = true;
    scene.collisionAccelMethod = static_cast<Scene::CollisionAcceleration>(collisionMethod);
    scene.constraintSolverMethod = static_cast<Scene::ConstraintSolver>(constraintMethod);
    
    // Load the scene
    if (!scene.load_scene(sceneFile, constraintFile)) {
        std::cerr << "Failed to load scene files: " << sceneFile << ", " << constraintFile << std::endl;
        return 1;
    }
    
    std::cout << "Running benchmark with scene: " << sceneFile 
              << ", collision method: " << collisionMethod 
              << ", constraint method: " << constraintMethod << std::endl;
    
    // Standard parameters
    double timeStep = 0.02; // 50 fps
    double CRCoeff = 1.0;
    double tolerance = 1e-3;
    int maxIterations = 100000000;
    
    // Reset metrics
    scene.metrics.resetAccumulativeMetrics();
    
    // Run the simulation for specified number of frames
    auto startTime = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < numFrames; i++) {
        scene.update_scene(timeStep, CRCoeff, maxIterations, tolerance);
    }

    auto endTime = std::chrono::high_resolution_clock::now();
    double totalDuration = std::chrono::duration<double, std::milli>(endTime - startTime).count();

    // Output results
    if (!outputFile.empty()) {
        std::ofstream out(outputFile, std::ios::app);
        if (out.is_open()) {
            out << sceneFile << "," << constraintFile << ","
                << numFrames << "," << collisionMethod << ","
                << constraintMethod << "," << threadCount << ","  // Add thread count
                // Use the scene metrics instead of manually calculated duration
                << scene.metrics.totalRunTime << ","
                << scene.metrics.totalFrames << ","
                << scene.metrics.accumulatedCollisionDetectionTime << ","
                << scene.metrics.accumulatedCollisionResolutionTime << ","
                << scene.metrics.accumulatedConstraintResolutionTime << ","
                << scene.metrics.totalCollisionChecks << ","
                << scene.metrics.totalCollisionsDetected << ","
                << scene.metrics.totalConstraintIterations << ","
                << scene.metrics.totalConstraintsResolved << std::endl;
        } else {
            std::cerr << "Failed to open output file: " << outputFile << std::endl;
        }
    }

    // Print summary to console
    std::cout << "=== Benchmark Results ===" << std::endl;
    std::cout << "Total run time: " << scene.metrics.totalRunTime << " ms" << std::endl;
    std::cout << "Wallclock time: " << totalDuration << " ms" << std::endl;
    std::cout << "Average frame time: " << scene.metrics.totalRunTime / numFrames << " ms" << std::endl;
    scene.metrics.printAccumulative();
    
    return 0;
}