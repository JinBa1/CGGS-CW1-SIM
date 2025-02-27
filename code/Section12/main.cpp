#include <polyscope/polyscope.h>
#include <polyscope/surface_mesh.h>
#include <polyscope/curve_network.h>
#include "readOFF.h"
#include "scene.h"
#include <Eigen/Dense>
#include <iostream>
#include <vector>
#include <set>
#include <array>
#include <chrono>

using namespace Eigen;
using namespace std;

bool isAnimating = false;

polyscope::SurfaceMesh* pMesh;
polyscope::CurveNetwork* pConstraints;

double currTime = 0;
double timeStep = 0.02; //assuming 50 fps
double CRCoeff = 1.0;
double tolerance = 1e-3;
int maxIterations = 100000; //init value 10 000

Scene scene;

void callback_function() {
  ImGui::PushItemWidth(150);
  
  ImGui::TextUnformatted("Animation Parameters");
  ImGui::Separator();
  bool changed = ImGui::Checkbox("isAnimating", &isAnimating);
  ImGui::Checkbox("Enable Metrics", &scene.enableMetrics);

  // Acceleration methods section
  if (ImGui::CollapsingHeader("Acceleration Techniques", ImGuiTreeNodeFlags_DefaultOpen)) {
    // Collision detection acceleration
    ImGui::Text("Collision Detection:");
    int collisionMethod = static_cast<int>(scene.collisionAccelMethod);
    if (ImGui::RadioButton("Brute Force (O(n²))", &collisionMethod, 0)) {
      scene.collisionAccelMethod = Scene::CollisionAcceleration::None;
    }
    if (ImGui::RadioButton("Spatial Hash Grid", &collisionMethod, 1)) {
      scene.collisionAccelMethod = Scene::CollisionAcceleration::SpatialHash;
    }
    // Uncomment if BVH is implemented
    /*
    if (ImGui::RadioButton("Bounding Volume Hierarchy", &collisionMethod, 2)) {
      scene.collisionAccelMethod = Scene::CollisionAcceleration::BVH;
    }
    */
    
    // Spatial hash grid options
    if (scene.collisionAccelMethod == Scene::CollisionAcceleration::SpatialHash) {
      ImGui::Indent();
      ImGui::Checkbox("Auto-adjust Grid Size", &scene.autoAdjustGridSize);
      if (!scene.autoAdjustGridSize) {
        ImGui::SliderFloat("Grid Cell Size", &scene.spatialGridCellSize, 0.1f, 10.0f);
      }
      ImGui::Unindent();
    }
    
    // Constraint solver acceleration
    ImGui::Text("Constraint Solver:");
    int solverMethod = static_cast<int>(scene.constraintSolverMethod);
    if (ImGui::RadioButton("Sequential Processing", &solverMethod, 0)) {
      scene.constraintSolverMethod = Scene::ConstraintSolver::Sequential;
    }
    if (ImGui::RadioButton("Island-based Processing", &solverMethod, 1)) {
      scene.constraintSolverMethod = Scene::ConstraintSolver::Islands;
    }
    if (ImGui::RadioButton("Parallel Island Processing", &solverMethod, 2)) {
      scene.constraintSolverMethod = Scene::ConstraintSolver::ParallelIslands;
    }
    if (ImGui::RadioButton("Propagation-based Processing", &solverMethod, 3)) {
      scene.constraintSolverMethod = Scene::ConstraintSolver::PropagationIslands;
    }



  }

  // Benchmark controls
  if (ImGui::CollapsingHeader("Benchmarking", ImGuiTreeNodeFlags_DefaultOpen)) {
    if (ImGui::Button("Start 500-Frame Benchmark")) {
      scene.benchmarkRunning = true;
      scene.benchmarkFrameCount = 0;
      scene.metrics.resetAccumulativeMetrics();
    }
    
    ImGui::InputInt("Benchmark Frames", &scene.benchmarkTargetFrames);
    
    if (scene.enableMetrics) {
      if (ImGui::Button("Show Accumulative Metrics")) {
        scene.metrics.printAccumulative();
      }
      ImGui::SameLine();
      if (ImGui::Button("Reset Accumulative Metrics")) {
        scene.metrics.resetAccumulativeMetrics();
      }
    }
  }

  ImGui::PopItemWidth();

  // Run simulation if animating or benchmark is running
  if (isAnimating || scene.benchmarkRunning) {
    scene.update_scene(timeStep, CRCoeff, maxIterations, tolerance);
    
    // For benchmark, count frames and stop when target is reached
    if (scene.benchmarkRunning) {
      scene.benchmarkFrameCount++;
      if (scene.benchmarkFrameCount >= scene.benchmarkTargetFrames) {
        scene.benchmarkRunning = false;
        scene.metrics.printAccumulative();
        std::cout << "Benchmark complete: " << scene.benchmarkFrameCount 
                  << " frames in " << scene.metrics.totalRunTime << " ms" << std::endl;
        std::cout << "Average ms per frame: " 
                  << scene.metrics.totalRunTime / scene.benchmarkFrameCount << std::endl;
      }
    }
  }
  
  pMesh->updateVertexPositions(scene.currV);
  pConstraints->updateNodePositions(scene.currConstVertices);
}


int main()
{
  
  scene.load_scene("cube_6-scene.txt","cube_6-constraints.txt");
  polyscope::init();
  
  scene.update_scene(0.0, CRCoeff, maxIterations, tolerance);
  
  // Visualization
  pMesh = polyscope::registerSurfaceMesh("Entire Scene", scene.currV, scene.allF);
  pConstraints = polyscope::registerCurveNetwork("Constraints", scene.currConstVertices, scene.constEdges);
  polyscope::options::groundPlaneHeightMode = polyscope::GroundPlaneHeightMode::Manual;
  polyscope::options::groundPlaneHeight = 0.; // in world coordinates along the up axis
  polyscope::state::userCallback = callback_function;
  
  polyscope::show();
  
}

