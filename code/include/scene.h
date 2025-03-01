#ifndef SCENE_HEADER_FILE
#define SCENE_HEADER_FILE

#include <vector>
#include <fstream>
#include "ccd.h"
#include "volInt.h"
#include "auxfunctions.h"
#include "readMESH.h"
#include "mesh.h"
#include "constraints.h"

#include <chrono>
#include <iomanip>

#include "spatial_hash.h"
#include "constraint_graph.h"

#include "parallel_constraint_solver.h"
#include "propagation_constraint_solver.h"

using namespace Eigen;
using namespace std;


//This class contains the entire scene operations, and the engine time loop.
class Scene{
private:
  SpatialHash spatialGrid;
  ConstraintGraph constraintGraph;


public:
  double currTime;
  vector<Mesh> meshes;
  vector<Constraint> constraints;
  Mesh groundMesh;
  
  //Mostly for visualization
  MatrixXi allF, constEdges;
  MatrixXd currV, currConstVertices;

  struct SimulationMetrics {
    // Timing in milliseconds
    double totalFrameTime = 0.0;
    double collisionDetectionTime = 0.0;
    double collisionResolutionTime = 0.0;
    double constraintResolutionTime = 0.0;
    
    // Operation counts
    int collisionChecksPerformed = 0;
    int actualCollisionsDetected = 0;
    int constraintIterations = 0;
    int constraintsResolved = 0;
    
    // Accumulative metrics
    double totalRunTime = 0.0;
    double accumulatedCollisionDetectionTime = 0.0;
    double accumulatedCollisionResolutionTime = 0.0;
    double accumulatedConstraintResolutionTime = 0.0;
    
    int totalFrames = 0;
    int totalCollisionChecks = 0;
    int totalCollisionsDetected = 0;
    int totalConstraintIterations = 0;
    int totalConstraintsResolved = 0;


    
    // Modified reset method - now only resets per-frame metrics
    void reset() {
        // Only reset per-frame metrics, not accumulative ones
        totalFrameTime = 0.0;
        collisionDetectionTime = 0.0;
        collisionResolutionTime = 0.0;
        constraintResolutionTime = 0.0;
        
        collisionChecksPerformed = 0;
        actualCollisionsDetected = 0;
        constraintIterations = 0;
        constraintsResolved = 0;
    }
    
    // New method to update accumulative metrics
    void updateAccumulativeMetrics() {
        totalFrames++;
        totalRunTime += totalFrameTime;
        accumulatedCollisionDetectionTime += collisionDetectionTime;
        accumulatedCollisionResolutionTime += collisionResolutionTime;
        accumulatedConstraintResolutionTime += constraintResolutionTime;
        
        totalCollisionChecks += collisionChecksPerformed;
        totalCollisionsDetected += actualCollisionsDetected;
        totalConstraintIterations += constraintIterations;
        totalConstraintsResolved += constraintsResolved;
    }
    
    // New method to reset accumulative metrics
    void resetAccumulativeMetrics() {
        totalRunTime = 0.0;
        accumulatedCollisionDetectionTime = 0.0;
        accumulatedCollisionResolutionTime = 0.0;
        accumulatedConstraintResolutionTime = 0.0;
        
        totalFrames = 0;
        totalCollisionChecks = 0;
        totalCollisionsDetected = 0;
        totalConstraintIterations = 0;
        totalConstraintsResolved = 0;
    }

    // Print metrics - per frame
    void print() const {
        std::cout << "===== SIMULATION METRICS =====" << std::endl;
        std::cout << std::fixed << std::setprecision(3);
        std::cout << "Total frame time: " << totalFrameTime << " ms" << std::endl;
        std::cout << "  - Collision detection:   " << collisionDetectionTime << " ms" << std::endl;
        std::cout << "  - Collision resolution:  " << collisionResolutionTime << " ms" << std::endl;
        std::cout << "  - Constraint resolution: " << constraintResolutionTime << " ms" << std::endl;
        
        std::cout << "Collision checks: " << collisionChecksPerformed << std::endl;
        std::cout << "Collisions detected: " << actualCollisionsDetected << std::endl;
        std::cout << "Constraint iterations: " << constraintIterations << std::endl;
        std::cout << "Constraints resolved: " << constraintsResolved << std::endl;
        std::cout << "==============================" << std::endl;
    }
    
    // Print accumulative metrics
    void printAccumulative() const {
        std::cout << "===== ACCUMULATIVE METRICS =====" << std::endl;
        std::cout << std::fixed << std::setprecision(3);
        std::cout << "Total frames: " << totalFrames << std::endl;
        std::cout << "Total run time: " << totalRunTime << " ms" << std::endl;
        std::cout << "  - Collision detection:   " << accumulatedCollisionDetectionTime << " ms" << std::endl;
        std::cout << "  - Collision resolution:  " << accumulatedCollisionResolutionTime << " ms" << std::endl;
        std::cout << "  - Constraint resolution: " << accumulatedConstraintResolutionTime << " ms" << std::endl;
        
        std::cout << "Total collision checks: " << totalCollisionChecks << std::endl;
        std::cout << "Total collisions detected: " << totalCollisionsDetected << std::endl;
        std::cout << "Total constraint iterations: " << totalConstraintIterations << std::endl;
        std::cout << "Total constraints resolved: " << totalConstraintsResolved << std::endl;
        
        // Averages per frame
        if (totalFrames > 0) {
            std::cout << "--- Average per frame ---" << std::endl;
            std::cout << "Avg frame time: " << totalRunTime / totalFrames << " ms" << std::endl;
            std::cout << "Avg collision checks: " << (double)totalCollisionChecks / totalFrames << std::endl;
            std::cout << "Avg collisions detected: " << (double)totalCollisionsDetected / totalFrames << std::endl;
        }
        std::cout << "==============================" << std::endl;
    }
};

  SimulationMetrics metrics;
  bool enableMetrics = true;

  int benchmarkFrameCount = 0;
  int benchmarkTargetFrames = 500;
  bool benchmarkRunning = false;
  
  enum class CollisionAcceleration {
    None,            // Brute force O(n²) collision checking
    SpatialHash     // Spatial hash grid
  };

  enum class ConstraintSolver {
    Sequential,      // Process constraints sequentially (original method)
    Islands,          // Process constraints by independent islands
    ParallelIslands,   // Process islands in parallel using OpenMP
    PropagationIslands // Process islands with dependency propagation
  };
  
  // Configuration for acceleration techniques
  CollisionAcceleration collisionAccelMethod = CollisionAcceleration::None;
  ConstraintSolver constraintSolverMethod = ConstraintSolver::Sequential;
  float spatialGridCellSize = 2.0f;
  bool autoAdjustGridSize = true;

  // Parallelism settings
  int numThreads = 16; // Default number of threads to use
  
  
  //adding an objects. You do not need to update this generally
  void add_mesh(const MatrixXd& V, const MatrixXi& F, const MatrixXi& T, const double density, const bool isFixed, const RowVector3d& COM, const RowVector4d& orientation){
    
    Mesh m(V,F, T, density, isFixed, COM, orientation);
    meshes.push_back(m);
    //cout<<"m.origV.row(0): "<<m.origV.row(0)<<endl;
    //cout<<"m.currV.row(0): "<<m.currV.row(0)<<endl;
    
    MatrixXi newAllF(allF.rows()+F.rows(),3);
    newAllF<<allF, (F.array()+currV.rows()).matrix();
    allF = newAllF;
    MatrixXd newCurrV(currV.rows()+V.rows(),3);
    newCurrV<<currV, m.currV;
    currV = newCurrV;
  }
  
  /*********************************************************************
   This function handles a collision between objects ro1 and ro2 when found, by assigning impulses to both objects.
   Input: RigidObjects m1, m2
   depth: the depth of penetration
   contactNormal: the normal of the conact measured m1->m2
   penPosition: a point on m2 such that if m2 <= m2 + depth*contactNormal, then penPosition+depth*contactNormal is the common contact point
   CRCoeff: the coefficient of restitution
   *********************************************************************/
  void handle_collision(Mesh& m1, Mesh& m2,const double& depth, const RowVector3d& contactNormal,const RowVector3d& penPosition, const double CRCoeff){
    /**************TODO: implement this function**************/

    // Skip collision handling if both objects are fixed
    if (m1.isFixed && m2.isFixed) {
        return;
    }
    // // Basic collision info
    // std::cout << "\n==== COLLISION DETECTED ====\n";
    // std::cout << "m1 fixed: " << m1.isFixed << ", m2 fixed: " << m2.isFixed << std::endl;
    // std::cout << "m1 invMass: " << m1.totalInvMass << ", m2 invMass: " << m2.totalInvMass << std::endl;
    // std::cout << "Depth: " << depth << ", CRCoeff: " << CRCoeff << std::endl;
    // std::cout << "Contact normal: " << contactNormal << std::endl;
    // std::cout << "Penetration pos: " << penPosition << std::endl;

    double invMass1 = m1.totalInvMass;
    double invMass2 = m2.totalInvMass;
    double w1 = invMass1/(invMass1+invMass2);
    double w2 = invMass2/(invMass1+invMass2);

    // std::cout << "Weight distribution - w1: " << w1 << ", w2: " << w2 << std::endl;
    // std::cout << "Pre-correction m1 COM: " << m1.COM << std::endl;
    // std::cout << "Pre-correction m2 COM: " << m2.COM << std::endl;

    m1.COM -= w1 * depth * contactNormal;
    m2.COM += w2 * depth * contactNormal;

    // std::cout << "Post-correction m1 COM: " << m1.COM << std::endl;
    // std::cout << "Post-correction m2 COM: " << m2.COM << std::endl;

    // 3) Calculate the actual contact point
    RowVector3d contactPoint = penPosition + depth * contactNormal; // Modified from w2 * depth
    // std::cout << "Contact point: " << contactPoint << std::endl;


    // 4) Calculate moment arms
    RowVector3d arm1 = contactPoint - m1.COM; // Changed order of subtraction
    RowVector3d arm2 = contactPoint - m2.COM;

    // std::cout << "arm1: " << arm1 << ", length: " << arm1.norm() << std::endl;
    // std::cout << "arm2: " << arm2 << ", length: " << arm2.norm() << std::endl;

    // 5) Calculate velocities at contact point

    // // Velocities
    // std::cout << "m1 velocity: " << m1.comVelocity << std::endl;
    // std::cout << "m2 velocity: " << m2.comVelocity << std::endl;
    // std::cout << "m1 ang velocity: " << m1.angVelocity << std::endl;
    // std::cout << "m2 ang velocity: " << m2.angVelocity << std::endl;
    RowVector3d v1AtContact = m1.comVelocity + m1.angVelocity.cross(arm1);
    RowVector3d v2AtContact = m2.comVelocity + m2.angVelocity.cross(arm2);
    RowVector3d relativeVelocity = v1AtContact - v2AtContact;

    // std::cout << "v1 at contact: " << v1AtContact << std::endl;
    // std::cout << "v2 at contact: " << v2AtContact << std::endl;
    // std::cout << "Relative velocity: " << relativeVelocity << std::endl;

    // 6) Calculate angular terms for impulse
    RowVector3d arm1CrossN = arm1.cross(contactNormal);
    RowVector3d arm2CrossN = arm2.cross(contactNormal);

    // std::cout << "Checkpoint 3" << std::endl;

    // 7) Calculate impulse denominator
    double denom = invMass1 + invMass2;
    Vector3d arm1CrossN_col = arm1CrossN.transpose();
    Vector3d arm2CrossN_col = arm2CrossN.transpose();

    Matrix3d invInertia1 = m1.get_curr_inv_IT();
    Matrix3d invInertia2 = m2.get_curr_inv_IT();

    // std::cout << "m1 invIT rank: " << invInertia1.rows() << "x" << invInertia1.cols() << std::endl;
    // std::cout << "m2 invIT rank: " << invInertia2.rows() << "x" << invInertia2.cols() << std::endl;

    denom += arm1CrossN.dot(invInertia1 * arm1CrossN_col);
    denom += arm2CrossN.dot(invInertia2 * arm2CrossN_col);

    // std::cout << "arm1CrossN: " << arm1CrossN << std::endl;
    // std::cout << "arm2CrossN: " << arm2CrossN << std::endl;

    // 8) Calculate impulse magnitude
    // std::cout << "==== IMPULSE CALCULATION ====\n";
    double velocityAlongNormal = relativeVelocity.dot(contactNormal);
    double j = -(1.0 + CRCoeff) * velocityAlongNormal / denom;

    // 9) Apply impulse
    RowVector3d impulse = j * contactNormal;

    // std::cout << "==== END COLLISION HANDLING ====\n\n";

    // Linear velocity updates
    m1.comVelocity += invMass1 * impulse;
    m2.comVelocity -= invMass2 * impulse;


    // Angular velocity updates
    Vector3d torque1 = (arm1.cross(impulse)).transpose();
    Vector3d torque2 = (arm2.cross(impulse)).transpose();



    m1.angVelocity += (invInertia1 * torque1).transpose();
    m2.angVelocity -= (invInertia2 * torque2).transpose();

    // std::cout << "==== END UPDATING VELOCITIES ====" << std::endl;
  }
  
  
  
  /*********************************************************************
   This function handles a single time step by:
   1. Integrating velocities, positions, and orientations by the timeStep
   2. detecting and handling collisions with the coefficient of restitutation CRCoeff
   3. updating the visual scene in fullV and fullT
   *********************************************************************/
  void update_scene(double timeStep, double CRCoeff, int maxIterations, double tolerance){

    if (enableMetrics) metrics.reset();
    auto frameStart = std::chrono::high_resolution_clock::now();
    
    //integrating velocity, position and orientation from forces and previous states
    for (int i=0;i<meshes.size();i++)
      meshes[i].integrate(timeStep);
    
    //detecting and handling collisions when found
    auto collisionDetectionStart = std::chrono::high_resolution_clock::now();


    // Detect and handle collisions using the selected acceleration method
    switch (collisionAccelMethod) {
      case CollisionAcceleration::SpatialHash: {
        // Use spatial hashing for broad-phase collision detection
        if (autoAdjustGridSize) {
          spatialGrid.optimizeCellSize(meshes);
        } else {
          spatialGrid.setCellSize(spatialGridCellSize);
        }
        
        // Clear and rebuild the spatial grid
        spatialGrid.clear();
        for (int i = 0; i < meshes.size(); i++) {
          spatialGrid.insert(i, meshes[i].currV);
        }
        
        // Get potential collision pairs
        std::vector<std::pair<int, int>> potentialPairs = spatialGrid.getPotentialCollisionPairs();
        
        if (enableMetrics) metrics.collisionChecksPerformed += potentialPairs.size();
        
        // Process collision pairs
        double depth;
        RowVector3d contactNormal, penPosition;
        
        for (const auto& pair : potentialPairs) {
          int i = pair.first;
          int j = pair.second;
          
          if (meshes[i].is_collide(meshes[j], depth, contactNormal, penPosition)) {
            if (enableMetrics) metrics.actualCollisionsDetected++;
            auto collisionResolutionStart = std::chrono::high_resolution_clock::now();
            
            handle_collision(meshes[i], meshes[j], depth, contactNormal, penPosition, CRCoeff);
            
            if (enableMetrics) {
              auto collisionResolutionEnd = std::chrono::high_resolution_clock::now();
              metrics.collisionResolutionTime += std::chrono::duration<double, std::milli>(
                collisionResolutionEnd - collisionResolutionStart).count();
            }
          }
        }
        
        break;
      }
        
      case CollisionAcceleration::None:
      default: {
        // Original brute force O(n²) approach
        double depth;
        RowVector3d contactNormal, penPosition;
        
        for (int i=0;i<meshes.size();i++) {
          for (int j=i+1;j<meshes.size();j++) {
            if (enableMetrics) metrics.collisionChecksPerformed++;
            
            if (meshes[i].is_collide(meshes[j], depth, contactNormal, penPosition)) {
              if (enableMetrics) metrics.actualCollisionsDetected++;
              auto collisionResolutionStart = std::chrono::high_resolution_clock::now();
              
              handle_collision(meshes[i], meshes[j], depth, contactNormal, penPosition, CRCoeff);
              
              if (enableMetrics) {
                auto collisionResolutionEnd = std::chrono::high_resolution_clock::now();
                metrics.collisionResolutionTime += std::chrono::duration<double, std::milli>(
                  collisionResolutionEnd - collisionResolutionStart).count();
                }
            }
          }
        }
        break;
      }
    }
    
    //colliding with the pseudo-mesh of the ground
    for (int i=0;i<meshes.size();i++){
      int minyIndex;
      double minY = meshes[i].currV.col(1).minCoeff(&minyIndex);
      if (enableMetrics) metrics.collisionChecksPerformed++; // record checks performed

      //linear resolution
      if (minY<=0.0) {
        if (enableMetrics) metrics.actualCollisionsDetected++;

        auto groundCollisionStart = std::chrono::high_resolution_clock::now();
        handle_collision(meshes[i], groundMesh, minY, {0.0,1.0,0.0},meshes[i].currV.row(minyIndex),CRCoeff); 
        auto groundCollisionEnd = std::chrono::high_resolution_clock::now();

        if (enableMetrics) {
          metrics.collisionResolutionTime += std::chrono::duration<double, std::milli>(
              groundCollisionEnd - groundCollisionStart).count();
        }
      }
    }

    // End collision detection timing (AFTER ground collision detection)
    if (enableMetrics) {
      auto collisionDetectionEnd = std::chrono::high_resolution_clock::now();
      metrics.collisionDetectionTime += std::chrono::duration<double, std::milli>(
          collisionDetectionEnd - collisionDetectionStart).count();
    }

    // CONSTRAINT RESOLUTION TIMING
    auto constraintResolutionStart = std::chrono::high_resolution_clock::now();
    
    switch (constraintSolverMethod) {
      case ConstraintSolver::ParallelIslands: {
        // Build the constraint graph and identify islands
        constraintGraph.buildGraph(constraints);
        
        // Use the new parallel solver
        solveConstraintsParallel(constraints, constraintGraph, meshes, maxIterations, tolerance, numThreads ,&metrics);
        break;
      }

      case ConstraintSolver::PropagationIslands: {
        // Build the constraint graph and identify islands
        constraintGraph.buildGraph(constraints);
        
        // Use the new propagation-based solver
        solveConstraintsPropagation(constraints, constraintGraph, meshes, maxIterations, tolerance, &metrics);
        break;
      }

      case ConstraintSolver::Islands: {
        // Build the constraint graph and identify islands
        constraintGraph.buildGraph(constraints);
        const auto& islands = constraintGraph.getIslands();
        
        // Process each island independently
        for (size_t islandIdx = 0; islandIdx < islands.size(); islandIdx++) {
          std::vector<int> islandConstraints = constraintGraph.getConstraintsForIsland(islandIdx, constraints);
          
          // Skip islands with no constraints
          if (islandConstraints.empty()) continue;
          
          // Process constraints for this island
          int currIteration = 0;
          int zeroStreak = 0;
          int currConstraintIdx = 0;
          
          while ((zeroStreak < islandConstraints.size()) && 
                 (currIteration * islandConstraints.size() < maxIterations)) {
            
            if (enableMetrics) metrics.constraintIterations++;
            
            int actualConstraintIdx = islandConstraints[currConstraintIdx];
            Constraint& currConstraint = constraints[actualConstraintIdx];
            
            RowVector3d origConstPos1 = meshes[currConstraint.m1].origV.row(currConstraint.v1);
            RowVector3d origConstPos2 = meshes[currConstraint.m2].origV.row(currConstraint.v2);
            
            RowVector3d currConstPos1 = QRot(origConstPos1, meshes[currConstraint.m1].orientation) + 
                                         meshes[currConstraint.m1].COM;
            RowVector3d currConstPos2 = QRot(origConstPos2, meshes[currConstraint.m2].orientation) + 
                                         meshes[currConstraint.m2].COM;
            
            MatrixXd currCOMPositions(2,3); 
            currCOMPositions << meshes[currConstraint.m1].COM, meshes[currConstraint.m2].COM;
            
            MatrixXd currConstPositions(2,3); 
            currConstPositions << currConstPos1, currConstPos2;
            
            MatrixXd correctedCOMPositions;
            bool positionWasValid = currConstraint.resolve_position_constraint(
                currCOMPositions, currConstPositions, correctedCOMPositions, tolerance);
            
            if (positionWasValid) {
              zeroStreak++;
            } else {
              if (enableMetrics) metrics.constraintsResolved++;
              
              zeroStreak = 0;
              
              meshes[currConstraint.m1].COM = correctedCOMPositions.row(0);
              meshes[currConstraint.m2].COM = correctedCOMPositions.row(1);
              
              // Resolving velocity
              currConstPos1 = QRot(origConstPos1, meshes[currConstraint.m1].orientation) + 
                               meshes[currConstraint.m1].COM;
              currConstPos2 = QRot(origConstPos2, meshes[currConstraint.m2].orientation) + 
                               meshes[currConstraint.m2].COM;
              
              currCOMPositions << meshes[currConstraint.m1].COM, meshes[currConstraint.m2].COM;
              currConstPositions << currConstPos1, currConstPos2;
              
              MatrixXd currCOMVelocities(2,3); 
              currCOMVelocities << meshes[currConstraint.m1].comVelocity, meshes[currConstraint.m2].comVelocity;
              
              MatrixXd currAngVelocities(2,3); 
              currAngVelocities << meshes[currConstraint.m1].angVelocity, meshes[currConstraint.m2].angVelocity;
              
              Matrix3d invInertiaTensor1 = meshes[currConstraint.m1].get_curr_inv_IT();
              Matrix3d invInertiaTensor2 = meshes[currConstraint.m2].get_curr_inv_IT();
              
              MatrixXd correctedCOMVelocities, correctedAngVelocities;
              
              bool velocityWasValid = currConstraint.resolve_velocity_constraint(
                  currCOMPositions, currConstPositions, currCOMVelocities, currAngVelocities,
                  invInertiaTensor1, invInertiaTensor2, 
                  correctedCOMVelocities, correctedAngVelocities, tolerance);
              
              if (!velocityWasValid) {
                meshes[currConstraint.m1].comVelocity = correctedCOMVelocities.row(0);
                meshes[currConstraint.m2].comVelocity = correctedCOMVelocities.row(1);
                
                meshes[currConstraint.m1].angVelocity = correctedAngVelocities.row(0);
                meshes[currConstraint.m2].angVelocity = correctedAngVelocities.row(1);
              }
            }
            
            currIteration++;
            currConstraintIdx = (currConstraintIdx + 1) % islandConstraints.size();
          }
          
          if (currIteration * islandConstraints.size() >= maxIterations) {
            std::cout << "Island " << islandIdx << " constraint resolution reached maxIterations!" << std::endl;
          }
        }
        break;
      }
      
      case ConstraintSolver::Sequential:
      default: {
        // Original sequential constraint resolution
        int currIteration = 0;
        int zeroStreak = 0;
        int currConstIndex = 0;
        
        while ((zeroStreak < constraints.size()) && (currIteration * constraints.size() < maxIterations)) {
          if (enableMetrics) metrics.constraintIterations++;
          
          Constraint& currConstraint = constraints[currConstIndex];
          
          RowVector3d origConstPos1 = meshes[currConstraint.m1].origV.row(currConstraint.v1);
          RowVector3d origConstPos2 = meshes[currConstraint.m2].origV.row(currConstraint.v2);
          
          RowVector3d currConstPos1 = QRot(origConstPos1, meshes[currConstraint.m1].orientation) + 
                                       meshes[currConstraint.m1].COM;
          RowVector3d currConstPos2 = QRot(origConstPos2, meshes[currConstraint.m2].orientation) + 
                                       meshes[currConstraint.m2].COM;
          
          MatrixXd currCOMPositions(2,3); 
          currCOMPositions << meshes[currConstraint.m1].COM, meshes[currConstraint.m2].COM;
          
          MatrixXd currConstPositions(2,3); 
          currConstPositions << currConstPos1, currConstPos2;
          
          MatrixXd correctedCOMPositions;
          bool positionWasValid = currConstraint.resolve_position_constraint(
              currCOMPositions, currConstPositions, correctedCOMPositions, tolerance);
          
          if (positionWasValid) {
            zeroStreak++;
          } else {
            if (enableMetrics) metrics.constraintsResolved++;
            
            zeroStreak = 0;
            
            meshes[currConstraint.m1].COM = correctedCOMPositions.row(0);
            meshes[currConstraint.m2].COM = correctedCOMPositions.row(1);
            
            // Resolving velocity
            currConstPos1 = QRot(origConstPos1, meshes[currConstraint.m1].orientation) + 
                             meshes[currConstraint.m1].COM;
            currConstPos2 = QRot(origConstPos2, meshes[currConstraint.m2].orientation) + 
                             meshes[currConstraint.m2].COM;
            
            currCOMPositions << meshes[currConstraint.m1].COM, meshes[currConstraint.m2].COM;
            currConstPositions << currConstPos1, currConstPos2;
            
            MatrixXd currCOMVelocities(2,3); 
            currCOMVelocities << meshes[currConstraint.m1].comVelocity, meshes[currConstraint.m2].comVelocity;
            
            MatrixXd currAngVelocities(2,3); 
            currAngVelocities << meshes[currConstraint.m1].angVelocity, meshes[currConstraint.m2].angVelocity;
            
            Matrix3d invInertiaTensor1 = meshes[currConstraint.m1].get_curr_inv_IT();
            Matrix3d invInertiaTensor2 = meshes[currConstraint.m2].get_curr_inv_IT();
            
            MatrixXd correctedCOMVelocities, correctedAngVelocities;
            
            bool velocityWasValid = currConstraint.resolve_velocity_constraint(
                currCOMPositions, currConstPositions, currCOMVelocities, currAngVelocities,
                invInertiaTensor1, invInertiaTensor2, 
                correctedCOMVelocities, correctedAngVelocities, tolerance);
            
            if (!velocityWasValid) {
              meshes[currConstraint.m1].comVelocity = correctedCOMVelocities.row(0);
              meshes[currConstraint.m2].comVelocity = correctedCOMVelocities.row(1);
              
              meshes[currConstraint.m1].angVelocity = correctedAngVelocities.row(0);
              meshes[currConstraint.m2].angVelocity = correctedAngVelocities.row(1);
            }
          }
          
          currIteration++;
          currConstIndex = (currConstIndex + 1) % constraints.size();
        }
        
        if (currIteration * constraints.size() >= maxIterations) {
          std::cout << "Constraint resolution reached maxIterations without resolving!" << std::endl;
        }
        break;
      }
    }

    // End constraint resolution timing
    if (enableMetrics) {
      auto constraintResolutionEnd = std::chrono::high_resolution_clock::now();
      metrics.constraintResolutionTime += std::chrono::duration<double, std::milli>(
          constraintResolutionEnd - constraintResolutionStart).count();
    }
    
    
    currTime+=timeStep;
    
    //updating meshes and visualization
    for (int i=0;i<meshes.size();i++)
      for (int j=0;j<meshes[i].currV.rows();j++)
        meshes[i].currV.row(j)<<QRot(meshes[i].origV.row(j), meshes[i].orientation)+meshes[i].COM;
    
    int currVOffset=0;
    for (int i=0;i<meshes.size();i++){
      currV.block(currVOffset, 0, meshes[i].currV.rows(), 3) = meshes[i].currV;
      currVOffset+=meshes[i].currV.rows();
    }
    for (int i=0;i<constraints.size();i+=2){   //jumping bc we have constraint pairs
      currConstVertices.row(i) = meshes[constraints[i].m1].currV.row(constraints[i].v1);
      currConstVertices.row(i+1) = meshes[constraints[i].m2].currV.row(constraints[i].v2);
    }

    // End frame timing
    if (enableMetrics) {
      auto frameEnd = std::chrono::high_resolution_clock::now();
      metrics.totalFrameTime = std::chrono::duration<double, std::milli>(
          frameEnd - frameStart).count();
      
      // Print metrics
      // metrics.print();
      metrics.updateAccumulativeMetrics();
    }
  }
  
  //loading a scene from the scene .txt files
  //you do not need to update this function
  bool load_scene(const std::string sceneFileName, const std::string constraintFileName){
    
    ifstream sceneFileHandle, constraintFileHandle;
    sceneFileHandle.open(DATA_PATH "/" + sceneFileName);
    if (!sceneFileHandle.is_open())
      return false;
    int numofObjects;
    
    currTime=0;
    sceneFileHandle>>numofObjects;
    for (int i=0;i<numofObjects;i++){
      MatrixXi objT, objF;
      MatrixXd objV;
      std::string MESHFileName;
      bool isFixed;
      double density;
      RowVector3d userCOM;
      RowVector4d userOrientation;
      sceneFileHandle>>MESHFileName>>density>>isFixed>>userCOM(0)>>userCOM(1)>>userCOM(2)>>userOrientation(0)>>userOrientation(1)>>userOrientation(2)>>userOrientation(3);
      userOrientation.normalize();

      // Print info before loading
      // std::cout << "Loading mesh " << i << ": " << MESHFileName << std::endl;
      readMESH(DATA_PATH "/" + MESHFileName,objV,objF, objT);
      // std::cout << "Loaded mesh " << i << ": " << MESHFileName << std::endl;
      
      //fixing weird orientation problem
      MatrixXi tempF(objF.rows(),3);
      tempF<<objF.col(2), objF.col(1), objF.col(0);
      objF=tempF;
      
      add_mesh(objV,objF, objT,density, isFixed, userCOM, userOrientation);
      // cout << "COM: " << userCOM <<endl;
      // cout << "orientation: " << userOrientation <<endl;

      // // Print dimensions
      // std::cout << "Mesh dimensions: V(" << objV.rows() << "x" << objV.cols() 
      // << "), F(" << objF.rows() << "x" << objF.cols() 
      // << "), T(" << objT.rows() << "x" << objT.cols() << ")" << std::endl;

    }
    
    //adding ground mesh artifically
    groundMesh = Mesh(MatrixXd(0,3), MatrixXi(0,3), MatrixXi(0,4), 0.0, true, RowVector3d::Zero(), RowVector4d::Zero());
    
    //Loading constraints
    int numofConstraints;
    constraintFileHandle.open(DATA_PATH "/" + constraintFileName);
    if (!constraintFileHandle.is_open())
      return false;
    constraintFileHandle>>numofConstraints;
    currConstVertices.resize(numofConstraints*2,3);
    constEdges.resize(numofConstraints,2);
    for (int i=0;i<numofConstraints;i++){
      int attachM1, attachM2, attachV1, attachV2;
      double lowerBound, upperBound;
      constraintFileHandle>>attachM1>>attachV1>>attachM2>>attachV2>>lowerBound>>upperBound;
      //cout<<"Constraints: "<<attachM1<<","<<attachV1<<","<<attachM2<<","<<attachV2<<","<<lowerBound<<","<<upperBound<<endl;
      
      double initDist=(meshes[attachM1].currV.row(attachV1)-meshes[attachM2].currV.row(attachV2)).norm();
      //cout<<"initDist: "<<initDist<<endl;
      double invMass1 = (meshes[attachM1].isFixed ? 0.0 : meshes[attachM1].totalInvMass);  //fixed meshes have infinite mass
      double invMass2 = (meshes[attachM2].isFixed ? 0.0 : meshes[attachM2].totalInvMass);
      constraints.push_back(Constraint(DISTANCE, INEQUALITY,false, attachM1, attachV1, attachM2, attachV2, invMass1,invMass2,RowVector3d::Zero(), lowerBound*initDist, 0.0));
      constraints.push_back(Constraint(DISTANCE, INEQUALITY,true, attachM1, attachV1, attachM2, attachV2, invMass1,invMass2,RowVector3d::Zero(), upperBound*initDist, 0.0));
      currConstVertices.row(2*i) = meshes[attachM1].currV.row(attachV1);
      currConstVertices.row(2*i+1) = meshes[attachM2].currV.row(attachV2);
      constEdges.row(i)<<2*i, 2*i+1;
    }
    
    return true;
  }
  
  
  Scene(){allF.resize(0,3); currV.resize(0,3);}
  ~Scene(){}
};



#endif
