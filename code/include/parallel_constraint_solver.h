#ifndef PARALLEL_CONSTRAINT_SOLVER_HEADER_FILE
#define PARALLEL_CONSTRAINT_SOLVER_HEADER_FILE

#include <vector>
#include <algorithm>
#ifdef _OPENMP
#include <omp.h>
#endif

/**
 * Enumeration for different constraint solver methods
 */
enum class ConstraintSolverMethod {
    Sequential,      // Original sequential processing
    Islands,         // Process constraint islands sequentially
    ParallelIslands  // Process constraint islands in parallel
};

/**
 * Function to process constraints using parallel island-based solving
 * 
 * @param constraints The list of all constraints
 * @param constraintGraph The graph holding constraint island information
 * @param meshes The list of all meshes
 * @param maxIterations Maximum number of iterations for constraint resolution
 * @param tolerance Tolerance for constraint satisfaction
 * @param metrics Optional metrics tracking
 * @return Number of constraints resolved
 */
template<typename MeshType, typename ConstraintType, typename GraphType, typename MetricsType>
int solveConstraintsParallel(
    std::vector<ConstraintType>& constraints,
    const GraphType& constraintGraph,
    std::vector<MeshType>& meshes,
    int maxIterations,
    double tolerance,
    MetricsType* metrics = nullptr) 
{
    const auto& islands = constraintGraph.getIslands();
    int totalConstraintsResolved = 0;
    
    // Set up shared metrics counters
    int constraintIterationsCount = 0;
    int constraintsResolvedCount = 0;
    
    #ifdef _OPENMP
    #pragma omp parallel reduction(+:constraintIterationsCount, constraintsResolvedCount)
    {
        #pragma omp for schedule(dynamic)
    #endif
        for (int islandIdx = 0; islandIdx < islands.size(); islandIdx++) {
            std::vector<int> islandConstraints = 
                constraintGraph.getConstraintsForIsland(islandIdx, constraints);
            
            // Skip islands with no constraints
            if (islandConstraints.empty()) continue;
            
            // Process constraints for this island
            int currIteration = 0;
            int zeroStreak = 0;
            int currConstraintIdx = 0;
            int localConstraintsResolved = 0;
            
            // Thread-local storage for intermediate values
            std::vector<double> tmpStorage;
            
            while ((zeroStreak < islandConstraints.size()) && 
                   (currIteration * islandConstraints.size() < maxIterations)) {
                
                constraintIterationsCount++;
                
                int actualConstraintIdx = islandConstraints[currConstraintIdx];
                ConstraintType& currConstraint = constraints[actualConstraintIdx];
                
                // Process position constraint
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
                    constraintsResolvedCount++;
                    localConstraintsResolved++;
                    
                    zeroStreak = 0;
                    
                    // Update mesh positions (this is thread-safe since islands are independent)
                    #ifdef _OPENMP
                    #pragma omp critical (mesh_update)
                    #endif
                    {
                        meshes[currConstraint.m1].COM = correctedCOMPositions.row(0);
                        meshes[currConstraint.m2].COM = correctedCOMPositions.row(1);
                    }
                    
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
                        #ifdef _OPENMP
                        #pragma omp critical (mesh_update)
                        #endif
                        {
                            meshes[currConstraint.m1].comVelocity = correctedCOMVelocities.row(0);
                            meshes[currConstraint.m2].comVelocity = correctedCOMVelocities.row(1);
                            
                            meshes[currConstraint.m1].angVelocity = correctedAngVelocities.row(0);
                            meshes[currConstraint.m2].angVelocity = correctedAngVelocities.row(1);
                        }
                    }
                }
                
                currIteration++;
                currConstraintIdx = (currConstraintIdx + 1) % islandConstraints.size();
            }
            
            totalConstraintsResolved += localConstraintsResolved;
        }
    #ifdef _OPENMP
    }
    #endif
    
    // Update metrics if provided
    if (metrics) {
        metrics->constraintIterations += constraintIterationsCount;
        metrics->constraintsResolved += constraintsResolvedCount;
    }
    
    return totalConstraintsResolved;
}

#endif // PARALLEL_CONSTRAINT_SOLVER_HEADER_FILE