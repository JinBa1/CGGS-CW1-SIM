#ifndef PROPAGATION_CONSTRAINT_SOLVER_HEADER_FILE
#define PROPAGATION_CONSTRAINT_SOLVER_HEADER_FILE

#include <vector>
#include <unordered_map>
#include <unordered_set>
#include <queue>
#include <algorithm>

/**
 * Solve constraints using dependency-based propagation (single-threaded version)
 * 
 * @param constraints List of all constraints
 * @param constraintGraph Graph of constraint islands
 * @param meshes List of all meshes
 * @param maxIterations Maximum iterations for constraint resolution
 * @param tolerance Tolerance for constraint satisfaction
 * @param metrics Optional metrics tracking
 * @return Number of constraints resolved
 */
template<typename MeshType, typename ConstraintType, typename GraphType, typename MetricsType>
int solveConstraintsPropagation(
    std::vector<ConstraintType>& constraints,
    const GraphType& constraintGraph,
    std::vector<MeshType>& meshes,
    int maxIterations,
    double tolerance,
    MetricsType* metrics = nullptr) 
{
    const auto& islands = constraintGraph.getIslands();
    int totalConstraintsResolved = 0;
    int constraintIterationsCount = 0;
    
    // Process each island sequentially
    for (int islandIdx = 0; islandIdx < islands.size(); islandIdx++) {
        std::vector<int> islandConstraints = 
            constraintGraph.getConstraintsForIsland(islandIdx, constraints);
        
        // Skip islands with no constraints
        if (islandConstraints.empty()) continue;
        
        // Create dependency map: mesh index -> constraints using that mesh
        std::unordered_map<int, std::vector<int>> meshToConstraints;
        for (int constraintIdx : islandConstraints) {
            ConstraintType& c = constraints[constraintIdx];
            meshToConstraints[c.m1].push_back(constraintIdx);
            meshToConstraints[c.m2].push_back(constraintIdx);
        }
        
        // Initialize active set with all constraints
        std::unordered_set<int> processedConstraints;
        std::queue<int> constraintQueue;
        for (int c : islandConstraints) {
            constraintQueue.push(c);
        }
        
        // Process constraints until no more active ones or we hit iteration limit
        int totalIterations = 0;
        int islandConstraintsResolved = 0;
        
        while (!constraintQueue.empty() && totalIterations < maxIterations) {
            // Get next constraint to process
            int currConstraintIdx = constraintQueue.front();
            constraintQueue.pop();
            
            // Skip if we've already processed this constraint in this round
            if (processedConstraints.find(currConstraintIdx) != processedConstraints.end()) {
                continue;
            }
            
            // Mark as processed for this round
            processedConstraints.insert(currConstraintIdx);
            
            // Count this as an iteration
            constraintIterationsCount++;
            totalIterations++;
            
            ConstraintType& currConstraint = constraints[currConstraintIdx];
            
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
            
            if (!positionWasValid) {
                // Constraint was resolved - add affected constraints to queue
                islandConstraintsResolved++;
                
                // Update mesh state
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
                
                // Add all dependent constraints to the queue
                std::unordered_set<int> alreadyQueued;
                for (int meshIdx : {currConstraint.m1, currConstraint.m2}) {
                    for (int depConstraintIdx : meshToConstraints[meshIdx]) {
                        // Skip the current constraint and already queued ones
                        if (depConstraintIdx != currConstraintIdx && 
                            alreadyQueued.find(depConstraintIdx) == alreadyQueued.end()) {
                            constraintQueue.push(depConstraintIdx);
                            alreadyQueued.insert(depConstraintIdx);
                        }
                    }
                }
            }
            
            // If queue is empty but we still have active constraints, 
            // Start a new round by clearing processed set and doing another sweep
            if (constraintQueue.empty() && islandConstraintsResolved > 0 && totalIterations < maxIterations / 2) {
                // Clear the processed set for a new round
                processedConstraints.clear();
                islandConstraintsResolved = 0;
                
                // Do one more full sweep to make sure everything is stable
                for (int c : islandConstraints) {
                    constraintQueue.push(c);
                }
            }
        }
        
        if (totalIterations >= maxIterations) {
            std::cout << "Island " << islandIdx << " constraint resolution reached maxIterations!" << std::endl;
        }
        
        totalConstraintsResolved += islandConstraintsResolved;
    }
    
    // Update metrics if provided
    if (metrics) {
        metrics->constraintIterations += constraintIterationsCount;
        metrics->constraintsResolved += totalConstraintsResolved;
    }
    
    return totalConstraintsResolved;
}

#endif // PROPAGATION_CONSTRAINT_SOLVER_HEADER_FILE