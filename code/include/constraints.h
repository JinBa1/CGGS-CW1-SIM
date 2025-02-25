#ifndef CONSTRAINTS_HEADER_FILE
#define CONSTRAINTS_HEADER_FILE

using namespace Eigen;
using namespace std;

typedef enum ConstraintType{DISTANCE, COLLISION} ConstraintType;   //You can expand it for more constraints
typedef enum ConstraintEqualityType{EQUALITY, INEQUALITY} ConstraintEqualityType;

//there is such constraints per two variables that are equal. That is, for every attached vertex there are three such constraints for (x,y,z);
class Constraint{
public:
  
  int m1, m2;                     //Two participating meshes (can be the same)  - auxiliary data for users (constraint class shouldn't use that)
  int v1, v2;                     //Two vertices from the respective meshes - auxiliary data for users (constraint class shouldn't use that)
  double invMass1, invMass2;       //inverse masses of two bodies
  double refValue;                //Reference values to use in the constraint, when needed (like distance)
  bool isUpper;                   //in case this is an inequality constraints, whether it's an upper or a lower bound
  RowVector3d refVector;             //Reference vector when needed (like vector)
  double CRCoeff;                 //velocity bias
  ConstraintType constraintType;  //The type of the constraint, and will affect the value and the gradient. This SHOULD NOT change after initialization!
  ConstraintEqualityType constraintEqualityType;  //whether the constraint is an equality or an inequality
  
  Constraint(const ConstraintType _constraintType, const ConstraintEqualityType _constraintEqualityType, const bool _isUpper, const int& _m1, const int& _v1, const int& _m2, const int& _v2, const double& _invMass1, const double& _invMass2, const RowVector3d& _refVector, const double& _refValue, const double& _CRCoeff):constraintType(_constraintType), constraintEqualityType(_constraintEqualityType), isUpper(_isUpper), m1(_m1), v1(_v1), m2(_m2), v2(_v2), invMass1(_invMass1), invMass2(_invMass2),  refValue(_refValue), CRCoeff(_CRCoeff){
    refVector=_refVector;
  }
  
  ~Constraint(){}
  
  
  
  //computes the impulse needed for all particles to resolve the velocity constraint, and corrects the velocities accordingly.
  //The velocities are a vector (vCOM1, w1, vCOM2, w2) in both input and output.
  //returns true if constraint was already valid with "currVelocities", and false otherwise (false means there was a correction done)
  bool resolve_velocity_constraint(const MatrixXd& currCOMPositions, const MatrixXd& currVertexPositions,
                                   const MatrixXd& currCOMVelocities, const MatrixXd& currAngVelocities,
                                   const Matrix3d& invInertiaTensor1, const Matrix3d& invInertiaTensor2,
                                   MatrixXd& correctedCOMVelocities, MatrixXd& correctedAngVelocities, double tolerance){
    
    
    /***************************TODO: implement this function**********************/
    // Check matrix dimensions
    if (currVertexPositions.rows() < 2 || currCOMPositions.rows() < 2 ||
        currCOMVelocities.rows() < 2 || currAngVelocities.rows() < 2) {
        // If matrices don't have enough rows, return without changes
        correctedCOMVelocities = currCOMVelocities;
        correctedAngVelocities = currAngVelocities;
        return true;
    }

    // Extract constrained positions and convert to fixed-size vectors for cross products
    Vector3d p1 = currVertexPositions.row(0).transpose();
    Vector3d p2 = currVertexPositions.row(1).transpose();

    // Calculate constraint gradient (normalized direction)
    Vector3d J = (p1 - p2).normalized();

    // Calculate moment arms from COM to constraint points
    Vector3d arm1 = p1 - currCOMPositions.row(0).transpose();
    Vector3d arm2 = p2 - currCOMPositions.row(1).transpose();

    // Convert angular velocities to fixed-size vectors for cross products
    Vector3d w1 = currAngVelocities.row(0).transpose();
    Vector3d w2 = currAngVelocities.row(1).transpose();
    Vector3d v1 = currCOMVelocities.row(0).transpose();
    Vector3d v2 = currCOMVelocities.row(1).transpose();

    // Compute current velocities at the constraint points
    Vector3d v1AtConstraint = v1 + w1.cross(arm1);
    Vector3d v2AtConstraint = v2 + w2.cross(arm2);

    // Calculate relative velocity at constraint points
    Vector3d relativeVelocity = v1AtConstraint - v2AtConstraint;

    // Check if constraint is already satisfied
    double violation = J.dot(relativeVelocity);
    if (std::abs(violation) <= tolerance) {
        correctedCOMVelocities = currCOMVelocities;
        correctedAngVelocities = currAngVelocities;
        return true;
    }

    // Calculate denominator terms
    double denom = invMass1 + invMass2;

    // Calculate cross products for angular terms
    Vector3d armJ1 = arm1.cross(J);
    Vector3d armJ2 = arm2.cross(J);

    // Add angular terms to denominator
    denom += armJ1.transpose() * invInertiaTensor1 * armJ1;
    denom += armJ2.transpose() * invInertiaTensor2 * armJ2;

    // Check if correction is possible
    if (denom == 0) {
        correctedCOMVelocities = currCOMVelocities;
        correctedAngVelocities = currAngVelocities;
        return true;
    }

    // Calculate Lagrange multiplier
    double lambda = -violation / denom;

    // Calculate velocity corrections
    Vector3d deltaV1 = lambda * invMass1 * J;
    Vector3d deltaV2 = -lambda * invMass2 * J;

    // Calculate angular velocity corrections
    Vector3d deltaW1 = invInertiaTensor1 * (lambda * armJ1);
    Vector3d deltaW2 = -invInertiaTensor2 * (lambda * armJ2);

    // Apply corrections
    correctedCOMVelocities = currCOMVelocities;
    correctedAngVelocities = currAngVelocities;

    correctedCOMVelocities.row(0) += deltaV1.transpose();
    correctedCOMVelocities.row(1) += deltaV2.transpose();

    correctedAngVelocities.row(0) += deltaW1.transpose();
    correctedAngVelocities.row(1) += deltaW2.transpose();

    return false;
    
    // //stub implementation
    // correctedCOMVelocities = currCOMVelocities;
    // correctedAngVelocities = currAngVelocities;
    // return true;
  }
  
  //projects the position unto the constraint
  //returns true if constraint was already good
  bool resolve_position_constraint(const MatrixXd& currCOMPositions, const MatrixXd& currConstPositions,
                                   MatrixXd& correctedCOMPositions, double tolerance){
    
    /***************************TODO: implement this function**********************/
    // Check matrix dimensions
    if (currConstPositions.rows() < 2 || currCOMPositions.rows() < 2) {
      correctedCOMPositions = currCOMPositions;
      return true;
    }

    // Extract constrained positions
    RowVector3d p1 = currConstPositions.row(0);
    RowVector3d p2 = currConstPositions.row(1);

    double displacement = (p1-p2).norm();
    double diff = displacement - refValue;
    bool isValid = isUpper ? diff <= tolerance : diff >= -tolerance;
    if(isValid) {
      correctedCOMPositions = currCOMPositions;
      return true;
    }

    RowVector3d J = (p1 - p2) / displacement;

    double denom = invMass1 + invMass2;

    if (denom == 0) {
      correctedCOMPositions = currCOMPositions;
      return true;
    }

    double lambda = -diff / denom;

    RowVector3d deltaP1 = lambda * invMass1 * J;
    RowVector3d deltaP2 = -lambda * invMass2 * J;

    correctedCOMPositions = currCOMPositions;
    correctedCOMPositions.row(0) += deltaP1;
    correctedCOMPositions.row(1) += deltaP2;

    return false;


    // //stub implementation
    // correctedCOMPositions = currCOMPositions;
    // return true;
  }
  
};



#endif /* constraints_h */
