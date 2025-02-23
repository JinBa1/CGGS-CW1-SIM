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

using namespace Eigen;
using namespace std;


//This class contains the entire scene operations, and the engine time loop.
class Scene{
public:
  double currTime;
  vector<Mesh> meshes;
  vector<Constraint> constraints;
  Mesh groundMesh;
  
  //Mostly for visualization
  MatrixXi allF, constEdges;
  MatrixXd currV, currConstVertices;
  
  
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

    double invMass1 = m1.totalInvMass;
    double invMass2 = m2.totalInvMass;

    double w1 = invMass1/(invMass1+invMass2);
    double w2 = invMass2/(invMass1+invMass2);

    m1.COM -= w1 * depth * contactNormal;
    m2.COM += w2 * depth * contactNormal;

    // 3) Calculate the actual contact point
    RowVector3d contactPoint = penPosition + depth * contactNormal; // Modified from w2 * depth

    // 4) Calculate moment arms
    RowVector3d arm1 = contactPoint - m1.COM; // Changed order of subtraction
    RowVector3d arm2 = contactPoint - m2.COM;

    // 5) Calculate velocities at contact point
    RowVector3d v1AtContact = m1.comVelocity + m1.angVelocity.cross(arm1);
    RowVector3d v2AtContact = m2.comVelocity + m2.angVelocity.cross(arm2);
    RowVector3d relativeVelocity = v1AtContact - v2AtContact;

    // 6) Calculate angular terms for impulse
    RowVector3d arm1CrossN = arm1.cross(contactNormal);
    RowVector3d arm2CrossN = arm2.cross(contactNormal);

    // 7) Calculate impulse denominator
    double denom = invMass1 + invMass2;
    Vector3d arm1CrossN_col = arm1CrossN.transpose();
    Vector3d arm2CrossN_col = arm2CrossN.transpose();

    Matrix3d invInertia1 = m1.get_curr_inv_IT();
    Matrix3d invInertia2 = m2.get_curr_inv_IT();

    denom += arm1CrossN.dot(invInertia1 * arm1CrossN_col);
    denom += arm2CrossN.dot(invInertia2 * arm2CrossN_col);

    // 8) Calculate impulse magnitude
    double velocityAlongNormal = relativeVelocity.dot(contactNormal);
    double j = -(1.0 + CRCoeff) * velocityAlongNormal / denom;

    // 9) Apply impulse
    RowVector3d impulse = j * contactNormal;

    // Linear velocity updates
    m1.comVelocity += invMass1 * impulse;
    m2.comVelocity -= invMass2 * impulse;

    // Angular velocity updates
    Vector3d torque1 = (arm1.cross(impulse)).transpose();
    Vector3d torque2 = (arm2.cross(impulse)).transpose();

    m1.angVelocity += (invInertia1 * torque1).transpose();
    m2.angVelocity -= (invInertia2 * torque2).transpose();
  }
  
  
  
  /*********************************************************************
   This function handles a single time step by:
   1. Integrating velocities, positions, and orientations by the timeStep
   2. detecting and handling collisions with the coefficient of restitutation CRCoeff
   3. updating the visual scene in fullV and fullT
   *********************************************************************/
  void update_scene(double timeStep, double CRCoeff, int maxIterations, double tolerance){
    
    //integrating velocity, position and orientation from forces and previous states
    for (int i=0;i<meshes.size();i++)
      meshes[i].integrate(timeStep);
    
    //detecting and handling collisions when found
    //This is done exhaustively: checking every two objects in the scene.
    double depth;
    RowVector3d contactNormal, penPosition;
    for (int i=0;i<meshes.size();i++)
      for (int j=i+1;j<meshes.size();j++)
        if (meshes[i].is_collide(meshes[j],depth, contactNormal, penPosition))
          handle_collision(meshes[i], meshes[j],depth, contactNormal, penPosition,CRCoeff);
    
    //colliding with the pseudo-mesh of the ground
    for (int i=0;i<meshes.size();i++){
      int minyIndex;
      double minY = meshes[i].currV.col(1).minCoeff(&minyIndex);
      //linear resolution
      if (minY<=0.0)
        handle_collision(meshes[i], groundMesh, minY, {0.0,1.0,0.0},meshes[i].currV.row(minyIndex),CRCoeff);
    }
    
    //Resolving constraints
    int currIteration=0;
    int zeroStreak=0;  //how many consecutive constraints are already below tolerance without any change; the algorithm stops if all are.
    int currConstIndex=0;
    while ((zeroStreak<constraints.size())&&(currIteration*constraints.size()<maxIterations)){
      
      Constraint currConstraint=constraints[currConstIndex];
      
      RowVector3d origConstPos1=meshes[currConstraint.m1].origV.row(currConstraint.v1);
      RowVector3d origConstPos2=meshes[currConstraint.m2].origV.row(currConstraint.v2);
      
      RowVector3d currConstPos1 = QRot(origConstPos1, meshes[currConstraint.m1].orientation)+meshes[currConstraint.m1].COM;
      RowVector3d currConstPos2 = QRot(origConstPos2, meshes[currConstraint.m2].orientation)+meshes[currConstraint.m2].COM;
      
      MatrixXd currCOMPositions(2,3); currCOMPositions<<meshes[currConstraint.m1].COM, meshes[currConstraint.m2].COM;
      MatrixXd currConstPositions(2,3); currConstPositions<<currConstPos1, currConstPos2;
      
      MatrixXd correctedCOMPositions;
      
      bool positionWasValid=currConstraint.resolve_position_constraint(currCOMPositions, currConstPositions,correctedCOMPositions, tolerance);
      
      if (positionWasValid){
        zeroStreak++;
      }else{
        //only update the COM and angular velocity, don't both updating all currV because it might change again during this loop!
        zeroStreak=0;
        
        meshes[currConstraint.m1].COM = correctedCOMPositions.row(0);
        meshes[currConstraint.m2].COM = correctedCOMPositions.row(1);
        
        //resolving velocity
        currConstPos1 = QRot(origConstPos1, meshes[currConstraint.m1].orientation)+meshes[currConstraint.m1].COM;
        currConstPos2 = QRot(origConstPos2, meshes[currConstraint.m2].orientation)+meshes[currConstraint.m2].COM;
        //cout<<"(currConstPos1-currConstPos2).norm(): "<<(currConstPos1-currConstPos2).norm()<<endl;
        //cout<<"(meshes[currConstraint.m1].currV.row(currConstraint.v1)-meshes[currConstraint.m2].currV.row(currConstraint.v2)).norm(): "<<(meshes[currConstraint.m1].currV.row(currConstraint.v1)-meshes[currConstraint.m2].currV.row(currConstraint.v2)).norm()<<endl;
        currCOMPositions<<meshes[currConstraint.m1].COM, meshes[currConstraint.m2].COM;
        currConstPositions<<currConstPos1, currConstPos2;
        MatrixXd currCOMVelocities(2,3); currCOMVelocities<<meshes[currConstraint.m1].comVelocity, meshes[currConstraint.m2].comVelocity;
        MatrixXd currAngVelocities(2,3); currAngVelocities<<meshes[currConstraint.m1].angVelocity, meshes[currConstraint.m2].angVelocity;
        
        Matrix3d invInertiaTensor1=meshes[currConstraint.m1].get_curr_inv_IT();
        Matrix3d invInertiaTensor2=meshes[currConstraint.m2].get_curr_inv_IT();
        MatrixXd correctedCOMVelocities, correctedAngVelocities, correctedCOMPositions;
        
        bool velocityWasValid=currConstraint.resolve_velocity_constraint(currCOMPositions, currConstPositions, currCOMVelocities, currAngVelocities, invInertiaTensor1, invInertiaTensor2, correctedCOMVelocities,correctedAngVelocities, tolerance);
        
        if (!velocityWasValid){
          meshes[currConstraint.m1].comVelocity =correctedCOMVelocities.row(0);
          meshes[currConstraint.m2].comVelocity =correctedCOMVelocities.row(1);
          
          meshes[currConstraint.m1].angVelocity =correctedAngVelocities.row(0);
          meshes[currConstraint.m2].angVelocity =correctedAngVelocities.row(1);
        }
      }
      
      currIteration++;
      currConstIndex=(currConstIndex+1)%(constraints.size());
    }
    
    if (currIteration*constraints.size()>=maxIterations)
      cout<<"Constraint resolution reached maxIterations without resolving!"<<endl;
    
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
      readMESH(DATA_PATH "/" + MESHFileName,objV,objF, objT);
      
      //fixing weird orientation problem
      MatrixXi tempF(objF.rows(),3);
      tempF<<objF.col(2), objF.col(1), objF.col(0);
      objF=tempF;
      
      add_mesh(objV,objF, objT,density, isFixed, userCOM, userOrientation);
      cout << "COM: " << userCOM <<endl;
      cout << "orientation: " << userOrientation <<endl;
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
