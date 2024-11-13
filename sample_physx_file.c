#include <PxPhysicsAPI.h>

using namespace physx;

int main() {
    // Initialize PhysX
    PxDefaultAllocator allocator;
    PxDefaultErrorCallback errorCallback;
    PxFoundation* foundation = PxCreateFoundation(PX_PHYSICS_VERSION, allocator, errorCallback);
    PxPhysics* physics = PxCreatePhysics(PX_PHYSICS_VERSION, *foundation, PxTolerancesScale());

    // Create a scene
    PxSceneDesc sceneDesc(physics->getTolerancesScale());
    sceneDesc.gravity = PxVec3(0.0f, -9.81f, 0.0f);
    PxScene* scene = physics->createScene(sceneDesc);

    // Add a dynamic actor (e.g., a falling cube)
    PxMaterial* material = physics->createMaterial(0.5f, 0.5f, 0.6f);
    PxTransform transform(PxVec3(0.0f, 10.0f, 0.0f));
    PxRigidDynamic* cube = physics->createRigidDynamic(transform);
    PxShape* shape = PxRigidActorExt::createExclusiveShape(*cube, PxBoxGeometry(1, 1, 1), *material);
    scene->addActor(*cube);

    // Simulate
    for (int i = 0; i < 100; i++) {
        scene->simulate(1.0f / 60.0f);
        scene->fetchResults(true);
    }

    // Cleanup
    scene->release();
    physics->release();
    foundation->release();
    return 0;
}

