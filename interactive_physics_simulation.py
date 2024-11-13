
import omni
from pxr import UsdPhysics, Sdf, Gf, UsdGeom

def run_physics_simulation():
    # Enable necessary extensions for PhysX
    omni.kit.app.get_app().get_extension_manager().set_extension_enabled("omni.physx", True)

    # Access or create stage
    stage = omni.usd.get_context().get_stage()

    # Define ground with collision properties
    ground = stage.DefinePrim("/World/Ground", "Xform")
    UsdPhysics.CollisionAPI.Apply(ground)
    ground.GetAttribute("xformOp:translate").Set(Gf.Vec3d(0, -10, 0))

    # Define an interactive object with mass and physics properties
    ball = stage.DefinePrim("/World/Ball", "Sphere")
    ball.GetAttribute("radius").Set(1)
    UsdPhysics.RigidBodyAPI.Apply(ball)

    # Position the ball above ground to simulate a drop
    ball.GetAttribute("xformOp:translate").Set(Gf.Vec3d(0, 10, 0))

    print("Physics simulation setup complete, ready to run.")

if __name__ == "__main__":
    run_physics_simulation()
