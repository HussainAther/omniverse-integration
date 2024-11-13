
import omni
from pxr import UsdPhysics, Sdf, Gf

# Basic setup for Omniverse Physics Simulation
def setup_simulation():
    # Ensure PhysX is enabled
    omni.kit.app.get_app().get_extension_manager().set_extension_enabled("omni.physx", True)
    
    # Initialize the stage
    stage = omni.usd.get_context().get_stage()
    
    # Add ground plane
    ground = stage.DefinePrim("/World/Ground", "Xform")
    ground.GetAttribute("xformOp:translate").Set(Gf.Vec3d(0, -10, 0))
    
    # Define physical properties using PhysX
    UsdPhysics.CollisionAPI.Apply(ground)
    ground.GetAttribute("collision:enabled").Set(True)
    
    print("Basic simulation setup with PhysX complete.")

if __name__ == "__main__":
    setup_simulation()
