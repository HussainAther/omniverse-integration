
import omni
from pxr import UsdGeom, UsdLux, Sdf, Gf

def setup_collaborative_scene():
    # Enable Omniverse extensions if necessary
    omni.kit.app.get_app().get_extension_manager().set_extension_enabled("omni.kit.collaboration", True)

    # Access or create stage
    stage = omni.usd.get_context().get_stage()

    # Create environment with ground plane
    ground = stage.DefinePrim("/World/Ground", "Xform")
    ground.GetAttribute("xformOp:translate").Set(Gf.Vec3d(0, -10, 0))
    
    # Add light
    light = UsdLux.DistantLight.Define(stage, "/World/Light")
    light.GetIntensityAttr().Set(1000)

    # Add example textured object
    textured_object = stage.DefinePrim("/World/HighResObject", "Xform")
    textured_object.GetAttribute("xformOp:translate").Set(Gf.Vec3d(0, 0, 0))
    
    print("Collaborative 3D scene setup complete.")

if __name__ == "__main__":
    setup_collaborative_scene()
