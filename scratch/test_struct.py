import warp as wp
from warp_jsb.ground_reactions import ContactPoint

wp.init()

try:
    c = ContactPoint(pos_body=wp.vec3(0,0,0), is_bogey=True, spring_k=1.0, damping_c=1.0, static_friction=1.0, dynamic_friction=1.0, max_steer=1.0)
    print("Keyword init worked")
except Exception as e:
    print(f"Keyword init failed: {e}")

try:
    c = ContactPoint()
    c.pos_body = wp.vec3(0,0,0)
    print("Default init + attribute setting worked")
except Exception as e:
    print(f"Default init failed: {e}")
