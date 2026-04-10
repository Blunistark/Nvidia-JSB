import jsbsim
import os

fdm = jsbsim.FGFDMExec('d:\\Nvidia-JSB')
fdm.load_model('c172p')
fdm.run_ic()

with open('jsb_properties.txt', 'w') as f:
    # Recursively list common property roots
    for p in fdm.query_property_catalog("forces"):
        f.write(p + "\n")
    for p in fdm.query_property_catalog("moments"):
        f.write(p + "\n")
    for p in fdm.query_property_catalog("propulsion"):
        f.write(p + "\n")

print("Properties written to jsb_properties.txt")
