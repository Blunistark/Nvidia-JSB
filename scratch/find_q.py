import jsbsim
import os

fdm = jsbsim.FGFDMExec('d:\\Nvidia-JSB')
fdm.load_model('c172p')
fdm.run_ic()

print("--- ALL 'Q' PROPERTIES ---")
for p in fdm.query_property_catalog(""):
    if 'q' in p.lower():
        print(p)
