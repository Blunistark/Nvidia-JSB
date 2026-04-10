import jsbsim
import os

ROOT_DIR = "d:/Nvidia-JSB"
AIRCRAFT = "c172p"

def extract():
    fdm = jsbsim.FGFDMExec(ROOT_DIR)
    fdm.load_model(AIRCRAFT)
    fdm.set_dt(1.0/100.0)
    
    # Matching the validation.py state
    fdm['ic/h-sl-ft'] = 5000.0
    fdm['ic/u-fps'] = 150.0
    fdm['ic/rpm'] = 2400.0
    fdm.run_ic()
    
    fdm['fcs/throttle-cmd-norm'] = 0.8
    fdm['fcs/mixture-cmd-norm'] = 0.9
    fdm['propulsion/magneto_cmd'] = 3
    fdm['propulsion/starter_cmd'] = 1
    fdm['propulsion/engine/set-running'] = 1
    
    # Warm-up phase: Run 3 seconds to let RPM stabilize
    print("Warming up engine...")
    for _ in range(300):
        fdm.run()
    
    print("Step, RPM, MAP, BHP, Thrust_lbs")
    for i in range(100):
        fdm.run()
        if i % 20 == 0:
            rpm = fdm['propulsion/engine/engine-rpm']
            map_inhg = fdm['propulsion/engine/map-inhg']
            bhp = fdm['propulsion/engine/power-hp']
            thrust = fdm['propulsion/engine/thrust-lbs']
            print(f"{i}, {rpm:.2f}, {map_inhg:.2f}, {bhp:.2f}, {thrust:.2f}")

if __name__ == "__main__":
    extract()
