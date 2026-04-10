import warp as wp

@wp.kernel
def calculate_atmosphere_kernel(
    altitudes: wp.array(dtype=wp.float32), # Geometric altitude (m)
    densities: wp.array(dtype=wp.float32), # Output: rho (kg/m^3)
    pressures: wp.array(dtype=wp.float32), # Output: p (Pa)
    temperatures: wp.array(dtype=wp.float32) # Output: T (K)
):
    tid = wp.tid()
    h = altitudes[tid]
    
    # Constants for ISA (Troposphere focus < 11km)
    T0 = 288.15      # Sea level temperature (K)
    P0 = 101325.0    # Sea level pressure (Pa)
    R = 287.058      # Gas constant for air (J/kg*K)
    g = 9.80665      # Gravity (m/s^2)
    L = 0.0065       # Temperature lapse rate (K/m)
    
    # Simple Troposphere model
    if h < 11000.0:
        T = T0 - L * h
        p = P0 * wp.pow(T / T0, g / (R * L))
    else:
        # Lower Stratosphere (Isothermal layer)
        T11 = T0 - L * 11000.0
        p11 = P0 * wp.pow(T11 / T0, g / (R * L))
        h_diff = h - 11000.0
        T = T11
        p = p11 * wp.exp(-g * h_diff / (R * T11))
        
    rho = p / (R * T)
    
    densities[tid] = rho
    pressures[tid] = p
    temperatures[tid] = T

@wp.func
def get_air_density(h: wp.float32):
    # Scalar version for use inside kernels
    T0 = 288.15
    P0 = 101325.0
    R = 287.058
    g = 9.80665
    L = 0.0065
    
    if h < 11000.0:
        T = T0 - L * h
        p = P0 * wp.pow(T / T0, g / (R * L))
    else:
        T11 = T0 - L * 11000.0
        p11 = P0 * wp.pow(T11 / T0, g / (R * L))
        h_diff = h - 11000.0
        T = T11
        p = p11 * wp.exp(-g * h_diff / (R * T11))
        
    return p / (R * T)
