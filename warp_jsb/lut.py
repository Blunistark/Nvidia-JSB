import warp as wp
import numpy as np
import os

@wp.func
def sample_lut_1d(table: wp.array(dtype=wp.float32), 
                  meta: wp.array(dtype=wp.float32), 
                  x: wp.float32):
    # meta = [x_min, x_max]
    x_min = meta[0]
    x_max = meta[1]
    
    n = len(table)
    # Normalize x to [0, n-1]
    u = (x - x_min) / (x_max - x_min) * float(n - 1)
    u = wp.clamp(u, 0.0, float(n - 1))
    
    i = wp.int32(wp.floor(u))
    j = wp.min(i + wp.int32(1), wp.int32(n - 1))
    frac = u - float(i)
    
    return wp.lerp(table[i], table[j], frac)

@wp.func
def sample_lut_2d(table: wp.array2d(dtype=wp.float32), 
                  meta: wp.array(dtype=wp.float32), 
                  r: wp.float32, 
                  c: wp.float32):
    # meta = [r_min, r_max, c_min, c_max]
    r_min = meta[0]
    r_max = meta[1]
    c_min = meta[2]
    c_max = meta[3]
    
    h = table.shape[0]
    w = table.shape[1]
    
    u = (r - r_min) / (r_max - r_min) * float(h - 1)
    v = (c - c_min) / (c_max - c_min) * float(w - 1)
    
    u = wp.clamp(u, 0.0, float(h - 1))
    v = wp.clamp(v, 0.0, float(w - 1))
    
    i0 = wp.int32(wp.floor(u))
    i1 = wp.min(i0 + wp.int32(1), wp.int32(h - 1))
    j0 = wp.int32(wp.floor(v))
    j1 = wp.min(j0 + wp.int32(1), wp.int32(w - 1))
    
    f_u = u - float(i0)
    f_v = v - float(j0)
    
    v00 = table[i0, j0]
    v01 = table[i0, j1]
    v10 = table[i1, j0]
    v11 = table[i1, j1]
    
    return wp.lerp(wp.lerp(v00, v01, f_v), wp.lerp(v10, v11, f_v), f_u)
