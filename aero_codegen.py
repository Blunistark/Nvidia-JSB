import json
import os

def generate_aero_code(manifest_path, output_path):
    with open(manifest_path, 'r') as f:
        manifest = json.load(f)
    
    code = [
        "import warp as wp",
        "from warp_jsb.lut import sample_lut_1d, sample_lut_2d",
        "",
        "# Generated Aerodynamic Model",
        ""
    ]
    
    # Identify all tables needed
    tables = []
    for axis, functions in manifest["axes"].items():
        for func in functions:
            if func["table_meta"]:
                safe_name = func["name"].replace('/', '_')
                tables.append(safe_name)
    
    # Generate the structure to hold table handles
    code += [
        "@wp.struct",
        "class AeroModelHandles:",
    ]
    for t in tables:
        code.append(f"    {t}_table: wp.array(dtype=wp.float32)")
        code.append(f"    {t}_meta: wp.array(dtype=wp.float32)")
    code.append("")

    func_lines = [
        "@wp.func",
        "def evaluate_aero_model(",
        "    alpha: wp.float32,",
        "    beta: wp.float32,",
        "    qbar: wp.float32,",
        "    p: wp.float32,",
        "    q: wp.float32,",
        "    r: wp.float32,",
        "    bi2vel: wp.float32,",
        "    ci2vel: wp.float32,",
        "    h_mac: wp.float32,",
        "    stall_hyst: wp.float32,",
        "    elevator: wp.float32,",
        "    aileron: wp.float32,",
        "    rudder: wp.float32,",
        "    flaps: wp.float32,",
        "    handles: AeroModelHandles",
        "):",
    ]
    
    prop_map = {
        "aero/qbar-psf": "qbar",
        "aero/alpha-rad": "alpha",
        "aero/beta-rad": "beta",
        "aero/bi2vel": "bi2vel",
        "aero/ci2vel": "ci2vel",
        "aero/h_b-mac-ft": "h_mac",
        "aero/mag-beta-rad": "wp.abs(beta)",
        "aero/stall-hyst-norm": "stall_hyst",
        "aero/alphadot-rad_sec": "0.0", # Simplified
        "fcs/flap-pos-deg": "(flaps * 30.0)",
        "fcs/elevator-pos-rad": "(elevator * 0.4)",
        "fcs/mag-elevator-pos-rad": "wp.abs(elevator * 0.4)",
        "fcs/left-aileron-pos-rad": "(aileron * 0.3)",
        "fcs/rudder-pos-rad": "(rudder * 0.3)",
        "metrics/Sw-sqft": "174.0",
        "metrics/bw-ft": "35.8",
        "metrics/cbarw-ft": "4.9",
        "velocities/p-aero-rad_sec": "p",
        "velocities/q-aero-rad_sec": "q",
        "velocities/r-aero-rad_sec": "r",
        "aero/function/kCDge": "1.0", # Placeholder for nested func
        "aero/function/kCLge": "1.0",
        "aero/function/qbar-induced-psf": "qbar", # Placeholder
    }
    
    def build_expr(item, tbl_name=None):
        if item["type"] == "value": return str(item["value"])
        if item["type"] == "property": return prop_map.get(item["name"], "0.0")
        if item["type"] == "table": return f"sample_table_{tbl_name}"
        if item["type"] == "product": return "(" + " * ".join([build_expr(i, tbl_name) for i in item["items"]]) + ")"
        if item["type"] == "sum": return "(" + " + ".join([build_expr(i, tbl_name) for i in item["items"]]) + ")"
        return "0.0"

    axis_results = {}
    for axis, functions in manifest["axes"].items():
        axis_results[axis] = []
        for func in functions:
            safe_name = func["name"].replace('/', '_')
            expr = build_expr(func["structure"], safe_name)
            if func["table_meta"]:
                meta = func["table_meta"]
                if meta["dim"] == 1:
                    v = prop_map.get(meta["var"], '0.0')
                    tbl_call = f"sample_lut_1d(handles.{safe_name}_table, handles.{safe_name}_meta, {v})"
                else:
                    v1 = prop_map.get(meta["vars"][0], '0.0')
                    v2 = prop_map.get(meta["vars"][1], '0.0')
                    tbl_call = f"sample_lut_2d(handles.{safe_name}_table, handles.{safe_name}_meta, {v1}, {v2})"
                expr = expr.replace(f"sample_table_{safe_name}", tbl_call)
            
            func_lines.append(f"    {safe_name} = {expr}")
            axis_results[axis].append(safe_name)

    func_lines.append("    # Total Sums")
    for axis, funcs in axis_results.items():
        func_lines.append(f"    total_{axis} = {' + '.join(funcs)}")
    
    func_lines.append("    return total_DRAG, total_SIDE, total_LIFT, total_ROLL, total_PITCH, total_YAW")

    with open(output_path, 'w') as f:
        f.write('\n'.join(code + func_lines))

if __name__ == "__main__":
    generate_aero_code('d:\\Nvidia-JSB\\data\\c172p\\manifest.json', 'd:\\Nvidia-JSB\\warp_jsb\\aero_generated.py')
    print("Code generation complete.")
