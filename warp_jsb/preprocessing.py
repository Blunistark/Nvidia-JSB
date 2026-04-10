import xml.etree.ElementTree as ET
import numpy as np
from scipy.interpolate import interp1d
import os
import json

def parse_jsbsim_table(table_node):
    ind_vars = []
    for var in table_node.findall('independentVar'):
        ind_vars.append(var.text.strip())
    
    table_data_node = table_node.find('tableData')
    if table_data_node is None: return None
    data_text = table_data_node.text.strip().split('\n')
    
    if len(ind_vars) == 1:
        rows = [list(map(float, line.split())) for line in data_text if line.strip()]
        rows = np.array(rows)
        return ind_vars, rows[:, 0], rows[:, 1]
    
    if len(ind_vars) == 2:
        lines = [line.split() for line in data_text if line.strip()]
        cols = np.array(list(map(float, lines[0])))
        rows, data = [], []
        for line in lines[1:]:
            rows.append(float(line[0]))
            data.append(list(map(float, line[1:])))
        return ind_vars, np.array(rows), cols, np.array(data)
    return None

def resample_1d(x, y, num_points=128):
    f = interp1d(x, y, kind='linear', fill_value="extrapolate")
    x_new = np.linspace(x.min(), x.max(), num_points)
    return x_new, f(x_new)

def resample_2d(rows, cols, data, num_rows=64, num_cols=64):
    from scipy.interpolate import RectBivariateSpline
    f = RectBivariateSpline(rows, cols, data, kx=1, ky=1)
    rows_new = np.linspace(rows.min(), rows.max(), num_rows)
    cols_new = np.linspace(cols.min(), cols.max(), num_cols)
    return rows_new, cols_new, f(rows_new, cols_new)

def get_function_structure(node):
    if node.tag == 'product':
        return {"type": "product", "items": [get_function_structure(c) for c in node]}
    elif node.tag == 'sum':
        return {"type": "sum", "items": [get_function_structure(c) for c in node]}
    elif node.tag == 'property':
        return {"type": "property", "name": node.text.strip()}
    elif node.tag == 'value':
        return {"type": "value", "value": float(node.text.strip())}
    elif node.tag == 'table':
        return {"type": "table"}
    return None

def process_full_model(xml_path, output_dir):
    tree = ET.parse(xml_path)
    root = tree.getroot()
    os.makedirs(output_dir, exist_ok=True)
    
    aero = root.find('aerodynamics')
    model_manifest = {"axes": {}}
    
    for axis in aero.findall('axis'):
        axis_name = axis.get('name')
        model_manifest["axes"][axis_name] = []
        
        for func in axis.findall('function'):
            func_name = func.get('name')
            print(f"Mapping: {func_name}")
            
            # Extract math structure
            structure = None
            product = func.find('product')
            if product is not None: structure = get_function_structure(product)
            sum_node = func.find('sum')
            if sum_node is not None: structure = get_function_structure(sum_node)
            
            # Extract table if exists
            table = func.find('.//table')
            table_meta = None
            if table is not None:
                res = parse_jsbsim_table(table)
                safe_name = func_name.replace('/', '_')
                if len(res) == 3:
                    xn, yn = resample_1d(res[1], res[2])
                    np.save(os.path.join(output_dir, f"{safe_name}.npy"), yn.astype(np.float32))
                    np.save(os.path.join(output_dir, f"{safe_name}_meta.npy"), np.array([res[1].min(), res[1].max()], dtype=np.float32))
                    table_meta = {"dim": 1, "var": res[0][0]}
                elif len(res) == 4:
                    rn, cn, dn = resample_2d(res[1], res[2], res[3])
                    np.save(os.path.join(output_dir, f"{safe_name}.npy"), dn.astype(np.float32))
                    np.save(os.path.join(output_dir, f"{safe_name}_meta.npy"), np.array([res[1].min(), res[1].max(), res[2].min(), res[2].max()], dtype=np.float32))
                    table_meta = {"dim": 2, "vars": res[0]}
            
            model_manifest["axes"][axis_name].append({
                "name": func_name,
                "structure": structure,
                "table_meta": table_meta
            })
            
    with open(os.path.join(output_dir, 'manifest.json'), 'w') as f:
        json.dump(model_manifest, f, indent=2)

if __name__ == "__main__":
    process_full_model('d:\\Nvidia-JSB\\c172p.xml', 'd:\\Nvidia-JSB\\data\\c172p')
    print("Full Aero Pre-processing complete.")
