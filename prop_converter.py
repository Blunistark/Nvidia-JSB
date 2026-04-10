import xml.etree.ElementTree as ET
import numpy as np
import os

def parse_table(table_node):
    data_node = table_node.find('tableData')
    if data_node is None:
        return None
    data_str = data_node.text.strip()
    rows = []
    for line in data_str.split('\n'):
        line = line.strip()
        if not line: continue
        rows.append([float(x) for x in line.split()])
    return np.array(rows)

def convert_prop():
    tree = ET.parse('d:\\Nvidia-JSB\\engine\\prop_75in2f.xml')
    root = tree.getroot()
    
    ct_data = None
    cp_data = None
    
    for table in root.findall('table'):
        name = table.get('name')
        if name == 'C_THRUST':
            ct_data = parse_table(table)
        elif name == 'C_POWER':
            cp_data = parse_table(table)
            
    output_dir = 'd:\\Nvidia-JSB\\data\\c172p'
    os.makedirs(output_dir, exist_ok=True)
    
    def save_prop_table(filename, data):
        if data is None:
            print(f"Warning: {filename} data not found.")
            return
        # data is [J, Value]
        x_min = float(data[0, 0])
        x_max = float(data[-1, 0])
        
        vals = data[:, 1].astype(np.float32)
        meta = np.array([x_min, x_max], dtype=np.float32)
        
        target_path = os.path.join(output_dir, filename)
        np.save(f"{target_path}.npy", vals)
        np.save(f"{target_path}_meta.npy", meta)
        print(f"Saved {filename} ({len(vals)} pts, J: {x_min:.2f}->{x_max:.2f})")

    save_prop_table("C_THRUST", ct_data)
    save_prop_table("C_POWER", cp_data)

if __name__ == "__main__":
    convert_prop()
