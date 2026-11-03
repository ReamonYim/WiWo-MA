"""Independent PLY vertex projection; no SDK, ray tracing, or CSI."""
from pathlib import Path
from collections import Counter
import hashlib
import json
import argparse
import xml.etree.ElementTree as ET
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from wiwo_ma.platform.config import load
from wiwo_ma.platform.scene_geometry import scene_xml



def main():
    root=Path(__file__).resolve().parents[2]
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,default=root/'configs/platform/town02_strict_itu_64tx_40ue.json')
    parser.add_argument('--output',type=Path,default=root/'artifacts/town02/map-overview.png')
    args=parser.parse_args()
    config_path=args.config.resolve()
    output=args.output.resolve()
    output.parent.mkdir(parents=True,exist_ok=True)
    config=load(config_path)
    xml_path=scene_xml(config['scene'])
    folder=xml_path.parent
    xml=ET.parse(xml_path).getroot()
    audit=json.loads((root/config['scene']['material_audit_path']).read_text())
    palette={'mat_brick':(176,104,83),'mat_wood':(151,112,72),'mat_floorboard':(145,149,151),
             'mat_glass':(88,130,190),'mat_metal':(112,123,138)}
    materials={node.get('id'):node.find("string[@name='type']").get('value') for node in xml.findall('bsdf')}
    bindings=Counter(shape.find('ref').get('id') for shape in xml.findall('shape'))
    assert set(bindings)==set(palette)==set(materials)
    colors=list(palette.values());points=[];labels=[];faces=0
    kinds={'float':'<f4','float32':'<f4','double':'<f8','float64':'<f8','uchar':'u1','uint8':'u1','int':'<i4','int32':'<i4','uint':'<u4','uint32':'<u4'}
    for shape in xml.findall('shape'):
        file=folder/shape.find("string[@name='filename']").get('value')
        with file.open('rb') as stream:
            assert stream.readline().strip()==b'ply'
            fields=[];element=None;nvertices=0;format_name=None
            while True:
                line=stream.readline().decode('ascii').strip();pieces=line.split()
                if not pieces:continue
                if pieces[0]=='format':format_name=pieces[1]
                elif pieces[0]=='element':
                    element=pieces[1]
                    if element=='vertex':nvertices=int(pieces[2])
                    elif element=='face':faces+=int(pieces[2])
                elif pieces[0]=='property' and element=='vertex':
                    assert pieces[1]!='list'
                    fields.append((pieces[2],kinds[pieces[1]]))
                elif pieces[0]=='end_header':break
            assert format_name=='binary_little_endian',format_name
            vertices=np.fromfile(stream,dtype=np.dtype(fields),count=nvertices)
            assert len(vertices)==nvertices
            xyz=np.stack([vertices[k] for k in ('x','y','z')],axis=1)
            assert np.isfinite(xyz).all()
            points.append(xyz)
            labels.append(np.full(len(xyz),list(palette).index(shape.find('ref').get('id')),dtype=np.uint8))
    points=np.concatenate(points);labels=np.concatenate(labels)
    lo=points.min(0);hi=points.max(0);step=max(hi[0]-lo[0],hi[1]-lo[1])/1400
    xy=np.floor((points[:,:2]-lo[:2])/step).astype(int)
    width,height=xy.max(0)+1
    # Keep the highest sampled vertex in each XY cell. This is a projection, not a rendered or ray-traced image.
    flat=xy[:,1]*width+xy[:,0]
    order=np.argsort(points[:,2],kind='stable')
    image=np.full((height*width,3),245,dtype=np.uint8)
    image[flat[order]]=np.array(colors,dtype=np.uint8)[labels[order]]
    image=image.reshape(height,width,3)
    bs=np.array(config['base_stations'][0]['position_m']);ue=np.array([x['position_m'] for x in config['users']])
    fig,ax=plt.subplots(figsize=(12,10))
    fig.subplots_adjust(left=.08,right=.985,bottom=.14,top=.90)
    ax.imshow(image,origin='lower',extent=[lo[0],hi[0],lo[1],hi[1]],interpolation='nearest')
    ax.scatter(ue[:,0],ue[:,1],c='#f4a723',s=12,zorder=3,label='40 UE road anchors')
    ax.scatter(bs[0],bs[1],c='#087fba',s=110,marker='*',zorder=4,label='BS / 64 physical TX ports')
    legend=[Patch(color=np.array(c)/255,label=materials[name]+f' ({bindings[name]:,} meshes)') for name,c in palette.items()]
    handles,text=ax.get_legend_handles_labels();ax.legend(handles=legend+handles,loc='upper left',fontsize=8)
    ax.set(xlabel='Original world x / m',ylabel='Original world y / m')
    fig.suptitle(f"Town02 overview | {len(materials)} Sionna ITU material groups | {config['radio']['carrier_hz']/1e9:g} GHz",fontsize=15,fontweight='bold',y=.96)
    counts=audit['counts']
    fig.text(.5,.065,f"Source-label matched: {counts['matched_meshes']:,} meshes; {counts['changed_meshes']:,} bindings corrected. "
             f"{counts['unchanged_unverified_meshes']:,} retain unverified baseline bindings.",ha='center',fontsize=9)
    fig.text(.5,.045,'Original PLY vertex projection; colors indicate assigned radio-material groups, not optical textures or a propagation result.',ha='center',fontsize=9,color='#555555')
    ax.set_aspect('equal');fig.savefig(output,dpi=200);plt.close(fig)
    report=dict(status='Passed_PLY_vertex_inspection',config=str(config_path.relative_to(root)),scene_xml=str(xml_path.relative_to(root)),
                scene_xml_sha256=hashlib.sha256(xml_path.read_bytes()).hexdigest(),materials=materials,material_bindings=dict(bindings),material_assignment_counts=counts,
                shapes=len(xml.findall('shape')),vertices=len(points),faces_from_PLY_headers=faces,bounds_m=[lo.tolist(),hi.tolist()],BS_position_m=bs.tolist(),UE_positions_inside_map_XY_bounds=bool(np.all(ue[:,:2]>=lo[:2]) and np.all(ue[:,:2]<=hi[:2])),scope='Independent binary PLY parsing and vertex projection; colors show XML material groups. No SDK scene load, GPU tracing, native CSI or mesh clearance certification.')
    output.with_suffix('.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__ == '__main__':
    main()
