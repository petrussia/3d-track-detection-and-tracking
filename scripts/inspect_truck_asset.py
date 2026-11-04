"""Inspect supplied FBX without assuming its actor scale in the video scene.

Optional analysis dependency: ufbx==0.0.5. No FBX loader is needed for inference.
Transforms: https://ufbx.github.io/elements/nodes/
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import ufbx


def inspect(path, output):
    scene = ufbx.load_file(str(path))
    settings = scene.settings
    unit = settings.unit_meters
    meshes, all_vertices = [], []
    # Keep native wrappers alive while reading their child arrays.
    nodes = scene.nodes
    native_refs = [scene, settings, nodes]
    for index in range(len(nodes)):
        node = nodes[index]
        mesh = node.mesh
        native_refs.extend([node, mesh])
        if mesh is None:
            continue
        matrix = np.asarray(node.geometry_to_world, dtype=float).T
        source_vertices = mesh.vertices
        native_refs.append(source_vertices)
        vertices = np.asarray([tuple(source_vertices[i]) for i in range(len(source_vertices))], dtype=float)
        world = (vertices @ matrix[:, :3].T + matrix[:, 3])*unit
        all_vertices.append(world)
        meshes.append(dict(name=node.name, vertices=len(world),
            min_m=world.min(axis=0).tolist(), max_m=world.max(axis=0).tolist(),
            node_origin_m=(np.asarray(node.node_to_world.c3)*unit).tolist()))
    vertices = np.concatenate(all_vertices)
    lo, hi = vertices.min(axis=0), vertices.max(axis=0)
    result = dict(source=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        unit_meters=unit, axes=str(scene.settings.axes),
        scene_min_m=lo.tolist(), scene_max_m=hi.tolist(),
        scene_extent_xyz_m=(hi-lo).tolist(), scene_center_m=((hi+lo)/2).tolist(),
        meshes=meshes, actor_scale_verified=False,
        caution='FBX asset-space dimensions do not establish actor scale, pivot or reference in the supplied videos.')
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding='utf-8')
    np.savez(output.with_suffix('.npz'), vertices_asset_m=vertices)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    # ufbx 0.0.5 requires child wrappers to be released before the owning scene.
    scene.free()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('asset', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    inspect(args.asset, args.output)
