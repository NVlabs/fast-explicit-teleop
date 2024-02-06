# Copyright (c) 2022-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Licensed under the MIT License [see LICENSE for details].

from typing import Union
from srl.teleop.assistance.transforms import T2pq_array
import warp as wp
import warp.render
import numpy as np
import time

from pxr import Usd, UsdGeom, UsdSkel, Gf
import trimesh
import quaternion
import carb

DEVICE = wp.get_preferred_device()
#DEVICE = 'cpu'

@wp.func
def cw_min(a: wp.vec3, b: wp.vec3):

    return wp.vec3(wp.min(a[0], b[0]),
                wp.min(a[1], b[1]),
                wp.min(a[2], b[2]))

@wp.func
def cw_max(a: wp.vec3, b: wp.vec3):
    return wp.vec3(wp.max(a[0], b[0]),
                wp.max(a[1], b[1]),
                wp.max(a[2], b[2]))


@wp.kernel
def intersect(query_mesh: wp.uint64,
              query_mesh_scale: wp.float32,
              query_xforms: wp.array(dtype=wp.transform),
              fixed_mesh: wp.uint64,
              result: wp.array(dtype=int, ndim=2)):

    batch, face = wp.tid()
    # mesh_0 is assumed to be the query mesh, we launch one thread
    # for each face in mesh_0 and test it against the opposing mesh's BVH
    # transforms from query -> fixed space
    xform = query_xforms[batch]
    # load query triangles points and transform to mesh_1's space
    # Local scale is useful for checking whether the interior (roughly) of the object would overlap.
    v0 = wp.transform_point(xform, wp.mesh_eval_position(query_mesh, face, 1.0, 0.0) * query_mesh_scale)
    v1 = wp.transform_point(xform, wp.mesh_eval_position(query_mesh, face, 0.0, 1.0) * query_mesh_scale)
    v2 = wp.transform_point(xform, wp.mesh_eval_position(query_mesh, face, 0.0, 0.0) * query_mesh_scale)

    # compute bounds of the query triangle
    lower = cw_min(cw_min(v0, v1), v2)
    upper = cw_max(cw_max(v0, v1), v2)

    query = wp.mesh_query_aabb(fixed_mesh, lower, upper)
    result[batch][face] = 0

    for f in query:
        u0 = wp.mesh_eval_position(fixed_mesh, f, 1.0, 0.0)
        u1 = wp.mesh_eval_position(fixed_mesh, f, 0.0, 1.0)
        u2 = wp.mesh_eval_position(fixed_mesh, f, 0.0, 0.0)

        # test for triangle intersection
        i = wp.intersect_tri_tri(v0, v1, v2,
                                u0, u1, u2)

        if i > 0:
            result[batch][face] = 1
            return
        # use if you want to count all intersections
        #wp.atomic_add(result, batch, i)

@wp.kernel
def grasp_contacts(
              mesh_1: wp.uint64,
              left_finger_pad_point: wp.vec3,
              right_finger_pad_point: wp.vec3,
              palm_point: wp.vec3,
              xforms: wp.array(dtype=wp.transform),
              result: wp.array(dtype=float, ndim=2),
              points: wp.array(dtype=wp.vec3, ndim=2)):

    batch = wp.tid()
    # mesh_0 is assumed to be the query mesh, we launch one thread
    # for each face in mesh_0 and test it against the opposing mesh's BVH
    # transforms from mesh_0 -> mesh_1 space
    xform = xforms[batch]
    # load query triangles points and transform to mesh_1's space
    left_ray_origin = wp.transform_point(xform, left_finger_pad_point)
    right_ray_origin = wp.transform_point(xform, right_finger_pad_point)
    palm_ray_origin = wp.transform_point(xform, palm_point)

    left_ray_dir = wp.transform_vector(xform, wp.vec3(0., -1., 0.))
    right_ray_dir = wp.transform_vector(xform, wp.vec3(0., 1., 0.))
    palm_ray_dir = wp.transform_vector(xform, wp.vec3(0., 0., 1.))

    left_ray_t = float(0.)
    left_ray_sign = float(0.)
    u = float(0.)
    v = float(0.0)
    normal = wp.vec3()
    face = int(0)
    left_hit = wp.mesh_query_ray(mesh_1, left_ray_origin, left_ray_dir, .1, left_ray_t, u, v, left_ray_sign, normal, face)

    right_ray_t = float(0.)
    right_ray_sign = float(0.)
    right_hit = wp.mesh_query_ray(mesh_1, right_ray_origin, right_ray_dir, .1, right_ray_t, u, v, right_ray_sign, normal, face)

    palm_ray_t = float(100.)
    palm_ray_sign = float(0.)
    palm_hit = wp.mesh_query_ray(mesh_1, palm_ray_origin, palm_ray_dir, .04, palm_ray_t, u, v, palm_ray_sign, normal, face)

    #points[batch][0] = left_ray_origin + left_ray_t * left_ray_dir
    #points[batch][1] = right_ray_origin + right_ray_t * right_ray_dir
    #points[batch][2] = palm_ray_origin + palm_ray_t * palm_ray_dir
    result[batch][2] = palm_ray_t
    if not left_hit and right_hit:
        # Usually, _both_ rays will hit. If only one doesn't, report both as zero anyways
        # to let the outside code assume as much.
        result[batch][0] = 0.
        result[batch][1] = 0.
    else:
        result[batch][0] = left_ray_t
        result[batch][1] = right_ray_t


class WarpGeometeryScene:

    def __init__(self):
        self._warp_mesh_cache = {}
        self._trimesh_cache = {}

    def query(self, Ts, from_mesh, to_mesh, render=False, query_name=None, from_mesh_scale=1.0):
        # Transforms take "from-mesh" coordinates into "to-mesh" coordinates

        from_mesh = self._load_and_cache_geometry(from_mesh, "warp")
        to_mesh = self._load_and_cache_geometry(to_mesh, "warp")

        pq_array = T2pq_array(Ts)
        xforms = wp.array(pq_array[:, (0,1,2,4,5,6,3)], dtype=wp.transform, device=DEVICE)

        with wp.ScopedTimer("intersect", active=False):
            carb.profiler.begin(1, f"collision check (N={len(Ts)})", active=True)
            query_num_faces = len(from_mesh.indices) // 3
            shape = (len(xforms),query_num_faces)
            array_results = wp.empty(shape=shape, dtype=int, device=DEVICE)
            wp.launch(kernel=intersect, dim=shape, inputs=[from_mesh.id, from_mesh_scale, xforms, to_mesh.id, array_results], device=DEVICE)
            wp.synchronize()
            # Get num contacts per transform by summing over all faces
            results = array_results.numpy()
            if len(Ts) == 0:
                # warp 0.5.1
                results = np.empty(shape)
            results = results.sum(1)
            carb.profiler.end(1, True)

        if render:
            if query_name is None:
                query_name = str(self._get_mesh_name(to_mesh)).split("/")[-1]
            self.viz_query(results, xforms, from_mesh, to_mesh, query_name)

        return results

    def query_grasp_contacts(self, Ts, from_mesh, to_mesh, render=False, query_name=None):
        # Transforms take "from-mesh" coordinates into "to-mesh" coordinates
        carb.profiler.begin(1, "Prep meshes", active=True)
        from_mesh = self._load_and_cache_geometry(from_mesh, "warp")
        to_mesh = self._load_and_cache_geometry(to_mesh, "warp")
        carb.profiler.end(1, True)
        carb.profiler.begin(1, "Prep transforms", active=True)
        carb.profiler.begin(1, "T2pq", active=True)
        pq_array = T2pq_array(Ts)
        carb.profiler.end(1, True)
        xforms = wp.array(pq_array[:, (0,1,2,4,5,6,3)], dtype=wp.transform, device=DEVICE)
        carb.profiler.end(1, True)

        with wp.ScopedTimer("intersect_and_contact", active=False):
            carb.profiler.begin(1, f"collision and contact measure (N={len(Ts)})", active=True)
            query_num_faces = len(from_mesh.indices) // 3
            shape = (len(xforms),query_num_faces)
            contacts_shape = (len(xforms), 3)
            contact_results = wp.empty(shape=contacts_shape, dtype=float, device=DEVICE)
            points = wp.empty(shape=(len(xforms), 3), dtype=wp.vec3, device=DEVICE)
            intersect_results = wp.empty(shape=shape, dtype=int, device=DEVICE)
            wp.launch(kernel=intersect, dim=shape, inputs=[from_mesh.id, 1.0, xforms, to_mesh.id, intersect_results], device=DEVICE)
            wp.launch(kernel=grasp_contacts, dim=(len(xforms),), inputs=[to_mesh.id, (0.0, 0.04, 0.005), (0.0, -0.04, 0.005), (0.0,0.0,-0.025), xforms, contact_results, points], device=DEVICE)
            wp.synchronize()
            # Get num contacts per transform by summing over all faces
            intersections = intersect_results.numpy()
            contacts = contact_results.numpy()
            if len(Ts) == 0:
                # warp 0.5.1
                intersections = np.empty(shape)
                contacts = np.empty(shape)

            intersections = intersections.sum(1)
            carb.profiler.end(1, True)

        if render:
            if query_name is None:
                query_name = str(self._get_mesh_name(to_mesh)).split("/")[-1]
            self.viz_query(intersections, xforms, from_mesh, to_mesh, query_name, contacts=points.numpy())

        return intersections, contacts

    def viz_query(self, collisions, xforms, from_mesh, to_mesh, target_name, contacts=None):
        if len(xforms) == 0:
            return
        renderer = wp.render.UsdRenderer(f"/tmp/collision_viz/{target_name}-{time.time()}.usd", upaxis="z")
        #renderer.render_ground()
        with wp.ScopedTimer("render", active=True):
            renderer.begin_frame(0.0)
            to_mesh_points = to_mesh.points.numpy()
            to_mesh_indices = to_mesh.indices.numpy()
            from_mesh_points = from_mesh.points.numpy()
            from_mesh_indices = from_mesh.indices.numpy()

            to_extents = np.max(to_mesh_points, axis=0) - np.min(to_mesh_points, axis=0)
            spacing_x = to_extents[0] + .3
            spacing_y = to_extents[1] + .3
            row_size = int(np.sqrt(len(xforms)))
            for i, xform in enumerate(xforms.numpy()):
                x_offset = (i % row_size) * spacing_x
                y_offset = (i // row_size) * spacing_y
                renderer.render_mesh(f"to_{target_name}_{i}", points=to_mesh_points, indices=to_mesh_indices, pos=wp.vec3(x_offset, y_offset, 0))
                p, q = xform[:3], xform[3:]
                renderer.render_mesh(f"frommesh_{i}", points=from_mesh_points, indices=from_mesh_indices, pos=wp.vec3(p[0] + x_offset, p[1] + y_offset, p[2]), rot=q)

                if contacts is not None:
                    for j, contact in enumerate(contacts[i]):
                        renderer.render_sphere(f"contact_{i}_{j}", pos=wp.vec3(contact[0] + x_offset, contact[1] + y_offset, contact[2]), rot=q, radius=.01)

                # if pair intersects then draw a small box above the pair
                if collisions[i] > 0:
                    renderer.render_box(f"result_{i}", pos=wp.vec3(x_offset, y_offset, .15), rot=wp.quat_identity(), extents=(0.01, 0.01, 0.02))

            renderer.end_frame()
            renderer.save()

    def get_support_surfaces(self, geom):
        as_trimesh = self._load_and_cache_geometry(geom, "trimesh")
        facet_centroids = np.empty((len(as_trimesh.facets), 3))
        for i, (facet, total_area) in enumerate(zip(as_trimesh.facets, as_trimesh.facets_area)):
            weighted_centroid = 0
            for tri_index in facet:
                weighted_centroid += as_trimesh.area_faces[tri_index] * as_trimesh.triangles_center[tri_index]
            facet_centroids[i] = weighted_centroid / total_area
        if len(facet_centroids) == 0:
            return facet_centroids, np.empty((0,3)), as_trimesh.facets, as_trimesh.facets_area, as_trimesh.facets_boundary
        return facet_centroids, as_trimesh.facets_normal, as_trimesh.facets, as_trimesh.facets_area, as_trimesh.facets_boundary

    def combine_geometries_to_mesh(self, geoms, xforms) -> wp.Mesh:
        tri = self.combine_geometries_to_trimesh(geoms, xforms)
        mesh = warp_from_trimesh(tri)
        return mesh

    def combine_geometries_to_trimesh(self, geoms, xforms) -> trimesh.Trimesh:
        assert len(geoms) == len(xforms)
        trimeshes = [self._load_and_cache_geometry(geom, target="trimesh").copy(include_cache=True).apply_transform(xform) for geom, xform in zip(geoms, xforms)]

        tri = trimesh.util.concatenate(trimeshes)
        return tri

    def _load_and_cache_geometry(self, obj, target='warp') -> Union[wp.Mesh, trimesh.Trimesh]:
        if target == 'warp':
            if isinstance(obj, wp.Mesh):
                return obj

            cached = self._warp_mesh_cache.get(obj.GetPath(), None)
            if cached is not None:
                return cached
            else:
                # Assume that the object is a usd geom
                tri = self._load_and_cache_geometry(obj, target='trimesh')
                processed = warp_from_trimesh(tri)
                self._warp_mesh_cache[obj.GetPath()] = processed
                return processed
        elif target == "trimesh":
            if isinstance(obj, trimesh.Trimesh):
                return obj

            cached = self._trimesh_cache.get(obj.GetPath(), None)
            if cached is not None:
                return cached
            else:
                # Assume that the object is a usd geom
                tri = geom_to_trimesh(obj)
                self._trimesh_cache[obj.GetPath()] = tri
                return tri
        else:
            assert(False)

    def _get_mesh_name(self, mesh):
        return list(self._warp_mesh_cache.keys())[list(self._warp_mesh_cache.values()).index(mesh)]


def warp_from_trimesh(trimesh: trimesh.Trimesh):
    mesh = wp.Mesh(
        points=wp.array(trimesh.vertices, dtype=wp.vec3, device=DEVICE),
        indices=wp.array(trimesh.faces.flatten(), dtype=int, device=DEVICE))
    return mesh


def get_support_surfaces_trimesh(mesh: trimesh.Trimesh, for_normal=None, threshold=None):
    # No caching at the moment so don't put this in any loops
    facet_centroids = []
    if for_normal:
        scores = mesh.facets_normal.dot(for_normal)
        support_mask = scores < threshold
    else:
        support_mask = np.ones((len(mesh.facets)))
    facets = []
    for facet, total_area, is_support in zip(mesh.facets, mesh.facets_area, support_mask):
        if not is_support:
            continue
        facets.append(facet)
        weighted_centroid = 0
        for tri_index in facet:
            weighted_centroid += mesh.area_faces[tri_index] * mesh.triangles_center[tri_index]
        facet_centroids.append(weighted_centroid / total_area)
    return facets, mesh.facets_area[support_mask], np.array(facet_centroids), mesh.facets_normal[support_mask]


def geom_to_trimesh(geom):
    if isinstance(geom, UsdGeom.Mesh):
        trimesh = load_trimesh_from_usdgeom(geom)
    elif isinstance(geom, UsdGeom.Cube):
        trimesh = get_trimesh_for_cube(geom)
    elif isinstance(geom, UsdGeom.Cylinder):
        trimesh = get_trimesh_for_cylinder(geom)
    elif isinstance(geom, UsdGeom.Cone):
        trimesh = get_trimesh_for_cone(geom)
    elif isinstance(geom, UsdGeom.Sphere):
        trimesh = get_trimesh_for_sphere(geom)
    else:
        raise Exception("No mesh representation for obj" + str(geom))
    return trimesh


def get_trimesh_for_cube(cube: UsdGeom.Cube):
    transform = cube.GetLocalTransformation()
    translate, rotation, scale = UsdSkel.DecomposeTransform(transform)
    transform = Gf.Matrix4d(Gf.Vec4d(scale[0], scale[1], scale[2], 1))
    size = cube.GetSizeAttr().Get()
    baked_trimesh = trimesh.creation.box(extents=(size, size, size))
    baked_trimesh.apply_transform(transform)
    return baked_trimesh


def get_trimesh_for_cylinder(cylinder: UsdGeom.Cylinder):
    transform = cylinder.GetLocalTransformation()
    translate, rotation, scale = UsdSkel.DecomposeTransform(transform)
    transform = Gf.Matrix4d(Gf.Vec4d(scale[0], scale[1], scale[2], 1))
    baked_trimesh = trimesh.creation.cylinder(radius=cylinder.GetRadiusAttr().Get(), height=cylinder.GetHeightAttr().Get())
    baked_trimesh.apply_transform(transform)
    return baked_trimesh


def get_trimesh_for_cone(cone: UsdGeom.Cone):
    baked_trimesh = trimesh.creation.cone(radius=cone.GetRadiusAttr().Get(), height=cone.GetHeightAttr().Get())
    baked_trimesh.apply_transform(trimesh.transformations.translation_matrix([0,0,-cone.GetHeightAttr().Get() / 2]))
    return baked_trimesh


def get_trimesh_for_sphere(shpere: UsdGeom.Sphere):
    transform = shpere.GetLocalTransformation()
    baked_trimesh = trimesh.creation.icosphere(radius=shpere.GetRadiusAttr().Get())
    baked_trimesh.apply_transform(transform)
    return baked_trimesh


def load_trimesh_from_usdgeom(mesh: UsdGeom.Mesh):
    transform = mesh.GetLocalTransformation()
    baked_trimesh = trimesh.Trimesh(vertices=mesh.GetPointsAttr().Get(), faces=np.array(mesh.GetFaceVertexIndicesAttr().Get()).reshape(-1,3))
    baked_trimesh.apply_transform(transform)
    return baked_trimesh
