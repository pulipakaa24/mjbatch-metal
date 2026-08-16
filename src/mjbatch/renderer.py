"""BatchRenderer — tiled, instanced MuJoCo observation rendering on Metal.

One frame = one background pass + one instanced geometry draw for ALL envs:
- an atlas of tiles, each tile one environment's camera view
- per-(env, geom) world transforms + RGBA colors in a storage buffer
- per-env view-projection, lighting, and tile rect in a storage buffer
- per-env background images in a texture array (greenscreen compositing is
  done in-shader: the color attachment IS the composited observation)
- fragment shader discards outside the instance's tile (prevents spill)

Geometry is uploaded once at construction (unindexed, flat per-face normals),
with optional quadric decimation of dense visual meshes (recommended: the
renderer is vertex-bound otherwise).

Fidelity scope (by design): Lambertian flat shading, no shadows, no textures.
Built for domain-randomized RL observations composited over backgrounds, not
for photorealism.
"""
from __future__ import annotations

import numpy as np
import mujoco
import wgpu

_SHADER = """
struct EnvData {
    view_proj: mat4x4<f32>,
    tile: vec4<f32>,
    light: vec4<f32>,
};
struct GeomXform {
    row0: vec4<f32>,
    row1: vec4<f32>,
    row2: vec4<f32>,
    color: vec4<f32>,
};
@group(0) @binding(0) var<storage, read> envs: array<EnvData>;
@group(0) @binding(1) var<storage, read> xforms: array<GeomXform>;
@group(0) @binding(2) var bg_tex: texture_2d_array<f32>;
@group(0) @binding(3) var bg_samp: sampler;

struct VSIn {
    @location(0) pos: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) geom_id: u32,
    @builtin(instance_index) env: u32,
};
struct VSOut {
    @builtin(position) clip: vec4<f32>,
    @location(0) normal_w: vec3<f32>,
    @location(1) color: vec4<f32>,
    @location(2) @interpolate(flat) env: u32,
};

const N_GEOMS: u32 = {N_GEOMS}u;

@vertex
fn vs_main(in: VSIn) -> VSOut {
    let e = envs[in.env];
    let g = xforms[in.env * N_GEOMS + in.geom_id];
    let p = vec3<f32>(
        dot(g.row0.xyz, in.pos) + g.row0.w,
        dot(g.row1.xyz, in.pos) + g.row1.w,
        dot(g.row2.xyz, in.pos) + g.row2.w,
    );
    let n = vec3<f32>(
        dot(g.row0.xyz, in.normal),
        dot(g.row1.xyz, in.normal),
        dot(g.row2.xyz, in.normal),
    );
    var clip = e.view_proj * vec4<f32>(p, 1.0);
    clip = vec4<f32>(clip.x * e.tile.z + e.tile.x * clip.w,
                     clip.y * e.tile.w + e.tile.y * clip.w,
                     clip.z, clip.w);
    var out: VSOut;
    out.clip = clip;
    out.normal_w = n;
    out.color = g.color;
    out.env = in.env;
    return out;
}

@fragment
fn fs_main(in: VSOut) -> @location(0) vec4<f32> {
    let e = envs[in.env];
    let tile_w = {TILE_W}.0;
    let tile_h = {TILE_H}.0;
    let col = f32(in.env % {TPR}u);
    let row = f32(in.env / {TPR}u);
    if (in.clip.x < col * tile_w || in.clip.x >= (col + 1.0) * tile_w ||
        in.clip.y < row * tile_h || in.clip.y >= (row + 1.0) * tile_h) {
        discard;
    }
    let n = normalize(in.normal_w);
    let ndl = clamp(dot(n, -e.light.xyz), 0.0, 1.0);
    let shade = 0.35 + 0.65 * ndl;
    return vec4<f32>(in.color.rgb * shade, 1.0);
}

struct BGOut {
    @builtin(position) clip: vec4<f32>,
    @location(0) uv: vec2<f32>,
    @location(1) @interpolate(flat) env: u32,
};

@vertex
fn bg_vs(@builtin(vertex_index) vi: u32, @builtin(instance_index) env: u32) -> BGOut {
    var quad = array<vec2<f32>, 6>(
        vec2(-1.0, -1.0), vec2(1.0, -1.0), vec2(1.0, 1.0),
        vec2(-1.0, -1.0), vec2(1.0, 1.0), vec2(-1.0, 1.0));
    let q = quad[vi];
    let e = envs[env];
    var out: BGOut;
    out.clip = vec4<f32>(q.x * e.tile.z + e.tile.x, q.y * e.tile.w + e.tile.y, 0.9999, 1.0);
    out.uv = vec2<f32>(q.x * 0.5 + 0.5, 0.5 - q.y * 0.5);
    out.env = env;
    return out;
}

@fragment
fn bg_fs(in: BGOut) -> @location(0) vec4<f32> {
    return textureSampleLevel(bg_tex, bg_samp, in.uv, i32(in.env), 0.0);
}
"""

_BOX_FACES = np.array([
    [[1,-1,-1],[1, 1,-1],[1, 1, 1]], [[1,-1,-1],[1, 1, 1],[1,-1, 1]],
    [[-1,-1,-1],[-1, 1, 1],[-1, 1,-1]], [[-1,-1,-1],[-1,-1, 1],[-1, 1, 1]],
    [[-1, 1,-1],[-1, 1, 1],[ 1, 1, 1]], [[-1, 1,-1],[ 1, 1, 1],[ 1, 1,-1]],
    [[-1,-1,-1],[ 1,-1,-1],[ 1,-1, 1]], [[-1,-1,-1],[ 1,-1, 1],[-1,-1, 1]],
    [[-1,-1, 1],[ 1,-1, 1],[ 1, 1, 1]], [[-1,-1, 1],[ 1, 1, 1],[-1, 1, 1]],
    [[-1,-1,-1],[-1, 1,-1],[ 1, 1,-1]], [[-1,-1,-1],[ 1, 1,-1],[ 1,-1,-1]],
], dtype=np.float64)


def _uv_sphere(radius, n_lat=8, n_lon=12):
    tris = []
    for i in range(n_lat):
        t0, t1 = np.pi * i / n_lat, np.pi * (i + 1) / n_lat
        for j in range(n_lon):
            p0, p1 = 2 * np.pi * j / n_lon, 2 * np.pi * (j + 1) / n_lon
            a = [np.sin(t0)*np.cos(p0), np.sin(t0)*np.sin(p0), np.cos(t0)]
            b = [np.sin(t1)*np.cos(p0), np.sin(t1)*np.sin(p0), np.cos(t1)]
            c = [np.sin(t1)*np.cos(p1), np.sin(t1)*np.sin(p1), np.cos(t1)]
            d = [np.sin(t0)*np.cos(p1), np.sin(t0)*np.sin(p1), np.cos(t0)]
            tris.append([a, b, c]); tris.append([a, c, d])
    return np.asarray(tris) * radius


def _cylinder(radius, half_len, n=16):
    tris = []
    for j in range(n):
        p0, p1 = 2*np.pi*j/n, 2*np.pi*(j+1)/n
        a = [radius*np.cos(p0), radius*np.sin(p0), -half_len]
        b = [radius*np.cos(p1), radius*np.sin(p1), -half_len]
        c = [radius*np.cos(p1), radius*np.sin(p1),  half_len]
        d = [radius*np.cos(p0), radius*np.sin(p0),  half_len]
        tris += [[a, b, c], [a, c, d]]
        tris += [[[0,0,half_len], d, c], [[0,0,-half_len], b, a]]
    return np.asarray(tris)


class BatchRenderer:
    """Tiled batch renderer for N MuJoCo environments sharing one model.

    Parameters
    ----------
    model : mujoco.MjModel
    n_envs : int
    width, height : per-env tile resolution (default 128x128)
    camera : camera name used for the projection matrix (fovy); per-env pose
        is passed at render time. Defaults to the model's first camera.
    max_group : draw geoms with geom_group < max_group (MuJoCo convention:
        visual geoms in low groups, collision in group 3).
    include_planes : draw plane geoms (default False — with compositing the
        background replaces the floor).
    decimate_faces : per-mesh face budget (0 disables decimation; requires
        fast_simplification when enabled and a mesh exceeds the budget).
    """

    def __init__(self, model: mujoco.MjModel, n_envs: int, *,
                 width: int = 128, height: int = 128, camera: str | None = None,
                 max_group: int = 3, include_planes: bool = False,
                 decimate_faces: int = 1500):
        self.m = model
        self.n = n_envs
        self.tw, self.th = width, height
        self.tpr = int(np.ceil(np.sqrt(n_envs)))
        self.aw = self.tpr * width
        self.ah = int(np.ceil(n_envs / self.tpr)) * height
        if camera is None:
            if model.ncam == 0:
                raise ValueError("model has no cameras; pass fovy via a camera")
            self._proj_cam = 0
        else:
            self._proj_cam = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, camera)
            if self._proj_cam < 0:
                raise ValueError(f"camera {camera!r} not found")
        adapter = wgpu.gpu.request_adapter_sync(power_preference="high-performance")
        self.device = adapter.request_device_sync()
        self._build_geometry(max_group, include_planes, decimate_faces)
        self._build_pipeline()

    # -- geometry ---------------------------------------------------------

    def _build_geometry(self, max_group, include_planes, decimate_faces):
        m = self.m
        verts, norms, gids = [], [], []
        self.geoms = []
        for g in range(m.ngeom):
            if m.geom_group[g] >= max_group:
                continue
            t = int(m.geom_type[g])
            size = m.geom_size[g]
            if t == mujoco.mjtGeom.mjGEOM_MESH:
                mid = int(m.geom_dataid[g])
                va, vn = int(m.mesh_vertadr[mid]), int(m.mesh_vertnum[mid])
                fa, fn = int(m.mesh_faceadr[mid]), int(m.mesh_facenum[mid])
                v_full = m.mesh_vert[va:va + vn].astype(np.float64)
                f_full = m.mesh_face[fa:fa + fn].astype(np.int64)
                if decimate_faces and len(f_full) > decimate_faces:
                    import fast_simplification
                    v_d, f_d = fast_simplification.simplify(
                        v_full, f_full,
                        target_reduction=1.0 - decimate_faces / len(f_full))
                    tri = np.asarray(v_d)[np.asarray(f_d)]
                else:
                    tri = v_full[f_full]
            elif t == mujoco.mjtGeom.mjGEOM_BOX:
                tri = _BOX_FACES * size
            elif t == mujoco.mjtGeom.mjGEOM_SPHERE:
                tri = _uv_sphere(size[0])
            elif t == mujoco.mjtGeom.mjGEOM_CYLINDER:
                tri = _cylinder(size[0], size[1])
            elif t == mujoco.mjtGeom.mjGEOM_CAPSULE:
                # capsule approximated as cylinder + end spheres
                tri = np.concatenate([
                    _cylinder(size[0], size[1]),
                    _uv_sphere(size[0]) + [0, 0, size[1]],
                    _uv_sphere(size[0]) - [0, 0, size[1]]])
            elif t == mujoco.mjtGeom.mjGEOM_PLANE:
                if not include_planes:
                    continue
                s = 1.0
                tri = np.array([[[-s,-s,0],[ s,-s,0],[ s, s,0]],
                                [[-s,-s,0],[ s, s,0],[-s, s,0]]], dtype=np.float64)
            else:
                continue
            n = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
            n /= (np.linalg.norm(n, axis=1, keepdims=True) + 1e-12)
            verts.append(tri.reshape(-1, 3))
            norms.append(np.repeat(n, 3, axis=0))
            gids.append(np.full(tri.shape[0] * 3, len(self.geoms), np.uint32))
            self.geoms.append(g)
        if not self.geoms:
            raise ValueError("no drawable geoms found (check max_group)")
        v = np.concatenate(verts).astype(np.float32)
        nrm = np.concatenate(norms).astype(np.float32)
        gid = np.concatenate(gids)
        self.n_verts = len(v)
        self.G = len(self.geoms)
        inter = np.zeros((self.n_verts, 7), np.float32)
        inter[:, 0:3] = v
        inter[:, 3:6] = nrm
        inter[:, 6] = gid.view(np.float32)
        self.vbuf = self.device.create_buffer_with_data(
            data=inter.tobytes(), usage=wgpu.BufferUsage.VERTEX)

    def default_colors(self) -> np.ndarray:
        """(G, 4) material-resolved RGBA per drawn geom (MuJoCo semantics)."""
        m = self.m
        out = np.zeros((self.G, 4), np.float32)
        for i, g in enumerate(self.geoms):
            mat = int(m.geom_matid[g])
            out[i] = m.mat_rgba[mat] if mat >= 0 else m.geom_rgba[g]
        return out

    # -- pipeline ---------------------------------------------------------

    def _build_pipeline(self):
        dev = self.device
        src = (_SHADER.replace("{N_GEOMS}", str(self.G))
               .replace("{TPR}", str(self.tpr))
               .replace("{TILE_W}", str(self.tw))
               .replace("{TILE_H}", str(self.th)))
        shader = dev.create_shader_module(code=src)
        self.ebuf = dev.create_buffer(size=self.n * 96,
                                      usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_DST)
        self.xbuf = dev.create_buffer(size=self.n * self.G * 64,
                                      usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_DST)
        self.bg_tex = dev.create_texture(
            size=(self.tw, self.th, self.n), format=wgpu.TextureFormat.rgba8unorm,
            usage=wgpu.TextureUsage.TEXTURE_BINDING | wgpu.TextureUsage.COPY_DST)
        samp = dev.create_sampler(mag_filter=wgpu.FilterMode.linear,
                                  min_filter=wgpu.FilterMode.linear)
        bgl = dev.create_bind_group_layout(entries=[
            {"binding": 0, "visibility": wgpu.ShaderStage.VERTEX | wgpu.ShaderStage.FRAGMENT,
             "buffer": {"type": wgpu.BufferBindingType.read_only_storage}},
            {"binding": 1, "visibility": wgpu.ShaderStage.VERTEX,
             "buffer": {"type": wgpu.BufferBindingType.read_only_storage}},
            {"binding": 2, "visibility": wgpu.ShaderStage.FRAGMENT,
             "texture": {"sample_type": wgpu.TextureSampleType.float,
                         "view_dimension": wgpu.TextureViewDimension.d2_array}},
            {"binding": 3, "visibility": wgpu.ShaderStage.FRAGMENT,
             "sampler": {"type": wgpu.SamplerBindingType.filtering}},
        ])
        self.bind = dev.create_bind_group(layout=bgl, entries=[
            {"binding": 0, "resource": {"buffer": self.ebuf, "offset": 0, "size": self.n * 96}},
            {"binding": 1, "resource": {"buffer": self.xbuf, "offset": 0, "size": self.n * self.G * 64}},
            {"binding": 2, "resource": self.bg_tex.create_view(dimension=wgpu.TextureViewDimension.d2_array)},
            {"binding": 3, "resource": samp},
        ])
        layout = dev.create_pipeline_layout(bind_group_layouts=[bgl])
        vlayout = [{"array_stride": 28, "step_mode": wgpu.VertexStepMode.vertex,
                    "attributes": [
                        {"format": wgpu.VertexFormat.float32x3, "offset": 0, "shader_location": 0},
                        {"format": wgpu.VertexFormat.float32x3, "offset": 12, "shader_location": 1},
                        {"format": wgpu.VertexFormat.uint32, "offset": 24, "shader_location": 2}]}]
        ds = {"format": wgpu.TextureFormat.depth32float,
              "depth_write_enabled": True, "depth_compare": wgpu.CompareFunction.less}
        self.geo_pipe = dev.create_render_pipeline(
            layout=layout,
            vertex={"module": shader, "entry_point": "vs_main", "buffers": vlayout},
            primitive={"topology": wgpu.PrimitiveTopology.triangle_list, "cull_mode": wgpu.CullMode.none},
            depth_stencil=ds, multisample={"count": 1},
            fragment={"module": shader, "entry_point": "fs_main",
                      "targets": [{"format": wgpu.TextureFormat.rgba8unorm}]})
        self.bg_pipe = dev.create_render_pipeline(
            layout=layout,
            vertex={"module": shader, "entry_point": "bg_vs", "buffers": []},
            primitive={"topology": wgpu.PrimitiveTopology.triangle_list, "cull_mode": wgpu.CullMode.none},
            depth_stencil={"format": wgpu.TextureFormat.depth32float,
                           "depth_write_enabled": False, "depth_compare": wgpu.CompareFunction.always},
            multisample={"count": 1},
            fragment={"module": shader, "entry_point": "bg_fs",
                      "targets": [{"format": wgpu.TextureFormat.rgba8unorm}]})
        self.color_tex = dev.create_texture(
            size=(self.aw, self.ah, 1), format=wgpu.TextureFormat.rgba8unorm,
            usage=wgpu.TextureUsage.RENDER_ATTACHMENT | wgpu.TextureUsage.COPY_SRC)
        self.depth_tex = dev.create_texture(
            size=(self.aw, self.ah, 1), format=wgpu.TextureFormat.depth32float,
            usage=wgpu.TextureUsage.RENDER_ATTACHMENT)

    # -- per-frame API ----------------------------------------------------

    def set_backgrounds(self, env_ids, images):
        """images: iterable of (height, width, 3) uint8 arrays."""
        for eid, im in zip(env_ids, images):
            rgba = np.dstack([im, np.full(im.shape[:2], 255, np.uint8)])
            self.device.queue.write_texture(
                {"texture": self.bg_tex, "mip_level": 0, "origin": (0, 0, int(eid))},
                rgba.tobytes(),
                {"offset": 0, "bytes_per_row": self.tw * 4, "rows_per_image": self.th},
                (self.tw, self.th, 1))

    def _proj(self, near=0.01, far=10.0):
        fovy = np.deg2rad(self.m.cam_fovy[self._proj_cam])
        f = 1.0 / np.tan(fovy / 2)
        aspect = self.tw / self.th
        P = np.zeros((4, 4))
        P[0, 0] = f / aspect
        P[1, 1] = f
        P[2, 2] = far / (near - far)
        P[2, 3] = near * far / (near - far)
        P[3, 2] = -1.0
        return P

    def render(self, datas, cam_pos, cam_quat, colors=None, light_dirs=None):
        """Render all envs; returns (N, height, width, 3) uint8 tiles.

        datas: list of N MjData (mj_forward'd).
        cam_pos/cam_quat: (N,3)/(N,4) world camera pose per env
            (MuJoCo camera convention: looks along -Z, +Y up).
        colors: (N, G, 4) RGBA per drawn geom; default = material colors.
        light_dirs: (N, 3) directional light per env.
        """
        dev = self.device
        if colors is None:
            colors = np.broadcast_to(self.default_colors(), (self.n, self.G, 4))
        if light_dirs is None:
            light_dirs = np.broadcast_to([0.3, 0.3, -0.9], (self.n, 3))
        P = self._proj()
        sx = self.tw / self.aw
        sy = self.th / self.ah
        gidx = np.asarray(self.geoms)
        Rall = np.stack([d.geom_xmat[gidx] for d in datas]).reshape(self.n, self.G, 3, 3)
        pall = np.stack([d.geom_xpos[gidx] for d in datas])
        xf = np.zeros((self.n, self.G, 16), np.float32)
        xf[:, :, 0:3] = Rall[:, :, 0, :]
        xf[:, :, 3] = pall[:, :, 0]
        xf[:, :, 4:7] = Rall[:, :, 1, :]
        xf[:, :, 7] = pall[:, :, 1]
        xf[:, :, 8:11] = Rall[:, :, 2, :]
        xf[:, :, 11] = pall[:, :, 2]
        xf[:, :, 12:16] = colors
        edata = np.zeros((self.n, 24), np.float32)
        for e in range(self.n):
            V = np.eye(4)
            R = _quat_to_mat(cam_quat[e])
            V[:3, :3] = R.T
            V[:3, 3] = -R.T @ cam_pos[e]
            vp = (P @ V).astype(np.float32)
            col = e % self.tpr
            row = e // self.tpr
            edata[e, 0:16] = vp.T.reshape(-1)
            edata[e, 16:20] = [-1.0 + (2 * col + 1) * sx, 1.0 - (2 * row + 1) * sy, sx, sy]
            ld = np.asarray(light_dirs[e], dtype=np.float64)
            edata[e, 20:23] = ld / (np.linalg.norm(ld) + 1e-9)
        dev.queue.write_buffer(self.ebuf, 0, edata.tobytes())
        dev.queue.write_buffer(self.xbuf, 0, xf.tobytes())

        enc = dev.create_command_encoder()
        rp = enc.begin_render_pass(
            color_attachments=[{"view": self.color_tex.create_view(),
                                "clear_value": (0, 0, 0, 1),
                                "load_op": wgpu.LoadOp.clear, "store_op": wgpu.StoreOp.store}],
            depth_stencil_attachment={"view": self.depth_tex.create_view(),
                                      "depth_clear_value": 1.0,
                                      "depth_load_op": wgpu.LoadOp.clear,
                                      "depth_store_op": wgpu.StoreOp.store})
        rp.set_pipeline(self.bg_pipe)
        rp.set_bind_group(0, self.bind)
        rp.draw(6, self.n)
        rp.set_pipeline(self.geo_pipe)
        rp.set_bind_group(0, self.bind)
        rp.set_vertex_buffer(0, self.vbuf)
        rp.draw(self.n_verts, self.n)
        rp.end()
        dev.queue.submit([enc.finish()])
        buf = dev.queue.read_texture(
            {"texture": self.color_tex, "mip_level": 0, "origin": (0, 0, 0)},
            {"offset": 0, "bytes_per_row": self.aw * 4, "rows_per_image": self.ah},
            (self.aw, self.ah, 1))
        atlas = np.frombuffer(buf, np.uint8).reshape(self.ah, self.aw, 4)[:, :, :3]
        tiles = np.empty((self.n, self.th, self.tw, 3), np.uint8)
        for e in range(self.n):
            r0 = (e // self.tpr) * self.th
            c0 = (e % self.tpr) * self.tw
            tiles[e] = atlas[r0:r0 + self.th, c0:c0 + self.tw]
        return tiles


def _quat_to_mat(q):
    w, x, y, z = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ])
