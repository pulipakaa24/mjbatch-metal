"""BatchRenderer — tiled, instanced MuJoCo observation rendering on Metal.

Architecture (Madrona/PyBatchRender-style, Metal-native via wgpu):
- each unique mesh/primitive is stored ONCE, indexed, with smooth normals
  and UVs; one instanced draw call per unique mesh covers every
  (environment, geom) pair that uses it — all inside a single render pass
- per-(env, geom) world transforms and RGBA colors in a storage buffer
- per-env view-projection, lighting, and tile rect in a storage buffer
- MuJoCo textures packed into one atlas; per-geom material info (atlas rect,
  texrepeat) sampled in-shader; per-env colors modulate texture or flat color
- optional per-env background compositing (greenscreen-style, in-shader)
- optional metric depth output

The same renderer serves low-complexity robot scenes and high-complexity
textured scenes; behavior follows the model file, not a mode switch.
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
struct GeomMat {
    rect: vec4<f32>,      // atlas u0, v0, du, dv
    par: vec4<f32>,       // texrepeat.xy, textured flag, unused
};
@group(0) @binding(0) var<storage, read> envs: array<EnvData>;
@group(0) @binding(1) var<storage, read> xforms: array<GeomXform>;
@group(0) @binding(2) var<storage, read> inst_table: array<vec2<u32>>;  // (env, slot)
@group(0) @binding(3) var<storage, read> mats: array<GeomMat>;
@group(0) @binding(4) var atlas_tex: texture_2d<f32>;
@group(0) @binding(5) var atlas_samp: sampler;
@group(0) @binding(6) var bg_tex: texture_2d_array<f32>;
@group(0) @binding(7) var bg_samp: sampler;

struct VSIn {
    @location(0) pos: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) uv: vec2<f32>,
    @builtin(instance_index) inst: u32,
};
struct VSOut {
    @builtin(position) clip: vec4<f32>,
    @location(0) normal_w: vec3<f32>,
    @location(1) color: vec4<f32>,
    @location(2) uv: vec2<f32>,
    @location(3) @interpolate(flat) env: u32,
    @location(4) @interpolate(flat) slot: u32,
};

const N_GEOMS: u32 = {N_GEOMS}u;

@vertex
fn vs_main(in: VSIn) -> VSOut {
    let pair = inst_table[in.inst];
    let env = pair.x;
    let slot = pair.y;
    let e = envs[env];
    let g = xforms[env * N_GEOMS + slot];
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
    out.uv = in.uv;
    out.env = env;
    out.slot = slot;
    return out;
}

struct FSOut {
    @location(0) color: vec4<f32>,
    @location(1) seg: u32,
};

@fragment
fn fs_main(in: VSOut) -> FSOut {
    let e = envs[in.env];
    let tile_w = {TILE_W}.0;
    let tile_h = {TILE_H}.0;
    let col = f32(in.env % {TPR}u);
    let row = f32(in.env / {TPR}u);
    if (in.clip.x < col * tile_w || in.clip.x >= (col + 1.0) * tile_w ||
        in.clip.y < row * tile_h || in.clip.y >= (row + 1.0) * tile_h) {
        discard;
    }
    let m = mats[in.slot];
    var base = in.color.rgb;
    if (m.par.z > 0.5) {
        let tuv = fract(in.uv * m.par.xy);
        let auv = m.rect.xy + tuv * m.rect.zw;
        base = base * textureSampleLevel(atlas_tex, atlas_samp, auv, 0.0).rgb;
    }
    let n = normalize(in.normal_w);
    let ndl = clamp(dot(n, -e.light.xyz), 0.0, 1.0);
    let shade = 0.35 + 0.65 * ndl;
    var out: FSOut;
    out.color = vec4<f32>(base * shade, 1.0);
    out.seg = in.slot + 1u;
    return out;
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

struct BGFSOut {
    @location(0) color: vec4<f32>,
    @location(1) seg: u32,
};

@fragment
fn bg_fs(in: BGOut) -> BGFSOut {
    var out: BGFSOut;
    out.color = textureSampleLevel(bg_tex, bg_samp, in.uv, i32(in.env), 0.0);
    out.seg = 0u;
    return out;
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


def _tris_to_indexed(tri, uv=None):
    """(F,3,3) triangle soup -> indexed verts with smooth normals (+uv)."""
    flat = tri.reshape(-1, 3)
    if uv is None:
        uv_flat = np.zeros((len(flat), 2))
    else:
        uv_flat = uv.reshape(-1, 2)
    key = np.round(np.concatenate([flat, uv_flat], axis=1), 7)
    _, first, inverse = np.unique(key, axis=0, return_index=True, return_inverse=True)
    verts = flat[first]
    uvs = uv_flat[first]
    idx = inverse.reshape(-1, 3).astype(np.uint32)
    fn = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    fn /= (np.linalg.norm(fn, axis=1, keepdims=True) + 1e-12)
    normals = np.zeros_like(verts)
    for k in range(3):
        np.add.at(normals, idx[:, k], fn)
    normals /= (np.linalg.norm(normals, axis=1, keepdims=True) + 1e-12)
    return verts, normals, uvs, idx


class _Atlas:
    """Naive shelf-packed texture atlas from mjModel textures (RGB role)."""

    def __init__(self, m: mujoco.MjModel):
        self.rects = {}                       # texid -> (u0, v0, du, dv)
        if m.ntex == 0:
            self.image = np.zeros((4, 4, 4), np.uint8)
            return
        pad = 1
        widths = [int(m.tex_width[t]) for t in range(m.ntex)]
        heights = [int(m.tex_height[t]) for t in range(m.ntex)]
        aw = max(widths) + 2 * pad
        order = sorted(range(m.ntex), key=lambda t: -heights[t])
        x = y = shelf_h = 0
        pos = {}
        for t in order:
            w, h = widths[t] + 2 * pad, heights[t] + 2 * pad
            if x + w > aw:
                y += shelf_h; x = 0; shelf_h = 0
            pos[t] = (x + pad, y + pad)
            x += w
            shelf_h = max(shelf_h, h)
        ah = y + shelf_h
        img = np.zeros((ah, aw, 4), np.uint8)
        img[..., 3] = 255
        for t in range(m.ntex):
            w, h = widths[t], heights[t]
            nc = int(m.tex_nchannel[t]) if hasattr(m, "tex_nchannel") else 3
            adr = int(m.tex_adr[t])
            data = m.tex_data[adr:adr + w * h * nc].reshape(h, w, nc)
            px, py = pos[t]
            # MuJoCo texture rows are bottom-up relative to our sampling
            img[py:py + h, px:px + w, :3] = data[::-1, :, :3]
            self.rects[t] = (px / aw, py / ah, w / aw, h / ah)
        self.image = img


class BatchRenderer:
    """Tiled batch renderer for N MuJoCo environments sharing one model.

    Parameters
    ----------
    model : mujoco.MjModel
    n_envs : int
    width, height : per-env tile resolution (default 128x128)
    camera : camera name used for the projection matrix (fovy); per-env pose
        is passed at render time. Defaults to the model's first camera.
    max_group : draw geoms with geom_group < max_group.
    include_planes : draw plane geoms (default True; set False for
        composited observations where the background replaces the floor).
    use_backgrounds : enable the per-env background compositing pass
        (default False; call set_backgrounds() and pass True for RL
        observation pipelines).
    decimate_faces : per-mesh face budget (0 disables decimation).
    """

    def __init__(self, model: mujoco.MjModel, n_envs: int, *,
                 width: int = 128, height: int = 128, camera: str | None = None,
                 max_group: int = 3, include_planes: bool = True,
                 use_backgrounds: bool = False, decimate_faces: int = 0):
        self.m = model
        self.n = n_envs
        self.tw, self.th = width, height
        self.use_bg = use_backgrounds
        self.tpr = int(np.ceil(np.sqrt(n_envs)))
        self.aw = self.tpr * width
        self.ah = int(np.ceil(n_envs / self.tpr)) * height
        if camera is None:
            if model.ncam == 0:
                raise ValueError("model has no cameras")
            self._proj_cam = 0
        else:
            self._proj_cam = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, camera)
            if self._proj_cam < 0:
                raise ValueError(f"camera {camera!r} not found")
        adapter = wgpu.gpu.request_adapter_sync(power_preference="high-performance")
        self.device = adapter.request_device_sync()
        self._atlas = _Atlas(model)
        self._build_geometry(max_group, include_planes, decimate_faces)
        self._build_pipeline()

    # -- geometry ---------------------------------------------------------

    def _geom_tris_uv(self, g, decimate_faces):
        m = self.m
        t = int(m.geom_type[g])
        size = m.geom_size[g]
        if t == mujoco.mjtGeom.mjGEOM_MESH:
            mid = int(m.geom_dataid[g])
            va, vn = int(m.mesh_vertadr[mid]), int(m.mesh_vertnum[mid])
            fa, fn = int(m.mesh_faceadr[mid]), int(m.mesh_facenum[mid])
            v_full = m.mesh_vert[va:va + vn].astype(np.float64)
            f_full = m.mesh_face[fa:fa + fn].astype(np.int64)
            uv = None
            tca = int(m.mesh_texcoordadr[mid])
            if tca >= 0:
                ftc = m.mesh_facetexcoord[fa:fa + fn].astype(np.int64)
                uv = m.mesh_texcoord[tca + ftc]                # (F,3,2)
            if decimate_faces and len(f_full) > decimate_faces:
                import fast_simplification
                v_d, f_d = fast_simplification.simplify(
                    v_full, f_full,
                    target_reduction=1.0 - decimate_faces / len(f_full))
                return np.asarray(v_d)[np.asarray(f_d)], None   # decimation drops UVs
            return v_full[f_full], uv
        if t == mujoco.mjtGeom.mjGEOM_BOX:
            tri = _BOX_FACES * size
            # planar per-face UVs: project each face onto its two varying axes
            uv = np.zeros((len(tri), 3, 2))
            for f in range(len(tri)):
                span = tri[f].max(axis=0) - tri[f].min(axis=0)
                axes = np.argsort(span)[-2:]
                lo = tri[f][:, axes].min(axis=0)
                rng_ = np.maximum(tri[f][:, axes].max(axis=0) - lo, 1e-9)
                uv[f] = (tri[f][:, axes] - lo) / rng_
            return tri, uv
        if t == mujoco.mjtGeom.mjGEOM_SPHERE:
            return _uv_sphere(size[0]), None
        if t == mujoco.mjtGeom.mjGEOM_CYLINDER:
            return _cylinder(size[0], size[1]), None
        if t == mujoco.mjtGeom.mjGEOM_CAPSULE:
            return np.concatenate([
                _cylinder(size[0], size[1]),
                _uv_sphere(size[0]) + [0, 0, size[1]],
                _uv_sphere(size[0]) - [0, 0, size[1]]]), None
        if t == mujoco.mjtGeom.mjGEOM_PLANE:
            s = max(float(size[0]), 1.0), max(float(size[1]), 1.0)
            tri = np.array([[[-s[0],-s[1],0],[ s[0],-s[1],0],[ s[0], s[1],0]],
                            [[-s[0],-s[1],0],[ s[0], s[1],0],[-s[0], s[1],0]]])
            uv = (tri[:, :, :2] / (2 * np.array(s)) + 0.5)
            return tri, uv
        return None, None

    def _build_geometry(self, max_group, include_planes, decimate_faces):
        m = self.m
        mesh_cache = {}       # key -> mesh index in self._meshes
        self._meshes = []     # dicts: v_off, i_off, i_count
        self.geoms = []       # model geom ids in slot order
        self._geom_mesh = []  # slot -> mesh index
        all_v, all_n, all_uv, all_i = [], [], [], []
        v_off = i_off = 0
        for g in range(m.ngeom):
            if m.geom_group[g] >= max_group:
                continue
            t = int(m.geom_type[g])
            if t == mujoco.mjtGeom.mjGEOM_PLANE and not include_planes:
                continue
            if t == mujoco.mjtGeom.mjGEOM_MESH:
                key = ("mesh", int(m.geom_dataid[g]))
            else:
                key = (t, tuple(np.round(m.geom_size[g], 6)))
            if key not in mesh_cache:
                tri, uv = self._geom_tris_uv(g, decimate_faces)
                if tri is None:
                    continue
                verts, normals, uvs, idx = _tris_to_indexed(tri, uv)
                mesh_cache[key] = len(self._meshes)
                self._meshes.append({"v_off": v_off, "i_off": i_off,
                                     "i_count": idx.size,
                                     "radius": float(np.linalg.norm(verts, axis=1).max())})
                all_v.append(verts); all_n.append(normals); all_uv.append(uvs)
                all_i.append(idx.reshape(-1) + v_off)
                v_off += len(verts)
                i_off += idx.size
            elif self._geom_tris_uv(g, decimate_faces)[0] is None:
                continue
            self._geom_mesh.append(mesh_cache[key])
            self.geoms.append(g)
        if not self.geoms:
            raise ValueError("no drawable geoms found (check max_group)")
        v = np.concatenate(all_v).astype(np.float32)
        nrm = np.concatenate(all_n).astype(np.float32)
        uvs = np.concatenate(all_uv).astype(np.float32)
        self.n_verts = len(v)
        self.G = len(self.geoms)
        inter = np.zeros((self.n_verts, 8), np.float32)
        inter[:, 0:3] = v
        inter[:, 3:6] = nrm
        inter[:, 6:8] = uvs
        self.vbuf = self.device.create_buffer_with_data(
            data=inter.tobytes(), usage=wgpu.BufferUsage.VERTEX)
        self.ibuf = self.device.create_buffer_with_data(
            data=np.concatenate(all_i).astype(np.uint32).tobytes(),
            usage=wgpu.BufferUsage.INDEX)
        # instance table: for each unique mesh, the (env, slot) pairs using it
        self._draws = []      # (i_off, i_count, first_inst, inst_count)
        table = []
        first = 0
        for mi, mesh in enumerate(self._meshes):
            slots = [s for s, mm in enumerate(self._geom_mesh) if mm == mi]
            pairs = [(e, s) for s in slots for e in range(self.n)]
            table.extend(pairs)
            self._draws.append((mesh["i_off"], mesh["i_count"], first, len(pairs)))
            first += len(pairs)
        self._inst_table = np.asarray(table, np.uint32)
        # per-slot material info
        mats = np.zeros((self.G, 8), np.float32)
        for slot, g in enumerate(self.geoms):
            mat = int(m.geom_matid[g])
            texid = -1
            if mat >= 0 and m.ntex:
                roles = m.mat_texid[mat].reshape(-1)
                texid = int(roles[1]) if len(roles) > 1 else int(roles[0])
            if texid >= 0 and texid in self._atlas.rects:
                mats[slot, 0:4] = self._atlas.rects[texid]
                mats[slot, 4:6] = m.mat_texrepeat[mat]
                mats[slot, 6] = 1.0
        self._mats = mats

    def default_colors(self) -> np.ndarray:
        """(G, 4) material-resolved RGBA per drawn geom (MuJoCo semantics).
        For textured geoms this is the modulation color (white unless the
        material tints)."""
        m = self.m
        out = np.zeros((self.G, 4), np.float32)
        for i, g in enumerate(self.geoms):
            mat = int(m.geom_matid[g])
            if mat >= 0:
                out[i] = m.mat_rgba[mat]
            else:
                out[i] = m.geom_rgba[g]
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
        self.tbuf = dev.create_buffer_with_data(
            data=self._inst_table.tobytes(),
            usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_DST)
        self.mbuf = dev.create_buffer_with_data(
            data=self._mats.tobytes(), usage=wgpu.BufferUsage.STORAGE)
        ai = self._atlas.image
        self.atlas_tex = dev.create_texture(
            size=(ai.shape[1], ai.shape[0], 1), format=wgpu.TextureFormat.rgba8unorm,
            usage=wgpu.TextureUsage.TEXTURE_BINDING | wgpu.TextureUsage.COPY_DST)
        dev.queue.write_texture(
            {"texture": self.atlas_tex, "mip_level": 0, "origin": (0, 0, 0)},
            ai.tobytes(), {"offset": 0, "bytes_per_row": ai.shape[1] * 4,
                           "rows_per_image": ai.shape[0]},
            (ai.shape[1], ai.shape[0], 1))
        self.bg_tex = dev.create_texture(
            size=(self.tw, self.th, self.n), format=wgpu.TextureFormat.rgba8unorm,
            usage=wgpu.TextureUsage.TEXTURE_BINDING | wgpu.TextureUsage.COPY_DST)
        samp = dev.create_sampler(mag_filter=wgpu.FilterMode.linear,
                                  min_filter=wgpu.FilterMode.linear)
        entries = [
            {"binding": 0, "visibility": wgpu.ShaderStage.VERTEX | wgpu.ShaderStage.FRAGMENT,
             "buffer": {"type": wgpu.BufferBindingType.read_only_storage}},
            {"binding": 1, "visibility": wgpu.ShaderStage.VERTEX,
             "buffer": {"type": wgpu.BufferBindingType.read_only_storage}},
            {"binding": 2, "visibility": wgpu.ShaderStage.VERTEX,
             "buffer": {"type": wgpu.BufferBindingType.read_only_storage}},
            {"binding": 3, "visibility": wgpu.ShaderStage.FRAGMENT,
             "buffer": {"type": wgpu.BufferBindingType.read_only_storage}},
            {"binding": 4, "visibility": wgpu.ShaderStage.FRAGMENT,
             "texture": {"sample_type": wgpu.TextureSampleType.float,
                         "view_dimension": wgpu.TextureViewDimension.d2}},
            {"binding": 5, "visibility": wgpu.ShaderStage.FRAGMENT,
             "sampler": {"type": wgpu.SamplerBindingType.filtering}},
            {"binding": 6, "visibility": wgpu.ShaderStage.FRAGMENT,
             "texture": {"sample_type": wgpu.TextureSampleType.float,
                         "view_dimension": wgpu.TextureViewDimension.d2_array}},
            {"binding": 7, "visibility": wgpu.ShaderStage.FRAGMENT,
             "sampler": {"type": wgpu.SamplerBindingType.filtering}},
        ]
        bgl = dev.create_bind_group_layout(entries=entries)
        self.bind = dev.create_bind_group(layout=bgl, entries=[
            {"binding": 0, "resource": {"buffer": self.ebuf, "offset": 0, "size": self.n * 96}},
            {"binding": 1, "resource": {"buffer": self.xbuf, "offset": 0, "size": self.n * self.G * 64}},
            {"binding": 2, "resource": {"buffer": self.tbuf, "offset": 0,
                                        "size": self._inst_table.nbytes}},
            {"binding": 3, "resource": {"buffer": self.mbuf, "offset": 0,
                                        "size": self._mats.nbytes}},
            {"binding": 4, "resource": self.atlas_tex.create_view()},
            {"binding": 5, "resource": samp},
            {"binding": 6, "resource": self.bg_tex.create_view(dimension=wgpu.TextureViewDimension.d2_array)},
            {"binding": 7, "resource": samp},
        ])
        layout = dev.create_pipeline_layout(bind_group_layouts=[bgl])
        vlayout = [{"array_stride": 32, "step_mode": wgpu.VertexStepMode.vertex,
                    "attributes": [
                        {"format": wgpu.VertexFormat.float32x3, "offset": 0, "shader_location": 0},
                        {"format": wgpu.VertexFormat.float32x3, "offset": 12, "shader_location": 1},
                        {"format": wgpu.VertexFormat.float32x2, "offset": 24, "shader_location": 2}]}]
        ds = {"format": wgpu.TextureFormat.depth32float,
              "depth_write_enabled": True, "depth_compare": wgpu.CompareFunction.less}
        self.geo_pipe = dev.create_render_pipeline(
            layout=layout,
            vertex={"module": shader, "entry_point": "vs_main", "buffers": vlayout},
            primitive={"topology": wgpu.PrimitiveTopology.triangle_list, "cull_mode": wgpu.CullMode.none},
            depth_stencil=ds, multisample={"count": 1},
            fragment={"module": shader, "entry_point": "fs_main",
                      "targets": [{"format": wgpu.TextureFormat.rgba8unorm},
                                  {"format": wgpu.TextureFormat.r32uint}]})
        self.bg_pipe = dev.create_render_pipeline(
            layout=layout,
            vertex={"module": shader, "entry_point": "bg_vs", "buffers": []},
            primitive={"topology": wgpu.PrimitiveTopology.triangle_list, "cull_mode": wgpu.CullMode.none},
            depth_stencil={"format": wgpu.TextureFormat.depth32float,
                           "depth_write_enabled": False, "depth_compare": wgpu.CompareFunction.always},
            multisample={"count": 1},
            fragment={"module": shader, "entry_point": "bg_fs",
                      "targets": [{"format": wgpu.TextureFormat.rgba8unorm},
                                  {"format": wgpu.TextureFormat.r32uint}]})
        self.color_tex = dev.create_texture(
            size=(self.aw, self.ah, 1), format=wgpu.TextureFormat.rgba8unorm,
            usage=wgpu.TextureUsage.RENDER_ATTACHMENT | wgpu.TextureUsage.COPY_SRC)
        self.depth_tex = dev.create_texture(
            size=(self.aw, self.ah, 1), format=wgpu.TextureFormat.depth32float,
            usage=wgpu.TextureUsage.RENDER_ATTACHMENT | wgpu.TextureUsage.COPY_SRC)
        self.seg_tex = dev.create_texture(
            size=(self.aw, self.ah, 1), format=wgpu.TextureFormat.r32uint,
            usage=wgpu.TextureUsage.RENDER_ATTACHMENT | wgpu.TextureUsage.COPY_SRC)
        # staging buffers for pipelined readback (unified memory: mapping is
        # effectively zero-copy on Apple Silicon)
        self._row_bytes = (self.aw * 4 + 255) // 256 * 256
        self._staging = [dev.create_buffer(size=self._row_bytes * self.ah,
                                           usage=wgpu.BufferUsage.COPY_DST | wgpu.BufferUsage.MAP_READ)
                         for _ in range(2)]
        self._staging_slot = 0
        self._pending = None       # slot with an in-flight frame
        # culling state
        self._mesh_radius = np.zeros(len(self._meshes))
        self._geom_radius = np.array([self._mesh_radius_of(mi) for mi in self._geom_mesh])

    def _mesh_radius_of(self, mi):
        return self._meshes[mi].get("radius", 0.0)

    # -- per-frame API ----------------------------------------------------

    def set_backgrounds(self, env_ids, images):
        """images: iterable of (height, width, 3) uint8 arrays. Enables the
        compositing pass if it wasn't already enabled."""
        self.use_bg = True
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

    def render(self, datas, cam_pos, cam_quat, colors=None, light_dirs=None,
               return_depth=False, return_seg=False, cull=False,
               pipelined=False):
        """Render all envs; returns (N, height, width, 3) uint8 tiles.

        datas: list of N MjData (mj_forward'd).
        cam_pos/cam_quat: (N,3)/(N,4) world camera pose per env
            (MuJoCo camera convention: looks along -Z, +Y up).
        colors: (N, G, 4) RGBA per drawn geom (modulates textures);
            default = material colors.
        light_dirs: (N, 3) directional light per env.
        return_depth: also return (N, height, width) float32 metric depth.
        return_seg: also return (N, height, width) uint32 segmentation ids
            (0 = background, i+1 = draw-order slot i; see .geoms for the
            model geom id of each slot).
        cull: frustum-cull (env, geom) instances on the CPU before drawing —
            wins when most instances are off-camera (large scenes).
        pipelined: submit this frame and return the PREVIOUS frame's tiles
            (one-frame latency, hides readback sync; returns None on the
            first call). Incompatible with return_depth/return_seg.
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
        Rc = _quats_to_mats(np.asarray(cam_quat, np.float64))          # (N,3,3)
        V = np.tile(np.eye(4), (self.n, 1, 1))
        V[:, :3, :3] = np.transpose(Rc, (0, 2, 1))
        V[:, :3, 3] = -np.einsum("nij,nj->ni", np.transpose(Rc, (0, 2, 1)),
                                 np.asarray(cam_pos, np.float64))
        vp = np.einsum("ij,njk->nik", P, V).astype(np.float32)         # (N,4,4)
        edata[:, 0:16] = np.transpose(vp, (0, 2, 1)).reshape(self.n, 16)
        ecol = np.arange(self.n) % self.tpr
        erow = np.arange(self.n) // self.tpr
        edata[:, 16] = -1.0 + (2 * ecol + 1) * sx
        edata[:, 17] = 1.0 - (2 * erow + 1) * sy
        edata[:, 18] = sx
        edata[:, 19] = sy
        ld = np.asarray(light_dirs, np.float64)
        edata[:, 20:23] = ld / (np.linalg.norm(ld, axis=1, keepdims=True) + 1e-9)
        dev.queue.write_buffer(self.ebuf, 0, edata.tobytes())
        dev.queue.write_buffer(self.xbuf, 0, xf.tobytes())

        draws = self._draws
        if cull:
            # conservative sphere-vs-clip test per (env, geom)
            centers = np.concatenate([pall, np.ones((self.n, self.G, 1))], axis=2)
            vp4 = vp  # (N,4,4) from above
            clip = np.einsum("nij,ngj->ngi", vp4.astype(np.float64), centers)
            w = clip[..., 3]
            rad = self._geom_radius[None, :]
            margin = w + rad * 2.0
            vis = ((clip[..., 0] > -margin) & (clip[..., 0] < margin) &
                   (clip[..., 1] > -margin) & (clip[..., 1] < margin) &
                   (w + rad > 0))
            table, draws = [], []
            first = 0
            for mi, mesh in enumerate(self._meshes):
                slots = [sl for sl, mm in enumerate(self._geom_mesh) if mm == mi]
                pairs = [(e, sl) for sl in slots for e in np.flatnonzero(vis[:, sl])]
                if pairs:
                    table.extend(pairs)
                    draws.append((mesh["i_off"], mesh["i_count"], first, len(pairs)))
                    first += len(pairs)
            if table:
                tbl = np.asarray(table, np.uint32)
                dev.queue.write_buffer(self.tbuf, 0, tbl.tobytes())

        enc = dev.create_command_encoder()
        rp = enc.begin_render_pass(
            color_attachments=[{"view": self.color_tex.create_view(),
                                "clear_value": (0, 0, 0, 1),
                                "load_op": wgpu.LoadOp.clear, "store_op": wgpu.StoreOp.store},
                               {"view": self.seg_tex.create_view(),
                                "clear_value": (0, 0, 0, 0),
                                "load_op": wgpu.LoadOp.clear, "store_op": wgpu.StoreOp.store}],
            depth_stencil_attachment={"view": self.depth_tex.create_view(),
                                      "depth_clear_value": 1.0,
                                      "depth_load_op": wgpu.LoadOp.clear,
                                      "depth_store_op": wgpu.StoreOp.store})
        if self.use_bg:
            rp.set_pipeline(self.bg_pipe)
            rp.set_bind_group(0, self.bind)
            rp.draw(6, self.n)
        rp.set_pipeline(self.geo_pipe)
        rp.set_bind_group(0, self.bind)
        rp.set_vertex_buffer(0, self.vbuf)
        rp.set_index_buffer(self.ibuf, wgpu.IndexFormat.uint32)
        for i_off, i_count, first, count in draws:
            rp.draw_indexed(i_count, count, i_off, 0, first)
        rp.end()
        if cull and draws is not self._draws:
            # restore static table for subsequent non-culled calls
            self._table_dirty = True
        elif getattr(self, "_table_dirty", False):
            dev.queue.write_buffer(self.tbuf, 0, self._inst_table.tobytes())
            self._table_dirty = False
        if pipelined:
            slot = self._staging_slot
            enc.copy_texture_to_buffer(
                {"texture": self.color_tex, "mip_level": 0, "origin": (0, 0, 0)},
                {"buffer": self._staging[slot], "offset": 0,
                 "bytes_per_row": self._row_bytes, "rows_per_image": self.ah},
                (self.aw, self.ah, 1))
            dev.queue.submit([enc.finish()])
            prev = self._pending
            self._pending = slot
            self._staging_slot = 1 - slot
            if prev is None:
                return None
            sb = self._staging[prev]
            sb.map_sync(wgpu.MapMode.READ)
            raw = np.frombuffer(sb.read_mapped(), np.uint8)
            sb.unmap()
            atlas = raw.reshape(self.ah, self._row_bytes)[:, :self.aw * 4]
            atlas = atlas.reshape(self.ah, self.aw, 4)[:, :, :3]
            tiles = np.empty((self.n, self.th, self.tw, 3), np.uint8)
            for e in range(self.n):
                r0 = (e // self.tpr) * self.th
                c0 = (e % self.tpr) * self.tw
                tiles[e] = atlas[r0:r0 + self.th, c0:c0 + self.tw]
            return tiles
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
        if return_seg:
            sgbuf = dev.queue.read_texture(
                {"texture": self.seg_tex, "mip_level": 0, "origin": (0, 0, 0)},
                {"offset": 0, "bytes_per_row": self.aw * 4, "rows_per_image": self.ah},
                (self.aw, self.ah, 1))
            satlas = np.frombuffer(sgbuf, np.uint32).reshape(self.ah, self.aw)
            seg = np.empty((self.n, self.th, self.tw), np.uint32)
            for e in range(self.n):
                r0 = (e // self.tpr) * self.th
                c0 = (e % self.tpr) * self.tw
                seg[e] = satlas[r0:r0 + self.th, c0:c0 + self.tw]
        if not return_depth and not return_seg:
            return tiles
        if not return_depth:
            return tiles, seg
        dbuf = dev.queue.read_texture(
            {"texture": self.depth_tex, "mip_level": 0, "origin": (0, 0, 0)},
            {"offset": 0, "bytes_per_row": self.aw * 4, "rows_per_image": self.ah},
            (self.aw, self.ah, 1))
        datlas = np.frombuffer(dbuf, np.float32).reshape(self.ah, self.aw)
        near, far = 0.01, 10.0
        metric = near * far / (far - datlas * (far - near))
        depth = np.empty((self.n, self.th, self.tw), np.float32)
        for e in range(self.n):
            r0 = (e // self.tpr) * self.th
            c0 = (e % self.tpr) * self.tw
            depth[e] = metric[r0:r0 + self.th, c0:c0 + self.tw]
        if return_seg:
            return tiles, depth, seg
        return tiles, depth


def _quat_to_mat(q):
    w, x, y, z = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ])


def _quats_to_mats(q):
    """(N,4) wxyz quaternions -> (N,3,3) rotation matrices, vectorized."""
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    M = np.empty((len(q), 3, 3))
    M[:, 0, 0] = 1 - 2 * (y * y + z * z)
    M[:, 0, 1] = 2 * (x * y - w * z)
    M[:, 0, 2] = 2 * (x * z + w * y)
    M[:, 1, 0] = 2 * (x * y + w * z)
    M[:, 1, 1] = 1 - 2 * (x * x + z * z)
    M[:, 1, 2] = 2 * (y * z - w * x)
    M[:, 2, 0] = 2 * (x * z - w * y)
    M[:, 2, 1] = 2 * (y * z + w * x)
    M[:, 2, 2] = 1 - 2 * (x * x + y * y)
    return M
