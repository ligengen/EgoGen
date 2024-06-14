import os
os.environ['PYOPENGL_PLATFORM'] = 'egl'
import torch
import numpy as np
import pyrender
import trimesh
import cv2
from yacs.config import CfgNode
from typing import List, Optional
from torchvision.utils import make_grid

def create_raymond_lights() -> List[pyrender.Node]:
    """
    Return raymond light nodes for the scene.
    """
    thetas = np.pi * np.array([1.0 / 6.0, 1.0 / 6.0, 1.0 / 6.0])
    phis = np.pi * np.array([0.0, 2.0 / 3.0, 4.0 / 3.0])

    nodes = []

    for phi, theta in zip(phis, thetas):
        xp = np.sin(theta) * np.cos(phi)
        yp = np.sin(theta) * np.sin(phi)
        zp = np.cos(theta)

        z = np.array([xp, yp, zp])
        z = z / np.linalg.norm(z)
        x = np.array([-z[1], z[0], 0.0])
        if np.linalg.norm(x) == 0:
            x = np.array([1.0, 0.0, 0.0])
        x = x / np.linalg.norm(x)
        y = np.cross(z, x)

        matrix = np.eye(4)
        matrix[:3,:3] = np.c_[x,y,z]
        nodes.append(pyrender.Node(
            light=pyrender.DirectionalLight(color=np.ones(3), intensity=1.0),
            matrix=matrix
        ))

    return nodes

class Renderer:

    def __init__(self, cfg: CfgNode, faces: np.array):
        """
        Wrapper around the pyrender renderer to render SMPL meshes.
        Args:
            cfg (CfgNode): Model config file.
            faces (np.array): Array of shape (F, 3) containing the mesh faces.
        """
        self.cfg = cfg
        self.focal_length = cfg.EXTRA.FOCAL_LENGTH
        self.img_res = cfg.MODEL.IMAGE_SIZE
        self.renderer = pyrender.OffscreenRenderer(viewport_width=self.img_res,
                                       viewport_height=self.img_res,
                                       point_size=1.0)

        self.camera_center = [self.img_res // 2, self.img_res // 2]
        self.faces = faces


    def visualize_full_img(self, vertices, camera_translation, images, nrow=3, padding=2, focal_length=5000, cam_cx=0, cam_cy=0, img_w=1920, img_h=1080):
        # images_np = np.transpose(images, (0,2,3,1))
        rend_imgs = []
        # images: np array,
        # import pdb; pdb.set_trace()
        images = images[0:img_h, 0:img_w]
        for i in range(vertices.shape[0]):
            # focal_length = self.focal_length
            rend_img = torch.from_numpy(np.transpose(self.__call__(vertices[i], camera_translation[i], images, side_view=False,
                                                                   focal_length=focal_length, cam_cx=cam_cx, cam_cy=cam_cy, img_w=img_w, img_h=img_h),
                                                     (2,0,1))).float()
            rend_img_side = torch.from_numpy(np.transpose(self.__call__(vertices[i], camera_translation[i], images, side_view=True,
                                                                        focal_length=focal_length, cam_cx=cam_cx, cam_cy=cam_cy, img_w=img_w, img_h=img_h),
                                                          (2, 0, 1))).float()
            rend_imgs.append(torch.from_numpy(images).permute(2,0,1))
            rend_imgs.append(rend_img)
            rend_imgs.append(rend_img_side)
        rend_imgs = make_grid(rend_imgs, nrow=nrow, padding=padding)
        rend_imgs = 255 * np.transpose(rend_imgs.numpy()[::-1], (1, 2, 0))
        h, w, _ = rend_imgs.shape
        rend_imgs = cv2.resize(src=rend_imgs, dsize=(int(w/3), int(h/3)), interpolation=cv2.INTER_AREA)
        return rend_imgs

    def visualize(self, vertices, camera_translation, images, focal_length=5000, cam_cx=0, cam_cy=0, img_w=1920, img_h=1080):
        # images_np = np.transpose(images, (0,2,3,1))
        # rend_imgs = []
        # images: np array,
        # import pdb; pdb.set_trace()
        images = images[0:img_h, 0:img_w]
        for i in range(vertices.shape[0]):
            # focal_length = self.focal_length
            rend_img = torch.from_numpy(np.transpose(self.__call__(vertices[i], camera_translation[i], images, side_view=False,
                                                                   focal_length=focal_length, cam_cx=cam_cx, cam_cy=cam_cy, img_w=img_w, img_h=img_h),
                                                     (2,0,1))).float()
            # rend_imgs.append(rend_img)
        # rend_imgs = make_grid(rend_imgs, nrow=nrow, padding=padding)
        rend_img = 255 * np.transpose(rend_img.numpy()[::-1], (1, 2, 0))
        h, w, _ = rend_img.shape
        # rend_imgs = cv2.resize(src=rend_imgs, dsize=(int(w/3), int(h/3)), interpolation=cv2.INTER_AREA)
        return rend_img

    def __call__(self,
                vertices: np.array,
                camera_translation: np.array,
                image: torch.Tensor,
                full_frame: bool = False,
                 resize=None, side_view=False,
                 rot_angle=90,
                 focal_length=5000,
                 cam_cx=0, cam_cy=0,
                 img_w=1920, img_h=1080,
                 imgname: Optional[str] = None) -> np.array:
        """
        Render meshes on input image
        Args:
            vertices (np.array): Array of shape (V, 3) containing the mesh vertices.
            camera_translation (np.array): Array of shape (3,) with the camera translation.
            image (torch.Tensor): Tensor of shape (3, H, W) containing the image crop with normalized pixel values.
            full_frame (bool): If True, then render on the full image.
            imgname (Optional[str]): Contains the original image filenamee. Used only if full_frame == True.
        """
        
        # if full_frame:
        #     image = cv2.imread(imgname).astype(np.float32)[:, :, ::-1] / 255.
        # else:
        #     image = image.clone() * torch.tensor(self.cfg.MODEL.IMAGE_STD, device=image.device).reshape(3,1,1)
        #     image = image + torch.tensor(self.cfg.MODEL.IMAGE_MEAN, device=image.device).reshape(3,1,1)
        #     image = image.permute(1, 2, 0).cpu().numpy()

        renderer = pyrender.OffscreenRenderer(viewport_width=img_w,
                                              viewport_height=img_h,
                                              point_size=1.0)
        material = pyrender.MetallicRoughnessMaterial(
            metallicFactor=0.0,
            alphaMode='OPAQUE',
            baseColorFactor=(1.0, 1.0, 0.9, 1.0))

        camera_translation[0] *= -1.

        mesh = trimesh.Trimesh(vertices.copy(), self.faces.copy())
        if side_view:
            rot = trimesh.transformations.rotation_matrix(np.radians(rot_angle), [0, 1, 0])
            mesh.apply_transform(rot)
        rot = trimesh.transformations.rotation_matrix(np.radians(180), [1, 0, 0])
        mesh.apply_transform(rot)
        mesh = pyrender.Mesh.from_trimesh(mesh, material=material)

        scene = pyrender.Scene(bg_color=[0.0, 0.0, 0.0, 0.0],
                               ambient_light=(0.3, 0.3, 0.3))
        scene.add(mesh, 'mesh')

        camera_pose = np.eye(4)
        camera_pose[:3, 3] = camera_translation
        # camera_center = [image.shape[1] / 2., image.shape[0] / 2.]
        camera_center = [cam_cx, cam_cy]
        if side_view:
            camera = pyrender.IntrinsicsCamera(fx=focal_length*0.4, fy=focal_length*0.4,
                                               cx=camera_center[0], cy=camera_center[1])
        else:
            camera = pyrender.IntrinsicsCamera(fx=focal_length, fy=focal_length,
                                               cx=camera_center[0], cy=camera_center[1])
        scene.add(camera, pose=camera_pose)


        light_nodes = create_raymond_lights()
        for node in light_nodes:
            scene.add_node(node)

        color, _ = renderer.render(scene, flags=pyrender.RenderFlags.RGBA)
        color = color.astype(np.float32) / 255.0
        valid_mask = (color[:, :, -1] > 0)[:, :, np.newaxis]
        if not side_view:
            output_img = (color[:, :, :3] * valid_mask +
                          (1 - valid_mask) * image)
        else:
            output_img = color[:, :, :3]
        if resize is not None:
            output_img = cv2.resize(output_img, resize)

        output_img = output_img.astype(np.float32)
        renderer.delete()
        return output_img

def render_with_scene(input_img, static_scene, body, fx, fy, cx, cy, zoom_out_factor, renderer, camera_pose, light, material):
    camera = pyrender.camera.IntrinsicsCamera(
        fx=fx, fy=fy,
        cx=cx, cy=cy)
    scene = pyrender.Scene(bg_color=[0.0, 0.0, 0.0, 0.0], ambient_light=(0.3, 0.3, 0.3))
    scene.add(camera, pose=camera_pose)
    scene.add(light, pose=camera_pose)
    body_mesh = pyrender.Mesh.from_trimesh(body, material=material)
    scene.add(body_mesh, 'body_mesh')
    color, _ = renderer.render(scene)
    valid_mask = (color[:, :, -1] > 0)[:, :, np.newaxis]
    pred_render_img = (color[:, :, :3] * valid_mask + (1 - valid_mask) * input_img)
    pred_render_img = np.ascontiguousarray(pred_render_img, dtype=np.uint8)


    ########## render body + scene
    camera = pyrender.camera.IntrinsicsCamera(
        fx=fx/zoom_out_factor, fy=fy/zoom_out_factor,
        cx=cx, cy=cy)

    body_mesh = pyrender.Mesh.from_trimesh(body, material=material)
    static_scene_mesh = pyrender.Mesh.from_trimesh(static_scene)

    scene = pyrender.Scene()
    scene.add(camera, pose=camera_pose)
    scene.add(light, pose=camera_pose)
    scene.add(static_scene_mesh, 'scene_mesh')
    scene.add(body_mesh, 'body_mesh')
    render_img_scene_pred, _ = renderer.render(scene)
    render_img_scene_pred = render_img_scene_pred[:, :, ::-1]
    render_img_scene_pred = np.ascontiguousarray(render_img_scene_pred)

    # rot = trimesh.transformations.rotation_matrix(np.radians(90), [0, 1, 0])
    # static_scene.apply_transform(rot)
    # body.apply_transform(rot)
    # body_mesh = pyrender.Mesh.from_trimesh(body, material=material)
    # static_scene_mesh = pyrender.Mesh.from_trimesh(static_scene)
    # scene = pyrender.Scene()
    # scene.add(camera, pose=camera_pose)
    # scene.add(light, pose=camera_pose)
    # scene.add(static_scene_mesh, 'scene_mesh')
    # scene.add(body_mesh, 'body_mesh')
    # render_img_scene_pred_rot, _ = renderer.render(scene)
    # render_img_scene_pred_rot = render_img_scene_pred_rot[:, :, ::-1]
    # render_img_scene_pred_rot = np.ascontiguousarray(render_img_scene_pred_rot)

    output = np.concatenate([input_img, pred_render_img, render_img_scene_pred], axis=1)
    return output


def convert_pare_to_full_img_cam(
        pare_cam, bbox_height, bbox_center,
        img_w, img_h, focal_length, crop_res=224):
    # Converts weak perspective camera estimated by PARE in
    # bbox coords to perspective camera in full image coordinates
    # from https://arxiv.org/pdf/2009.06549.pdf

    # import pdb; pdb.set_trace()
    s, tx, ty = pare_cam[:, 0], pare_cam[:, 1], pare_cam[:, 2]  # pare_cam: [bs, 3]
    # res = 224
    r = bbox_height / crop_res
    tz = 2 * focal_length / (r * crop_res * s)  # [bs]

    cx = 2 * (bbox_center[:, 0] - (img_w / 2.)) / (s * bbox_height)
    cy = 2 * (bbox_center[:, 1] - (img_h / 2.)) / (s * bbox_height)

    cam_t = torch.stack([tx + cx, ty + cy, tz], dim=-1)

    return cam_t
