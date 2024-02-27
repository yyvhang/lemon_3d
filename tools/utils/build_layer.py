import torch
from tools.smplx import build_layer
from tools import smplx
import numpy as np
import open3d as o3d

def build_smplh_mesh(params):
    model = build_layer(
        model_path = 'smpl_models/smplh/SMPLH_NEUTRAL.pkl',
        model_type = "smplh",
        use_pca=False,
        gender = 'neutral',
        ext = 'pkl',
        num_betas=16
    )
    model.eval()
    output = model(betas = params['shape'].float(), body_pose = params['body_pose'].float(), left_hand_pose = params['left_hand_pose'].float(), 
                   right_hand_pose = params['right_hand_pose'].float(), global_orient = params['global_orient'].float(), return_verts=True)
    vertices = output.vertices.detach()
    faces = model.faces.astype(np.int32)

    return vertices, faces
    
def build_smplh_template(root_pose, body_pose, lhand_pose, rhand_pose, shape, cam_trans, device):

    model = smplx.create(
        model_path = 'smpl_models/smplh/SMPLH_NEUTRAL.pkl',
        model_type = "smplh",
        use_pca=False,
        gender = 'neutral',
        ext = 'pkl',
        num_betas=10
    )
    model.eval()
    model.cuda()
    B = root_pose.shape[0]
    betas = torch.from_numpy(shape).view(B, -1).float().to(device)
    transl = torch.from_numpy(cam_trans).view(B, -1).float().to(device)
    body_pose = torch.from_numpy(body_pose).view(B, -1).float().to(device)
    left_hand_pose = torch.from_numpy(lhand_pose).view(B, -1).float().to(device)
    right_hand_pose = torch.from_numpy(rhand_pose).view(B, -1).float().to(device)
    root_pose = torch.from_numpy(root_pose).view(B, -1).float().to(device)
    output = model(betas = betas, body_pose = body_pose, left_hand_pose = left_hand_pose, right_hand_pose = right_hand_pose,
    global_orient = root_pose, return_verts=True)

    vertices = output.vertices
    faces = model.faces.astype(np.int32)
    return vertices,faces

Vector3dVector = o3d.utility.Vector3dVector
Vector3iVector = o3d.utility.Vector3iVector
TriangleMesh = o3d.geometry.TriangleMesh

def create_mesh(vertices, faces, colors=None, **kwargs):
    mesh = TriangleMesh()
    mesh.vertices = Vector3dVector(vertices)
    mesh.triangles = Vector3iVector(faces)
    mesh.compute_vertex_normals()
    if colors is not None:
        mesh.vertex_colors = Vector3dVector(colors)
    else:
        r_c = np.random.random(3)
        mesh.paint_uniform_color(r_c)
    return mesh

def get_faces():

    model = build_layer(
        model_path = 'smpl_models/smplh/SMPLH_NEUTRAL.pkl',
        model_type = "smplh",
        use_pca=False,
        gender = 'neutral',
        ext = 'pkl',
        num_betas=10
    )

    faces = model.faces.astype(np.int32)
    return faces

def Pelvis_norm(vertices, device, H36M_correct_path='tools/utils/data/J_regressor_h36m_correct.npy'):

    J_regressor_h36m_correct = torch.from_numpy(np.load(H36M_correct_path)).float().to(device)
    joints = torch.einsum('bik,ji->bjk', [vertices, J_regressor_h36m_correct])
    pelvis = joints[:,0,:]
    norm_vertices = vertices - pelvis[:, None, :]
    return norm_vertices, pelvis