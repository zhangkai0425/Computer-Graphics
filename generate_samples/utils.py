import os
import torch
import numpy as np
from pytorch3d.io import load_obj, save_obj
from pytorch3d.structures import Meshes, Pointclouds, join_meshes_as_scene, join_meshes_as_batch
from pytorch3d.renderer.mesh.rasterizer import Fragments
from pytorch3d.renderer.mesh.textures import TexturesUV
'''
    全局注意:务必注意点的坐标 X,Y中 X对应列,Y对应行！！！！！！
'''
'''    
    获得采样点全体UV(XY)坐标的函数
    get_all_samples(resol,inner_edge)
    输入:inner_edge:已经处理完的边界采样后的二值图(且只内采样); resol:图像分辨率
    输出:采样点的坐标集合 shape:(num_samples,2)
'''

def get_all_samples(resol,inner_edge,edge_value):
    Image_flat = inner_edge.reshape(-1)
    XY_circle = np.arange(resol*resol)
    X = (XY_circle%resol)[Image_flat==edge_value]
    Y = (XY_circle//resol)[Image_flat==edge_value]
    XY = np.zeros((len(X),2))
    XY[:,0] = X
    XY[:,1] = Y
    return XY.astype(int)
# TODO:Tested !测试完成,函数无误!

'''
    计算点和点集最小距离的函数:
    cal_set_distance(point,point_set):
    输入:单点:point shape(1,2);点集point_set shape(set_size,2)
    输出:最小距离

'''
def cal_set_distance(point,point_set):
    d = np.sqrt(np.sum((point_set - point)**2,axis=1))
    return min(d)

# TODO:Tested !测试完成,函数无误!

''' 
    计算某一点(X,Y)最近的atlas
    divide_samples(XY,atlas_edges,verts_uvs)
    输入:
    XY 是所有采样点的坐标集合,XY.shape : (num_samples,2),
    atlas_edges 是某个atlas的严格边界集合,由getedges得到,是getedges的某一行
    verts_uvs:存储点的编号和uv坐标的矩阵,是全程序最重要的变量
    输出:给出边界点坐标集和对应的相邻的atlas标号 矩阵 Samples_with_atlas shape(nums_samples,3)
    前两列代表XY坐标,第三列代表对应的相邻的atlas编号
'''
def divide_samples(XY,atlas_edges,verts_uvs):
    num_points = XY.shape[0]
    # 这里用循环,第一重循环理论上也许可以去掉,第二和三重没有必要去掉,也似乎不容易去掉
    belong = np.zeros((num_points,1)) # 记录每个点集的归属:属于哪一段边界
    for i in range(num_points):
        point = XY[i,:]
        d = []
        atlas_id = []
        for j in range(len(atlas_edges)):
            # 如果没有相邻的边界点,continue
            if atlas_edges[j] == [-1]:
                continue
            else:
                atlas_edges_1 = [(i-1) for i in atlas_edges[j]]
                point_set = verts_uvs[atlas_edges_1,:]
                d.append(cal_set_distance(point=point,point_set=point_set))
                atlas_id.append(j)
        # 根据最近的点和点集的距离判断point属于同哪个atlas相邻的边界    
        belong_id = d.index(min(d))
        belong[i] = atlas_id[belong_id]
        Samples_with_atlas = np.concatenate((XY,belong),axis = 1).astype(int)
    return Samples_with_atlas

# TODO:Tested !测试完成,函数无误!

def convert_edges(edges,verts_to_uv):
    atlas_num = len(edges)
    for i in range(atlas_num):
        for j in range(atlas_num):
            if edges[i][j] == [-1]:
                continue
            else:
                uv_list = []
                for verts in edges[i][j]:
                    uv = verts_to_uv[verts_to_uv[:,0]==verts][:,1]
                    uv_list += list(set(uv.tolist()))
                edges[i][j] = uv_list
    return edges

'''
原来triutils.py中的函数,为了不需要在原文件夹下运行,因此单独拿出来了
'''
def coarse_atlas_import(file_path, mesh_name_pattern='#.obj', mesh_resol=3,
                        device=torch.device("cpu"), save_texture_atlas=False):

    verts, faces, aux = load_obj(
        f=os.path.join(file_path, mesh_name_pattern.replace('#', str(mesh_resol))),
        load_textures=True, create_texture_atlas=False, device=device)

    verts_idx, normals_idx, textures_idx, materials_idx = faces

    normals, verts_uvs, material_colors, texture_images, texture_atlas = aux

    face_verts_uv = verts_uvs[textures_idx]  # (N_f, 3, 2)

    textures = torch.stack(
        [texture_images[idx] for idx in list(texture_images.keys())],
        dim=0
    )  # (N_a, H, W, 3)

    embedding_idx_list = torch.FloatTensor(
        np.array(list(range(textures.shape[0])))
    ).to(device)

    # fake 3x3 texture-atlas-map for embedding indexing.
    embedding_textures = torch.stack(
        [
            coeff * torch.ones(
                # textures.shape[1], textures.shape[2], textures.shape[3],
                3, 3, textures.shape[3],
                dtype=embedding_idx_list.dtype, device=device)
            for coeff in embedding_idx_list
        ]
    )

    # Atlas partition:
    materials_indices_list = list(set(materials_idx.tolist()))
    # print(materials_indices_list)
    faces_uvs_atlas = []
    verts_uvs_atlas = []
    faces_atlas = []
    verts_atlas = []

    # partition the mesh in terms of altas.
    for i in materials_indices_list:
        mask = (materials_idx == i)
        # max_num_faces = mask.sum() if mask.sum() > max_num_faces else max_num_faces

        # re-index the textures_idx in order.
        textures_idx_revise = torch.LongTensor(
            np.array(list(range(textures_idx[mask].shape[0] * textures_idx[mask].shape[1])))
        ).to(device)

        faces_uvs_atlas.append(textures_idx_revise.view(-1, 3))
        verts_uvs_atlas.append(verts_uvs[textures_idx[mask]].view(-1, 2))

        # re-index the verts_idx in order.
        verts_idx_revise = torch.LongTensor(
            np.array(list(range(verts_idx[mask].shape[0] * verts_idx[mask].shape[1])))
        ).to(device)

        faces_atlas.append(verts_idx_revise.view(-1, 3))
        verts_atlas.append(verts[verts_idx[mask]].view(-1, 3))

    return verts_atlas, faces_atlas, verts_uvs_atlas, faces_uvs_atlas, textures.shape[0]

'''
原来uv_warping.py中的函数,为了不需要在原文件夹下运行,因此单独拿出来了
'''
def mesh_from_list(verts_atlas, faces_atlas, verts_uvs_atlas, faces_uvs_atlas,
                    z_embedding_length,selected_atlas_indices,device=torch.device("cpu")):
    """
        Returns:
            list of meshes and faces_atlas_id.
            atlas_mesh_list: list of pytorch3d Meshes object, each corresponding to a single Atlas.
                [Mesh_0, Mesh_1, Mesh_2, ...]

            (abandoned) selected_faces_atlas_id: list of atlas indicators defined on each face.

    """
    atlas_mesh_list = []

    # selected_atlas_indices = [2] # 选取的 atlas编号,范围是1-40
    for i, atlas_idx in enumerate(selected_atlas_indices):
        # faces_atlas_id = i * torch.ones(faces_uvs_atlas[atlas_idx].shape[0],
        #                                 dtype=verts_atlas[atlas_idx].dtype, device=device)
        embedding_textures = i * torch.ones(1, 3, 3, 3, dtype=verts_atlas[atlas_idx].dtype, device=device)

        # save fake texture-atlas (for embedding indexing)
        Textures_atlas = TexturesUV(
            # maps=textures,
            maps=embedding_textures,
            faces_uvs=faces_uvs_atlas[atlas_idx][None, ...],
            verts_uvs=verts_uvs_atlas[atlas_idx][None, ...]
        )

        mesh = Meshes(verts=verts_atlas[atlas_idx][None, ...],
                        faces=faces_atlas[atlas_idx][None, ...],
                        textures=Textures_atlas)

        atlas_mesh_list.append(mesh)

    return atlas_mesh_list