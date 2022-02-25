import os
import sys
import math
import cv2 as cv
sys.path.append("../../")
import torch
from utils import *
import numpy as np
import torch.nn as nn
from PIL import Image
import torch.nn.functional as F
import matplotlib.pyplot as plt
from getedges import getedges
from generate_samples import generate_samples
from pytorch3d.structures import Meshes, Pointclouds, join_meshes_as_scene, join_meshes_as_batch
from pytorch3d.renderer.mesh.rasterizer import Fragments
from pytorch3d.renderer.mesh.textures import TexturesUV


def generate_3D_samples(file_path='/hdd1/zhangkai/256X40',mesh_resol=3,atlas_resol=128,num_atlas = 40,sample_width = 2):
    edges_samples,valid_id = generate_samples(file_path = file_path,mesh_resol=mesh_resol,num_atlas=num_atlas,sample_width = sample_width,atlas_resol = atlas_resol)

    # file_path = '/hdd1/zhangkai/256X40'
    # mesh_resol = 3 # 读取3.obj文件
    # atlas_resol = 128
    # num_atlas = 40

    # '''
    # 上面是全局的参数
    # '''
    selected_atlas_indices = [i for i in range(num_atlas)]
    # Set the device
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")
        print("WARNING: CPU only, this will be slow!")

    verts, faces, aux = load_obj(file_path+'/%s.obj'%mesh_resol)
    verts_uvs = aux.verts_uvs
    verts_uvs = (verts_uvs * (atlas_resol - 1.)).type(torch.cuda.LongTensor).cpu().numpy()
    print('debug here:',verts_uvs.shape)
    verts_atlas, faces_atlas, verts_uvs_atlas, faces_uvs_atlas, z_embedding_length = coarse_atlas_import(
            file_path = file_path,
            mesh_resol = mesh_resol)
    atlas_mesh_list = mesh_from_list(
        verts_atlas, faces_atlas, verts_uvs_atlas, faces_uvs_atlas, z_embedding_length,selected_atlas_indices = selected_atlas_indices
    )
    print('atlas_mesh_list_num:',len(atlas_mesh_list))
    edges_samples_GM = []
    edges_samples_NM = []
    for i in range(len(atlas_mesh_list)):
        if i not in valid_id:
            edges_samples_GM.append([[-1]]*len(atlas_mesh_list))
            edges_samples_NM.append([[-1]]*len(atlas_mesh_list))
            continue
            
        atlas_mesh = atlas_mesh_list[i]
        outlier_mask, pixel_facet_IDM, barycentric = initialize_atlas_uv_map(resol=atlas_resol,mesh=atlas_mesh)
        image_verts_normal = generate_init_NM(barycentric=barycentric,outlier_mask=outlier_mask,pixel_facet_IDM=pixel_facet_IDM,
            resol=atlas_resol,mesh = atlas_mesh,atlas_id=i)
        image_interpolate = generate_init_GM(barycentric=barycentric,outlier_mask=outlier_mask,
            pixel_facet_IDM=pixel_facet_IDM,resol=atlas_resol,mesh=atlas_mesh,atlas_id=i)
        
        each_samples_GM = []
        each_samples_NM = []
        
        for j in range(len(edges_samples)):
            if len(edges_samples[i][j]) == 1:
                each_samples_GM += [[-1]]
                each_samples_NM += [[-1]]     
                continue
            else:
                XY = (edges_samples[i][j] * (atlas_resol-1)).astype(int)
                each_samples_GM += [image_interpolate[XY[:,1],XY[:,0],:]]
                each_samples_NM += [image_verts_normal[XY[:,1],XY[:,0],:]]
        edges_samples_GM.append(each_samples_GM)
        edges_samples_NM.append(each_samples_NM)
        print("atlas",i,":3D采样已完成!")

    return edges_samples_GM,edges_samples_NM,valid_id


if __name__ == '__main__':
    edges_samples_GM,edges_samples_NM,valid_id = generate_3D_samples(file_path='/hdd1/zhangkai/256X40',mesh_resol=3,atlas_resol=128,num_atlas = 40,sample_width = 2)
    print(len(edges_samples_GM))