import os
import re
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

def generate_3D_edges(file_path='/hdd1/zhangkai/256X40',mesh_resol=3,atlas_resol=1024,num_atlas = 40,Edge_Id = None,ALL_Edge = None):

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
    Inner_Edge_3D = []
    Out_Edge_3D = []
    Inner_Edge_NM = []
    Out_Edge_NM = []
    for i in range(len(Edge_Id)):

        # 内边界
        atlas_mesh = atlas_mesh_list[Edge_Id[i][0]]
        outlier_mask, pixel_facet_IDM, barycentric = initialize_atlas_uv_map(resol=atlas_resol,mesh=atlas_mesh)
        image_interpolate = generate_init_GM(barycentric=barycentric,outlier_mask=outlier_mask,
            pixel_facet_IDM=pixel_facet_IDM,resol=atlas_resol,mesh=atlas_mesh,atlas_id=Edge_Id[i][0])
        image_verts_normal = generate_init_NM(barycentric=barycentric,outlier_mask=outlier_mask,
            pixel_facet_IDM=pixel_facet_IDM,resol=atlas_resol,mesh = atlas_mesh,atlas_id=Edge_Id[i][0])
        XY = (ALL_Edge[0][i] * (atlas_resol-1)).astype(int)
        Inner_Edge_3D += [image_interpolate[XY[:,1],XY[:,0],:]]
        Inner_Edge_NM += [image_verts_normal[XY[:,1],XY[:,0],:]]

        # 外边界
        atlas_mesh = atlas_mesh_list[Edge_Id[i][1]]
        outlier_mask, pixel_facet_IDM, barycentric = initialize_atlas_uv_map(resol=atlas_resol,mesh=atlas_mesh)
        image_interpolate = generate_init_GM(barycentric=barycentric,outlier_mask=outlier_mask,
            pixel_facet_IDM=pixel_facet_IDM,resol=atlas_resol,mesh=atlas_mesh,atlas_id=Edge_Id[i][1])
        image_verts_normal = generate_init_NM(barycentric=barycentric,outlier_mask=outlier_mask,
            pixel_facet_IDM=pixel_facet_IDM,resol=atlas_resol,mesh = atlas_mesh,atlas_id=Edge_Id[i][1])

        XY = (ALL_Edge[1][i] * (atlas_resol-1)).astype(int)
        Out_Edge_3D += [image_interpolate[XY[:,1],XY[:,0],:]]
        Out_Edge_NM += [image_verts_normal[XY[:,1],XY[:,0],:]]
        print("atlas_id:",Edge_Id[i],"done!","当前进度:%s/%s"%(i,len(Edge_Id)))

    ALL_Edge_3D = []
    ALL_Edge_3D.append(Inner_Edge_3D)
    ALL_Edge_3D.append(Out_Edge_3D)
    ALL_Edge_NM = []
    ALL_Edge_NM.append(Inner_Edge_NM)
    ALL_Edge_NM.append(Out_Edge_NM)

    return ALL_Edge_3D,ALL_Edge_NM

if __name__ == '__main__':
    Edge_Id = torch.load("Edge_Id_1024.pt")
    ALL_Edge = torch.load("ALL_Edge_1024.pt")
    ALL_Edge_3D,ALL_Edge_NM = generate_3D_edges(file_path='/hdd1/zhangkai/256X40',mesh_resol=3,atlas_resol=1024,num_atlas = 40,Edge_Id = Edge_Id,ALL_Edge = ALL_Edge)
    for i in range(2):
        for j in range(len(Edge_Id)):
            ALL_Edge_3D[i][j] = torch.from_numpy(ALL_Edge_3D[i][j])
            ALL_Edge_NM[i][j] = torch.from_numpy(ALL_Edge_NM[i][j])
            
    torch.save(ALL_Edge_3D,"ALL_Edge_3D_1024_torch.pt")
    torch.save(ALL_Edge_NM,"ALL_Edge_NM_1024_torch.pt")

    ### 上色保存点云
    XYZ = np.array([[0,0,0]])
    RGB = np.array([[0,0,0]])

    for i in range(len(Edge_Id)):
        # 内边界上色
        XYZ = np.concatenate((XYZ,ALL_Edge_3D[0][i]),axis = 0)
        row = len(ALL_Edge_3D[0][i])
        rgb = np.zeros((row,3))
        rgb[:,0] = (10 + 5*i + 10 * Edge_Id[i][0]/200)%255
        rgb[:,1] = (255 - 10 * Edge_Id[i][0])%230
        rgb[:,2] = 10 + (33 * i)%240
        RGB = np.concatenate((RGB,rgb),axis = 0)
        # 外边界上色
        XYZ = np.concatenate((XYZ,ALL_Edge_3D[1][i]),axis = 0)
        row = len(ALL_Edge_3D[1][i])
        rgb = np.zeros((row,3))
        rgb[:,0] = (10 + 5*i + 10 * Edge_Id[i][1]/200)%255
        rgb[:,1] = (255 - 10 * Edge_Id[i][1])%230
        rgb[:,2] = 10 + (33 * i)%240
        RGB = np.concatenate((RGB,rgb),axis = 0)   
    
    XYZ = XYZ[1:]
    RGB = RGB[1:]
    print(XYZ.shape,RGB.shape)

    save2ply(ply_filePath="visulize/3D/edges_1024.ply",xyz_np=XYZ,rgb_np=RGB)


    # pad:
    maxsize = 1
    for i in range(2):
        for j in range(len(Edge_Id)):
            if len(ALL_Edge_3D[i][j]) > maxsize:
                maxsize = len(ALL_Edge_3D[i][j])
    
    for i in range(len(Edge_Id)):
        Inner_Edge_i = ALL_Edge_3D[0][i]
        pad_size = maxsize - Inner_Edge_i.shape[0]
        pad = nn.ZeroPad2d(padding=(0,0,0,pad_size))
        ALL_Edge_3D[0][i] = pad(Inner_Edge_i)
    
        Outer_Edge_i = ALL_Edge_3D[1][i]
        pad_size = maxsize - Outer_Edge_i.shape[0]
        pad = nn.ZeroPad2d(padding=(0,0,0,pad_size))
        ALL_Edge_3D[1][i] = pad(Outer_Edge_i)

    for i in range(len(Edge_Id)):
        Inner_Edge_i = ALL_Edge_NM[0][i]
        pad_size = maxsize - Inner_Edge_i.shape[0]
        pad = nn.ZeroPad2d(padding=(0,0,0,pad_size))
        ALL_Edge_NM[0][i] = pad(Inner_Edge_i)
    
        Outer_Edge_i = ALL_Edge_NM[1][i]
        pad_size = maxsize - Outer_Edge_i.shape[0]
        pad = nn.ZeroPad2d(padding=(0,0,0,pad_size))
        ALL_Edge_NM[1][i] = pad(Outer_Edge_i)

    torch.save(ALL_Edge_3D,"ALL_Edge_3D_1024_torch_padded.pt")
    torch.save(ALL_Edge_NM,"ALL_Edge_NM_1024_torch_padded.pt")

    print("All done!")

