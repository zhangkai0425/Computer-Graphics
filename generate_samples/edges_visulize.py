import os
import sys
import math
import cv2 as cv
import torch
from utils import *
import numpy as np
import torch.nn as nn
from PIL import Image
import torch.nn.functional as F
import matplotlib.pyplot as plt
from getedges import getedges
from pytorch3d.structures import Meshes, Pointclouds, join_meshes_as_scene, join_meshes_as_batch
from pytorch3d.renderer.mesh.rasterizer import Fragments
from pytorch3d.renderer.mesh.textures import TexturesUV

def visulize_edges(file_path = '/hdd1/zhangkai/256X40',mesh_resol = 3,atlas_resol = 128,selected_atlas_indices = [0]):
    # Set the device
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")
        print("WARNING: CPU only, this will be slow!")

    # set parameters:之后可以改成调用self.parameters的形式,目前还是单独设置方便些
    '''
    file_path = '/hdd1/zhangkai/256X40'
    mesh_resol = 3 # 读取3.obj文件

    atlas_resol = 128
    selected_atlas_indices = [0]
    '''
    '''
    file_path = '/hdd1/zhangkai/256X40'
    mesh_resol = 3 # 读取3.obj文件

    atlas_resol = 128
    selected_atlas_indices = [0]

    '''

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

    edges,valid_id = getedges(file_path+'/%s.obj'%mesh_resol)
    # edges = convert_edges(edges=edges,verts_to_uv=verts_to_uv)
    # 生成空白黑图
    picture = np.zeros((atlas_resol,atlas_resol))

    thisedge = edges[selected_atlas_indices[0]]
    edges_list,color_i = get_color(thisedge)
    i = 0
    print("debug2:",edges_list)
    for j in edges_list:
        atlas_edges_1 = [(i-1) for i in thisedge[j]]
        point_set = verts_uvs[atlas_edges_1,:]
        picture[point_set[:,1],point_set[:,0]] = 50+(200*i)//len(edges_list)
        # picture[point_set[:,1],point_set[:,0]] = 255
        i += 1
    visulize_save_folder = 'visulize/edges'
    # widthImg = cv.GaussianBlur(gaussImg,(3,3),0)
    if not os.path.exists(visulize_save_folder):
        os.makedirs(visulize_save_folder)
    plt.imsave(os.path.join(visulize_save_folder, 'edges_points_atlas%s.png'%selected_atlas_indices[0]), picture)

if __name__ == '__main__':
    # visulize_edges(file_path = '/hdd1/zhangkai/256X40',mesh_resol = 3,atlas_resol = 128,selected_atlas_indices = [4])
    for i in range(30):
        visulize_edges(file_path = '/hdd1/zhangkai/256X40',mesh_resol = 3,atlas_resol = 128,selected_atlas_indices = [i])
