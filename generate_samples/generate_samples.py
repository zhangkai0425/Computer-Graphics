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
from pytorch3d.structures import Meshes, Pointclouds, join_meshes_as_scene, join_meshes_as_batch
from pytorch3d.renderer.mesh.rasterizer import Fragments
from pytorch3d.renderer.mesh.textures import TexturesUV

def subfunction_generate_samples(atlas_mesh,atlas_id,resol,sample_width):

    # Set the device
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")
        print("WARNING: CPU only, this will be slow!")

    mesh=atlas_mesh.to(device)	
    faces_uvs = mesh.textures.faces_uvs_padded()[0] # (num_view, num_faces, 3) --> (num_faces, 3)
    verts_uvs = mesh.textures.verts_uvs_padded()[0] # (num_view, num_verts, 2) --> (num_verts, 2)
    face_verts_uvs = verts_uvs[faces_uvs]   # (num_faces, 3, 2)

    selected_atlas_indices = [5] # 选取的 atlas编号,范围是1-40

    num_atlas = len(selected_atlas_indices)

    atlas_uv_map = np.zeros((resol, resol, 1), dtype=np.float64)
    # TODO:之后要记得GPU上运行
    face_verts_uvs_atlas_image = (face_verts_uvs * (resol - 1.))\
    .type(torch.cuda.LongTensor).cpu().numpy()  # (num_faces, 3, 2)
    # face_verts_uvs_atlas_image = (face_verts_uvs * (resol - 1.)).type(torch.LongTensor).cpu().numpy()  # (num_faces, 3, 2)
    # print(face_verts_uvs_atlas_image.shape)
    # print(face_verts_uvs_atlas_image.min(), face_verts_uvs_atlas_image.max())

    atlas_uv_map[face_verts_uvs_atlas_image.reshape(-1, 2)[:, 1], face_verts_uvs_atlas_image.reshape(-1, 2)[:, 0], :] = 1.0
    # debug_save_folder = 'debug/atlas_%s'%atlas_id
    # # TODO:1.图1
    # if not os.path.exists(debug_save_folder):
    # 	os.makedirs(debug_save_folder)
    image = atlas_uv_map[..., 0]	

    # TODO:2.图2
    x = np.arange(resol)
    y = np.arange(resol)
    grid_x, grid_y = np.meshgrid(x, y)
    map_q_coords = np.concatenate((grid_x[..., None], grid_y[..., None]), axis=-1).reshape(-1, 2)    # (H*W, 2)
    verts_coords = (face_verts_uvs * (resol - 1.)).cpu().numpy()   # (N_f, 3, 2)

    # use vector cross-product to identify the point-triangle correspondence.
    AP = map_q_coords[:, None, :] - verts_coords[None, :, 0, :]
    BP = map_q_coords[:, None, :] - verts_coords[None, :, 1, :]
    CP = map_q_coords[:, None, :] - verts_coords[None, :, 2, :]
    CA = verts_coords[None, :, 0, :] - verts_coords[None, :, 2, :]
    AB = verts_coords[None, :, 1, :] - verts_coords[None, :, 0, :]
    BC = verts_coords[None, :, 2, :] - verts_coords[None, :, 1, :]
    cross_A = np.cross(BP, BC)  # the cross-product for identifying vert v0.
    cross_B = np.cross(CP, CA)  # the cross-product for identifying vert v1.
    cross_C = np.cross(AP, AB)  # the cross-product for identifying vert v2.
    flag_A = cross_A < 0.0
    flag_B = cross_B < 0.0
    flag_C = cross_C < 0.0
    flag_total = (flag_A == flag_B) * (flag_A == flag_C)    # (H*W, N_f)
    # filter outlier pixels that belong to no facets.
    outlier_mask = (flag_total.sum(axis=-1) == 1.0)
    flag_total = flag_total[outlier_mask]
    vis_mask_flag = flag_total.sum(axis=-1)
    print(vis_mask_flag.min(), vis_mask_flag.max())

    if vis_mask_flag.max() == 0.0:
        raise ValueError('pixels belong to no projected triangle facets !')

    image = np.zeros((resol, resol), dtype=np.float64).flatten()
    image[outlier_mask] = 1.0
    image = image.reshape((resol, resol))

    plt.imsave('tmp/origin_pic.png', image)
    ######## 图像处理部分 #########
    # TODO: 提取uv_map中atlas的边界
    # DONE 
    img = cv.imread('tmp/origin_pic.png')
    # 显示原图
    # 转化为灰度图-二值图
    gray_img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    gray_img[gray_img>100] = 255
    gray_img[gray_img<100] = 0

    # 提取边缘 Canny边缘检测算法
    # https://blog.csdn.net/weixin_42709563/article/details/105700972?utm_medium=distribute.pc_relevant.none-task-blog-2~default~baidujs_title~default-4.pc_relevant_default&spm=1001.2101.3001.4242.3&utm_relevant_index=7
    # print(img.shape)
    # print(gray_img.shape)
    blur = cv.GaussianBlur(gray_img,(5,5),0)
    gaussImg = cv.Canny(blur,5,5)
    # TODO: 平滑将边界拓展为固定的宽度
    # DONE
    # 实现给定宽度的边缘生成
    kernel_size = math.ceil(2 * sample_width - 1)
    widthImg = cv.GaussianBlur(gaussImg,(kernel_size,kernel_size),0)

    gray_edge = widthImg
    gray_edge[gray_edge>0] = 255
    gray_edge[gray_edge<0] = 0
    # 阈值二值化
    # 内部取点,取交集
    inner_edge = np.zeros(gray_edge.shape)
    inner_edge[(gray_edge==255)*(gray_img==255)] = 255

    return inner_edge

def generate_samples(file_path,mesh_resol=3,num_atlas=40,sample_width = 2,atlas_resol = 128):
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
    num_atlas = 40
    
    sample_width = 2 # sample_width:决定边界采样点的宽度,即决定高斯核的宽度:
    # 为了保证高斯核的阶为奇数:kernel_size = math.ceil(2 * sample_width - 1)
    atlas_resol = 128
    '''
    ### TODO:此处是选取了全部atlas,以后可以改
    selected_atlas_indices = [i for i in range(num_atlas)]

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

    # print(edges)
    # print(edges[0])
    edges_samples = []
    for i in range(len(atlas_mesh_list)):
        if i not in valid_id:
            edges_samples.append([[-1]]*len(atlas_mesh_list))
            continue
        each_samples = []
        atlas_mesh = atlas_mesh_list[i]
        atlas_edges = edges[i] # 获得某一个atlas的边界列表
        inner_edge = subfunction_generate_samples(atlas_mesh = atlas_mesh,atlas_id = i,resol = atlas_resol,sample_width=sample_width)
        
        XY = get_all_samples(resol = atlas_resol,inner_edge=inner_edge,edge_value=255)
        print("XY的长度:",len(XY))
        Samples_with_atlas = divide_samples(XY=XY,atlas_edges=atlas_edges,verts_uvs=verts_uvs,atlas_resol=atlas_resol,mesh=atlas_mesh)

        # print(Samples_with_atlas[Samples_with_atlas[:,2]==7][:,0:2]/(atlas_resol-1))

        for j in range(len(edges)):
            if edges[i][j] == [-1]:
                each_samples += [[-1]]
                continue
            else:
                each_samples += [Samples_with_atlas[Samples_with_atlas[:,2]==j][:,0:2]/(atlas_resol-1)]
        edges_samples.append(each_samples)
        print("atlas:",i,"已完成!")
    return edges_samples,valid_id

if __name__ == '__main__':
    edges_samples,valid_id = generate_samples(file_path = '/hdd1/zhangkai/256X40',mesh_resol=3,num_atlas=40,sample_width = 2,atlas_resol = 128)
    print(valid_id)
    # print(edges_samples[38])
    print(edges_samples[38][0])
    
