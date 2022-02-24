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
from generate_samples import *
import torch.nn.functional as F
import matplotlib.pyplot as plt
from getedges import getedges
from pytorch3d.structures import Meshes, Pointclouds, join_meshes_as_scene, join_meshes_as_batch
from pytorch3d.renderer.mesh.rasterizer import Fragments
from pytorch3d.renderer.mesh.textures import TexturesUV

def visulize(sample_width = 2,selected_atlas_indices=[5],mesh_resol = 3):
    edges_samples,valid_id = generate_samples(file_path = '/hdd1/zhangkai/256X40',mesh_resol=3,num_atlas=40,sample_width = sample_width,atlas_resol = 128)

    # 可视化程序,只可视化atlas[5]了,可以调节
    verts_atlas, faces_atlas, verts_uvs_atlas, faces_uvs_atlas, z_embedding_length = coarse_atlas_import(
            file_path = '/hdd1/zhangkai/256X40',
            mesh_resol = mesh_resol)
    atlas_mesh_list = mesh_from_list(
        verts_atlas, faces_atlas, verts_uvs_atlas, faces_uvs_atlas, z_embedding_length,selected_atlas_indices=selected_atlas_indices)  
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")
        print("WARNING: CPU only, this will be slow!")
    print(len(atlas_mesh_list))
    # prepare the parameters:
    # resol=self.atlas_map_resol, mesh=atlas_mesh_i.to(self.device), atlas_id=atlas_i
    # debug 时 for 循环其实可以去掉
    for atlas_i in range(len(atlas_mesh_list)):

        atlas_map_resol = 128
        atlas_mesh_i = atlas_mesh_list[atlas_i]

        resol = atlas_map_resol
        mesh=atlas_mesh_i.to(device)
        atlas_id = atlas_i


        faces_uvs = mesh.textures.faces_uvs_padded()[0] # (num_view, num_faces, 3) --> (num_faces, 3)
        verts_uvs = mesh.textures.verts_uvs_padded()[0] # (num_view, num_verts, 2) --> (num_verts, 2)
        face_verts_uvs = verts_uvs[faces_uvs]   # (num_faces, 3, 2)

        print(face_verts_uvs.shape)


        num_atlas = len(atlas_mesh_list)
        print(num_atlas)
        atlas_uv_map = np.zeros((resol, resol, 1), dtype=np.float64)
        # TODO:之后要记得GPU上运行
        face_verts_uvs_atlas_image = (face_verts_uvs * (resol - 1.))\
        .type(torch.cuda.LongTensor).cpu().numpy()  # (num_faces, 3, 2)
        # face_verts_uvs_atlas_image = (face_verts_uvs * (resol - 1.)).type(torch.LongTensor).cpu().numpy()  # (num_faces, 3, 2)
        print(face_verts_uvs_atlas_image.shape)
        print(face_verts_uvs_atlas_image.min(), face_verts_uvs_atlas_image.max())

        atlas_uv_map[face_verts_uvs_atlas_image.reshape(-1, 2)[:, 1], face_verts_uvs_atlas_image.reshape(-1, 2)[:, 0], :] = 1.0
        visulize_save_folder = 'visulize'
        # TODO:1.图1
        if not os.path.exists(visulize_save_folder):
            os.makedirs(visulize_save_folder)
        image = atlas_uv_map[..., 0]	

        plt.imsave(os.path.join(visulize_save_folder, 'proj_atlas{}.png'.format(atlas_id)), image)

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

        plt.imsave(os.path.join(visulize_save_folder, 'map_mask{}.png'.format(atlas_id)), image)

        # TODO: 提取uv_map中atlas的边界

        img = cv.imread('visulize/map_mask%s.png'%atlas_id)
        # 转化为灰度图-二值图
        gray_img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        gray_img[gray_img>100] = 255
        gray_img[gray_img<100] = 0

        plt.imsave(os.path.join(visulize_save_folder, 'gray_pic.png'.format(atlas_id)), gray_img)

        # 提取边缘 Canny边缘检测算法
        # https://blog.csdn.net/weixin_42709563/article/details/105700972?utm_medium=distribute.pc_relevant.none-task-blog-2~default~baidujs_title~default-4.pc_relevant_default&spm=1001.2101.3001.4242.3&utm_relevant_index=7
        # print(img.shape)
        # print(gray_img.shape)
        blur = cv.GaussianBlur(gray_img,(5,5),0)
        gaussImg = cv.Canny(blur,5,5)

        plt.imsave(os.path.join(visulize_save_folder, 'edge.png'.format(atlas_id)), gaussImg)


        widthImg = cv.GaussianBlur(gaussImg,(3,3),0)

        gray_edge = widthImg
        gray_edge[gray_edge>0] = 255
        gray_edge[gray_edge<0] = 0
        # gray_edge[50:70,:] = 100
        # print((np.sum(gray_edge==255)+np.sum(gray_edge==0))/128)
        # 阈值二值化

        plt.imsave(os.path.join(visulize_save_folder, 'edge.png'.format(atlas_id)), gray_edge)

        # 内部取点,取交集
        inner_edge = np.zeros(gray_edge.shape)
        inner_edge[(gray_edge==255)*(gray_img==255)] = 255

        plt.imsave(os.path.join(visulize_save_folder, 'inner.png'.format(atlas_id)), inner_edge)

        this_edge_samples = edges_samples[selected_atlas_indices[0]]
        # print(this_edge_samples)
        color_num = 0
        edges_list = []
        print(len(this_edge_samples[4]))
        for i in range(len(this_edge_samples)):
            if len(this_edge_samples[i])!=1:
                edges_list.append(i)
                color_num += 1
        # print(color_num,edges_list)

        sample_edge = np.zeros(inner_edge.shape)
        for i in range(len(edges_list)):
            color_i = 20+(200*i)//color_num
            XY = (this_edge_samples[edges_list[i]]*(128-1)).astype(int)
            # print(XY)
            sample_edge[XY[:,1],XY[:,0]] = color_i

        print(inner_edge.shape)
        plt.imsave(os.path.join(visulize_save_folder, 'edge_samples.png'.format(atlas_id)),sample_edge)

if __name__ == '__main__':
    visulize(sample_width = 3,selected_atlas_indices = [0],mesh_resol = 3)