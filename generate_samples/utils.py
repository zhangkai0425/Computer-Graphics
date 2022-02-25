import os
import torch
import numpy as np
from plyfile import PlyData, PlyElement
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

'''
用于3D可视化,原来uv_warping.py中的函数,为了不需要在原文件夹下运行,因此单独拿出来了
'''

def initialize_atlas_uv_map(resol=128, mesh=None):
    """
        Args:
            mesh: pytorch3d Meshes object.
    """
    faces_uvs = mesh.textures.faces_uvs_padded()[0] # (num_view, num_faces, 3) --> (num_faces, 3)
    verts_uvs = mesh.textures.verts_uvs_padded()[0] # (num_view, num_verts, 2) --> (num_verts, 2)

    face_verts_uvs = verts_uvs[faces_uvs]   # (num_faces, 3, 2)

    atlas_uv_map = np.zeros((resol, resol, 1), dtype=np.float64)
    face_verts_uvs_atlas_image = (face_verts_uvs * (resol - 1.))\
        .type(torch.cuda.LongTensor).cpu().numpy()  # (num_faces, 3, 2)

    atlas_uv_map[face_verts_uvs_atlas_image.reshape(-1, 2)[:, 1], face_verts_uvs_atlas_image.reshape(-1, 2)[:, 0], :] = 1.0


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

    if vis_mask_flag.max() == 0.0:
        raise ValueError('pixels belong to no projected triangle facets !')

    pixel_facet_IDM = flag_total.argmax(axis=-1)

    # use vector cross-product to calculate the barycentric coordinates.
    area = np.cross(-CA, AB) + 1e-8
    indices = np.argwhere(flag_total)
    area_selected = area[0, indices[:, -1]]  # (N_Sf)
    w0 = cross_A[outlier_mask][flag_total] / area_selected   # (N_Sf)
    w1 = cross_B[outlier_mask][flag_total] / area_selected
    w2 = cross_C[outlier_mask][flag_total] / area_selected
    barycentric = np.concatenate((w0[..., None], w1[..., None], w2[..., None]), axis=-1)
    checker = barycentric.sum(axis=-1)
    
    return outlier_mask, pixel_facet_IDM, barycentric

def generate_init_NM(barycentric, outlier_mask, pixel_facet_IDM,
						resol=128, mesh=None, atlas_id=None,debug_visualize=True):
	"""
		compute the normal map for the input coarse triangulation.
		Returns:
			pixel_interpolate_normals: normals for every pixel in 2D-atlas-uv-map.
				shape: (N_Sf, 3)
	"""
	faces_uvs = mesh.textures.faces_uvs_padded()[0]  # (num_view, num_faces, 3) --> (num_faces, 3)
	verts_uvs = mesh.textures.verts_uvs_padded()[0]  # (num_view, num_verts, 2) --> (num_verts, 2)
	verts = mesh.verts_padded()[0]
	faces = mesh.faces_padded()[0]
	verts_normals = mesh.verts_normals_padded()[0]
	faces_normals = mesh.faces_normals_padded()[0]

	face_verts_normals = verts_normals[faces]   # (num_faces, 3, 3)
	pixel_verts_normals = face_verts_normals[pixel_facet_IDM].cpu().numpy() # (N_Sf, 3, 3)
	pixel_faces_normals = faces_normals[pixel_facet_IDM].cpu().numpy()  # (N_Sf, 3)
	pixel_interpolate_normals = (pixel_verts_normals * barycentric[..., None]).sum(axis=1)

	# image_faces_normal = np.zeros((resol, resol, 3), dtype=np.float64).reshape(-1, 3)
	
	image_verts_normal = np.zeros((resol, resol, 3), dtype=np.float64).reshape(-1, 3)

	image_verts_normal[outlier_mask] = pixel_interpolate_normals

	image_verts_normal = image_verts_normal.reshape(resol,resol,3)
	
	# print('Today_debug here:',image_verts_normal.shape)
	# if debug_visualize:
	# 	debug_save_folder = os.path.join('debug', 'atlas_normal_map')
	# 	if not os.path.exists(debug_save_folder):
	# 		os.makedirs(debug_save_folder)

	# 	image_verts_normal[outlier_mask] = pixel_interpolate_normals


	# 	image_verts_normal -= image_verts_normal.min()
	# 	image_verts_normal /= image_verts_normal.max()
	# 	plt.imsave(os.path.join(debug_save_folder, 'verts_normal_atlas{}.png'.format(atlas_id)),
	# 				image_verts_normal.reshape(resol, resol, 3))

	# 	image_faces_normal[outlier_mask] = pixel_faces_normals
	# 	image_faces_normal -= image_faces_normal.min()
	# 	image_faces_normal /= image_faces_normal.max()
	# 	plt.imsave(os.path.join(debug_save_folder, 'faces_normal_atlas{}.png'.format(atlas_id)),
	# 				image_faces_normal.reshape(resol, resol, 3))

	return image_verts_normal


def generate_init_GM(barycentric, outlier_mask, pixel_facet_IDM,
						resol=128, mesh=None, atlas_id=None, debug_visualize=True):
	"""
		compute the geometry image for the input coarse triangulation.
		Returns:
			pixel_interpolate_coords: global 3D coords for every pixel in 2D-atlas-uv-map.
				shape: (N_Sf, 3)
	"""
	faces_uvs = mesh.textures.faces_uvs_padded()[0] # (num_view, num_faces, 3) --> (num_faces, 3)
	verts_uvs = mesh.textures.verts_uvs_padded()[0] # (num_view, num_verts, 2) --> (num_verts, 2)
	verts = mesh.verts_padded()[0]
	faces = mesh.faces_padded()[0]

	face_verts_uvs = verts_uvs[faces_uvs]   # (num_faces, 3, 2)
	face_verts = verts[faces]   # (num_faces, 3, 3)

	pixel_face_verts_uvs = face_verts_uvs[pixel_facet_IDM].cpu().numpy()  # (N_Sf, 3, 2)
	pixel_face_verts = face_verts[pixel_facet_IDM].cpu().numpy()  # (N_Sf, 3, 3)
	pixel_interpolate_coords = (pixel_face_verts * barycentric[..., None]).sum(axis=1)  # (N_Sf, 3)
	pixel_interpolate_uv_coords = (pixel_face_verts_uvs * barycentric[..., None]).sum(axis=1)  # (N_Sf, 3)

	image = np.zeros((resol, resol, 3), dtype=np.float64).reshape(-1, 3)
	image_uv = np.zeros((resol, resol, 3), dtype=np.float64).reshape(-1, 3)
	image_interpolate = np.zeros((resol, resol, 3), dtype=np.float64).reshape(-1, 3)
	image_uv_interpolate = np.zeros((resol, resol, 3), dtype=np.float64).reshape(-1, 3)

	image_interpolate[outlier_mask] = pixel_interpolate_coords

	image_interpolate = image_interpolate.reshape(resol,resol,3)

	# print("today debug here:",image_interpolate.shape)

	# if debug_visualize:
	# 	debug_save_folder = os.path.join('debug', 'atlas_geo_map')
	# 	if not os.path.exists(debug_save_folder):
	# 		os.makedirs(debug_save_folder)
	# 	image_interpolate[outlier_mask] = pixel_interpolate_coords
	# 	# TODO:
	# 	# abandone the large value in z-axis for high contrast visualization.
	# 	image_interpolate[..., -1] = 0.0
	# 	# image_interpolate[..., -1] /= image_interpolate[..., -1].mean()
	# 	image_interpolate -= image_interpolate.min()
	# 	image_interpolate /= image_interpolate.max()
	# 	plt.imsave(os.path.join(debug_save_folder, 'face_verts_interpolate_atlas{}.png'.format(atlas_id)),
	# 				image_interpolate.reshape(resol, resol, 3))

	# 	uv_zeros = np.zeros((pixel_face_verts_uvs.shape[0], 1), dtype=np.float64)
	# 	image_uv_interpolate[outlier_mask] = np.concatenate((
	# 		(pixel_face_verts_uvs * barycentric[..., None]).sum(axis=1), uv_zeros), axis=-1)
	# 	image_uv_interpolate -= image_uv_interpolate.min()
	# 	image_uv_interpolate /= image_uv_interpolate.max()
	# 	plt.imsave(os.path.join(debug_save_folder, 'face_uvs_verts_interpolate_atlas{}.png'.format(atlas_id)),
	# 				image_uv_interpolate.reshape(resol, resol, 3))

	# 	# blend 3 verts coords value for visualization
	# 	image[outlier_mask] = pixel_face_verts.mean(axis=1)
	# 	image_uv[outlier_mask] = np.concatenate((pixel_face_verts_uvs.mean(axis=1), uv_zeros), axis=-1)

	# 	# abandone the large value in z-axis for high contrast visualization.
	# 	image[..., -1] = 0.0
	# 	# image[..., -1] /= image[..., -1].mean()
	# 	image = image - image.min()
	# 	image = image / image.max()
	# 	plt.imsave(os.path.join(debug_save_folder, 'face_verts_GIM_atlas{}.png'.format(atlas_id)),
	# 				image.reshape(resol, resol, 3))

	# 	image_uv = image_uv - image_uv.min()
	# 	image_uv = image_uv / image_uv.max()
	# 	plt.imsave(os.path.join(debug_save_folder, 'face_uvs_verts_GIM_atlas{}.png'.format(atlas_id)),
	# 				image_uv.reshape(resol, resol, 3))

	return image_interpolate
'''
用于保存3D点云的函数
'''
def save2ply(ply_filePath, xyz_np, rgb_np=None, normal_np=None):
    """
    save data to ply file, xyz (rgb, normal)

    ---------
    inputs:
        xyz_np: (N_voxels, 3)
        rgb_np: None / (N_voxels, 3)
        normal_np: None / (N_voxels, 3)

        ply_filePath: 'xxx.ply'
    outputs:
        save to .ply file
    """
    N_voxels = xyz_np.shape[0]
    atributes = [('x', '<f4'), ('y', '<f4'), ('z', '<f4')]
    if normal_np is not None:
        atributes += [('nx', '<f4'), ('ny', '<f4'), ('nz', '<f4')]
    if rgb_np is not None:
        atributes += [('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]
    saved_pts = np.zeros(shape=(N_voxels,), dtype=np.dtype(atributes))

    saved_pts['x'], saved_pts['y'], saved_pts['z'] = xyz_np[:, 0], xyz_np[:, 1], xyz_np[:, 2]
    if rgb_np is not None:
        # print('saveed', saved_pts)
        saved_pts['red'], saved_pts['green'], saved_pts['blue'] = rgb_np[:, 0], rgb_np[:, 1], rgb_np[:, 2]
    if normal_np is not None:
        saved_pts['nx'], saved_pts['ny'], saved_pts['nz'] = normal_np[:, 0], normal_np[:, 1], normal_np[:, 2]

    el_vertex = PlyElement.describe(saved_pts, 'vertex')
    outputFolder = os.path.dirname(ply_filePath)
    if not os.path.exists(outputFolder):
        os.makedirs(outputFolder)
    PlyData([el_vertex]).write(ply_filePath)
    # print('saved ply file: {}'.format(ply_filePath))
    return 1

'''
上色函数,用于可视化
'''
def get_color(this_edge_samples):
    color_num = 0
    edges_list = []
    color_i = []
    for i in range(len(this_edge_samples)):
        if len(this_edge_samples[i])!=1:
            edges_list.append(i)
            color_num += 1
    for i in range(len(edges_list)):
        color_i.append([20+(200*i)//color_num,255-20-(200*i)//color_num,i*10%255])
    return edges_list,color_i