import os
import pdb
import torch
import numpy as np
from pytorch3d.io import load_obj, save_obj

'''find the number of atlas and the faces of each atlas'''
def get_atlas_num(objFilePath):
    with open(objFilePath) as file:
        atlas_num = 0
        line_num = 0
        atlas_faces = []
        while 1:
            line = file.readline()
            if not line:
                endline = line_num
                atlas_faces.append(endline)
                break
            line_num += 1
            strs = line.split(" ")
            if strs[0] == "usemtl":
                atlas_num += 1
                atlas_faces.append(line_num)
        for i in range(len(atlas_faces)-1):
            atlas_faces[i] = atlas_faces[i+1] - atlas_faces[i] - 1
        atlas_faces = atlas_faces[0:-1]
        atlas_faces[-1] += 1
    return atlas_num,atlas_faces

'''get begin and end index'''
def getbegin_end(atlas_faces):
    begin = []
    end = []
    for i in range(len(atlas_faces)):
        begin.append(sum(atlas_faces[0:i]))
        end.append(sum(atlas_faces[0:i+1]))
    return begin,end

'''get the Intersection of two list and return the Intersection as list type'''
def getIntersection_l(list_a,list_b):
    Intersection_l = list(set(list_a).intersection(set(list_b)))
    if len(Intersection_l)==0:
        Intersection_l = [-1]
    return Intersection_l

'''transform a tensor to a list(reshape as 1 dim)'''
def TensorToList(tensor):
    list = tensor.reshape(-1).cpu().numpy().tolist()
    return list

'''find the edges of each atlas and return the list of the edges of each atlas'''
''' 
    input:
        file_name : xxx.obj
    return:
        list of shared points of each atlas.
        size : (num_atlas,num_atlas)
        [
         [[0],[],[],[],[],[],...],
         [[],[0],[],[],[],[],...],
         ......
        ]
'''
def getedges(objFilePath):
    # get devices: if gpu not available,use cpu
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")
        print("WARNING: CPU only, this will be slow!")
    # load .obj file and load face_verts_idx to device 
    objFilePath = os.path.join(objFilePath)
    verts, faces, aux = load_obj(objFilePath)
    # TODO:怀疑这一行应该删去
    # face_idx = faces.verts_idx.to(device)
    # print("look here:::!",face_idx.shape,face_idx.max())
    # debug
    face_idx = faces.verts_idx.to(device)
    uv_idx = faces.textures_idx.to(device)
    face_idx += 1
    uv_idx += 1
    
    # get atlas number and atlas_faces number of each atlas
    atlas_num,atlas_faces = get_atlas_num(objFilePath)
    begin,end = getbegin_end(atlas_faces)
    # build the list of shared points of each atlas (Symmetric Matrices)
    edges = [] # atlas_num * atlas_num
    for i in range(atlas_num):
        Tensor_this = face_idx[begin[i]:end[i]]
        face_part_idx = Tensor_this
        uv_part_idx = uv_idx[begin[i]:end[i]]
        verts_to_uv = np.concatenate((face_part_idx.reshape(-1).cpu().numpy()[:,None],uv_part_idx.reshape(-1).cpu().numpy()[:,None]),axis=1)
        each_list = []
        for j in range(atlas_num):
            if i == j:
                each_list += [[-1]]
                continue
            if j < i:
                each_list += [edges[j][i]]
            else:
                Tensor_that = face_idx[begin[j]:end[j]]
                verts_list = getIntersection_l(TensorToList(Tensor_this),TensorToList(Tensor_that))
                if verts_list == [-1]:
                    each_list += [[-1]]
                    continue
                uv_list = []
                for verts in verts_list:
                    uv = verts_to_uv[verts_to_uv[:,0]==verts][:,1]
                    uv_list += list(set(uv.tolist()))
                print("debug here:",len(verts_list),len(uv_list))
                each_list += [uv_list]
        edges.append(each_list)
    ### 新加的部分

    valid_id = []
    for id in range(atlas_num):
        valid = False
        for j in range(atlas_num):
            if edges[id][j] != [-1]:
                valid = True
                break
        if valid == True:
            valid_id.append(id)
    # verts_idx =face_idx
    # verts_to_uv = np.concatenate((verts_idx.reshape(-1).cpu().numpy()[:,None],uv_idx.reshape(-1).cpu().numpy()[:,None]),axis=1)
    return edges,valid_id

if __name__ == '__main__':
    mesh_resol = 3
    edges,valid_id = getedges('/hdd1/zhangkai/256X40/%s.obj'%mesh_resol)
    # print(len(edges))
    # print(type(edges))
    # print(len(edges[2]))
    # print(len(edges))
    # print(valid_id)
    print(edges[0])
    X = len(edges)
    Y = len(edges[0])
    print("X=",X,",Y=",Y)
    Adjacency = np.zeros((X,Y))
    for i in range(X):
        for j in range(Y):
            if edges[i][j]!=[-1]:
                Adjacency[i][j] = 1
            elif i==j:
                Adjacency[i][j] = 1
    Adjacency = torch.from_numpy(Adjacency).int()
    # print(Adjacency[0])
    torch.save(Adjacency,"/hdd1/zhangkai/record/experiment/caches/3_128/adjacency/Adjacency_%s_obj.pt"%mesh_resol)
    print("save done!")
    # for i in range(len(edges[0])):
    #     if edges[0][i] != [-1]:
    #         print(i)
    # print(verts_to_uv)
