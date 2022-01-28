import os
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
        Intersection_l = [0]
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
    face_idx = faces.verts_idx.to(device)
    face_idx += 1
    # get atlas number and atlas_faces number of each atlas
    atlas_num,atlas_faces = get_atlas_num(objFilePath)
    begin,end = getbegin_end(atlas_faces)
    # build the list of shared points of each atlas (Symmetric Matrices)
    edges = [] # atlas_num * atlas_num
    for i in range(atlas_num):
        Tensor_this = face_idx[begin[i]:end[i]]
        each_list = []
        for j in range(atlas_num):
            if i == j:
                each_list += [[0]]
                continue
            if j < i:
                each_list += [edges[j][i]]
            else:
                Tensor_that = face_idx[begin[j]:end[j]]
                each_list += [getIntersection_l(TensorToList(Tensor_this),TensorToList(Tensor_that))]
        edges.append(each_list)
    return edges

if __name__ == '__main__':
    edges = getedges('xxx.obj')
    print(len(edges))
    print(type(edges))
    print(edges)