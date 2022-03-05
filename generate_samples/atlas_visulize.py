import os
import pdb
import torch
import numpy as np
import matplotlib.pyplot as plt
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

'''transform a tensor to a list(reshape as 1 dim)'''
def TensorToList(tensor):
    list = tensor.reshape(-1).cpu().numpy().tolist()
    return list

def visulize_atlas(file_path = '/hdd1/zhangkai/256X40',mesh_resol = 3,atlas_resol = 128,atlas_id = 0):
    # Set the device
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")
        print("WARNING: CPU only, this will be slow!") 
    atlas_num,atlas_faces = get_atlas_num(file_path+'/%s.obj'%mesh_resol)
    begin,end = getbegin_end(atlas_faces)
    verts, faces, aux = load_obj(file_path+'/%s.obj'%mesh_resol)
    face_idx = faces.verts_idx.to(device)
    uv_idx = faces.textures_idx.to(device)
    verts_uvs = aux.verts_uvs
    verts_uvs = (verts_uvs * (atlas_resol - 1.)).type(torch.cuda.LongTensor).cpu().numpy()   

    # # debug验证: 
    face_idx = face_idx[begin[atlas_id]:end[atlas_id]]
    uv_idx = uv_idx[begin[atlas_id]:end[atlas_id]]
    verts_to_uv = np.concatenate((face_idx.reshape(-1).cpu().numpy()[:,None],uv_idx.reshape(-1).cpu().numpy()[:,None]),axis=1)
    
    face_idx = face_idx.reshape(-1).cpu().numpy()
    # verts_to_uv = torch.from_numpy(verts_to_uv)
    uv_list = []
    for i in range(len(face_idx.reshape(-1))):
        uv = verts_to_uv[verts_to_uv[:,0]==face_idx.reshape(-1)[i]][:,1]
        uv_list += list(set(uv.tolist()))

    print(len(uv_list))
    print(len(face_idx))

    # Tensor_this = uv_idx[begin[atlas_id]:end[atlas_id]]
    # Tensor_list = TensorToList(Tensor_this)

    picture = np.zeros((atlas_resol,atlas_resol))
    point_set = verts_uvs[uv_list,:]
    picture[point_set[:,1],point_set[:,0]] = 255
    visulize_save_folder = 'visulize/atlas'
    picture[30,30] = 255
    # widthImg = cv.GaussianBlur(gaussImg,(3,3),0)
    if not os.path.exists(visulize_save_folder):
        os.makedirs(visulize_save_folder)
    plt.imsave(os.path.join(visulize_save_folder, 'atlas%s.png'%atlas_id), picture)

if __name__ == '__main__':
    visulize_atlas(file_path = '/hdd1/zhangkai/256X40',mesh_resol = 3,atlas_resol = 128,atlas_id = 0)