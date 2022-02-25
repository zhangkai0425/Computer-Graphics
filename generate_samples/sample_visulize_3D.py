from generate_3D_samples import *

if __name__ == '__main__':
    edges_samples_GM,edges_samples_NM,valid_id = generate_3D_samples(file_path='/hdd1/zhangkai/256X40',
        mesh_resol=3,atlas_resol=1024,num_atlas = 40,sample_width = 4)
    
    XYZ = np.array([[0,0,0]])
    RGB = np.array([[0,0,0]])
    for i in range(len(edges_samples_GM)):
        if i not in valid_id:
            continue
        this_edge_samples = edges_samples_GM[i]
        edges_list,color_i = get_color(this_edge_samples)
        for j in range(len(edges_list)):

            XYZ = np.concatenate((XYZ,this_edge_samples[edges_list[j]]),axis = 0)
            row = this_edge_samples[edges_list[j]].shape[0]
            rgb = np.zeros((row,3))
            rgb[:,0] = color_i[j][0]
            rgb[:,1] = color_i[j][1]
            rgb[:,2] = color_i[j][2]
            RGB = np.concatenate((RGB,rgb),axis = 0)
    XYZ = XYZ[1:]
    RGB = RGB[1:]
    save2ply(ply_filePath="visulize/3D/samples.ply",xyz_np=XYZ,rgb_np=RGB)




