### Atlas 采样点程序使用及可视化指南:

#### 环境配置：

主要需要pytorch3d即可，实验室中SurRF环境即可满足要求

若本地无SurRF环境，运行

```bash
conda env create -f SurRF.yml
```

有SurRF环境之后，运行

```bash
conda activate SurRF
```

#### 程序说明：

##### 1.严格的的边界点获取程序:

`getedges.py`包括了如何根据输入的`xxx.obj`文件获取严格的边界点

运行`python getedges.py`可以获得所有atlas相邻关系的列表矩阵，存储严格的各atlas的边界点集合

可设置的参数：`xxx.obj`

##### 2.拓展的边界采样程序:

`generate_samples.py`包括如何如何根据输入的`xxx.obj`文件获取采样后的边界点集合并输出的函数

运行`python generate_samples.py `可以获取包括相邻关系的所有atlas边界的边界采样点集合

可设置的参数：`xxx.obj,mesh_resol,atlas_resol,atlas_selected,sample_width`

#### 可视化效果：

##### 1.可视化效果：

为了方便可视化，可视化的时候，仅仅针对一个atlas进行可视化。期望达到的可视化效果是获得某个atlas边界采样图和原图，并通过不同颜色区分集合的边界

##### 2.可视化程序：

运行`python sample_visualize.py`可视化结果，结果保存在`visualize`文件夹中，可以据此验证`generate_samples.py`程序运行是否正确

运行`python sample_visualize_3D.py`可视化3D结果，结果保存在`visualize/3D`文件夹中，可以据此验证`generate_samples.py`程序运行是否正确

运行`python edges_visulize.py`可视化边界点uv坐标结果，结果保存在`visualize/edges`文件夹中，可以据此验证`generate_samples.py`程序运行是否正确
