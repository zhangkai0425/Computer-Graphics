from generate_samples import *
if __name__ == '__main__':
    edges_samples = torch.load("samples_1024.pt")
    # calculate the number of edges_samples:Nb
    Nb = 0
    for i in range(40):
        for j in range(40):
            if isinstance(edges_samples[i][j],list) or len(edges_samples[i][j])<100 or len(edges_samples[j][i])<100:
                continue
            else:
                Nb += 1
    Nb //= 2
    print(Nb)
    Edge_Id = np.zeros((Nb,2)).astype(int)
    pair = 0
    for i in range(40):
        for j in range(40):
            if j>=i:
                continue
            elif isinstance(edges_samples[i][j],list) or len(edges_samples[i][j])<100 or len(edges_samples[j][i])<100:
                continue
            else:
                Edge_Id[pair][0] = i
                Edge_Id[pair][1] = j
                pair += 1
    
    Inner_Edge = []
    Out_Edge = []
    for pair in range(Nb):
        Inner_Edge += [edges_samples[Edge_Id[pair][0]][Edge_Id[pair][1]]]
        Out_Edge += [edges_samples[Edge_Id[pair][1]][Edge_Id[pair][0]]]
    ALL_Edge = []
    ALL_Edge.append(Inner_Edge)
    ALL_Edge.append(Out_Edge)
    torch.save(Edge_Id,"Edge_Id_1024.pt")
    torch.save(ALL_Edge,"ALL_Edge_1024.pt")