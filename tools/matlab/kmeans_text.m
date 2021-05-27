function C = kmeans_text(I_index,k)
[row,col]=find(I_index~=0);%取出文本区域
index_t = [row,col];
[idx_t,C_T] = kmeans(index_t,k/2);%聚类

B=[1 1 1
   1 1 1
   1 1 1];
for i = 1:30
    I_index = imdilate(I_index,B);
end

[row,col]=find(I_index==0);%取出图像区域
index_p = [row,col];
[idx_p,C_P] = kmeans(index_p,k/2);
C = [C_T;C_P];

