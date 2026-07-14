function Img = Matrix2Image(ImgMatrix,m,n)
    R   = ImgMatrix(1:m      ,1:n);
    G   = ImgMatrix(m+1:2*m  ,1:n);
    B   = ImgMatrix(2*m+1:3*m,1:n);
    Img = uint8(cat(3,R,G,B));
end