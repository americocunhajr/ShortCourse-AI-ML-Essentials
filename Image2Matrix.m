function [ImgMatrix,m,n] = Image2Matrix(Img)
    [m,n,c]   = size(Img);
    R         = double(Img(:,:,1));
    G         = double(Img(:,:,2));
    B         = double(Img(:,:,3));
    ImgMatrix = [R; G; B];
end