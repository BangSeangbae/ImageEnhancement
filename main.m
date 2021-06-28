%% Main 
filename = strcat('Boat.png');
Img = imread(filename);

[v, h, c] = size(Img);

Red = double(Img(:,:,1)); Green = double(Img(:,:,2)); Blue = double(Img(:,:,3));
Y = (0.299 * Red) + (0.587 * Green) + (0.114 * Blue);
U = -(0.16874 * Red) - (0.3313 * Green) + (0.500 * Blue) + 128;
V =  (0.500 * Red) - (0.4187 * Green) - (0.0813 * Blue) + 128;   

Yout = Y;
Y = Y(1:floor(v/32)*32, 1:floor(h/32)*32,:);

%% Enhancement
Yout_Proposed = enhancment(Y,1.2,0.8,16);
Yout(1:floor(v/32)*32, 1:floor(h/32)*32) = Yout_Proposed(1:floor(v/32)*32, 1:floor(h/32)*32);
Red = Yout + 1.402*(V - 128);
Green = Yout - 0.34414*(U - 128) - 0.71414*(V - 128);
Blue = Yout + 1.772*(U - 128);
ImgEnh = uint8(cat(3,Red, Green, Blue));

outfilename = strcat('EnhOut.png');   
imwrite(ImgEnh,outfilename);       
    