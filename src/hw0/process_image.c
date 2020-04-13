#include <stdio.h>
#include <string.h>
#include <assert.h>
#include <math.h>
#include "image.h"

float get_pixel(image im, int x, int y, int c)
{
    // TODO Fill this in
    if (x<0) 
		x=0;
	if (x>=im.w)
		x=im.w-1;
	if (y<0)
		y=0;
	if (y>=im.h)
		y=im.h-1;
	return im.data[c*im.h*im.w + y*im.w + x];
}

void set_pixel(image im, int x, int y, int c, float v)
{
    // TODO Fill this in
	if (x<0 || x>=im.w || y<0 || y>=im.h || c<0 || c>im.c) return;
	assert(x<im.w && y<im.h && c<im.c);
	im.data[c*im.w*im.h + y*im.w +x] =v;
}

image copy_image(image im)
{
    image copy = make_image(im.w, im.h, im.c);
	int i,j,k,count=0;
	for (k=0; k<im.c; ++k){
		for (i=0; i<im.w; ++i){
			for (j=0; j<im.h; ++j) {
				copy.data[count]=im.data[count];
				count++;
			}
		}
	}
    // TODO Fill this in
    return copy;
}

image rgb_to_grayscale(image im)
{
    assert(im.c == 3);
    image gray = make_image(im.w, im.h, 1);
	for (int i=0; i<im.w; i++) {
		for (int j=0; j<im.h; j++) {
			gray.data[i+j*im.w]=0.299*get_pixel(im, i, j, 0)+0.587*get_pixel(im, i, j, 1)+0.114*get_pixel(im, i, j, 2);
		}
	}
    // TODO Fill this in
    return gray;
}

void shift_image(image im, int c, float v)
{
    // TODO Fill this in
	for (int i=0; i<im.w; i++) {
		for (int j=0; j<im.h; j++) {
			float val=get_pixel(im, i, j, c);
			set_pixel(im, i, j, c, val+v);
		}
	}
}

void clamp_image(image im)
{
    // TODO Fill this in
	for (int i=0; i<im.w; i++) {
		for (int j=0; j<im.h; j++){
			for (int k=0; k<im.c ;k++) {
				if (get_pixel(im, i, j, k)<0) set_pixel(im, i, j, k, 0);
				else if (get_pixel(im, i, j,k)>1) set_pixel(im, i, j, k,1);
			}
		}
	}
}


// These might be handy
float three_way_max(float a, float b, float c)
{
    return (a > b) ? ( (a > c) ? a : c) : ( (b > c) ? b : c) ;
}

float three_way_min(float a, float b, float c)
{
    return (a < b) ? ( (a < c) ? a : c) : ( (b < c) ? b : c) ;
}

void rgb_to_hsv(image im)
{
    // TODO Fill this in
	assert(im.c==3);
	for (int i=0; i<im.w; i++) {
		for (int j=0; j<im.h; j++) {
			float R= get_pixel(im, i, j, 0);
			float G= get_pixel(im, i, j, 1);
			float B= get_pixel(im, i, j, 2);
			float V = three_way_max(R,G,B);
			float m = three_way_min(R,G,B);
			float C = V-m;
			float S= (R==0 && G==0 && B==0) ? 0 : C/V;
			float H1;
			if (C==0) H1=0;
			else if (V==R) H1 = (G-B)/C;
			else if (V==G) H1 = (B-R)/C + 2;
			else if (V==B) H1 = (R-G)/C + 4;
			float H = (H1<0) ? H1/6 + 1 : H1/6;
			set_pixel(im, i, j, 0, H);
			set_pixel(im, i, j, 1, S);
			set_pixel(im, i, j, 2, V);
		}
	}	 

}

void hsv_to_rgb(image im)
{
    // TODO Fill this in
	assert(im.c==3);
	for (int i=0; i<im.w; i++) {
		for (int j=0; j<im.h; j++) {
			float H = get_pixel(im, i, j, 0);
			float S = get_pixel(im, i, j, 1);
			float V = get_pixel(im, i, j, 2);
			float C = V * S;
			float m = V-C;
			float H1 = H * 6;
			float X = C*(1-fabs(fmod(H1, 2) - 1));
			float R1, G1, B1;
			float R,G,B;
/*			if (H==0) {
				R1=0;
				G1=0;
				B1=0;
			}
*/
			if (0<=H1 && H1<=1) {
				R1=C;
				G1=X;
				B1=0;
			}
			else if (H1<=2) {
				R1=X;
				G1=C;
				B1=0;
			}
			else if (H1<=3) {
				R1=0;
				G1=C;
				B1=X;
			}
			else if (H1<=4) {
				R1=0;
				G1=X;
				B1=C;
			}
			else if (H1<=5) {
				R1=X;
				G1=0;
				B1=C;
			}
			else if (H1<=6) {
				R1=C;
				G1=0;
				B1=X;
			}
			R=R1+m;
			G=G1+m;
			B=B1+m;
			set_pixel(im, i, j, 0 ,R);
			set_pixel(im, i, j, 1, G);
			set_pixel(im, i, j, 2, B);
		}
	}

}

void scale_image(image im, int c, float v)
{
    for (int i=0; i<im.w; i++) {
        for (int j=0; j<im.h; j++) {
            float val=get_pixel(im, i, j, c);
			float newVal=val*v;
            set_pixel(im, i, j, c, newVal);
        }
    }
}

/*
void rgb_to_hcl(image im)
{
    float HCLgamma=3;
	float HCLy0=100;
	float HCLmaxL=0.530454533953517;
	float PI=3.1415926536;
	for (int i=0; i<im.w; i++) {
		for (int j=0; j<im.h; j++) {
			
    		float H = 0;
			float R=get_pixel(im, i, j, 0);
			float G=get_pixel(im, i, j, 1);
			float B=get_pixel(im, i, j, 2);
    		float U = three_way_min(R, G, B);
    		float V = three_way_max(R, G, B);
    		float Q = HCLgamma / HCLy0;
    		float y = V - U;
			float x;
			float z;
    		if (y != 0)
    		{
      			H = atan2(G - B, R - G) / PI;
      			Q *= U / V;
    		}
    		Q = exp(Q);
    		x = frac(H / 2 - min(frac(H), frac(-H)) / 6);
    		y *= Q;
    		z = lerp(-U, V, Q) / (HCLmaxL * 2);
    		set_pixel(im, i, j, 0, x);
			set_pixel(im, i, j, 1, y);
			set_pixel(im, i, j, 2, z);
		}
	}
}
*/
