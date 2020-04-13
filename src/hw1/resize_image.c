#include <math.h>
#include "image.h"


float nn_interpolate(image im, float x, float y, int c)
{
    // TODO Fill in
	int x_, y_;
	if (x<0) x_=0;
	else {
		float intNum=floor(x);
		x_ = (x - intNum <=0.5)?intNum:intNum+1;
	}
	if (y<0) 
		y_=0;
    else  {
        float intNum =floor(y);
		y_ = (y - intNum <0.5)?intNum:intNum+1;
	}
	return im.data[c*im.h*im.w + y_*im.w + x_];
}



image nn_resize(image im, int w, int h)
{
    // TODO Fill in (also fix that first line)
	image newimage = make_image(w, h, im.c);
	float a_w = (float)(im.w) / w;
	// printf("a_w = %f\n", a_w);
	float b_w = (float)(im.w) /(2*w) - 0.5;
	float a_h = (float)(im.h) / h;
	float b_h = (float)(im.h) /(2*h) - 0.5;
	// printf("a_w = %f, maxX = %f, w = %i\n", a_w, maxX, w);
	for (int k=0; k<im.c; k++) {
		for (int x = 0; x<w; x++) {
			for (int y = 0; y<h; y++) {
				float x1 = a_w*x + b_w;
				float y1 = a_h*y + b_h;
				newimage.data[k*w*h+y*w+x] = nn_interpolate(im, x1, y1, k);
			}
		}
	}
    return newimage;
}

float bilinear_interpolate(image im, float x, float y, int c)
{
    // TODO
	int left, right, top, bot;
	left = (x<0)?0:floor(x);
	if (x<0 || x>im.w-1 || left==x)
		right = left;
	else
		right = left+1;
	top = (y<0)?0:floor(y);
	if (y<0 || y>im.h-1 || top == y)
		bot = top;
	else
		bot = top + 1;
	float v1 = im.data[c*im.h*im.w + top*im.w + left];
	float v2 = im.data[c*im.h*im.w + top*im.w + right];
	float v3 = im.data[c*im.h*im.w + bot*im.w + left];
	float v4 = im.data[c*im.h*im.w + bot*im.w + right];
	float d1 = (left==right)?0.5:x-left;
	float d2 = (left==right)?0.5:right-x;
	float d3 = (top==bot)?0.5:y-top;
	float d4 = (top==bot)?0.5:bot-y;	
	float output = v1*d2*d4 + v2*d1*d4 + v3*d2*d3 + v4*d1*d3;
    return output;
}

image bilinear_resize(image im, int w, int h)
{
    // TODO Fill in (also fix that first line)
    image newimage = make_image(w, h, im.c);
    float a_w = (float)(im.w) / w;
    // printf("a_w = %f\n", a_w);
    float b_w = (float)(im.w) /(2*w) - 0.5;
    float a_h = (float)(im.h) / h;
    float b_h = (float)(im.h) /(2*h) - 0.5;
    // printf("a_w = %f, maxX = %f, w = %i\n", a_w, maxX, w);
    for (int k=0; k<im.c; k++) {
        for (int x = 0; x<w; x++) {
            for (int y = 0; y<h; y++) {
                float x1 = a_w*x + b_w;
                float y1 = a_h*y + b_h;
                newimage.data[k*w*h+y*w+x] = bilinear_interpolate(im, x1, y1, k);
            }
        }
    }
    return newimage;

    return make_image(1,1,1);
}

