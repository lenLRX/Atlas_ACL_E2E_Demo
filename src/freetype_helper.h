#ifndef __FREETYPE_HELPER_H__
#define __FREETYPE_HELPER_H__

#include <string>

class YUVColor;
class YUV420SPImage;

void RenderText(int x, int y, const std::string& text, const YUVColor* color, YUV420SPImage* image);

#endif//__FREETYPE_HELPER_H__